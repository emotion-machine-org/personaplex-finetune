# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""
PersonaPlex/Moshi live inference server using Pipecat + WebRTC.

Replaces the WebSocket-based server.py with WebRTC transport for
proper real-time audio (jitter buffering, PLC, no manual packet management).

Usage:
    python -m moshi.pipecat_server --device cuda \
        --voice-prompt-dir /path/to/voices \
        --host 0.0.0.0 --port 8998
"""

import argparse
import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece
import torch
from huggingface_hub import hf_hub_download

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    StartFrame,
    EndFrame,
    CancelFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCRequestHandler,
    SmallWebRTCRequest,
    SmallWebRTCPatchRequest,
    IceCandidate,
    ConnectionMode,
)

from .models import loaders, MimiModel, LMModel, LMGen
from .utils.logging import setup_logger

logger = setup_logger(__name__)

SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples = 80ms


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


@dataclass
class SessionConfig:
    """Per-session configuration passed from the client."""
    text_prompt: str = (
        "You are Emma, a friendly and supportive AI companion. Your role is to "
        "have natural, engaging conversations \u2014 listening actively, offering "
        "thoughtful responses, and helping the user think through whatever is on "
        "their mind. Be honest about what you do not know rather than guessing or "
        "making things up. You do not have access to day-to-day information such "
        "as what you did today, recent events in your own life, or personal "
        "experiences \u2014 if the user asks questions like these, gently redirect "
        "the conversation back to them. Be respectful, avoid offensive or hurtful "
        "language, and never pressure the user or make judgments about their choices."
    )
    voice_prompt: str = "NATF0.pt"
    audio_temperature: float = 0.65
    text_temperature: float = 0.6
    audio_topk: int = 150
    text_topk: int = 20
    greeting: str = "Hello there."
    seed: int = -1


class ModelState:
    """Shared model state. Wraps loading and warmup — one instance per server."""

    def __init__(
        self,
        mimi: MimiModel,
        other_mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        device: torch.device,
        voice_prompt_dir: Optional[str] = None,
        default_greeting: str = "",
    ):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.default_greeting = default_greeting
        self.frame_size = FRAME_SIZE

        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * FRAME_RATE),
            sample_rate=SAMPLE_RATE,
            device=device,
            frame_rate=FRAME_RATE,
        )

        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def warmup(self):
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])
        if self.device.type == "cuda":
            torch.cuda.synchronize()


class MoshiProcessor(FrameProcessor):
    """
    Full-duplex audio processor wrapping Moshi/PersonaPlex inference.

    Receives InputAudioRawFrame (PCM int16 @ 24kHz mono) from WebRTC,
    accumulates into 1920-sample frames, runs Mimi encode → LM step → Mimi decode,
    and pushes OutputAudioRawFrame back to the client.
    """

    def __init__(self, model: ModelState, config: SessionConfig, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.config = config
        self._pcm_buffer = np.array([], dtype=np.float32)
        self._session_active = False

    async def _init_session(self):
        """Configure model for this session and run system prompts."""
        m = self.model
        cfg = self.config

        # Set sampling params
        m.lm_gen.temp = cfg.audio_temperature
        m.lm_gen.temp_text = cfg.text_temperature
        m.lm_gen.top_k = max(1, cfg.audio_topk)
        m.lm_gen.top_k_text = max(1, cfg.text_topk)

        # Load voice prompt
        if m.voice_prompt_dir and cfg.voice_prompt:
            vp_path = os.path.join(m.voice_prompt_dir, cfg.voice_prompt)
            if not os.path.exists(vp_path):
                logger.error(f"Voice prompt not found: {vp_path}")
            elif m.lm_gen.voice_prompt != vp_path:
                if vp_path.endswith(".pt"):
                    logger.info("Skipping .pt voice prompt (incompatible with finetuned model)")
                    m.lm_gen.voice_prompt = vp_path
                    m.lm_gen.voice_prompt_audio = None
                    m.lm_gen.voice_prompt_embeddings = None
                    m.lm_gen.voice_prompt_cache = None
                else:
                    m.lm_gen.load_voice_prompt(vp_path)

        # Text prompt
        if cfg.text_prompt:
            m.lm_gen.text_prompt_tokens = m.text_tokenizer.encode(
                wrap_with_system_tags(cfg.text_prompt)
            )
        else:
            m.lm_gen.text_prompt_tokens = None

        # Greeting
        greeting_text = cfg.greeting or m.default_greeting
        if greeting_text:
            m.lm_gen.greeting_tokens = m.text_tokenizer.encode(greeting_text)
        else:
            m.lm_gen.greeting_tokens = None

        # Seed
        if cfg.seed >= 0:
            seed_all(cfg.seed)

        # Reset streaming state
        m.mimi.reset_streaming()
        m.other_mimi.reset_streaming()
        m.lm_gen.reset_streaming()

        # Run system prompts (voice → silence → text → silence)
        await m.lm_gen.step_system_prompts_async(m.mimi)
        m.mimi.reset_streaming()

        # Queue greeting
        if m.lm_gen.greeting_tokens:
            m.lm_gen.prepare_greeting()

        logger.info("Session initialized, system prompts done")
        self._session_active = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Acquire model lock and initialize session
            await self.model.lock.acquire()
            try:
                await self._init_session()
            except Exception:
                self.model.lock.release()
                raise
            await self.push_frame(frame, direction)

        elif isinstance(frame, (EndFrame, CancelFrame)):
            # Release model lock on disconnect
            self._session_active = False
            self._pcm_buffer = np.array([], dtype=np.float32)
            if self.model.lock.locked():
                self.model.lock.release()
            await self.push_frame(frame, direction)

        elif isinstance(frame, InputAudioRawFrame):
            if not self._session_active:
                return

            # Convert int16 PCM bytes → float32 [-1, 1]
            pcm_int16 = np.frombuffer(frame.audio, dtype=np.int16)
            pcm_f32 = pcm_int16.astype(np.float32) / 32768.0
            self._pcm_buffer = np.concatenate([self._pcm_buffer, pcm_f32])

            # Process complete 80ms frames
            while len(self._pcm_buffer) >= self.model.frame_size:
                chunk = self._pcm_buffer[: self.model.frame_size]
                self._pcm_buffer = self._pcm_buffer[self.model.frame_size :]

                chunk_t = torch.from_numpy(chunk).to(device=self.model.device)[None, None]
                codes = self.model.mimi.encode(chunk_t)
                _ = self.model.other_mimi.encode(chunk_t)

                for c in range(codes.shape[-1]):
                    tokens = self.model.lm_gen.step(codes[:, :, c : c + 1])
                    if tokens is None:
                        continue

                    main_pcm = self.model.mimi.decode(tokens[:, 1:9])
                    _ = self.model.other_mimi.decode(tokens[:, 1:9])
                    out_f32 = main_pcm[0, 0].cpu().numpy()

                    # Convert float32 → int16 PCM bytes
                    out_int16 = np.clip(out_f32 * 32767, -32768, 32767).astype(np.int16)

                    await self.push_frame(
                        OutputAudioRawFrame(
                            audio=out_int16.tobytes(),
                            sample_rate=SAMPLE_RATE,
                            num_channels=1,
                        )
                    )

                    # Text token output (sent via data channel)
                    text_token = tokens[0, 0, 0].item()
                    if text_token not in (0, 3) and not self.model.lm_gen._injecting_context:
                        _text = self.model.text_tokenizer.id_to_piece(text_token)
                        _text = _text.replace("▁", " ")
                        # TODO: send text via data channel when pipecat supports it
        else:
            await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

def create_app(model: ModelState, static_dir: Optional[str] = None):
    """Create the FastAPI application with WebRTC signaling endpoints."""
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse

    # Use SINGLE mode — model can only serve one session at a time
    request_handler = SmallWebRTCRequestHandler(
        connection_mode=ConnectionMode.SINGLE,
    )

    @asynccontextmanager
    async def lifespan(app):
        yield
        await request_handler.close()

    app = FastAPI(title="PersonaPlex Pipecat Server", lifespan=lifespan)

    # Stash for passing session config to the connection callback
    _pending_configs: dict[str, SessionConfig] = {}

    async def _on_new_connection(webrtc_connection: SmallWebRTCConnection):
        """Called by SmallWebRTCRequestHandler when a new peer connects."""
        pc_id = webrtc_connection.pc_id
        config = _pending_configs.pop(pc_id, SessionConfig())

        transport_params = TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            audio_in_channels=1,
            audio_out_channels=1,
            camera_in_enabled=False,
            camera_out_enabled=False,
        )

        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=transport_params,
        )

        processor = MoshiProcessor(model=model, config=config)

        pipeline = Pipeline([
            transport.input(),
            processor,
            transport.output(),
        ])

        task = PipelineTask(pipeline)
        runner = PipelineRunner(handle_sigint=False)

        # Run pipeline in background — lives until the WebRTC connection closes
        asyncio.create_task(runner.run(task))

    @app.post("/api/offer")
    async def offer(request: Request):
        body = await request.json()

        # Extract session config
        config = SessionConfig(
            text_prompt=body.get("text_prompt", ""),
            voice_prompt=body.get("voice_prompt", ""),
            audio_temperature=float(body.get("audio_temperature", 0.8)),
            text_temperature=float(body.get("text_temperature", 0.7)),
            audio_topk=int(body.get("audio_topk", 250)),
            text_topk=int(body.get("text_topk", 25)),
            greeting=body.get("greeting", ""),
            seed=int(body.get("seed", -1)),
        )

        # Build the signaling request
        webrtc_request = SmallWebRTCRequest(
            sdp=body.get("sdp", ""),
            type=body.get("type", "offer"),
            pc_id=body.get("pc_id"),
        )

        # Pre-stash config so the callback can find it.
        # For new connections pc_id is None; the handler creates one internally.
        # We stash under a sentinel and retrieve in the callback.
        # The callback gets the connection *after* initialize(), so pc_id is set.
        # Use a temporary holder that the callback consumes.
        _pending_configs["__next__"] = config

        original_callback = _on_new_connection

        async def _callback_with_config(conn: SmallWebRTCConnection):
            # Move config from sentinel to real pc_id
            cfg = _pending_configs.pop("__next__", SessionConfig())
            _pending_configs[conn.pc_id] = cfg
            await original_callback(conn)

        answer = await request_handler.handle_web_request(
            webrtc_request, _callback_with_config
        )

        return JSONResponse(answer)

    @app.patch("/api/offer")
    async def ice_candidate(request: Request):
        body = await request.json()

        candidates = []
        # Support both single-candidate and batch formats
        if "candidate" in body:
            candidates.append(IceCandidate(
                candidate=body["candidate"],
                sdp_mid=body.get("sdpMid", body.get("sdp_mid", "")),
                sdp_mline_index=int(body.get("sdpMLineIndex", body.get("sdp_mline_index", 0))),
            ))
        elif "candidates" in body:
            for c in body["candidates"]:
                candidates.append(IceCandidate(
                    candidate=c["candidate"],
                    sdp_mid=c.get("sdpMid", c.get("sdp_mid", "")),
                    sdp_mline_index=int(c.get("sdpMLineIndex", c.get("sdp_mline_index", 0))),
                ))

        if candidates:
            patch_request = SmallWebRTCPatchRequest(
                pc_id=body.get("pc_id", ""),
                candidates=candidates,
            )
            await request_handler.handle_patch_request(patch_request)

        return JSONResponse({"status": "ok"})

    # Serve static client
    if static_dir and os.path.isdir(static_dir):
        @app.get("/", response_class=HTMLResponse)
        async def index():
            index_path = os.path.join(static_dir, "index.html")
            with open(index_path) as f:
                return f.read()
    else:
        @app.get("/", response_class=HTMLResponse)
        async def index():
            return _get_builtin_client_html()

    return app


def _get_builtin_client_html() -> str:
    """Return the built-in WebRTC client HTML."""
    html_path = Path(__file__).parent / "static" / "webrtc_client.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Client not found. Place webrtc_client.html in moshi/static/</h1>"


# ---------------------------------------------------------------------------
# Model loading (reused from server.py)
# ---------------------------------------------------------------------------

def load_models(args) -> ModelState:
    """Load Mimi, Moshi LM, and tokenizer. Returns a ModelState."""
    device = args.device

    # Download from HF if paths not provided
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)

    logger.info("Loading Mimi...")
    mimi = loaders.get_mimi(args.mimi_weight, device)
    other_mimi = loaders.get_mimi(args.mimi_weight, device)
    logger.info("Mimi loaded")

    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    logger.info("Loading Moshi LM...")
    lm = loaders.get_moshi_lm(args.moshi_weight, device=device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("Moshi LM loaded")

    # Get voice prompt directory
    voice_prompt_dir = args.voice_prompt_dir
    if voice_prompt_dir is None:
        try:
            voices_tgz = hf_hub_download(args.hf_repo, "voices.tgz")
            voices_tgz = Path(voices_tgz)
            voices_dir = voices_tgz.parent / "voices"
            if not voices_dir.exists():
                import tarfile
                with tarfile.open(voices_tgz, "r:gz") as tar:
                    tar.extractall(path=voices_tgz.parent)
            if voices_dir.exists():
                voice_prompt_dir = str(voices_dir)
        except Exception as e:
            logger.warning(f"Could not download voice prompts: {e}")

    state = ModelState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=device,
        voice_prompt_dir=voice_prompt_dir,
        default_greeting=args.greeting,
    )

    logger.info("Warming up model...")
    state.warmup()
    logger.info("Model ready")

    return state


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PersonaPlex Pipecat WebRTC Server")
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str, help="Path to static files directory")
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--moshi-weight", type=str)
    parser.add_argument("--mimi-weight", type=str)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--voice-prompt-dir", type=str)
    parser.add_argument("--greeting", type=str, default="")

    args = parser.parse_args()

    # Resolve device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(args.device)

    seed_all(42424242)

    with torch.no_grad():
        model = load_models(args)
        app = create_app(model, static_dir=args.static)

        import uvicorn
        logger.info(f"Starting server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
