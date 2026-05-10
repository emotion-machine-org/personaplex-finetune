# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
import base64
from dataclasses import dataclass
import glob as glob_mod
import json
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
from typing import Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import sphn
import torch

from .client_utils import make_log, colorize
from .models import loaders, MimiModel, LMModel, LMGen
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog


logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"] #| Literal["mps"]

def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    #elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #    return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing.
    Example: "<system> You enjoy having a good conversation. Have a deep conversation about technology. Your name is Jane. <system>"
    """
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


@dataclass
class ServerState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: LMModel, device: str | torch.device, voice_prompt_dir: str | None = None,
                 save_voice_prompt_embeddings: bool = False, greeting: str = "",
                 skip_pt_voice_prompts: bool = True):
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.skip_pt_voice_prompts = skip_pt_voice_prompts
        self.active_ws = None  # Track active WebSocket so we can evict it
        self.voice_prompt_dir = voice_prompt_dir
        self.greeting_tokens = text_tokenizer.encode(greeting) if greeting else None
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(lm,
                            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
                            sample_rate=self.mimi.sample_rate,
                            device=device,
                            frame_rate=self.mimi.frame_rate,
                            save_voice_prompt_embeddings=save_voice_prompt_embeddings,
        )
        
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
    
    def warmup(self):
        # Set the CUDA device context so CUDA graph capture targets the right GPU.
        # Without this, the second model's graphs capture on the wrong device → empty/broken.
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)


    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        clog = ColorizedLog.randomize()
        peer = request.remote  # IP
        peer_port = request.transport.get_extra_info("peername")[1]  # Port
        clog.log("info", f"Incoming connection from {peer}:{peer_port}")

        # Evict previous active session so the lock is released
        if self.active_ws is not None and not self.active_ws.closed:
            clog.log("info", "evicting previous active session")
            await self.active_ws.close()
        self.active_ws = ws
        clog.log("info", f"Query params: {dict(request.query)}")

        self.lm_gen.temp = float(request.query["audio_temperature"])
        self.lm_gen.temp_text = float(request.query["text_temperature"])
        self.lm_gen.top_k_text = max(1, int(request.query["text_topk"]))
        self.lm_gen.top_k = max(1, int(request.query["audio_topk"]))
        
        # Construct full voice prompt path
        requested_voice_prompt_path = None
        voice_prompt_path = None
        if self.voice_prompt_dir is not None:
            voice_prompt_filename = request.query["voice_prompt"]
            requested_voice_prompt_path = None
            if voice_prompt_filename is not None:
                requested_voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
            # If the voice prompt file does not exist, find a valid (s0) voiceprompt file in the directory
            if requested_voice_prompt_path is None or not os.path.exists(requested_voice_prompt_path):
                raise FileNotFoundError(
                    f"Requested voice prompt '{voice_prompt_filename}' not found in '{self.voice_prompt_dir}'"
                )
            else:
                voice_prompt_path = requested_voice_prompt_path
                
        if self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith('.pt'):
                if self.skip_pt_voice_prompts:
                    clog.log("info", "skipping .pt voice prompt (incompatible with finetuned model)")
                    self.lm_gen.voice_prompt = voice_prompt_path
                    self.lm_gen.voice_prompt_audio = None
                    self.lm_gen.voice_prompt_embeddings = None
                    self.lm_gen.voice_prompt_cache = None
                else:
                    clog.log("info", "loading .pt voice prompt embeddings")
                    self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)
        self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(request.query["text_prompt"])) if len(request.query["text_prompt"]) > 0 else None
        # Greeting: query param overrides CLI default
        greeting_text = request.query.get("greeting", "").strip()
        if greeting_text:
            self.lm_gen.greeting_tokens = self.text_tokenizer.encode(greeting_text)
            clog.log("info", f"greeting from query: '{greeting_text}' ({len(self.lm_gen.greeting_tokens)} tokens)")
        elif self.greeting_tokens:
            self.lm_gen.greeting_tokens = self.greeting_tokens
            clog.log("info", f"greeting from CLI default ({len(self.greeting_tokens)} tokens)")
        else:
            self.lm_gen.greeting_tokens = None
        seed = int(request["seed"]) if "seed" in request.query else None

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"{ws.exception()}")
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSED:
                        break
                    elif message.type == aiohttp.WSMsgType.CLOSE:
                        break
                    elif message.type == aiohttp.WSMsgType.TEXT:
                        # JSON control messages (e.g. context injection from puppeteer)
                        try:
                            msg = json.loads(message.data)
                            if msg.get("type") == "context":
                                self.lm_gen.inject_context(msg["text"], self.text_tokenizer)
                                clog.log("info", f"context injected ({len(msg['text'])} chars)")
                            else:
                                clog.log("warning", f"unknown JSON message type: {msg.get('type')}")
                        except (json.JSONDecodeError, KeyError) as e:
                            clog.log("error", f"bad JSON message: {e}")
                        continue
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        clog.log("error", f"unexpected message type {message.type}")
                        continue
                    message = message.data
                    if not isinstance(message, bytes):
                        clog.log("error", f"unsupported message type {type(message)}")
                        continue
                    if len(message) == 0:
                        clog.log("warning", "empty message")
                        continue
                    kind = message[0]
                    if kind == 1:  # audio
                        payload = message[1:]
                        opus_reader.append_bytes(payload)
                    else:
                        clog.log("warning", f"unknown message kind {kind}")
            finally:
                close = True
                clog.log("info", "connection closed")

        async def opus_loop():
            all_pcm_data = None

            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm is None or pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= self.frame_size:
                    chunk = all_pcm_data[: self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        out_f32 = main_pcm[0, 0].cpu().numpy()
                        # Convert to int16 and buffer raw PCM (no Opus)
                        out_int16 = np.clip(out_f32 * 32767, -32768, 32767).astype(np.int16)
                        pcm_out_buf.extend(out_int16.tobytes())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3) and not self.lm_gen._injecting_context:
                            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            await ws.send_bytes(msg)
                        else:
                            text_token_map = ['EPAD', 'BOS', 'EOS', 'PAD']

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                if len(pcm_out_buf) > 0:
                    msg = bytes(pcm_out_buf)
                    pcm_out_buf.clear()
                    await ws.send_bytes(b"\x01" + msg)

        clog.log("info", "accepted connection")
        if len(request.query["text_prompt"]) > 0:
            clog.log("info", f"text prompt: {request.query['text_prompt']}")
        if len(request.query["voice_prompt"]) > 0:
            clog.log("info", f"voice prompt: {voice_prompt_path} (requested: {requested_voice_prompt_path})")
        close = False
        async with self.lock:
            # Ensure CUDA operations target the correct GPU for this model
            if self.device.type == 'cuda':
                torch.cuda.set_device(self.device)
            if seed is not None and seed != -1:
                seed_all(seed)

            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            pcm_out_buf = bytearray()  # raw PCM output buffer shared between opus_loop and send_loop
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            async def is_alive():
                if close or ws.closed:
                    return False
                try:
                    # Check for disconnect without waiting too long
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        return False
                except asyncio.TimeoutError:
                    # No messages → client probably still alive
                    return True
                except aiohttp.ClientConnectionError:
                    return False
                return True
            # Reuse mimi for encoding voice prompt and then reset it before conversation starts
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            clog.log("info", "done with system prompts")
            # Set up greeting tokens to force at the start of conversation
            if self.lm_gen.greeting_tokens:
                self.lm_gen.prepare_greeting()
                clog.log("info", f"greeting queued: {len(self.lm_gen.greeting_tokens)} tokens")
            # Send the handshake.
            if await is_alive():
                await ws.send_bytes(b"\x00")
                clog.log("info", "sent handshake bytes")
                # Clean cancellation manager
                tasks = [
                    asyncio.create_task(recv_loop()),
                    asyncio.create_task(opus_loop()),
                    asyncio.create_task(send_loop()),
                ]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                # Force-kill remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                await ws.close()
                clog.log("info", "session closed")
                # await asyncio.gather(opus_loop(), recv_loop(), send_loop())
        clog.log("info", "done with connection")
        return ws


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """
    If voice_prompt_dir is None:
      - download voices.tgz from HF
      - extract it once
      - return extracted directory
    If voice_prompt_dir is provided:
      - just return it
    """
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts")

    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        # When set to the "none" string, we don't serve any static content.
        return static
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults PersonaPlex. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient. "
                             "Requires 'accelerate' package.")
    # A/B testing: load a second model on a different GPU
    parser.add_argument("--ab-moshi-weight", type=str, default=None,
                        help="Path to second Moshi model for A/B testing.")
    parser.add_argument("--ab-device", type=str, default=None,
                        help="Device for the A/B model (e.g. 'cuda:1').")
    parser.add_argument("--ab-label", type=str, default="base",
                        help="Label for the A/B model in results (default: 'base').")
    parser.add_argument("--model-label", type=str, default="finetuned",
                        help="Label for the primary model in results (default: 'finetuned').")
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        help=(
            "Directory containing voice prompt files. "
            "If omitted, voices.tgz is downloaded from HF and extracted."
            "Voice prompt filenames from client requests will be joined with this directory path."
        )
    )
    parser.add_argument(
        "--greeting",
        type=str,
        default="",
        help=(
            "Preset greeting text the model will speak at the start of each session. "
            "E.g. 'Hi, this is Alex from Insurance Corp calling John.'"
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help=(
            "Directory containing training data (with stereo_wav/*.json for text prompts). "
            "Enables /api/random-prompt endpoint and a 'Random' button in the UI."
        )
    )
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    static_path: None | str = _get_static_path(args.static)
    assert static_path is None or os.path.exists(static_path), \
        f"Static path does not exist: {static_path}."
    logger.info(f"static_path = {static_path}")
    args.device = torch_auto_device(args.device)

    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            logger.error("Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Download config.json to increment download counter
    # No worries about double-counting since config.json will be cached the second time
    hf_hub_download(args.hf_repo, "config.json")

    # --- Load primary model ---
    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)  # type: ignore

    logger.info("loading moshi (primary)")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")
    state = ServerState(
        mimi=mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        save_voice_prompt_embeddings=False,
        greeting=args.greeting,
    )
    logger.info("warming up primary model")
    state.warmup()

    # --- Load A/B model if specified ---
    ab_state = None
    if args.ab_moshi_weight and args.ab_device:
        ab_device = torch_auto_device(args.ab_device)
        logger.info(f"loading A/B model on {ab_device}")
        ab_mimi = loaders.get_mimi(args.mimi_weight, ab_device)
        ab_lm = loaders.get_moshi_lm(args.ab_moshi_weight, device=ab_device, cpu_offload=args.cpu_offload)
        ab_lm.eval()
        ab_state = ServerState(
            mimi=ab_mimi,
            text_tokenizer=text_tokenizer,
            lm=ab_lm,
            device=ab_device,
            voice_prompt_dir=args.voice_prompt_dir,
            save_voice_prompt_embeddings=False,
            greeting=args.greeting,
            skip_pt_voice_prompts=False,  # base model loads .pt via load_voice_prompt_embeddings
        )
        logger.info("warming up A/B model")
        ab_state.warmup()
        logger.info(f"A/B mode enabled: primary='{args.model_label}' on {args.device}, ab='{args.ab_label}' on {ab_device}")

    # Load random prompt data if --data-dir is provided
    text_prompts = []
    voice_filenames = []
    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        # Load text prompts from stereo_wav/*.json
        for json_path in sorted(data_dir.glob("stereo_wav/*.json")):
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                if "text_prompt" in meta and meta["text_prompt"]:
                    text_prompts.append(meta["text_prompt"])
            except Exception:
                pass
        # Load voice filenames from voice_prompt_dir
        if args.voice_prompt_dir is not None:
            voice_filenames = [
                f.name for f in Path(args.voice_prompt_dir).iterdir()
                if f.suffix in (".pt", ".wav")
            ]
        logger.info(f"Loaded {len(text_prompts)} text prompts, {len(voice_filenames)} voices for random selection")

    # A/B test results storage
    ab_results_path = Path(args.data_dir or ".") / "ab_results.json"
    async def handle_ab_result(request):
        try:
            data = await request.json()
            # Append to JSON lines file
            with open(ab_results_path, "a") as f:
                f.write(json.dumps(data) + "\n")
            logger.info(f"A/B result saved: preference={data.get('preference')}, tester={data.get('tester_id')}")
            return web.json_response({"ok": True})
        except Exception as e:
            logger.error(f"Failed to save A/B result: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_ab_results(request):
        results = []
        if ab_results_path.exists():
            with open(ab_results_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
        return web.json_response(results)

    # Route /api/chat to the right model based on ?model= query param
    async def handle_chat_route(request):
        model_param = request.query.get("model", "primary")
        if model_param == "ab" and ab_state is not None:
            logger.info(f">>> Routing to A/B model (label='{args.ab_label}', device={args.ab_device})")
            return await ab_state.handle_chat(request)
        logger.info(f">>> Routing to PRIMARY model (label='{args.model_label}', device={args.device})")
        return await state.handle_chat(request)

    # --- Gemini Live proxy (Vertex AI) ---
    GEMINI_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "")
    GEMINI_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    GEMINI_MODEL_ID = "gemini-live-2.5-flash-native-audio"
    GEMINI_VOICES = ["Aoede", "Charon", "Fenrir", "Kore", "Puck", "Leda", "Orus", "Zephyr"]

    def _get_gcloud_access_token() -> str:
        """Get a fresh access token from gcloud ADC."""
        import subprocess
        return subprocess.check_output(
            ["gcloud", "auth", "application-default", "print-access-token"],
            text=True
        ).strip()

    # Check if ADC is available at startup
    gemini_enabled = False
    try:
        _test_token = _get_gcloud_access_token()
        gemini_enabled = bool(_test_token)
        if gemini_enabled:
            logger.info(f"Vertex AI auth OK (project={GEMINI_PROJECT}, location={GEMINI_LOCATION})")
    except Exception as e:
        logger.warning(f"Vertex AI auth not available: {e}")

    async def handle_gemini_chat(request):
        """WebSocket proxy: client ↔ our server ↔ Gemini Live via Vertex AI.
        Same binary protocol as Moshi (0x00=handshake, 0x01=audio, 0x02=text)."""
        ws_client = web.WebSocketResponse()
        await ws_client.prepare(request)

        text_prompt = request.query.get("text_prompt", "You are a helpful assistant.")
        gemini_voice = request.query.get("gemini_voice", "Aoede")
        greeting = request.query.get("greeting", "")

        logger.info(f">>> Gemini Live session starting (voice={gemini_voice}, model={GEMINI_MODEL_ID})")

        # Get fresh access token
        try:
            access_token = _get_gcloud_access_token()
        except Exception as e:
            logger.error(f"Failed to get access token: {e}")
            await ws_client.close()
            return ws_client

        # Connect to Vertex AI Gemini Live
        vertex_url = (
            f"wss://{GEMINI_LOCATION}-aiplatform.googleapis.com/ws/"
            f"google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
        )
        vertex_model = (
            f"projects/{GEMINI_PROJECT}/locations/{GEMINI_LOCATION}/"
            f"publishers/google/models/{GEMINI_MODEL_ID}"
        )

        session = aiohttp.ClientSession()
        try:
            gemini_ws = await session.ws_connect(
                vertex_url,
                headers={"Authorization": f"Bearer {access_token}"},
            )
        except Exception as e:
            logger.error(f"Failed to connect to Gemini: {e}")
            await ws_client.close()
            await session.close()
            return ws_client

        # Send setup message
        setup_msg = {
            "setup": {
                "model": vertex_model,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": gemini_voice}
                        }
                    },
                },
                "systemInstruction": {
                    "parts": [{"text": text_prompt}]
                },
            }
        }
        await gemini_ws.send_bytes(json.dumps(setup_msg).encode("utf-8"))

        # Wait for setupComplete (Gemini sends JSON as binary frames)
        setup_done = False
        async for msg in gemini_ws:
            if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                data = json.loads(msg.data)
                if "setupComplete" in data:
                    setup_done = True
                    break
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                break
        if not setup_done:
            logger.error("Gemini setup failed")
            await gemini_ws.close()
            await session.close()
            await ws_client.close()
            return ws_client

        logger.info("Gemini Live setup complete")
        # Send handshake to client
        await ws_client.send_bytes(b"\x00")

        close = False
        opus_reader = sphn.OpusStreamReader(24000)

        async def client_recv_loop():
            """Receive Opus audio from client, decode, forward PCM to Gemini."""
            nonlocal close
            try:
                async for msg in ws_client:
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        if len(msg.data) == 0:
                            continue
                        kind = msg.data[0]
                        if kind == 1:  # audio
                            opus_reader.append_bytes(msg.data[1:])
                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
            finally:
                close = True

        async def opus_to_gemini_loop():
            """Read decoded PCM from opus_reader, resample 24→16kHz, send to Gemini."""
            nonlocal close
            while not close:
                await asyncio.sleep(0.02)  # 20ms batches
                pcm = opus_reader.read_pcm()
                if pcm is None or pcm.shape[-1] == 0:
                    continue
                # Resample 24kHz → 16kHz (Gemini expects 16kHz input)
                # Simple decimation: take every 1.5th sample via linear interpolation
                src_len = pcm.shape[-1]
                dst_len = int(src_len * 16000 / 24000)
                indices = np.arange(dst_len) * (src_len - 1) / max(dst_len - 1, 1)
                left = np.floor(indices).astype(int)
                frac = indices - left
                right = np.minimum(left + 1, src_len - 1)
                pcm_16k = pcm[left] * (1 - frac) + pcm[right] * frac
                # Convert float32 to int16 bytes, then base64
                pcm_int16 = np.clip(pcm_16k * 32767, -32768, 32767).astype(np.int16)
                b64 = base64.b64encode(pcm_int16.tobytes()).decode("ascii")
                try:
                    await gemini_ws.send_bytes(json.dumps({
                        "realtimeInput": {
                            "audio": {
                                "data": b64,
                                "mimeType": "audio/pcm;rate=16000",
                            }
                        }
                    }).encode("utf-8"))
                except Exception:
                    close = True
                    break

        # Throttled audio output buffer — Gemini sends audio faster than real-time.
        # We buffer it here and drip-feed to the client at ~real-time rate.
        gemini_pcm_queue = asyncio.Queue()
        CHUNK_SAMPLES = 1920  # 80ms at 24kHz = 3840 bytes (matches Moshi frame size)
        CHUNK_BYTES = CHUNK_SAMPLES * 2  # int16

        async def gemini_recv_loop():
            """Receive audio/text from Gemini, enqueue PCM chunks."""
            nonlocal close
            try:
                async for msg in gemini_ws:
                    if close:
                        break
                    if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                        data = json.loads(msg.data)
                        server_content = data.get("serverContent", {})

                        # Handle interruption — user barged in, clear queued audio
                        if server_content.get("interrupted"):
                            while not gemini_pcm_queue.empty():
                                try:
                                    gemini_pcm_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            continue

                        model_turn = server_content.get("modelTurn", {})
                        parts = model_turn.get("parts", [])
                        for part in parts:
                            inline = part.get("inlineData", {})
                            if inline.get("data"):
                                pcm_bytes = base64.b64decode(inline["data"])
                                # Break into 80ms chunks for smooth playback
                                offset = 0
                                while offset < len(pcm_bytes):
                                    chunk = pcm_bytes[offset:offset + CHUNK_BYTES]
                                    await gemini_pcm_queue.put(chunk)
                                    offset += CHUNK_BYTES
                            text = part.get("text")
                            if text:
                                try:
                                    await ws_client.send_bytes(
                                        b"\x02" + text.encode("utf-8")
                                    )
                                except Exception:
                                    pass

                        # turnComplete — Gemini finished speaking, nothing more to do
                    elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
            finally:
                close = True

        async def throttled_send_loop():
            """Send PCM chunks to client at real-time rate (~80ms per chunk)."""
            nonlocal close
            while not close:
                try:
                    chunk = await asyncio.wait_for(gemini_pcm_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if close:
                    break
                try:
                    await ws_client.send_bytes(b"\x01" + chunk)
                except Exception:
                    close = True
                    break
                # Pace at real-time: 80ms per 1920-sample chunk
                chunk_duration = (len(chunk) / 2) / 24000
                await asyncio.sleep(chunk_duration)

        tasks = [
            asyncio.create_task(client_recv_loop()),
            asyncio.create_task(opus_to_gemini_loop()),
            asyncio.create_task(gemini_recv_loop()),
            asyncio.create_task(throttled_send_loop()),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await ws_client.close()
        await gemini_ws.close()
        await session.close()
        logger.info("Gemini Live session closed")
        return ws_client

    # Expose model labels so the client knows what's available
    async def handle_ab_config(request):
        config = {
            "enabled": ab_state is not None,
            "primary_label": args.model_label,
            "ab_label": args.ab_label,
            "gemini_enabled": gemini_enabled,
        }
        return web.json_response(config)

    # Hot-swap primary model from a run directory path
    async def handle_load_model(request):
        try:
            data = await request.json()
            run_path = data.get("path", "").strip()
            if not run_path:
                return web.json_response({"error": "No path provided"}, status=400)

            run_dir = Path(run_path)
            # Find model.safetensors: direct file, or search checkpoint subdirs
            if run_dir.is_file() and run_dir.suffix == ".safetensors":
                model_file = run_dir
            else:
                model_file = None
                # Look for checkpoints/checkpoint_*/model.safetensors (or in merged/ or consolidated/ subdirs)
                # Search from latest checkpoint backwards until we find one with a model file
                ckpt_dirs = sorted(run_dir.glob("checkpoints/checkpoint_*"), reverse=True)
                for ckpt_dir in ckpt_dirs:
                    for candidate in [
                        ckpt_dir / "model.safetensors",
                        ckpt_dir / "merged" / "model.safetensors",
                        ckpt_dir / "consolidated" / "model.safetensors",
                    ]:
                        if candidate.exists():
                            model_file = candidate
                            break
                    if model_file is not None:
                        break
                if model_file is None and (run_dir / "model.safetensors").exists():
                    model_file = run_dir / "model.safetensors"
                if model_file is None:
                    return web.json_response(
                        {"error": f"No model.safetensors found in {run_path}"}, status=400
                    )

            if not model_file.exists():
                return web.json_response({"error": f"File not found: {model_file}"}, status=400)

            logger.info(f"Hot-swapping primary model to: {model_file}")

            # Wait for any active session to finish
            async with state.lock:
                import gc

                # Step 1: Load new model to CPU (safe — uses system RAM only)
                logger.info("Loading new model to CPU...")
                new_lm = loaders.get_moshi_lm(str(model_file), device="cpu")
                new_lm.eval()

                # Step 2: Move old model OFF GPU to CPU, then delete
                logger.info("Moving old model off GPU...")
                old_lm = state.lm_gen.lm_model
                old_lm.cpu()  # moves all tensors to CPU, freeing GPU memory
                state.lm_gen.lm_model = None
                state.lm_gen = None
                del old_lm
                gc.collect()
                if args.device.type == 'cuda':
                    torch.cuda.set_device(args.device)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(args.device)
                    free_mb = torch.cuda.mem_get_info(args.device)[0] // (1024 * 1024)
                    logger.info(f"GPU memory freed: {free_mb} MiB available")

                # Step 3: Move new model from CPU → GPU
                logger.info("Moving new model to GPU...")
                new_lm = new_lm.to(device=args.device, dtype=torch.bfloat16)

                state.lm_gen = LMGen(
                    new_lm,
                    audio_silence_frame_cnt=int(0.5 * state.mimi.frame_rate),
                    sample_rate=state.mimi.sample_rate,
                    device=args.device,
                    frame_rate=state.mimi.frame_rate,
                )
                state.lm_gen.streaming_forever(1)

                # Swapped models may support .pt voice prompts — don't skip them
                state.skip_pt_voice_prompts = False
                # Reset cached voice prompt so it reloads on next connection
                state.lm_gen.voice_prompt = None

                # Warmup new model
                state.warmup()

            label = run_dir.name if run_dir.is_dir() else run_dir.parent.name
            args.model_label = label
            logger.info(f"Primary model swapped to: {model_file} (label='{label}')")
            return web.json_response({"ok": True, "path": str(model_file), "label": label})

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return web.json_response({"error": str(e)}, status=500)

    app = web.Application()
    app.router.add_get("/api/chat", handle_chat_route)
    app.router.add_get("/api/ab-config", handle_ab_config)
    app.router.add_post("/api/ab-result", handle_ab_result)
    app.router.add_get("/api/ab-results", handle_ab_results)
    app.router.add_post("/api/load-model", handle_load_model)
    if gemini_enabled:
        app.router.add_get("/api/gemini-chat", handle_gemini_chat)
        logger.info("Gemini Live proxy enabled (Vertex AI)")

    if text_prompts or voice_filenames:
        async def handle_random_prompt(_):
            result = {}
            if text_prompts:
                result["text_prompt"] = random.choice(text_prompts)
            if voice_filenames:
                result["voice_prompt"] = random.choice(voice_filenames)
            return web.json_response(result)
        app.router.add_get("/api/random-prompt", handle_random_prompt)

    if static_path is not None:
        async def handle_root(_):
            # Inject a script into index.html that adds a Random button
            index_path = os.path.join(static_path, "index.html")
            with open(index_path) as f:
                html = f.read()
            if text_prompts or voice_filenames:
                inject_script = """
<script>
document.addEventListener('DOMContentLoaded', function() {
  function tryInject() {
    var btns = document.querySelectorAll('button');
    if (btns.length === 0) { setTimeout(tryInject, 500); return; }
    if (document.getElementById('random-prompt-btn')) return;
    var btn = document.createElement('button');
    btn.id = 'random-prompt-btn';
    btn.textContent = 'Random Prompt';
    btn.style.cssText = 'margin:8px;padding:8px 16px;background:#6366f1;color:white;border:none;border-radius:6px;cursor:pointer;font-size:14px;';
    btn.addEventListener('click', async function() {
      btn.textContent = 'Loading...';
      try {
        var resp = await fetch('/api/random-prompt');
        var data = await resp.json();
        if (data.text_prompt) {
          var textEl = document.querySelector('textarea');
          if (textEl) {
            var nativeSet = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            nativeSet.call(textEl, data.text_prompt);
            textEl.dispatchEvent(new Event('input', {bubbles: true}));
          }
        }
        if (data.voice_prompt) {
          var inputs = document.querySelectorAll('input[type="text"]');
          for (var inp of inputs) {
            if (inp.value && (inp.value.includes('.pt') || inp.value.includes('.wav') || inp.value.includes('NAT') || inp.value.includes('VAR'))) {
              var nativeSetInp = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
              nativeSetInp.call(inp, data.voice_prompt);
              inp.dispatchEvent(new Event('input', {bubbles: true}));
              break;
            }
          }
        }
        btn.textContent = 'Random Prompt';
      } catch(e) {
        btn.textContent = 'Error!';
        setTimeout(function(){ btn.textContent = 'Random Prompt'; }, 2000);
      }
    });
    var header = document.querySelector('h1');
    if (header && header.parentElement) {
      header.parentElement.appendChild(btn);
    }
  }
  setTimeout(tryInject, 1500);
});
</script>
"""
                html = html.replace('</body>', inject_script + '</body>')
            return web.Response(text=html, content_type='text/html')

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")
    if setup_tunnel is not None:
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
