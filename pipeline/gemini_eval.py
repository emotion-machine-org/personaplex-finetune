#!/usr/bin/env python3
"""
Gemini Live ↔ Moshi bridge for voice model evaluation.

Connects Gemini Live API (as patient) to a running Moshi server (as care
assistant) via websockets.  Audio is resampled between 16kHz (Gemini) and
24kHz (Moshi) and transcoded between raw PCM (Gemini) and Opus (Moshi).

Usage:
    # 1. Start Moshi server with finetuned weights
    python -m moshi.server --moshi-weight merged.safetensors --port 8998

    # 2. Run this bridge
    python gemini_eval.py \
        --moshi-url ws://localhost:8998/api/chat \
        --system-prompt "You are Carmen Gutierrez, a 44-year-old woman..." \
        --text-prompt "PROGRAM: Wegovy Care Support\\nPATIENT: Carmen..." \
        --output eval_output.wav \
        --duration 180
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import struct
import time
from pathlib import Path

import numpy as np
def _load_env():
    """Load .env file from project root into os.environ."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

_load_env()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("gemini_eval")

MOSHI_SAMPLE_RATE = 24000
GEMINI_SAMPLE_RATE = 16000
MOSHI_FRAME_RATE = 12.5  # frames per second
MOSHI_FRAME_SIZE = int(MOSHI_SAMPLE_RATE / MOSHI_FRAME_RATE)  # 1920 samples


def resample_pcm(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample PCM audio using linear interpolation."""
    if from_rate == to_rate:
        return pcm
    ratio = to_rate / from_rate
    n_out = int(len(pcm) * ratio)
    indices = np.arange(n_out) / ratio
    indices = np.clip(indices, 0, len(pcm) - 1)
    # Linear interpolation
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(pcm) - 1)
    frac = indices - idx_floor
    return pcm[idx_floor] * (1 - frac) + pcm[idx_ceil] * frac


def pcm_float_to_int16_bytes(pcm_float: np.ndarray) -> bytes:
    """Convert float32 PCM [-1, 1] to 16-bit LE PCM bytes."""
    pcm_int16 = np.clip(pcm_float * 32767, -32768, 32767).astype(np.int16)
    return pcm_int16.tobytes()


def pcm_int16_bytes_to_float(data: bytes) -> np.ndarray:
    """Convert 16-bit LE PCM bytes to float32 [-1, 1]."""
    pcm_int16 = np.frombuffer(data, dtype=np.int16)
    return pcm_int16.astype(np.float32) / 32767.0


async def run_bridge(args, context_injections=None):
    """Main bridge loop connecting Gemini Live and Moshi server.

    Args:
        args: Namespace with connection/sampling params.
        context_injections: Optional list of {"frame": int, "text": str} dicts.
            Injections are sent to Moshi via JSON websocket at the specified
            frame offset (12.5 Hz).

    Returns:
        dict with keys: "wav_path", "moshi_transcript", "gemini_transcript",
        "elapsed_s", or None on fatal error.
    """
    try:
        import aiohttp
        import websockets
    except ImportError:
        logger.error("Install: pip install aiohttp websockets")
        return None

    try:
        import sphn
    except ImportError:
        logger.error("Install sphn for Opus encoding: pip install 'sphn>=0.1.4,<0.2'")
        return None

    gemini_key = os.environ.get("GEMINI_API_KEY", args.gemini_key if hasattr(args, "gemini_key") else None)
    if not gemini_key:
        logger.error("Set GEMINI_API_KEY env var or pass --gemini-key")
        return None

    # --- Gemini connection ---
    gemini_model = getattr(args, "gemini_model", "models/gemini-3.1-flash-live-preview")
    gemini_voice = getattr(args, "gemini_voice", "Puck")
    gemini_url = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
        f"?key={gemini_key}"
    )

    gemini_config = {
        "setup": {
            "model": gemini_model,
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": gemini_voice}
                    }
                },
            },
            "systemInstruction": {
                "parts": [{"text": args.system_prompt}]
            },
            "realtimeInputConfig": {
                "automaticActivityDetection": {
                    "disabled": False,
                    "startOfSpeechSensitivity": "START_SENSITIVITY_HIGH",
                    "endOfSpeechSensitivity": "END_SENSITIVITY_LOW",
                },
            },
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
        }
    }

    # --- Moshi connection params ---
    from urllib.parse import urlencode
    moshi_params = {
        "audio_temperature": str(args.audio_temperature),
        "text_temperature": str(args.text_temperature),
        "audio_topk": str(args.audio_topk),
        "text_topk": str(args.text_topk),
        "voice_prompt": args.voice_prompt or "",
        "text_prompt": args.text_prompt or "",
        "greeting": args.greeting or "",
    }
    moshi_url = args.moshi_url + "?" + urlencode(moshi_params)

    # --- Context injection schedule (sorted by frame) ---
    injection_schedule = []
    if context_injections:
        injection_schedule = sorted(context_injections, key=lambda x: x["frame"])

    # --- State ---
    gemini_audio_buffer = np.array([], dtype=np.float32)  # incoming from Gemini at 24kHz
    moshi_to_gemini_queue = asyncio.Queue()  # Moshi audio chunks to forward to Gemini
    moshi_pcm_out = []  # recorded Moshi output at 24kHz
    gemini_pcm_out = []  # recorded Gemini output at 24kHz (after resample)
    moshi_transcript_parts = []  # Gemini's inputTranscription (what it heard from Moshi)
    gemini_transcript_parts = []  # Gemini's outputTranscription (what Gemini said)
    done = asyncio.Event()
    t0 = time.time()
    total_duration = args.duration

    # Opus codec for encoding audio TO Moshi (Moshi sends raw PCM back)
    opus_writer = sphn.OpusStreamWriter(MOSHI_SAMPLE_RATE)

    logger.info(f"Connecting to Gemini ({gemini_model})...")
    try:
        async with websockets.connect(gemini_url) as gemini_ws:
            # Send setup
            await gemini_ws.send(json.dumps(gemini_config))
            setup_response = await gemini_ws.recv()
            logger.info(f"Gemini setup response: {str(setup_response)[:200]}")

            logger.info(f"Connecting to Moshi ({args.moshi_url})...")
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(moshi_url) as moshi_ws:
                    logger.info("Both connected. Waiting for Moshi handshake...")

                    # Wait for Moshi handshake (b"\x00")
                    while True:
                        msg = await moshi_ws.receive(timeout=120)
                        if msg.type == aiohttp.WSMsgType.BINARY and msg.data == b"\x00":
                            t0 = time.time()  # Reset timer after handshake
                            logger.info("Moshi handshake received. Starting bridge...")

                            # Send initial text to Gemini to trigger first response
                            # (audio VAD may not detect Moshi's synthesized speech)
                            if args.greeting:
                                await gemini_ws.send(json.dumps({
                                    "realtimeInput": {
                                        "text": args.greeting
                                    }
                                }))
                                logger.info(f"Sent greeting as text to Gemini: {args.greeting}")
                            break
                        elif msg.type == aiohttp.WSMsgType.BINARY and len(msg.data) > 0 and msg.data[0] == 1:
                            # Raw PCM audio from system prompt phase — discard
                            pass
                        elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.error("Moshi connection closed before handshake")
                            return None

                    async def gemini_recv_loop():
                        """Receive audio from Gemini, buffer it."""
                        nonlocal gemini_audio_buffer
                        try:
                            async for msg in gemini_ws:
                                if done.is_set():
                                    break
                                try:
                                    data = json.loads(msg)
                                except (json.JSONDecodeError, TypeError):
                                    continue

                                # Extract audio from server response
                                # Gemini outputs 24kHz PCM — same as Moshi, no resample needed
                                server_content = data.get("serverContent", {})
                                model_turn = server_content.get("modelTurn", {})
                                parts = model_turn.get("parts", [])
                                for part in parts:
                                    inline = part.get("inlineData", {})
                                    audio_b64 = inline.get("data")
                                    if audio_b64:
                                        pcm_bytes = base64.b64decode(audio_b64)
                                        pcm_24k = pcm_int16_bytes_to_float(pcm_bytes)
                                        gemini_audio_buffer = np.concatenate([gemini_audio_buffer, pcm_24k])
                                        if len(gemini_audio_buffer) % 24000 < len(pcm_24k):
                                            logger.info(f"Gemini audio buffer: {len(gemini_audio_buffer)/24000:.1f}s")

                                # Collect transcriptions
                                if server_content.get("inputTranscription"):
                                    text = server_content["inputTranscription"].get("text", "")
                                    if text:
                                        moshi_transcript_parts.append(text)
                                        logger.info(f"[Gemini heard]: {text}")
                                if server_content.get("outputTranscription"):
                                    text = server_content["outputTranscription"].get("text", "")
                                    if text:
                                        gemini_transcript_parts.append(text)
                                        logger.info(f"[Gemini said]: {text}")

                                # Log turn completion
                                if server_content.get("turnComplete"):
                                    logger.info("Gemini turn complete")

                                # Log non-content messages
                                if not server_content and data:
                                    msg_str = json.dumps(data)[:300]
                                    logger.info(f"Gemini msg: {msg_str}")
                        except Exception as e:
                            if not done.is_set():
                                logger.error(f"Gemini recv error: {e}")

                    async def moshi_recv_loop():
                        """Receive PCM audio from Moshi, queue for Gemini forwarding."""
                        while not done.is_set():
                            try:
                                msg = await moshi_ws.receive(timeout=5.0)
                            except asyncio.TimeoutError:
                                continue  # timeout on receive is normal during pauses
                            except Exception as e:
                                if not done.is_set():
                                    logger.error(f"Moshi recv error: {e}")
                                break

                            if msg.type == aiohttp.WSMsgType.BINARY:
                                data = msg.data
                                if len(data) > 0 and data[0] == 1:
                                    # Raw int16 PCM audio from Moshi server
                                    pcm_bytes = data[1:]
                                    if len(pcm_bytes) > 0:
                                        pcm_24k = pcm_int16_bytes_to_float(pcm_bytes)
                                        moshi_pcm_out.append(pcm_24k.copy())

                                        # Queue for Gemini forwarding (avoid concurrent ws sends)
                                        pcm_16k = resample_pcm(pcm_24k, MOSHI_SAMPLE_RATE, GEMINI_SAMPLE_RATE)
                                        pcm_16k = np.clip(pcm_16k * 8.0, -1.0, 1.0)
                                        await moshi_to_gemini_queue.put(pcm_16k)
                            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break

                    async def feed_gemini_to_moshi():
                        """Take buffered Gemini audio (24kHz), Opus encode, send to Moshi.
                        Also forwards queued Moshi audio to Gemini and handles context injections."""
                        nonlocal gemini_audio_buffer
                        # Gemini outputs 24kHz — same as Moshi, no resampling needed
                        frame_size_24k = MOSHI_FRAME_SIZE  # 1920 samples per 80ms frame
                        injection_idx = 0  # next injection to send

                        while not done.is_set():
                            current_frame = len(gemini_pcm_out)
                            elapsed = time.time() - t0
                            if elapsed >= total_duration:
                                logger.info(f"Duration {total_duration}s reached")
                                done.set()
                                break

                            # Forward any queued Moshi audio to Gemini
                            while not moshi_to_gemini_queue.empty():
                                try:
                                    pcm_16k = moshi_to_gemini_queue.get_nowait()
                                    pcm_bytes = pcm_float_to_int16_bytes(pcm_16k)
                                    audio_b64 = base64.b64encode(pcm_bytes).decode("utf-8")
                                    await gemini_ws.send(json.dumps({
                                        "realtimeInput": {
                                            "audio": {
                                                "data": audio_b64,
                                                "mimeType": "audio/pcm;rate=16000"
                                            }
                                        }
                                    }))
                                except Exception as e:
                                    logger.warning(f"Failed to forward Moshi audio to Gemini: {e}")

                            # Send any due context injections
                            while injection_idx < len(injection_schedule):
                                inj = injection_schedule[injection_idx]
                                if current_frame >= inj["frame"]:
                                    inj_msg = json.dumps({"type": "context", "text": inj["text"]})
                                    await moshi_ws.send_str(inj_msg)
                                    logger.info(
                                        f"Injected context at frame {current_frame} "
                                        f"(scheduled {inj['frame']}): {inj['text'][:80]}..."
                                    )
                                    injection_idx += 1
                                else:
                                    break

                            if len(gemini_audio_buffer) >= frame_size_24k:
                                # Take one frame worth of audio
                                chunk_24k = gemini_audio_buffer[:frame_size_24k]
                                gemini_audio_buffer = gemini_audio_buffer[frame_size_24k:]
                                gemini_pcm_out.append(chunk_24k.copy())

                                # Opus encode and send to Moshi
                                opus_writer.append_pcm(chunk_24k)
                                opus_bytes = opus_writer.read_bytes()
                                if opus_bytes:
                                    await moshi_ws.send_bytes(b"\x01" + opus_bytes)
                            else:
                                # No Gemini audio yet — send silence to Moshi
                                silence_24k = np.zeros(frame_size_24k, dtype=np.float32)
                                gemini_pcm_out.append(silence_24k.copy())
                                opus_writer.append_pcm(silence_24k)
                                opus_bytes = opus_writer.read_bytes()
                                if opus_bytes:
                                    await moshi_ws.send_bytes(b"\x01" + opus_bytes)

                            # Pace at ~12.5 Hz (80ms per frame)
                            target_time = t0 + (len(gemini_pcm_out)) * (1.0 / MOSHI_FRAME_RATE)
                            sleep_time = target_time - time.time()
                            if sleep_time > 0:
                                await asyncio.sleep(sleep_time)

                    # Run all loops concurrently, cancel when done
                    tasks = [
                        asyncio.create_task(gemini_recv_loop()),
                        asyncio.create_task(moshi_recv_loop()),
                        asyncio.create_task(feed_gemini_to_moshi()),
                    ]
                    # Wait for feed loop to finish (it sets done on duration)
                    await tasks[2]
                    # Cancel the recv loops
                    for t in tasks[:2]:
                        t.cancel()
                    # Wait for cancellation to complete
                    for t in tasks[:2]:
                        try:
                            await t
                        except asyncio.CancelledError:
                            pass

    except Exception as e:
        logger.error(f"Bridge error: {e}")
        return None

    elapsed = time.time() - t0
    logger.info(f"Bridge finished in {elapsed:.0f}s.")

    # --- Save output ---
    if moshi_pcm_out:
        moshi_audio = np.concatenate(moshi_pcm_out)
        logger.info(f"Moshi audio: {len(moshi_audio)} samples ({len(moshi_audio)/MOSHI_SAMPLE_RATE:.1f}s)")
    else:
        moshi_audio = np.array([], dtype=np.float32)

    if gemini_pcm_out:
        gemini_audio = np.concatenate(gemini_pcm_out)
        logger.info(f"Gemini audio: {len(gemini_audio)} samples ({len(gemini_audio)/MOSHI_SAMPLE_RATE:.1f}s)")
    else:
        gemini_audio = np.array([], dtype=np.float32)

    wav_path = getattr(args, "output", None)
    if wav_path and (len(moshi_audio) > 0 or len(gemini_audio) > 0):
        import soundfile as sf
        # Stereo: Moshi=left, Gemini=right
        max_len = max(len(moshi_audio), len(gemini_audio))
        moshi_padded = np.pad(moshi_audio, (0, max_len - len(moshi_audio)))
        gemini_padded = np.pad(gemini_audio, (0, max_len - len(gemini_audio)))
        stereo = np.stack([moshi_padded, gemini_padded], axis=-1)
        sf.write(wav_path, stereo, MOSHI_SAMPLE_RATE)
        logger.info(f"Saved stereo WAV to {wav_path}")

    # Build transcripts from collected parts
    moshi_transcript = " ".join(moshi_transcript_parts).strip()
    gemini_transcript = " ".join(gemini_transcript_parts).strip()

    return {
        "wav_path": wav_path,
        "moshi_transcript": moshi_transcript,
        "gemini_transcript": gemini_transcript,
        "elapsed_s": elapsed,
    }


def run_dialogue(
    moshi_url: str,
    system_prompt: str,
    text_prompt: str,
    greeting: str,
    voice_prompt: str,
    output_path: str,
    duration: float,
    context_injections: list[dict] | None = None,
    gemini_model: str = "models/gemini-3.1-flash-live-preview",
    gemini_voice: str = "Puck",
    audio_temperature: float = 0.55,
    text_temperature: float = 0.7,
    audio_topk: int = 100,
    text_topk: int = 30,
) -> dict | None:
    """Programmatic entry point for gen_eval integration.

    Runs one Gemini↔Moshi dialogue and returns structured results.

    Returns:
        dict with "wav_path", "moshi_transcript", "gemini_transcript",
        "elapsed_s", or None on error.
    """
    args = argparse.Namespace(
        moshi_url=moshi_url,
        gemini_key=None,  # uses GEMINI_API_KEY env var
        gemini_model=gemini_model,
        gemini_voice=gemini_voice,
        system_prompt=system_prompt,
        text_prompt=text_prompt,
        greeting=greeting,
        voice_prompt=voice_prompt,
        output=output_path,
        duration=duration,
        audio_temperature=audio_temperature,
        text_temperature=text_temperature,
        audio_topk=audio_topk,
        text_topk=text_topk,
    )
    return asyncio.run(run_bridge(args, context_injections=context_injections))


def main():
    parser = argparse.ArgumentParser(description="Gemini Live ↔ Moshi bridge for evaluation")

    parser.add_argument("--moshi-url", default="ws://localhost:8998/api/chat",
                        help="Moshi server WebSocket URL")
    parser.add_argument("--gemini-key", default=None,
                        help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--gemini-model", default="models/gemini-3.1-flash-live-preview",
                        help="Gemini Live model name")
    parser.add_argument("--gemini-voice", default="Puck",
                        help="Gemini voice name")
    parser.add_argument("--system-prompt", required=True,
                        help="System prompt for Gemini (patient role)")
    parser.add_argument("--text-prompt", default="",
                        help="Text prompt for Moshi (system prompt / call brief)")
    parser.add_argument("--greeting", default="",
                        help="Greeting text for Moshi to speak first")
    parser.add_argument("--voice-prompt", default="NATF0.pt",
                        help="Voice prompt filename for Moshi")
    parser.add_argument("--output", default="gemini_eval.wav",
                        help="Output stereo WAV path")
    parser.add_argument("--duration", type=float, default=180.0,
                        help="Conversation duration in seconds")

    # Moshi sampling params
    parser.add_argument("--audio-temperature", type=float, default=0.55)
    parser.add_argument("--text-temperature", type=float, default=0.7)
    parser.add_argument("--audio-topk", type=int, default=100)
    parser.add_argument("--text-topk", type=int, default=30)

    # Context injections (JSON file with [{frame, text}, ...])
    parser.add_argument("--context-injections", default=None,
                        help="Path to JSON file with context injections")

    args = parser.parse_args()

    # Load context injections if provided
    context_injections = None
    if args.context_injections:
        with open(args.context_injections) as f:
            context_injections = json.load(f)

    result = asyncio.run(run_bridge(args, context_injections=context_injections))
    if result:
        logger.info(f"Moshi transcript ({len(result['moshi_transcript'])} chars): {result['moshi_transcript'][:200]}")
        logger.info(f"Gemini transcript ({len(result['gemini_transcript'])} chars): {result['gemini_transcript'][:200]}")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
