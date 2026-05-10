"""Test both A/B models by connecting via WebSocket and checking for audio/text responses."""
import argparse
import asyncio
import ssl
import time
from urllib.parse import urlencode

import numpy as np
import aiohttp

PARAMS = {
    "text_temperature": "0.6",
    "text_topk": "20",
    "audio_temperature": "0.65",
    "audio_topk": "150",
    "pad_mult": "0",
    "text_seed": "12345",
    "audio_seed": "12345",
    "repetition_penalty_context": "64",
    "repetition_penalty": "1",
    "text_prompt": "You are Emma, a friendly AI companion.",
    "voice_prompt": "NATF0.pt",
    "greeting": "Hello there.",
}

async def test_model(ws_base: str, model_name: str, model_param: str):
    """Connect to a model, send silence, check for response."""
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    params = {**PARAMS, "model": model_param}
    url = ws_base + "?" + urlencode(params)

    print(f"\n{'='*60}")
    print(f"Testing {model_name} (model={model_param})")
    print(f"{'='*60}")

    try:
        import sphn
        opus_writer = sphn.OpusStreamWriter(24000)
        opus_reader = sphn.OpusStreamReader(24000)
    except ImportError:
        print("sphn not available, using raw test")
        opus_writer = None
        opus_reader = None

    session = aiohttp.ClientSession()
    try:
        ws = await session.ws_connect(url, ssl=ssl_ctx, timeout=30)
        print(f"  WebSocket connected")

        # Wait for handshake
        handshake_received = False
        audio_received = 0
        text_received = []
        start = time.time()

        # Send some silence frames to keep the connection alive
        if opus_writer:
            silence = np.zeros(1920, dtype=np.float32)  # 80ms of silence
            for _ in range(5):
                opus_writer.append_pcm(silence)
            silence_bytes = opus_writer.read_bytes()
            if len(silence_bytes) > 0:
                await ws.send_bytes(b"\x01" + silence_bytes)
                print(f"  Sent {len(silence_bytes)} bytes of silence")

        # Listen for responses for up to 15 seconds
        while time.time() - start < 15:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                # Send more silence to keep connection alive
                if opus_writer:
                    silence = np.zeros(1920 * 4, dtype=np.float32)
                    opus_writer.append_pcm(silence)
                    silence_bytes = opus_writer.read_bytes()
                    if len(silence_bytes) > 0:
                        await ws.send_bytes(b"\x01" + silence_bytes)
                continue

            if msg.type == aiohttp.WSMsgType.BINARY:
                data = msg.data
                kind = data[0]
                if kind == 0x00:
                    handshake_received = True
                    elapsed = time.time() - start
                    print(f"  Handshake received ({elapsed:.1f}s)")
                elif kind == 0x01:
                    audio_received += len(data) - 1
                elif kind == 0x02:
                    text = data[1:].decode("utf-8")
                    text_received.append(text)
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                print(f"  Connection closed: {msg.type}")
                break

        await ws.close()
    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        await session.close()

    elapsed = time.time() - start
    print(f"\n  Results after {elapsed:.1f}s:")
    print(f"  Handshake: {'YES' if handshake_received else 'NO'}")
    print(f"  Audio bytes received: {audio_received}")
    print(f"  Text tokens received: {len(text_received)}")
    if text_received:
        print(f"  Text: {''.join(text_received)}")

    ok = handshake_received and audio_received > 0
    print(f"  STATUS: {'PASS' if ok else 'FAIL'}")
    return ok


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ws-base",
        default="wss://localhost:8998/api/chat",
        help="Moshi WebSocket API base URL.",
    )
    args = parser.parse_args()

    print("Testing A/B models on the server...")

    # Test primary (finetuned) model
    primary_ok = await test_model(args.ws_base, "Finetuned (primary)", "primary")

    # Small gap between tests
    await asyncio.sleep(2)

    # Test ab (base) model
    ab_ok = await test_model(args.ws_base, "Base (ab)", "ab")

    print(f"\n{'='*60}")
    print(f"SUMMARY: primary={'PASS' if primary_ok else 'FAIL'}, ab={'PASS' if ab_ok else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
