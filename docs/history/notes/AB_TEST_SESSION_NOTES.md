# A/B Test — Session Notes (2026-04-07)

## What Was Fixed

### 1. Audio Noise / Sizzle — Root Cause: Corrupt Opus Decoder State
The client's `decoderWorker.ts` sent a **fake Ogg BOS page** during "prewarm" with:
- Invalid CRC (all zeros)
- Wrong sample rate (48kHz header vs 24kHz decoder config)
- No comment page or audio following it

When the real Opus stream arrived from the server, the Ogg decoder was already in a broken state, causing sizzle/noise artifacts. Model B was worse because `advanceToB()` called `prewarmDecoderWorker()` which sent another corrupt BOS page to the new worker.

**Fix**: Removed the entire prewarm/BOS machinery. Each conversation gets a fresh decoder worker initialized with just the `init` command (no fake audio data).

### 2. Eliminated Opus Codec Entirely — Raw PCM Over WebSocket
The fundamental architecture problem: the audio pipeline had two unnecessary codec layers:
```
Server: PCM → sphn.OpusStreamWriter → Ogg/Opus pages → WebSocket
Client: WebSocket → Ogg pages → WASM Opus decoder → Float32 → AudioWorklet
```

Both an internal reference impl (`useWebSocketAudio.ts` + `WavStreamPlayer`) and the pipecat_server.py use **raw PCM** — no Opus at all. On LAN/Tailscale, raw PCM at 24kHz mono Int16 = 48 KB/s. Zero reason for Opus.

**Fix**: 
- **Server** (`server.py`): Replaced `sphn.OpusStreamWriter` with direct int16 PCM output. After `mimi.decode()`, convert float32→int16 and send raw bytes over WebSocket with `0x01` prefix.
- **Client** (`useServerAudio.ts`): Removed the decoder worker entirely. Audio messages are now raw int16 PCM at 24kHz — converted to Float32 and upsampled to 48kHz via linear interpolation, then fed directly to the AudioWorklet. No WASM, no Ogg parsing, no decode latency.
- Mic input still uses Opus (opus-recorder → server's `sphn.OpusStreamReader`) since that path works fine.

### 3. Removed Wasteful `other_mimi` From Server
Each `ServerState` had TWO mimi codec instances (`self.mimi` + `self.other_mimi`). Every audio frame was encoded and decoded through both, with the second one's output discarded (`_ = self.other_mimi.encode(chunk)`). This doubled GPU codec compute per frame for no reason.

**Fix**: Removed `other_mimi` from `ServerState.__init__`, `warmup()`, `handle_chat()`, and all `main()` instantiation. Each model now has exactly one mimi instance.

### 4. Fixed Audio Message Encoding (Client)
`encoder.ts` used `new Uint8Array([0x01, ...message.data])` which spread the entire audio payload byte-by-byte. Replaced with proper `Uint8Array.set()` concatenation.

### 5. Jitter Buffer Underrun Fix
The AudioWorklet's `audio-processor.ts` called `resetStart()` when the buffer emptied, which required re-accumulating 160ms+40ms=200ms before playing again. This caused periodic ~200ms silence gaps (the "spotty" pattern).

**Fix**: On underrun, just fade out gracefully and set `needsFadeIn=true`. Playback resumes immediately when the next packet arrives (~2ms later) instead of waiting 200ms.

### 6. Clean Model B Transition
`advanceToB()` now creates a **fresh AudioWorkletNode** before mounting Model B's conversation. This prevents stale buffered audio from Model A bleeding into Model B's start.

## Current Architecture

### Audio Pipeline (Output: Server → Client)
```
Server GPU: Mimi decode → float32 PCM → clip & convert to int16
  → WebSocket binary: 0x01 + raw int16 bytes (24kHz mono)
Client: receive bytes → int16→float32 → linear interpolation 24kHz→48kHz
  → AudioWorkletProcessor (jitter buffer) → speakers
```

### Audio Pipeline (Input: Client → Server)  
```
Client Mic: getUserMedia → opus-recorder (Opus encode, 24kHz, 20ms frames)
  → WebSocket binary: 0x01 + Opus/Ogg pages
Server: sphn.OpusStreamReader → PCM float32 → Mimi encode → LM step
```

### Server Launch Command
```bash
cd <REPO>/personaplex/moshi
NO_CUDA_GRAPH=1 CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m moshi.server \
  --host 0.0.0.0 --port 8998 --device cuda:0 \
  --moshi-weight <REPO>/runs/<your_run>/checkpoints/<ckpt>/model.safetensors \
  --ab-moshi-weight <HF_CACHE>/personaplex-7b-v1/model.safetensors \
  --ab-device cuda:1 \
  --static <REPO>/personaplex/client/dist \
  --voice-prompt-dir <HF_CACHE>/personaplex-7b-v1/voices \
  --ssl <ssl_dir>
```

Access at `https://<host>:8998` over a self-signed cert — accept the browser warning.

### Model Assignment
- `--moshi-weight` = **finetuned** (companion_training-v0-22, LoRA merged checkpoint) on cuda:0
- `--ab-moshi-weight` = **base** (PersonaPlex 7B from HF) on cuda:1
- Client randomizes order: `?model=primary` routes to finetuned, `?model=ab` routes to base
- Server logs `>>> Routing to PRIMARY/AB model` for each connection

### SSL Certs
Generate a self-signed cert and pass `--ssl <dir>` (the dir must contain `key.pem` + `cert.pem`). Required because AudioWorklet needs HTTPS when accessed via non-localhost IP.

## Files Changed
- `personaplex/moshi/moshi/server.py` — removed `other_mimi`, raw PCM output, model routing logs
- `personaplex/client/src/decoder/decoderWorker.ts` — removed prewarm/BOS page, simplified to init-only
- `personaplex/client/src/pages/Conversation/hooks/useServerAudio.ts` — raw PCM decode, no decoder worker
- `personaplex/client/src/audio-processor.ts` — graceful underrun handling
- `personaplex/client/src/protocol/encoder.ts` — proper typed array concatenation
- `personaplex/client/src/pages/Queue/Queue.tsx` — fresh worklet on Model B transition, removed prewarm calls

## Remaining Issues
- Some cracks/pops still reported on Model B (may be worklet transition related or model-specific)
- Could further improve by also switching mic input from Opus to raw PCM (eliminates last codec layer)
- The `buildURL` function in `Conversation.tsx` uses `useMemo` inside a non-component function and logs on every render — noisy but not a bug
