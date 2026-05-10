# A/B Testing UI — Handoff & Fresh Start Plan

## What We're Building
An A/B testing UI for PersonaPlex voice models. Users talk to two models sequentially (randomized order), then rate which they preferred. Results are stored as JSON for analysis.

## Current State (What Works)
- **A/B UI flow**: Config → Model A conversation → Model B conversation → Rate → Submit. Randomized assignment. Step indicators. All in `Queue.tsx`.
- **Concentric ring orb visualizers**: Replaced old bar equalizers. Server orb (green), user mic orb (blue). Files: `ServerVisualizer.tsx`, `ClientVisualizer.tsx`.
- **Chat bubble transcript**: Assistant text shown as left-aligned gray bubbles, grouped by 800ms pauses. Files: `TextDisplay.tsx`, `useServerText.ts`, `ChatContext.ts`, `useChat.ts`.
- **Dual model server**: Single server loads two models on cuda:0 and cuda:1. Routes via `?model=primary` or `?model=ab` query param. Session eviction when new connection arrives. Files: `server.py`.
- **A/B results storage**: POST `/api/ab-result` → appends to `ab_results.json`. GET `/api/ab-results` to read.
- **AirPods mic fix**: `useUserAudio.ts` passes `sourceNode` to opus-recorder instead of `mediaTrackConstraints`, so both the visualizer and recorder use the same `getUserMedia` stream.
- **MediaStream cleanup**: `UserAudio.tsx` stops all MediaStream tracks on unmount so the next conversation gets a clean mic.
- **Decoder pre-warm**: `advanceToB()` in `Queue.tsx` pre-warms a decoder worker before Model B mounts.
- **Null check fix**: `server.py` line ~253 checks `if pcm is None` before accessing `.shape` (sphn's `read_pcm()` can return None).
- **CUDA graph disable**: Server must be launched with `NO_CUDA_GRAPH=1` env var, otherwise the second GPU's model generates silence due to broken CUDA graph capture in multi-GPU processes.

## What's Broken / Needs Fixing
1. **Audio sizzle/noise came back.** The initial jitter buffer fixes (audio-processor.ts) were good but something regressed. The internal Pipecat-based reference implementation (raw PCM, Pipecat transport) has clean audio. Our Opus encode/decode chain needs careful buffer management.
2. **Model B sometimes doesn't respond** after switching from Model A. This may be related to (1) — if the audio pipeline state (worklet, decoder, opus streams) isn't cleanly reset between conversations, the second session gets garbage.

## Key Audio Pipeline (Where Noise Comes From)
```
Client Mic → getUserMedia → opus-recorder (Opus encode, 24kHz, 20ms frames)
  → WebSocket binary (0x01 + Opus pages)
  → Server: sphn.OpusStreamReader → PCM float32 → Mimi encode → LM step → Mimi decode → sphn.OpusStreamWriter
  → WebSocket binary (0x01 + Opus pages)  
  → Client: libopus WASM decoder → Float32 PCM → AudioWorkletProcessor (jitter buffer) → speakers
```

### audio-processor.ts — Current Buffer Settings (Modified)
- `initialBufferSamples = 2 * frameSize` (160ms, was 80ms)
- `partialBufferSamples = 40ms` (was 10ms)  
- `maxBufferSamples = 300ms` (was 10ms)
- Added crossfade on packet drops (64-sample fade-in)
- These were the RIGHT changes for network jitter. But something else may have introduced noise.

### decoderWorker.ts — Opus Decoder
- `resampleQuality: 5` (was 0) — this is good, keep it
- Resamples from 24kHz → browser AudioContext rate (typically 48kHz)

### useUserAudio.ts — Opus Encoder  
- `encoderComplexity: 5` (was 0) — this is good, keep it
- `sourceNode: source` instead of `mediaTrackConstraints` — prevents dual getUserMedia

### Queue.tsx — AudioContext
- Currently `new AudioContext()` (browser default rate, typically 48kHz)
- We tried `new AudioContext({ sampleRate: 24000 })` but it caused no audio output on macOS

## Emotion-Machine Reference (Clean Audio)
The working reference (a separate Pipecat-based prototype, not in this repo) uses:
- **Raw PCM Int16** over WebSocket (no Opus codec at all)
- Pipecat's `FastAPIWebsocketTransport` with `add_wav_header=False`
- Input: 16kHz, Output: 24kHz
- `RawAudioSerializer` handles serialization
- No manual jitter buffer — Pipecat handles it

## Files Changed (from git diff)
### Client (personaplex/client/src/)
- `audio-processor.ts` — jitter buffer sizes, crossfade
- `index.css` — simplified layout (removed split-layout)
- `pages/Queue/Queue.tsx` — full rewrite for A/B testing UI
- `pages/Conversation/Conversation.tsx` — layout changes, ChatContext, modelQuery prop
- `pages/Conversation/ChatContext.ts` — NEW: shared chat message state
- `pages/Conversation/hooks/useChat.ts` — NEW: manages assistant + user messages
- `pages/Conversation/hooks/useServerText.ts` — uses ChatContext
- `pages/Conversation/hooks/useUserAudio.ts` — sourceNode fix
- `pages/Conversation/components/AudioVisualizer/ServerVisualizer.tsx` — concentric rings
- `pages/Conversation/components/AudioVisualizer/ClientVisualizer.tsx` — concentric rings
- `pages/Conversation/components/ServerAudio/ServerAudio.tsx` — sizing
- `pages/Conversation/components/UserAudio/UserAudio.tsx` — removed SpeechRecognition, MediaStream cleanup
- `pages/Conversation/components/TextDisplay/TextDisplay.tsx` — chat bubbles

### Server (personaplex/moshi/moshi/)
- `server.py` — dual model loading (--ab-moshi-weight, --ab-device), model routing, session eviction, .pt voice prompt handling, null check for opus_reader, A/B result endpoints

## How to Launch
```bash
NO_CUDA_GRAPH=1 CUDA_VISIBLE_DEVICES=0,1 python -m moshi.server \
  --host 0.0.0.0 --port 8998 --device cuda:0 \
  --moshi-weight /path/to/finetuned/model.safetensors \
  --ab-moshi-weight /path/to/base/model.safetensors \
  --ab-device cuda:1 --model-label finetuned --ab-label base \
  --static /path/to/client/dist \
  --voice-prompt-dir /path/to/voices \
  --ssl /path/to/certs
```

## What To Do Next (Fresh Start Priorities)
1. **Fix audio noise/sizzle.** Compare the current audio-processor.ts against the internal Pipecat-based's approach. Consider whether the Opus encode/decode chain is adding artifacts, or if the jitter buffer behavior is still too aggressive. Test with the browser's built-in audio tools (chrome://media-internals).
2. **Fix Model B reliability.** When switching from A→B, ensure the AudioWorkletProcessor state, decoder worker, and opus stream are fully clean. The `worklet.port.postMessage({type: "reset"})` may not be sufficient.
3. **Consider simplifying the audio path.** The internal Pipecat-based uses raw PCM (no Opus). If Opus is causing artifacts, consider switching to raw PCM for the WebSocket audio — simpler, no codec artifacts, at the cost of ~10x more bandwidth (which is fine on LAN).
