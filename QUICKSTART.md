# Voice Training Quickstart

## Running the Inference Server

### Prerequisites

```bash
# Install Opus codec
sudo apt install libopus-dev

# Install PersonaPlex server
cd personaplex/moshi
pip install .

# Build the client UI (once, or after UI changes)
cd personaplex/client
npm install && npm run build
```

### Launch

```bash
cd personaplex/moshi
.venv/bin/python -m moshi.server \
  --moshi-weight /mnt/data/runs/<run_name>/merged/model.safetensors \
  --static ../../personaplex/client/dist
```

Access at `https://localhost:8998`. The UI has a config panel on the left (system prompt, voice, generation params) and the conversation on the right.

## Merging a LoRA

After training, merge LoRA adapter weights into the base model:

```bash
python pipeline/merge_lora.py \
  --checkpoint runs/<run_name>/checkpoints/checkpoint_003000/consolidated \
  --output runs/<run_name>/merged/model.safetensors \
  --hf-repo nvidia/personaplex-7b-v1
```

This:

1. Loads LoRA weights (`lora.safetensors`) and config from the checkpoint dir
2. Downloads the base PersonaPlex model from HuggingFace
3. Computes `delta = (B @ A) * scaling` for each LoRA layer and adds it to base weights
4. Saves a single merged `model.safetensors` in bfloat16

The merged file can be used directly with `--moshi-weight` in both `moshi.server` and `moshi.offline`.

## Differences from Upstream PersonaPlex

This fork adds a custom client UI and several training/inference features on top of the NVIDIA PersonaPlex codebase.

### Client UI Changes

- **Two-column layout**: Config panel (system prompt, voice, greeting, generation params) always visible on the left; conversation on the right
- **Start/stop without reload**: Change prompt and start a new conversation without refreshing the page
- **Preset support**: One-click preset buttons that set both prompt and greeting
- **Generation controls**: Text/audio temperature, top-k, and repetition penalty editable from the UI

### Inference Changes

- **Context injection (puppeteer)**: Inject text context mid-conversation via `LMGen.inject_context()`. Agent audio is forced to silence tokens during injection to prevent corruption. Supported in `bot_to_bot.py` and `server.py` via WebSocket JSON messages.
- **Screech detection & recovery** (`screech_detector.py`): Spectral analysis detects high-frequency audio corruption and auto-recovers.

### Training Changes

- **LoRA with depformer freezing** (`skip_depformer: true`): Freezes the 6-layer depformer audio codec LoRA (43% of params) to preserve audio quality. Only backbone transformer LoRA is trained.
- **Context injection in training** (`interleaver.py`): Splices `<context>` tokens into the text stream at frame offsets. Loss is masked for prompt and injection regions. User audio is preserved during injection (not blanked).
- **System prompt masking**: Loss is masked over the system prompt region so the model doesn't learn to generate the prompt itself.
- **Generation eval** (`gen_eval.py`): Gemini Live ↔ Moshi evaluation with Claude LLM review for automated quality scoring.

### Data Pipeline (`pipeline/`)

End-to-end synthetic data generation: dialogue generation (Claude API) -> TTS rendering (VibeVoice) -> WhisperX alignment -> stereo routing -> injection offset computation -> manifest creation.
