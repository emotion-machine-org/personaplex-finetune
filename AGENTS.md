# AGENTS.md

Guidance for AI coding agents (Claude Code, Cursor, Codex, Aider, Gemini
CLI, Copilot) working in this repository. Humans should also read this —
it's the fastest way to get the lay of the land.

> **The two source-of-truth docs you should follow for the workflow are
> [`README.md`](README.md) (pipeline overview, commands, architecture)
> and [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) (zero-to-trained on
> a fresh GCP VM).** This file is a *navigation aid* — it points you at
> the right places and codifies conventions. It is not a replacement for
> the README.

## What this repo is

A finetuning workspace for **Moshi 7B** ([kyutai-labs/moshi](https://github.com/kyutai-labs/moshi))
and its NVIDIA variant **PersonaPlex** ([nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1)).
Two domains have been used to validate the pipeline: a *companion* chat
finetune and a *pharma* patient-support finetune with mid-conversation
context injection (a "puppeteer" LLM splices facts into the model's text
stream while audio plays). Architectural quirks (frame layout,
`dep_q=16`, weight-loader hook chain, system-prompt prefix) are
described in [`docs/history/notes/DUPLEX_AND_FINETUNING_NOTES.md`](docs/history/notes/DUPLEX_AND_FINETUNING_NOTES.md)
— read that before changing anything in `moshi-finetune/finetune/` or
`personaplex/moshi/moshi/`.

## End-to-end workflow (and where each step is documented)

The full pipeline is **8 stages**, all documented in detail in
[`README.md`](README.md) under "Pipeline steps". Brief map:

| # | Stage | Script | Output | README §  |
|---|---|---|---|---|
| 1 | Generate synthetic dialogues (Claude API, LHS-sampled personas) | `pipeline/generate_dialogues_sync.py` | `<dataset>.jsonl` | 1 |
| 2 | Parse JSONL → "Speaker N:" plain-text scripts | `pipeline/parse_dialogues.py` | `scripts/<id>.txt` | 2 |
| 3 | Render scripts → mono audio (VibeVoice 7B TTS, multi-GPU) | `pipeline/generate_audio.py` | `mono_wav/<id>.wav` | 3 |
| 4 | WhisperX align + route to stereo channels | `pipeline/create_stereo.py` | `stereo_wav/<id>.{wav,json}` | 4 |
| 5 | Compute frame offsets for context injections | `pipeline/compute_injection_offsets.py` | annotates `stereo_wav/<id>.json` | 5 |
| 6 | Build train/eval manifest | `pipeline/create_manifest.py` | `dataset/{train,eval}.jsonl` | 6 |
| 7 | Finetune (FSDP + LoRA, multi-GPU torchrun) | `moshi-finetune` | `runs/<run-name>/` | 7 |
| 8 | Merge LoRA + serve via WebSocket | `pipeline/merge_lora.py`, `personaplex/moshi/moshi/server.py` | merged checkpoint, live server | 8 |

Eval (LLM judge over generated transcripts, A/B preference) and run
comparison are documented in README §"Evaluation".

For a fresh-machine setup (NVIDIA drivers, CUDA 12.8, Python envs,
HuggingFace cache, gated model downloads, training launch on a GCP VM),
follow [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md). It walks the entire
journey from a blank Debian 12 box through a finished training run.

## Project map

```
configs/                        Training recipes (YAML). One file = one experiment.
data/                           (gitignored) Training audio + manifests. See "Data structure" below.
docs/
├── SETUP_GUIDE.md              Fresh-VM zero-to-trained walkthrough.
└── history/                    Frozen development artifacts (notes + wandb runs).
moshi-finetune/                 Vendored Kyutai finetune code (Apache-2.0). DO NOT pull from upstream.
personaplex/                    Vendored NVIDIA PersonaPlex code (MIT). DO NOT pull from upstream.
  moshi/                        Patched moshi runtime (dep_q=16, context injection).
  client/                       React/TS UI for serving + A/B testing.
pipeline/                       Top-level orchestration scripts (data prep, eval, manifest).
  utils/                        One-off / debugging utilities.
plans/                          (gitignored) Local planning scratch.
runs/                           (gitignored) Training outputs. See "Run structure" below.
VibeVoice/                      Vendored TTS used for synthetic data generation.
```

## Data structure (canonical example: `data/adhery-short/`)

A single dataset is a directory under `data/` with these subdirs, each
produced by a different pipeline stage:

```
data/<dataset>/
├── <dataset>.jsonl              Stage 1 output: source synthetic dialogues
├── scripts/<id>.txt             Stage 2: VibeVoice-format "Speaker 1:/Speaker 2:" plain text
├── speaker_samples/*.wav        Voice library (one wav per voice; deterministic per-dialogue assignment)
├── mono_wav/<id>.wav            Stage 3: VibeVoice-rendered single-channel audio (24 kHz)
├── stereo_wav/<id>.wav          Stage 4: 2-channel WAV (L=agent, R=user)
├── stereo_wav/<id>.json         Stage 4 sidecar: alignments + turns + prompts (see below)
└── dataset/
    ├── train.jsonl              Stage 6: training manifest
    └── eval.jsonl               Stage 6: held-out eval manifest
```

### Source dialogue row (`<dataset>.jsonl`, one row per dialogue)

```json
{
  "id": "adhv3-00000",
  "assistant_name": "Mia",
  "seed": {
    "patient_name": "...", "patient_age": 72, "drug": "Enzalutamide ...",
    "scenario_type": "edge-case", "tenor": "urgent ...",
    "outcome": {"label": "deferred_resolution", "description": "..."},
    "injection_count": 0, "turn_count_target": 24, /* ... */
  },
  "system_prompt": "...",
  "dialogue": [{"speaker": "BROKER", "text": "Hi, ..."}, ...],
  "context_injections": [{"after_turn": 5, "text": "directive coaching ..."}, ...]
}
```

### Stereo sidecar (`stereo_wav/<id>.json`)

```json
{
  "text_prompt": "...",                      // system prompt prepended at training time
  "turns": [
    {"index": 0, "speaker": "SPEAKER_BROKER", "start": 0.318, "end": 2.101, "text": "..."},
    {"index": 1, "speaker": "SPEAKER_CLIENT", "start": 2.741, "end": 4.925, "text": "..."}
  ],
  "alignments": [
    ["Hi,", [0.318, 0.638], "SPEAKER_BROKER"],          // word-level, WhisperX
    /* ... */
  ],
  "context_injections": [
    {"after_turn": 5, "text": "...", "frame_offset": 312}    // frame_offset added by stage 5
  ]
}
```

Speaker labels are **fixed**: `SPEAKER_BROKER` (agent / left channel) and
`SPEAKER_CLIENT` (user / right channel). The interleaver in
`moshi-finetune/train.py` builds with `keep_main_only=True` and defaults
the main speaker to `SPEAKER_BROKER` — do not rename these.

### Manifest row (`dataset/{train,eval}.jsonl`, one row per sample)

```json
{"path": "/abs/path/to/stereo_wav/adhv3-00114.wav", "duration": 82.5333}
```

The trainer reads `path`, opens the sibling `.json` for prompts +
alignments, and uses `duration` to slice into chunks of `duration_sec`
(see config).

## Run structure (canonical example: `runs/adhery-v2-22/`)

A training run produces a directory under `runs/` with this layout:

```
runs/<run-name>/
├── args.yaml                              Frozen launch config (full training args)
├── metrics.train.jsonl                    1 row per step: loss, lr, mem, audio_cb*_loss, ctx_*, etc.
├── metrics.eval.jsonl                     1 row per eval step: eval_loss, perplexity, eval_*_pct
├── tb/                                    TensorBoard event files (train + eval)
├── checkpoints/checkpoint_<NNNNNN>/       Per-`ckpt_freq` checkpoint directories
│   ├── config.json                          model config snapshot
│   ├── lora.safetensors                     LoRA adapter weights ONLY (not the full model)
│   ├── train_state.pt                       optimizer + scheduler state for `--resume`
│   └── consolidated/                        full merged weights (created post-hoc by `pipeline/merge_lora.py`)
├── gen_eval/step_<NNNNNN>/                Per-step generation eval (every `gen_eval.freq`)
│   ├── results.json                         LLM judge scores
│   └── dialogues/                           generated audio + per-prompt JSON + stderr
│       ├── eval_<prompt_name>.mp3
│       ├── eval_<prompt_name>_prompt.json
│       ├── eval_<prompt_name>_injections.json    (if puppeteer enabled)
│       └── eval_<prompt_name>_stderr.log
└── wandb/run-<YYYYMMDD_HHMMSS>-<run_id>/  One dir per resume (multiple if the run was restarted)
```

Notes:

- `lora.safetensors` is **just the adapter** (~500 MB at rank 128).
  Use `pipeline/merge_lora.py` to produce a `consolidated/model.safetensors`
  if you want a single deployable weight file.
- `metrics.train.jsonl` rows include per-codebook losses
  (`audio_cb0_loss` through `audio_cb15_loss`), context-injection counters
  (`inj_total`, `inj_placed`, `inj_truncated`, `ctx_injected_pct`), and
  PAD-token diagnostics (`pred_pad_pct`, `real_token_pct`,
  `text_loss`/`audio_loss` split). Useful when diagnosing PAD collapse
  or screech.
- `gen_eval/step_NNNNNN/` is also gitignored implicitly (`runs/` is
  excluded). For historical eval results refer to
  [`docs/history/notes/combined_experiment_report.md`](docs/history/notes/combined_experiment_report.md)
  and the exported wandb archive in [`docs/history/wandb/`](docs/history/wandb/).

## Conventions

- **Configs are flat** in `configs/`. One YAML = one named experiment.
  Always include a top comment block: purpose, hardware required,
  expected runtime.
- **Paths in configs are placeholders** (`<EDIT_ME>/data/...`). Don't
  hardcode personal paths. Don't use `~` or env-var expansion —
  moshi-finetune doesn't support either.
- **Environment variables** in `.env` (gitignored). Template at
  `.env.example`. Read via `os.environ.get(...)` — no `dotenv` import in
  scripts.
- **Secrets**: never commit. If you see a key in code, treat it as
  compromised and rotate at the issuing console.
- **Vendored code** (`moshi-finetune/`, `personaplex/`, `VibeVoice/`) is
  intentionally frozen. Edit it freely for local needs; do **not** treat
  it as upstream to merge from.
- **Logs**: wandb is the canonical training log. Per-run `args.yaml` +
  `metrics.{train,eval}.jsonl` are the local mirror.

## Do / don't

**Do:**

- Use `uv` for Python deps. Pin numerics-affecting deps exactly (`torch`,
  `flash-attn`, `transformers`); use ranges for the rest.
- Add a smoke variant for every config (`<config>_smoke.yaml`, ≤10 min on
  1 GPU).
- Cite this repo and the base models (Moshi, PersonaPlex) when you
  publish results — see the Citation section in `MODEL_CARD.md`.

**Don't:**

- Commit `.env`, `runs/`, `data/`, `*.pem`, `certs/`, `*.wav` (already
  gitignored).
- Train on `dep_q=8` with PersonaPlex weights — PersonaPlex needs
  `dep_q=16` (see `personaplex/moshi/moshi/loaders.py` `_load_hook`
  chain).
- Push large binaries. `*.pt`, `*.wav`, `*.mp4` are LFS-tracked but the
  default is "don't commit binaries" — regenerate from the pipeline.

## Where to look first

| You want to... | Look at |
|---|---|
| Run the whole pipeline end-to-end | [`README.md`](README.md) "Pipeline steps" |
| Set up a fresh VM | [`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md) |
| Understand the architecture (3 streams, 17 channels, 12.5 Hz) | [`README.md`](README.md) "Architecture" + [`docs/history/notes/DUPLEX_AND_FINETUNING_NOTES.md`](docs/history/notes/DUPLEX_AND_FINETUNING_NOTES.md) |
| Understand context injection (puppeteer) | `personaplex/moshi/moshi/lm.py::LMGen.inject_context`, [`docs/history/notes/puppeteer.md`](docs/history/notes/puppeteer.md) |
| Understand PersonaPlex weight loading (`dep_q=16`) | `personaplex/moshi/moshi/loaders.py::_load_hook` |
| Add a new training experiment | Copy a `configs/*.yaml`, edit `<EDIT_ME>` paths, name distinctly |
| Add a new eval | Extend `pipeline/batch_eval.py` or `moshi-finetune/finetune/gen_eval.py` |
| Debug inference | [`docs/history/notes/INFERENCE_DEBUG_NOTES.md`](docs/history/notes/INFERENCE_DEBUG_NOTES.md) |
| See past experiment results | [`docs/history/notes/combined_experiment_report.md`](docs/history/notes/combined_experiment_report.md), [`docs/history/wandb/`](docs/history/wandb/) |
| Check a previous run's exact config | `docs/history/wandb/<project>/<run_id>/config.yaml` |
| Skim 59 historical runs in a spreadsheet | [`docs/history/wandb/summary.csv`](docs/history/wandb/summary.csv) |
