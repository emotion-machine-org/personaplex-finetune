# Third-Party Notices

This repository vendors source from several third-party projects. Each tree
retains its original license file. Local modifications have been applied —
see the "Modifications" notes below for what we changed.

---

## `personaplex/`

Vendored from NVIDIA's PersonaPlex reference implementation, which itself is
derived from Kyutai's `moshi` package. Includes Kyutai code under
`personaplex/moshi/` and a custom React/TypeScript client under
`personaplex/client/`.

- **Upstream — model weights:** [`nvidia/personaplex-7b-v1`](https://huggingface.co/nvidia/personaplex-7b-v1) (HuggingFace)
- **Upstream — base codebase (Moshi):** https://github.com/kyutai-labs/moshi
- **Licenses:**
  - `personaplex/moshi/LICENSE.moshi` — MIT (Copyright © Kyutai)
  - `personaplex/moshi/LICENSE.audiocraft` — MIT (Copyright © Meta Platforms, Inc.)
  - `personaplex/client/LICENSE` — see file
- **Modifications in this repo:**
  - `personaplex/moshi/moshi/models/loaders.py` — patched `get_moshi_lm()` to
    auto-detect `model_type == "personaplex"` and set `dep_q = 16`. Patched
    `get_mimi()` to cap `num_codebooks` at `n_q // 2` so the codec doesn't
    bloat when `dep_q` doubles.
  - Checkpoint weight conversion (`StreamingMultiheadAttention`,
    `LoRALinear`) — added `_load_hook` chain to convert monolithic
    `in_proj_weight` to per-step `in_projs.{i}.frozen_W.weight`.
  - `personaplex/moshi/moshi/models/lm.py` — added `LMGen.inject_context()`
    plus `_injecting_context` / `_context_queue` machinery so an external
    puppeteer LLM can splice `<context>...</context>` tokens into the text
    stream mid-conversation. Agent audio is forced to silence tokens during
    injection.
  - `personaplex/moshi/moshi/server.py` — added a JSON WebSocket message
    type for context injection.
  - `personaplex/moshi/moshi/bot_to_bot.py` — added `--context-injections`
    flag and the puppeteer scaffolding.
  - `personaplex/moshi/moshi/screech_detector.py` — new module: spectral
    detection + recovery for high-frequency audio corruption.
  - `personaplex/client/` — extensively rewritten UI: two-column layout with
    a config panel (system prompt, voice, generation params), preset
    buttons, A/B testing UI, and start/stop without page reload.

## `moshi-finetune/`

Vendored from Kyutai's `moshi-finetune`.

- **Upstream:** https://github.com/kyutai-labs/moshi-finetune
- **License:** `moshi-finetune/LICENSE` — Apache License 2.0 (Copyright © Kyutai)
- **Modifications in this repo:**
  - `finetune/args.py` — added `SystemPromptArgs`, `PuppeteerArgs`,
    `GenEvalArgs` configs.
  - `finetune/data/interleaver.py` — added voice + text system-prompt
    prefixing and mid-conversation `<context>` token splicing at frame
    offsets.
  - `finetune/loss.py` — masks loss over the system-prompt region (via
    `prompt_lengths`) and over context-injection regions (text loss only,
    via `context_masks`).
  - `finetune/train.py`, `finetune/eval.py` — wired the new args/masks.
  - `finetune/gen_eval.py` — Gemini Live ↔ Moshi generation eval at
    checkpoint time with WhisperX conversation profiling and LLM transcript
    review.
  - `example/moshi_7B.yaml` — added the new system-prompt block.

## `VibeVoice/`

Community-maintained fork of Microsoft VibeVoice TTS, used for synthetic
training audio generation (step 3 of the data pipeline).

- **Upstream:** the original Microsoft repo was removed; this is the
  community fork referenced in `VibeVoice/README.md`.
- **License:** MIT — see [`VibeVoice/LICENSE`](VibeVoice/LICENSE)
  (Microsoft's original MIT license, preserved by the community fork).
- **Modifications:** none of substance — used as-is via `pipeline/generate_audio.py`.

---

## Other dependencies

The `requirements.txt` and `pyproject.toml` files pull additional packages
(WhisperX, pyannote, anthropic, google-generativeai, flash-attn, triton,
torchcodec, sentencepiece, sphn, etc.) under their respective licenses. See
each package's homepage for license terms.

## Pretrained model weights

This repository does **not** redistribute model weights. Users download:

- `nvidia/personaplex-7b-v1` from HuggingFace (subject to NVIDIA's model
  license on that page).
- `vibevoice/VibeVoice-7B` from the VibeVoice fork (subject to its
  license).
- `pyannote/segmentation` and Whisper / WhisperX models on first use of
  the alignment pipeline.

Read each model card before using the weights for anything other than
research.
