---
language:
  - en
license: other
license_name: synthetic-mixed
pretty_name: "PersonaPlex Finetune Datasets"
size_categories:
  - 1K<n<10K
task_categories:
  - text-to-speech
  - automatic-speech-recognition
tags:
  - synthetic
  - duplex
  - voice
  - context-injection
---

# Dataset Card — Voice Training Workspace

This directory holds the on-disk training datasets. **It is gitignored**
— the data is too large to track and its license depends on which sources
you used to build it. The pipeline in [`pipeline/`](../pipeline/) is the
canonical recipe for regenerating equivalent data from scratch.

## On-disk layout

```
data/
├── <dataset>/
│   ├── mono_wav/<id>.wav                 VibeVoice-rendered mono audio
│   ├── stereo_wav/<id>.{wav,json}        WhisperX-aligned stereo + sidecar
│   └── dataset/{train,eval}.jsonl        Manifest pointing at stereo files
└── scripts/
    └── generate_system_prompts.py        System-prompt generator (Claude)
```

Each sidecar JSON carries:

- `text_prompt` — system prompt prepended at training time.
- `voice_prompt` — optional voice-conditioning frames.
- `alignments` — WhisperX word-level timing and speaker labels
  (`SPEAKER_BROKER` for agent, `SPEAKER_CLIENT` for user).
- `context_injections` (pharma only) — list of `{after_turn, text}`
  records that splice context into the agent's text stream at runtime;
  frame offsets are computed by `pipeline/compute_injection_offsets.py`.

## Datasets used in the published experiments

| Dataset | Domain | Samples | Mean duration | Source |
|---|---|---|---|---|
| `companion-v6/` | Casual chat | 1,000 | 180 s | Synthetic (Claude → VibeVoice) |
| `podcast-curated/` | Conversational | 564 | varies | Public podcast clips, curated index |
| `pharma-v2/` | Patient support | 2,003 | 296 s | Synthetic with context injection |

## Generation recipe

See [`README.md`](../README.md) "Pipeline steps" section. Briefly:

1. `pipeline/generate_dialogues_sync.py` — Claude API, LHS-sampled persona
   parameters, produces JSONL of dialogues + (optional) `context_injections`.
2. `pipeline/parse_dialogues.py` — JSONL → VibeVoice script format.
3. `pipeline/generate_audio.py` — multi-GPU VibeVoice TTS to mono WAV.
4. `pipeline/create_stereo.py` — WhisperX alignment + channel routing.
5. `pipeline/compute_injection_offsets.py` — frame-offset annotation.
6. `pipeline/create_manifest.py` — dataset manifest generation.

## Provenance / what we did NOT include

- **No real customer or patient data.** All dialogues are synthetic.
- **No personally identifying information** in the synthetic transcripts —
  any names / phone numbers / addresses are fabricated by the LLM.
- **No copyrighted podcast audio** is redistributed; the `podcast-curated`
  set is a list of public IDs + alignment, not the audio itself.

## Bias / limitations

- English only; American English over-represented (VibeVoice voice
  library bias).
- Synthetic dialogues are fluent but stylized — distributional shift vs.
  real human conversation is real.
- Sample length distribution is bimodal (short companion ≈180 s vs. long
  pharma ≈300 s). Models trained on one length may not transfer.

## Licensing

- **Code / scripts in `pipeline/`**: MIT (this repo's license).
- **Synthetic transcripts you generate**: bound by the
  [Anthropic API usage terms](https://www.anthropic.com/legal/aup) (you
  may use outputs commercially; see the Anthropic Commercial Terms).
- **Synthetic audio**: bound by the
  [VibeVoice license](https://github.com/microsoft/VibeVoice).
- **Curated podcast indices**: redistribute as IDs only, not as audio.
