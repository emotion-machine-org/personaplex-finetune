# Development history

Frozen artifacts from the original development of this repo. Useful as
**context** for understanding why the code looks the way it does, what
was tried, and what failed. **Not** a workflow you're meant to reproduce
— modern users start from `README.md` and the `pipeline/` scripts.

## What's here

```
docs/history/
├── notes/    Session notes, debugging logs, design docs, A/B handoffs.
└── wandb/    Raw exports of every training run from the dev period (59 runs).
```

Both directories are read-only history. New work doesn't go here.

## Where to look first

| If you want to understand... | Read |
|---|---|
| Why training defaults look the way they do | [`notes/DUPLEX_AND_FINETUNING_NOTES.md`](notes/DUPLEX_AND_FINETUNING_NOTES.md) |
| The puppeteer / context-injection design | [`notes/puppeteer.md`](notes/puppeteer.md) |
| Inference debugging tales (PAD collapse, voice prompts, websockets) | [`notes/INFERENCE_DEBUG_NOTES.md`](notes/INFERENCE_DEBUG_NOTES.md) |
| Day-by-day timeline of what was tried | [`notes/logbook.md`](notes/logbook.md) |
| The A/B testing rig | [`notes/AB_TEST_HANDOFF.md`](notes/AB_TEST_HANDOFF.md), [`notes/AB_TEST_SESSION_NOTES.md`](notes/AB_TEST_SESSION_NOTES.md) |
| Every training run, with config + full metric history | [`wandb/`](wandb/) |
| A spreadsheet of all runs at a glance | [`wandb/summary.csv`](wandb/summary.csv) |

## Curated run highlights

Cross-references the wandb archive with the consolidated experiment
report at [`notes/combined_experiment_report.md`](notes/combined_experiment_report.md).

| Discovery | Run(s) | What it shows |
|---|---|---|
| **PAD-token collapse fix** | `companion-plex/keacuv5o` (v0.2) vs `gwg99nq7` (v0.1) | `text_padding_weight=0.0` eliminates PAD-collapse degeneracy |
| **L2 regularization on lora_B** | `adhery-demo/0irtky6i` (v2-7) | First stable pharma run; lr=1e-6, rank=32, L2=1e-4 |
| **LR stability boundary (pharma)** | `adhery-demo/m6uq6my8` (v2-8) | lr=5e-6 collapses by step 64, even with L2=4e-4 |
| **Best completed pharma run** | `adhery-demo/0irtky6i` (v2-7) | Eval scores 2.2/2.2/3.1, stable through 192 steps |
| **Rank 64 vs 32 at lr=1e-6** | `adhery-demo/woyp70qf` (v2-22) | Rank 64 learns ~10% faster without stability hit |
| **Old chunking strategy is NOT the magic** | `adhery-demo/1v6fvqoa` (v2-16) | Direct test ruled out chunking as the explanation for the original lr=2e-5 success |

Open any run's `config.yaml` to see the exact hyperparameters; load
`history.parquet` with pandas/polars to replot.

## Why we keep this

- **Provenance** — explains hyperparameter choices that look arbitrary.
- **Failure record** — the crashed runs are as informative as the
  successful ones; saves you from re-running the same dead end.
- **Reproducibility** — config + full unsampled metric history per run
  means you can replay any experiment.

This archive is a snapshot. It will not be updated as the repo evolves.
For ongoing work, refer to the live wandb projects (if you have access)
and any newer notes added under `docs/`.
