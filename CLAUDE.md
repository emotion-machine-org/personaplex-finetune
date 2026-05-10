# CLAUDE.md

This project follows the cross-agent [`AGENTS.md`](AGENTS.md) standard.
**Read [`AGENTS.md`](AGENTS.md) first** — it covers the workflow, project
map, conventions, and pointers into [`README.md`](README.md) and
[`docs/SETUP_GUIDE.md`](docs/SETUP_GUIDE.md), which are the
source-of-truth docs.

## Claude-specific notes

- **Model recommendation**: Use Opus 4.7 (1M context) when working in
  `moshi-finetune/` or `personaplex/moshi/` — the streaming-model code
  has enough cross-file invariants that smaller models sometimes propose
  changes that break weight-loading. Sonnet 4.6 is fine for
  `pipeline/` scripts and config edits.
- **Skills used in this repo**: `init`, `simplify`, `review`,
  `security-review`. Run `/security-review` before any push to `main`.
- **MCP servers**: `wandb` is configured (see `claude_desktop_config.json`
  if present) — useful for `mcp__wandb__query_wandb_tool` when
  investigating past runs without leaving the editor. The local mirror
  of the historical wandb runs is in `docs/history/wandb/`; new training
  should log to your own wandb entity.
