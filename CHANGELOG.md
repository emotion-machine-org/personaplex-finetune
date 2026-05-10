# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Initial public release scaffolding: `LICENSE`, `THIRD_PARTY_NOTICES.md`,
  `SECURITY.md`, `CHANGELOG.md`, `AGENTS.md`, `CLAUDE.md`,
  `.cursor/rules/{training,pipeline,client}.mdc`.
- Top-level `pyproject.toml` (uv-managed, ruff-configured) with optional
  extras for `eval`, `train`, `dev`, `wandb-export`.
- License section in `README.md` clarifying base-model weight licenses
  (Moshi CC-BY-4.0, PersonaPlex NVIDIA Open Model License) vs. code
  licenses.

### Changed
- Renamed `adhery` → `pharma` across configs, eval prompts, reports, and
  docs to remove partner-specific naming.
- Genericized hardcoded GCP project default in
  `personaplex/moshi/moshi/server.py` (now reads `GOOGLE_CLOUD_PROJECT_ID`
  with empty fallback).
- Moved frozen session notes into `docs/history/notes/` (was top-level).
- Moved one-off utilities into `pipeline/utils/`.

### Removed
- Obsolete config variants (`adhery_v2_*`, `companion_training_v*`,
  `outbound_insurance`, test configs).
- Vendored arxiv markdown copies (`resources/{moshi,personaplex}_paper.md`).
- Internal planning notes that referenced infrastructure (`plans/companion_v6_podcast_training.md`).

## [0.1.0] — TBD

First public release. See [Unreleased] above; this entry will be filled
when we tag.
