# Security Policy

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Instead use
GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/emotion-machine-org/personaplex-finetune/security)
  of this repo.
2. Click **Report a vulnerability**.
3. Describe the issue, reproduction steps, and impact.

We aim to acknowledge within 7 days. This repo is research code released
as-is — there is no SLA for fixes, but we will coordinate disclosure.

## Scope

In scope:

- Supply-chain risks in our pipeline scripts (`pipeline/`).
- Auth / token handling in the inference server (`personaplex/moshi/moshi/server.py`).
- Secret leakage in committed code or git history.

Out of scope (report upstream):

- Vulnerabilities in vendored Kyutai code (`moshi-finetune/`, `personaplex/moshi/`).
Report at [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi/issues).
- Vulnerabilities in PyTorch, transformers, or other third-party deps.  
Report to the project upstream.

&nbsp;