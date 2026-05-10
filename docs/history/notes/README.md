# Frozen session notes

These files are **not maintained**. They are session-time snapshots of
debugging work, handoffs between contributors, and architecture deep dives
that captured useful context at the moment they were written. They may be
out of date or contradicted by later changes — read them as historical
artifacts, not as current spec.

| File | What's in it |
|------|--------------|
| [`logbook.md`](logbook.md) | Day-by-day experiment log: what changed, what broke, what we learned. Earliest entry March 2026. |
| [`INFERENCE_DEBUG_NOTES.md`](INFERENCE_DEBUG_NOTES.md) | RTX 4090 inference-time debugging: PAD-token collapse, voice-prompt incompatibility, WebSocket timeouts. |
| [`DUPLEX_AND_FINETUNING_NOTES.md`](DUPLEX_AND_FINETUNING_NOTES.md) | Architecture deep-dive on Moshi's duplex (3-stream) attention pattern, depformer, dep_q, and what LoRA touches. |
| [`AB_TEST_HANDOFF.md`](AB_TEST_HANDOFF.md) | What was working / broken on the A/B test UI when the previous contributor handed it over. |
| [`AB_TEST_SESSION_NOTES.md`](AB_TEST_SESSION_NOTES.md) | Companion notes from the A/B test debugging session. |

If a fact in here matters for current work, copy it into [`README.md`](../../README.md)
or [`SETUP_GUIDE.md`](../../SETUP_GUIDE.md) and delete it from here.
