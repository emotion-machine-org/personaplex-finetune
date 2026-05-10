# Plan: Dynamic Puppeteer Context Injection for PersonaPlex

## Context

The finetuned PersonaPlex broker model hallucinates specifics (dollar amounts, policy forms, client history) because it learned that "brokers cite specifics" but has no grounding mechanism. The solution: an external LLM ("puppeteer") that runs async during conversations, makes tool calls to retrieve real information, and injects it into PersonaPlex's text stream mid-conversation. PersonaPlex learns to pause/filler during injection, then ground its responses on the injected data.

This requires changes across four areas: model inference, training data format, training pipeline, and synthetic data generation.

---

## Key Architectural Insight

PersonaPlex already has `inject_continuation()` (`lm.py:1027`) which forces text tokens into the text channel unconditionally, one per frame at 12.5 Hz. The model continues generating audio freely during injection. This is the exact hook we need — we just need to:
1. Wrap injected content in `<context>...</context>` tags (distinct from `<system>` opening prompts)
2. Train the model on data containing these mid-stream injections so it learns to generate silence during them and ground subsequent speech on the content
3. Mask text loss on injection frames (model didn't produce these tokens) but keep audio loss (model should learn silence behavior during injection)

### Injection Timing: During Client Speech (Primary Strategy)

The **preferred injection window is while the client is speaking**. At each frame, the model:
- Receives user audio (channels 9-16) — unchanged, client speech flows in normally
- Generates agent text (channel 0) — normally PAD/EPAD while listening; we replace this with context tokens
- Generates agent audio (channels 1-8) — already silence while listening

This means: injecting context while the client talks has **zero perceptible latency**. The broker was already silent. We're replacing empty PAD tokens with meaningful context tokens — the model reads the context AND listens to the client simultaneously through the transformer's summed embeddings.

The puppeteer should be **proactive and anticipatory**: it monitors the conversation, predicts what information the broker will need, retrieves it, and injects it during client speech windows — BEFORE the broker needs to respond. By the time the client finishes their question, the context is already loaded in the model's attention state.

If the client finishes speaking before the full context has been consumed, the injection should **spill over into the immediate post-client response gap** while the broker is still silent. The broker should not begin speaking until the injection completes. This is still considered **proactive placement**. We reserve **reactive placement** for a different case: the broker already hedged ("let me check that"), then context loads during the resulting pause.

### Critical: Text Stream Is Public Speech, Not Hidden Memory

In this codebase, non-PAD tokens on channel 0 are treated as **broker speech** everywhere:
- `bot_to_bot.py:367` — silence detection counts non-PAD as speech
- `bot_to_bot.py:404` — transcript logging emits non-PAD tokens
- `server.py:248` — websocket sends non-PAD tokens to client
- `gen_eval.py:260` / `batch_eval.py:155` — silence metrics count them

Injecting `<context>` tokens into the text channel raw would make the broker appear to be "speaking" context text. We need a **hidden-context runtime path**:

1. **Separate `_context_queue`** on LMGen (not reusing `_continuation_queue`) with its own processing in `process_transformer_output`
2. **`_injecting_context` flag** on LMGen — set `True` while context tokens are being consumed from the queue
3. **All output consumers check this flag**: when `_injecting_context` is True, the text token is suppressed from transcripts, websocket emission, silence metrics, and logging. The token still flows through the transformer (attention) and depformer (which generates silence), but it's invisible externally.
4. **Queue priority**: Context queue is separate from continuation/nudge queue. They don't clobber each other. Priority order: context > nudge > greeting. If both context and nudge have tokens, context wins.

### Vocalization: Model Learns Silence During `<context>` Tags

The text channel drives both attention (main transformer) and audio generation (depformer). During system prompt setup, text tokens are injected and audio is silence — the model already knows this pattern. By training with `<context>` tokens paired with silence audio (broker listening), the model learns the same behavior mid-conversation: read context, stay silent.

### Latency Budget

Each token = 1 frame = 80ms. Injections are directive coaching (action + facts) and must be short:
- **During client speech** (primary, ~70%): No perceived latency. Target ~20-30 tokens (~1.6-2.4s), comfortably within a client utterance.
- **During broker pause** (fallback, ~30%): Keep to 10-20 tokens (~0.8-1.6s). A brief pause after "let me check on that" is natural.
- Hard cap: **50 tokens** per injection. If more context is needed, split across multiple injection points.
- **Observed**: Injections average ~40 tokens in practice. At 12.5 Hz, that's ~3.2s to drain the context queue.

### Injection Drop Rate

~40% of in-window injections are dropped during training data preparation, primarily due to broker speech overlap at the anchor frame. The interleaver tracks detailed drop reasons via `InjectionStats` (see Observability section). The `ctx%` metric during training runs 0.6–1.7%, consistent with the expected ~1.5% given the drop rate and chunk windowing.

---

## Phase 0: Data Schema (Must Come First)

### 0a. Extend create_stereo.py to retain both-speaker timings

**File:** `pipeline/create_stereo.py`

Currently, the alignment writer only stores broker words as `SPEAKER_MAIN` (line 202). Client timings are dropped. Without client timings, we cannot reliably place injections during client speech or verify placement.

Change: store both speakers' alignments with distinct labels:
- `SPEAKER_BROKER` (was SPEAKER_MAIN) — broker word timings
- `SPEAKER_CLIENT` — client word timings
- Preserve **explicit raw turn boundaries** in a `turns` array alongside alignments. This avoids downstream turn-index drift after `[INJECT: ...]` markers are stripped; reconstructing turns from consecutive same-speaker word runs can incorrectly merge broker→broker turns and shift later `after_turn` indices.

This gives us the timestamp data needed to: (a) compute injection frame offsets during client speech, (b) validate injection placement doesn't overlap broker speech, (c) enable richer eval metrics.

### 0b. Add `context_injections` field to per-sample JSON

The injection metadata lives alongside existing alignment data, not in a separate file.

---

## Phase 1: Training Data Format + Loss Masking

### 1a. Extend JSON metadata format

Add `context_injections` to per-sample JSON files:

```json
{
  "alignments": [...],
  "turns": [
    {"index": 0, "speaker": "SPEAKER_BROKER", "start": 0.00, "end": 1.42, "text": "..."}, 
    {"index": 1, "speaker": "SPEAKER_CLIENT", "start": 1.42, "end": 3.88, "text": "..."}
  ],
  "text_prompt": "ROLE: Sandra Osei, Northbridge E&S... KNOWN: ... UNKNOWN: ... GOAL: ... STYLE: ...",
  "voice_prompt": "NATF2.pt",
  "context_injections": [
    {"frame_offset": 312, "text": "Quote the mid-term addition at $847 prorated on Kemper policy 4471. Emphasize no lapse in coverage."},
    {"frame_offset": 875, "text": "Agreed value rider approved, cap $85K. Push to bind today — underwriter hold expires Monday."}
  ]
}
```

`frame_offset` is relative to conversation start (after system prompt prefix). Injection text gets `<context>...</context>` wrapping during tokenization. Injections are directive coaching instructions (action + key facts), not raw data dumps. Each should be under 50 tokens. The `turns` array preserves the original transcript turn order used by `after_turn`.

### 1b. Extend Sample/Batch dataclasses

**File:** `moshi-finetune/finetune/data/interleaver.py`

```python
@dataclass
class InjectionStats:
    total: int = 0          # injections in the JSON
    in_window: int = 0      # fell within the chunk window
    placed: int = 0         # successfully spliced
    truncated: int = 0      # placed but truncated to fit
    drop_no_offset: int = 0
    drop_out_of_window: int = 0
    drop_anchor_overlap: int = 0
    drop_no_space: int = 0
    drop_final_overlap: int = 0
    tokens_placed: int = 0
    tokens_requested: int = 0

@dataclass
class Sample:
    codes: torch.Tensor
    condition_attributes: ConditionAttributes | None = None
    prompt_length: int = 0
    context_mask: torch.Tensor | None = None  # [T] boolean, True = injection frame
    injection_stats: InjectionStats | None = None

@dataclass
class Batch:
    codes: torch.Tensor
    condition_attributes: list[ConditionAttributes] | None = None
    prompt_lengths: list[int] | None = None
    context_masks: torch.Tensor | None = None  # [B, T] boolean
    injection_stats: InjectionStats | None = None  # aggregated across batch
```

Collation: pad/stack `context_mask` tensors across batch, defaulting to all-False for samples without injections. `InjectionStats` fields are summed across samples in the batch.

### 1c. Splice injection tokens into codes tensor

**File:** `moshi-finetune/finetune/data/interleaver.py:365-426` — in `InterleavedTokenizer.__call__`

After building `conv_codes` (line 418) and before prepending system prompt (line 421):

1. Read `context_injections` from JSON data
2. For each injection:
   - Wrap text: `f"<context> {text} </context>"`
   - Tokenize via `tokenize(self.interleaver.tokenizer, wrapped, bos=False)`
   - Treat the anchored `frame_offset` as the **start of an intended listening window**, not a fixed hard placement
   - Find the maximal contiguous non-broker span around that anchor (client speech + immediate post-client silent gap)
   - If the injection fits, place it so it starts during client speech and finishes before broker speech resumes; if needed, slide it earlier within that window
   - Set `context_mask[...] = True` for the placed region
   - Audio channels remain untouched — during client-speech injections these contain the client's actual audio (agent channels are silence since broker is listening). During pause injections these contain silence on both channels.
3. Adjust offsets for `start_sec` chunk windowing; skip injections outside the current chunk
4. Never let an injection overlap broker-speaking frames. If the original client turn is too short, allow **spillover through the immediate post-client response gap**. If the full injection still cannot fit before broker speech, use the entire available listening window rather than silently dropping the injection.

### 1d. Extend loss masking for context regions

**File:** `moshi-finetune/finetune/loss.py:5-58`

Add `context_masks: torch.Tensor | None = None` parameter. After the `prompt_lengths` block (line 30), add:

```python
if context_masks is not None:
    if mode == "text":
        # Zero text loss on injection frames — model didn't produce these tokens
        for b in range(context_masks.shape[0]):
            weights[b, :, context_masks[b]] = 0.0
    # Audio loss stays active — model learns filler/silence during injections
```

Also apply to the `per_codebook` mask block (line 45-48) for text mode.

### 1e. Wire context_masks through train.py

**File:** `moshi-finetune/train.py:364-384`

Pass `context_masks=batch.context_masks` to the text loss call. Pass `context_masks=None` to the audio loss call (audio learns during injections).

### 1f. Training observability ✅ DONE

**File:** `moshi-finetune/train.py`, `moshi-finetune/finetune/monitoring/metrics_logger.py`

Training loop now logs per-step:
- **`text_loss` / `audio_loss`** — loss breakdown (previously only combined `loss` was logged)
- **`ctx_injected_pct`** — % of text frames that are injection frames
- **`ctx_samples`** — number of samples in the step that had injections
- **`inj_total` / `inj_in_window` / `inj_placed`** — injection pipeline counts
- **`inj_truncated`** — placed but truncated to fit available space
- **`inj_drop_anchor` / `inj_drop_space` / `inj_drop_final`** — drop reasons
- **`inj_place_rate`** — % of in-window injections successfully placed
- **`inj_token_yield`** — % of requested tokens actually spliced

Eval loop now logs **`text_eval_loss` / `audio_eval_loss`** separately (data was already computed but not passed to the logger).

Console log line format:
```
step: 000010 - loss: 5.43 - txt: 2.87 - aud: 2.56 - ctx%: 1.2 - inj_place%: 58 - inj_yield%: 72 - lr: 2.0e-05 - ...
```

All metrics logged to wandb under `train/` and `eval/` prefixes.

---

## Phase 2: Structured Prompt Format

### 2a. Initial system prompt with KNOWN/UNKNOWN/GOAL ✅ DONE

Implemented in `pipeline/generate_dialogues_sync.py`. System prompt is generated alongside the dialogue in a single Claude API call, ensuring the broker's behavior is causally driven by the prompt (not reverse-engineered from a finished dialogue as before).

Format:

```
ROLE: Sandra Osei, Northbridge E&S Solutions, excess and surplus lines specialist
CLIENT: Rosa Antonelli, facilities manager, Greenstate Environmental Consulting
RELATIONSHIP: New client, under 1 year
KNOWN: 14 staff. Phase I/II environmental site assessments. Remediation oversight. 1-2 locations. Prior general commercial policy flagged inadequate. New submission for professional liability, $1M/$2M limits quoted at $4,800/yr.
UNKNOWN: Specific endorsement pricing. Subcontractor pollution exposure details. Competitor quote client mentioned.
GOAL: Present professional liability recommendation and get verbal commitment to proceed with application.
STYLE: Direct, efficient, measured Canadian manner
```

Key design choices:
- KNOWN contains 5-8 **concrete** facts with dollar amounts, dates, policy numbers
- UNKNOWN lists 2-4 items the broker anticipates needing — but is **not exhaustive** (client will also ask surprise questions not in UNKNOWN, teaching the model that UNKNOWN is a hint, not a complete list)
- GOAL states the broker's specific call objective (all calls are broker-initiated outbound)
- `<system>` tags are added during tokenization in the training pipeline, not stored in the data

### 2b. Mid-conversation context injections (directive coaching style) ✅ DONE

Injections are generated as part of dialogue generation. They are **directive coaching instructions** — like a senior broker whispering in a junior's ear — not raw data dumps. Each injection tells the broker what to DO with the information:

```
<context> Quote the Berkshire alternative at $13,100 — 3.6% increase vs their current 12.3%. Emphasize the savings. </context>
```

```
<context> Client's Markel quote doesn't include completed ops tail. Flag that gap — it's a dealbreaker for municipal contracts. </context>
```

```
<context> Umbrella approved — $2M, $10K SIR, $3,400/yr. Push to bind today before underwriter hold expires Friday. </context>
```

Each injection is under 50 tokens. Lead with action, follow with just enough facts.

---

## Phase 3: Synthetic Data Generation

### 3a. Merged generation script ✅ DONE

**File:** `pipeline/generate_dialogues_sync.py`

Dialogue generation and system prompt generation are merged into a single pipeline step. One Claude API call produces both the structured pre-call brief (text_prompt) and the dialogue transcript with embedded injection markers. This replaced the old two-step approach where `generate_system_prompts.py` reverse-engineered prompts from finished dialogues.

**8-dimensional Latin Hypercube Sampling** across seed axes:

| Dim | Axis | Count | Mapping |
|-----|------|-------|---------|
| 0 | Broker persona | 60 | uniform |
| 1 | Client persona | 80 | uniform |
| 2 | Broker outbound goal | 32 | uniform |
| 3 | Emotional tenor | 7 | uniform |
| 4 | Domain vocabulary | 14 | uniform |
| 5 | Complexity level | 4 | uniform |
| 6 | Call outcome | 6 | weighted 80/20 success/failure |
| 7 | Injection count | 4 | weighted (see 3d) |

Key design choices:
- **All calls are broker-initiated outbound** with a clear goal (not inbound inquiries)
- **Outcome axis**: 80% success (full/partial/deferred/conditional) and 20% failure (soft/hard) — teaches the model to handle both winning and losing gracefully
- **Mandatory naturalness**: backchanneling every 3-4 turns, 1-3 interruptions per dialogue, disfluency throughout (false starts, self-corrections, filler clusters, verbal tics)
- **Information discipline**: broker cites KNOWN confidently, hedges on UNKNOWN, is genuinely surprised by questions outside both lists
- **Injection markers** `[INJECT: ...]` are placed inline in the transcript, then parsed out into structured `context_injections` metadata. The clean dialogue (markers stripped) goes to TTS.

**Output JSONL format** per record:
```json
{
  "id": "dial-00042",
  "seed": {"broker": {...}, "client": {...}, "goal": "...", "outcome": {...}, "injection_count": 3, ...},
  "text_prompt": "ROLE: ... KNOWN: ... UNKNOWN: ... GOAL: ... STYLE: ...",
  "dialogue": "[Call start]\n\nBROKER (Name): ...\n\n[Call end]",
  "context_injections": [
    {"after_turn": 12, "text": "Quote the Berkshire at $13,100. Emphasize savings vs current 12.3% increase."},
    {"after_turn": 22, "text": "Flag that their Markel quote excludes completed ops tail — dealbreaker for municipal."},
    {"after_turn": 31, "text": "Offer 3-year lock at $11,800/yr. Beats their shopping quote."}
  ],
  "model": "claude-sonnet-4-6",
  "usage": {...},
  "generated_at": "..."
}
```

### 3b. Computing frame_offset from preserved turns (TODO — downstream step)

The JSONL contains `after_turn` (0-indexed turn number). A downstream script (post-audio, post-alignment) converts this to `frame_offset` using the preserved `turns` array written into the stereo JSON. Word-level alignments are a fallback only for older data.

Two injection placement strategies, determined by which speaker precedes the marker:

**Proactive (~70% of injections):** The turn before the marker is a **client** turn (no broker hedge). Place `frame_offset` at the START of that client utterance: `frame_offset = int(start_sec * 12.5)`. Context starts loading while the client speaks and may spill into the immediate post-client gap before the broker responds.

**Reactive (~30% of injections):** The turn before the marker is a **broker hedge** ("let me check on that"). Place `frame_offset` right after the hedge: `frame_offset = ceil(end_sec * 12.5)`. Context loads during a brief natural pause.

The proactive/reactive split is determined automatically from the transcript structure — no separate annotation needed. Preserving explicit turn boundaries is important here; reconstructing turns from merged speaker runs can shift later `after_turn` indices after markers are stripped.

### 3c. Audio generation (unchanged)

TTS synthesis via VibeVoice 7B: generate dialogue text → render to audio → WhisperX alignment → stereo routing. Same pipeline as before (`parse_dialogues.py` → `generate_audio.py` → `create_stereo.py`).

Note: `generate_system_prompts.py` is **no longer needed** — `text_prompt` comes directly from the JSONL. It just needs to be copied into the stereo JSON alongside alignments (during `create_stereo.py` or a simple post-step).

### 3d. Mix ratio ✅ DONE

Weighted via LHS dim 7:

| Range | Injection count | Share |
|-------|----------------|-------|
| [0.0, 0.5) | 3 injections | 50% |
| [0.5, 0.75) | 0 injections | 25% |
| [0.75, 0.9) | 6 injections | 15% |
| [0.9, 1.0) | 9 injections | 10% |

75% of dialogues have injections, 25% without. The no-injection dialogues teach the model to hedge on UNKNOWN items and never fabricate. Higher injection counts (6, 9) train the model on sustained puppeteer coaching throughout a complex call.

---

## Phase 4: Inference — Hidden-Context Runtime Path

### 4a. Separate context queue and injection flag on LMGen

**File:** `personaplex/moshi/moshi/models/lm.py`

Add alongside existing `_continuation_queue` (line 700) and `inject_continuation` (line 1027):

```python
# In __init__:
self._context_queue = deque()       # separate from _continuation_queue
self._injecting_context = False     # flag for downstream consumers

def inject_context(self, text: str, tokenizer):
    """Inject grounding context via a hidden channel.

    Uses a dedicated queue (not _continuation_queue) so nudges
    and context injections don't clobber each other.
    """
    wrapped = f"<context> {text.strip()} </context>"
    token_ids = tokenizer.encode(wrapped)
    self._context_queue.extend(token_ids)  # NOTE: extend, not clear+extend
```

In `process_transformer_output` (line 895-922), add context queue processing with **higher priority** than nudge:

```python
# Context injection — highest priority, uses dedicated queue
if self._context_queue:
    forced_id = self._context_queue.popleft()
    forced_tensor = torch.full_like(next_text_token, forced_id)
    next_text_token = forced_tensor
    sampled_text_token = forced_tensor
    self._injecting_context = True
elif self._injecting_context:
    self._injecting_context = False  # context queue just emptied
# Nudge — only fires when context is not active
elif self._continuation_queue:
    forced_id = self._continuation_queue.popleft()
    ...
```

The `_injecting_context` flag is readable by all output consumers.

### 4b. Suppress context tokens at all output points

Every place that treats non-PAD text as broker speech must check `_injecting_context`:

**`bot_to_bot.py:367`** — silence detection:
```python
a_silent = prev_text_id_a in (0, 3) or prev_text_id_a is None or a_injecting_context
```

**`bot_to_bot.py:404`** — transcript logging: skip token, or record as `"<CTX>"` placeholder

**`server.py:248`** — websocket emission: suppress text token when `_injecting_context`

**`gen_eval.py:260` / `batch_eval.py:155`** — silence metrics: treat context frames as silence

### 4c. Extend server.py with mid-session context channel

**File:** `personaplex/moshi/moshi/server.py`

Currently accepts only binary audio frames (line 207). Add a JSON message type for context injection:

```python
# In websocket handler:
if isinstance(message, str):
    msg = json.loads(message)
    if msg.get("type") == "context":
        lm_gen.inject_context(msg["text"], tokenizer)
        continue
# else: treat as binary audio frame (existing path)
```

This gives the puppeteer a live channel to inject context over the websocket.

### 4d. Extend bot_to_bot worker protocol and nudge system ✅ DONE

**File:** `personaplex/moshi/moshi/bot_to_bot.py`

Worker handles `"context"` and `"nudge"` messages. Output is 3-tuple `(pcm, text_id, injecting_flag)`.

**Nudge system changes:**
- **Bidirectional nudges**: Alternates 50/50 between Bot A and Bot B (even nudges → A, odd → B). Previously only nudged Bot A.
- **2-second threshold**: Default `nudge_after` reduced from 5.0s to 2.0s for faster silence breaking.
- **Unlimited nudges**: Default `max_nudges=0` means unlimited (was 5). `max_nudges=0` is the unlimited sentinel.
- **No nudge during injection**: `prev_injecting_a` flag gates the nudge condition — nudges are suppressed while context queue is draining, preventing wasted nudge messages during the ~3.2s injection window.
- **Scheduled context injections**: `--context-injections` CLI arg loads a JSON file of `[{frame, text}, ...]` for puppeteered eval. Scheduled injections have priority over nudges in the orchestrator's if/elif chain.

### 4e. Puppeteer process

**New file:** `personaplex/moshi/moshi/puppeteer.py`

Async process that runs proactively and anticipatorily:

1. Receives transcript fragments from the orchestrator each frame (both Bot A and Bot B text)
2. Accumulates conversation state and tracks what KNOWN/UNKNOWN items have been discussed
3. **Proactive mode**: When the client starts asking about an UNKNOWN item, the puppeteer fires off tool calls immediately — before the broker even needs to respond. It injects context during the client's speech so the broker has it ready.
4. **Reactive mode** (fallback): If the puppeteer didn't anticipate the need, it detects the broker hedging and injects during the resulting pause.
5. Calls Claude with tool definitions (database lookup, policy search, CRM query, etc.)
6. Returns context string to orchestrator for injection via `inject_context`

The puppeteer runs fully async — it doesn't block the 12.5 Hz frame loop. When it produces a result, it's queued and injected at the next available frame. The preferred injection window is during client speech (zero perceived latency).

---

## Phase 5: Evaluation

### 5a. Wire context_masks through eval.py ✅ DONE

**File:** `moshi-finetune/finetune/eval.py`

`compute_loss_with_mask()` passes `context_masks` through eval. Text/audio eval loss now logged separately.

### 5b. Grounding metric in LLM review with provenance ✅ DONE

**File:** `moshi-finetune/finetune/gen_eval.py`

4-dimension review: COHERENCE, NATURALNESS, EFFECTIVENESS, **GROUNDING** (1-5). The review prompt includes the **full system prompt AND all injected context texts** as reference material. Provenance (system_prompt + context_injections) persisted in result records.

### 5c. Provenance logging in batch_eval ✅ DONE

**File:** `pipeline/batch_eval.py`

Result records include `system_prompt` and `context_injections` for downstream grounding analysis.

### 5d. Automated hallucination detector ✅ DONE

**File:** `pipeline/compare_runs.py`

`compute_grounding_rate()`: regex extraction of factual claims ($amounts, percentages, dates, policy references) from broker transcript, checked against provenance. Reports per-dialogue and aggregate `grounding_rate`.

### 5e. Puppeteered batch_eval ✅ DONE

**File:** `pipeline/batch_eval.py`

`--puppeteer PATH` flag with knowledge base JSON (`{scenario_id: [{frame, text}, ...]}`). Writes per-dialogue injection files and passes `--context-injections` to bot_to_bot.

### 5f. Gen eval with scheduled context injections ✅ DONE

**File:** `moshi-finetune/finetune/gen_eval.py`, `moshi-finetune/finetune/eval_prompts.json`

gen_eval now supports `context_injections` in eval prompts. When a prompt has `context_injections: [{frame, text}, ...]`, gen_eval writes a temp JSON and passes `--context-injections` to bot_to_bot. Provenance logged in results for the GROUNDING review.

**3 eval scenarios (300s each):**

| Scenario | Injections | Tests |
|----------|-----------|-------|
| `eval_renewal` | 4 @ frames 750, 1375, 2000, 2625 | Pricing data usage — does broker cite injected numbers instead of hallucinating? |
| `eval_claims` | 5 @ frames 625, 1250, 1875, 2500, 3125 | Technical claim data + relationship building — can broker weave in specific details? |
| `eval_baseline` | 0 | Hallucination control — does broker hedge properly on unknowns without injections? |

Injection text is **proactive coaching** — things the broker should work into conversation at the next opportunity, not responses to specific questions. Spaced ~50s apart to land in plausible listening windows regardless of conversation flow.

---

## Implementation Order (revised per review)

**Stage 1: Data schema (prerequisite for everything)** ✅ DONE
1. **create_stereo.py** — ✅ Both-speaker timings (SPEAKER_BROKER + SPEAKER_CLIENT), preserved raw `turns` metadata, `--dialogues` arg propagates text_prompt + context_injections from JSONL
2. **Re-process existing data** to include client alignments

**Stage 2: Hidden-context runtime path + queue scheduler** ✅ DONE
3. **lm.py** — ✅ `_context_queue`, `_injecting_context` flag, `inject_context()` method, priority processing in `process_transformer_output` (context > nudge > greeting)
4. **bot_to_bot.py** — ✅ Worker context message handling, 3-tuple output with injecting flag, `<CTX>` placeholder in transcripts, suppressed from silence tracking and readable output. Bidirectional nudges (50/50 A/B), 2s threshold, unlimited, no nudge during injection. `--context-injections` for scheduled eval.
5. **server.py** — ✅ JSON `TEXT` message type for mid-session context injection over websocket, text output suppressed during injection

**Stage 3: Training pipeline + loss masking** ✅ DONE
6. **interleaver.py** — ✅ Sample/Batch dataclass extensions (context_mask/context_masks, InjectionStats), context injection splicing with per-injection drop-reason tracking
7. **loss.py** — ✅ `context_masks` parameter, text-only masking for injection regions (audio loss stays active)
8. **train.py + eval.py** — ✅ Pass `context_masks` through to text loss in both paths. Training logs text/audio loss split, ctx%, injection pipeline stats (place_rate, token_yield, drop reasons). Eval logs text/audio eval loss separately.
9. **args.py** — ✅ PuppeteerArgs dataclass (enable, max_tokens_per_injection, context_tag), GenEvalArgs updated (nudge_after=2.0, max_nudges=0=unlimited)

**Stage 4: Data generation** ✅ PARTIALLY DONE
10. ~~generate_system_prompts.py~~ — **Eliminated.** System prompts now generated alongside dialogues.
11. **generate_dialogues_sync.py** — ✅ Merged dialogue + system prompt + injection generation into single script. Outbound broker calls, 8-dim LHS, KNOWN/UNKNOWN/GOAL, directive coaching injections, backchanneling/interruptions.
12. **compute_injection_offsets.py** — ✅ Converts `after_turn` to `frame_offset` using preserved turn metadata (alignments fallback only for old data). Proactive (client turn → start) vs reactive (broker turn → end) placement.
13. **copy text_prompt into stereo JSON** — ✅ Integrated into create_stereo.py via `--dialogues` arg.

**Stage 5: Evaluation + puppeteer** ✅ DONE
14. **batch_eval.py** — ✅ Provenance logging (system_prompt + context_injections in results), `--puppeteer` flag with knowledge base JSON, `--context-injections` arg on bot_to_bot for scheduled frame-based injection
15. **gen_eval.py + compare_runs.py** — ✅ GROUNDING dimension (1-5) in LLM review with full system prompt + injected context as reference material; automated `grounding_rate` in compare_runs (regex claim extraction + provenance check)
16. **gen_eval.py** — ✅ Supports `context_injections` in eval prompts, writes temp injection JSON and passes `--context-injections` to bot_to_bot. 3 eval scenarios: 2 with scheduled injections (renewal + claims), 1 baseline (no injections).
17. **puppeteer.py** — ✅ SCAFFOLD (I/O contract + pseudocode, implemented in Phase 4)

---

## Key Files

| File | Status | Change |
|------|--------|--------|
| `pipeline/generate_dialogues_sync.py` | ✅ DONE | Merged dialogue + system prompt + injection generation. 8-dim LHS, outbound calls, directive coaching injections, backchanneling. |
| ~~`pipeline/generate_system_prompts.py`~~ | ✅ ELIMINATED | No longer needed — text_prompt generated alongside dialogue |
| `pipeline/create_stereo.py` | ✅ DONE | Both-speaker timings (SPEAKER_BROKER/CLIENT), preserved raw `turns` metadata; `--dialogues` propagates text_prompt + context_injections |
| `pipeline/compute_injection_offsets.py` | ✅ DONE | Converts `after_turn` to `frame_offset` using preserved turn metadata (alignments fallback for old JSON) |
| `personaplex/moshi/moshi/models/lm.py` | ✅ DONE | `_context_queue`, `_injecting_context` flag, `inject_context()`, priority queue processing (context > nudge > greeting) |
| `personaplex/moshi/moshi/bot_to_bot.py` | ✅ DONE | Worker context messages, 3-tuple output, `<CTX>` placeholder. Bidirectional nudges (50/50 A/B), 2s threshold, unlimited, gated on injection flag. `--context-injections` for scheduled eval. |
| `personaplex/moshi/moshi/server.py` | ✅ DONE | JSON TEXT message for context injection, text output suppressed during injection |
| `moshi-finetune/finetune/data/interleaver.py` | ✅ DONE | Sample/Batch context_mask + InjectionStats, injection splicing with per-drop-reason tracking, spillover into listening window |
| `moshi-finetune/finetune/loss.py` | ✅ DONE | context_masks param, text-only masking (audio loss stays active) |
| `moshi-finetune/train.py` | ✅ DONE | Pass context_masks to text loss. Logs text/audio loss split, ctx%, injection pipeline stats (place_rate, token_yield, drop reasons) |
| `moshi-finetune/finetune/eval.py` | ✅ DONE | Pass context_masks to text loss. Text/audio eval loss logged separately. |
| `moshi-finetune/finetune/args.py` | ✅ DONE | PuppeteerArgs config. GenEvalArgs: nudge_after=2.0, max_nudges=0 (unlimited) |
| `pipeline/batch_eval.py` | ✅ DONE | GROUNDING dimension, provenance logging (system_prompt + injections), `--puppeteer` knowledge base flag, `<CTX>` silence handling |
| `moshi-finetune/finetune/gen_eval.py` | ✅ DONE | GROUNDING dimension, provenance logging, `<CTX>` handling. Supports `context_injections` in eval prompts → passes `--context-injections` to bot_to_bot. |
| `moshi-finetune/finetune/eval_prompts.json` | ✅ DONE | 3 scenarios: eval_renewal (4 inj), eval_claims (5 inj), eval_baseline (0 inj, hallucination control) |
| `moshi-finetune/finetune/monitoring/metrics_logger.py` | ✅ DONE | Console log includes txt/aud/ctx%/inj_place%/inj_yield%. Tolerant of missing optional keys. |
| `pipeline/compare_runs.py` | ✅ DONE | `grounding` in metric extraction, `compute_grounding_rate` (regex claim extraction + provenance check) |
| `personaplex/moshi/moshi/puppeteer.py` | ✅ SCAFFOLD | I/O contract + pseudocode, ConversationState, Puppeteer class with feed/get_context/run stubs |

## Verification

1. **Unit test**: Create a sample JSON with `context_injections`, run through interleaver, verify text channel has `<context>` tokens at correct offsets and `context_mask` is set
2. **Loss test**: Verify text loss is zero on masked frames, audio loss is nonzero
3. **Training smoke test**: Train 100 steps on a small batch with injections, verify loss curves
4. **Inference test**: Run bot-to-bot with manual `inject_context` call, verify model generates filler then references injected data
5. **Eval test**: Run puppeteered batch_eval, compare grounding_rate vs non-puppeteered baseline

## Related work (added retrospectively, 2026-05-10)

After this design and the implementation in `personaplex/moshi/moshi/lm.py`
landed (March–April 2026), Sakana AI published **KAME — Tandem
Architecture for Enhancing Knowledge in Real-Time Speech-to-Speech
Conversational AI** ([pub.sakana.ai/kame](https://pub.sakana.ai/kame/),
[arXiv:2510.02327](https://arxiv.org/abs/2510.02327), accepted at ICASSP
2026, posted 2026-04-30). KAME tackles the same problem we tackled here
— a fast S2S model that needs to be grounded by a slower backend LLM
during the user's turn — but with a different architecture: KAME runs
the backend LLM asynchronously in parallel and feeds "oracle signals"
into the speech model continuously, while our puppeteer splices
discrete `<context>` text tokens into the model's own text stream at
specific frame offsets. KAME is the more thoroughly evaluated reference
for this class of approach; readers interested in async knowledge
injection for duplex models should start there.
