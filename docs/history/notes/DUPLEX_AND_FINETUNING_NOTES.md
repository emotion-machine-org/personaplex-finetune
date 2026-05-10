# Duplex Architecture & Finetuning Deep Dive — Notes for Inference Quality Improvement

**Date:** 2026-04-04
**Context:** Understanding root causes of PAD/silence bias in LoRA-finetuned PersonaPlex models

---

## Part 0: Glossary & Core Concepts (Voice Models, Codecs, Codebooks)

### Terminology

| Term | Definition |
|------|-----------|
| **Codec** | A neural network that compresses raw audio waveform into compact discrete tokens and back. Think of it as a learned MP3. Examples: Mimi, DAC, EnCodec, SoundStream. |
| **Codebook** | A lookup table of learned vectors (centroids). Each audio frame gets matched to the nearest vector in the codebook, producing an integer token ID. One codebook = one level of quantization. Vocabulary size typically 1024 or 2048. |
| **RVQ (Residual Vector Quantization)** | Hierarchical quantization: first codebook captures the "big picture" (semantics), then each subsequent codebook captures the *residual error* left by the previous ones (acoustic detail). Like JPEG quality layers — cb0 is the blurry version, cb1-7 add sharpness. |
| **Codebook depth / n_q** | Number of RVQ codebooks. Mimi uses 8 per audio stream. More codebooks = higher fidelity but more tokens to predict. |
| **dep_q** | Number of codebooks the **depformer** handles. Moshi: dep_q=8 (agent audio only). PersonaPlex: dep_q=16 (agent + user audio). |
| **Frame** | One time-step of encoded audio. At 12.5 Hz frame rate, each frame = 80ms of audio = 1920 samples at 24kHz. Each frame produces `n_q` tokens (one per codebook). |
| **Frame rate** | How many frames per second the codec produces. Mimi = 12.5 Hz. DAC = ~86 Hz. Lower = fewer tokens to model but coarser time resolution. |
| **Sample rate** | Raw audio samples per second. Mimi operates at 24kHz (24,000 samples/sec). |
| **Bitrate** | Bits per second of compressed audio. Mimi: ~1.1 kbps (8 codebooks x 12.5 Hz x ~11 bits). Extraordinarily low — phone calls are 64 kbps. |
| **Encoder** | Codec component: waveform → continuous latent → quantized tokens. Convolutional with strided downsampling. |
| **Decoder** | Codec component: quantized tokens → continuous latent → reconstructed waveform. Convolutional with upsampling. |
| **Temporal Transformer** | The main 7B backbone. Processes all 17 channels autoregressively across time. Predicts the text token + provides context for the depformer. This is what our LoRA finetunes. |
| **Depformer (Depth Transformer)** | Small 6-layer transformer that generates audio codebook tokens *within* a single frame, conditioned on the temporal transformer's output and the text token. Runs sequentially through codebooks (cb0→cb1→...→cb15). |
| **Inner Monologue** | Moshi's key trick: generate a text token first, then use it to condition audio generation. Text acts as a semantic "plan" that constrains what audio should sound like. |
| **Delay Pattern** | Technique to handle the sequential dependency between codebooks. Higher codebooks are shifted right by N steps, creating manufactured conditional independence so they can be predicted "jointly" by the temporal transformer. The depformer then refines within each frame. |
| **PAD / Silence Token** | Token ID 3 (`zero_text_code`). Represents "model is not speaking." In the text stream, PAD = silence. In audio, silence has its own codec tokens (SILENCE_TOKENS = `[948, 243, 1178, 546, 1736, 1030, 1978, 2008]`). |
| **EPAD** | Token ID 0 (`end_of_text_padding_id`). Marks the last padding step before a word begins. Semantically similar to PAD for inference filtering. |
| **Streaming** | Processing audio chunk-by-chunk as it arrives, not waiting for the full utterance. Requires causal attention (can only see past, not future). |
| **KV Cache** | Stored key/value pairs from previous transformer steps. Enables O(1) per-step inference instead of recomputing attention over the full history. |
| **LoRA** | Low-Rank Adaptation. Inserts small trainable matrices A (down-project) and B (up-project) alongside frozen weight matrices. Output = W(x) + scaling * B(A(x)). Trains ~5% of params. |
| **FSDP** | Fully Sharded Data Parallel. Distributes model parameters across GPUs, gathering them on-demand for computation. Enables training models larger than single-GPU memory. |
| **CFG (Classifier-Free Guidance)** | Technique from diffusion models adapted for AR: trains both conditional and unconditional models, then amplifies the difference. Parakeet's CFG-filter variant uses CFG logits as a mask over conditional logits to avoid speech speed-up artifacts. |
| **RQ-Transformer** | Transformer that operates over residual-quantized tokens. Parakeet/SoundStorm style: predict all codebook levels jointly using delay patterns instead of separate hierarchical models. |

### How a Neural Audio Codec Works (Mimi)

**Encode (one 80ms frame):**

```
Raw Audio [1920 samples @ 24kHz]
  |
  v
Conv Downsampler (strided convolutions)
  |
  v
Continuous Latent [1, D, 1]
  |
  v
RVQ: Residual Vector Quantization
  |
  |  Step 1: latent --> nearest in codebook_0 (1024 entries)
  |           token_0 = 948
  |           residual = latent - codebook_0[948]
  |
  |  Step 2: residual --> nearest in codebook_1
  |           token_1 = 243
  |           residual -= codebook_1[243]
  |
  |  ... repeat for codebook_2 through codebook_7 ...
  |
  v
Output: 8 integer tokens [948, 243, 1178, 546, 1736, 1030, 1978, 2008]
```

### From Tokens to Audio Waveform: The Full Decode Path

The model **doesn't synthesize audio** — it picks entries from a learned dictionary, sums them, and runs the result through a learned upsampler (convolutional decoder). The "intelligence" is in the dictionary and the decoder, both trained end-to-end.

**Step 1: Each token is a lookup into its codebook**

Each codebook is a table of 1024 learned vectors, each of dimension D (e.g., 256):

```
codebook_0[948]  = [0.23, -0.81, 0.44, ...]  (256 floats)
codebook_1[243]  = [0.05,  0.12, -0.33, ...]  (256 floats)
codebook_2[1178] = [-0.11, 0.03,  0.67, ...]  (256 floats)
...
codebook_7[2008] = [0.01, -0.02,  0.08, ...]  (256 floats)
```

These vectors live in a continuous latent space learned during codec training. Each one represents a "sound atom."

**Step 2: Sum the vectors (RVQ reconstruction)**

```
latent = codebook_0[948]     <-- rough sketch (pitch, phoneme)
       + codebook_1[243]     <-- add shading (formants)
       + codebook_2[1178]    <-- add color (spectral detail)
       + codebook_3[546]     <-- refine harmonics
       + codebook_4[1736]    <-- consonant transients
       + codebook_5[1030]    <-- room acoustics
       + codebook_6[1978]    <-- noise floor
       + codebook_7[2008]    <-- fine highlights (sibilance)

Result: one vector [256 floats] representing 80ms of audio
```

This is why it's called *residual* VQ — cb0 captures the bulk, each subsequent codebook adds a correction to the residual error. Summing them reconstructs the full representation.

**Step 3: Convolutional decoder upsamples to a waveform**

The Mimi decoder is a stack of transposed (upsampling) convolutions:

```
latent [1, 256, 1]           <-- 1 time-step, 256 channels
  |
TransposedConv1d (stride 8)
  --> [1, 128, 8]
TransposedConv1d (stride 6)
  --> [1, 64, 48]
TransposedConv1d (stride 5)
  --> [1, 32, 240]
TransposedConv1d (stride 8)
  --> [1, 1, 1920]           <-- 1920 raw samples = 80ms @ 24kHz
```

Total stride = 8 x 6 x 5 x 8 = 1920, which is exactly why one codec frame = 1920 audio samples. The strides/layers are approximate — the point is that the decoder learned to turn a 256-dim vector into a 1920-sample waveform.

**These transposed convolutions are NOT doing Fourier synthesis or classical DSP.** They're learned nonlinear functions (with activations between layers) that map latent vectors to waveforms. The decoder learned during training that codebook_0[948] corresponds to a particular pitch/phoneme pattern and should produce those specific oscillations.

### How the Codec Was Trained

The codec (Mimi) was trained as an autoencoder with adversarial + reconstruction losses:

```
Training loop (on thousands of hours of speech):
  1. Take real audio waveform
  2. Encode: waveform -> conv downsampler -> latent -> RVQ -> 8 tokens
  3. Decode: 8 tokens -> lookup + sum -> latent -> conv upsampler -> reconstructed waveform
  4. Loss = |original - reconstructed|
          + adversarial loss (discriminator tells real vs fake)
          + perceptual loss (multi-scale spectrograms match)
  5. Backprop through everything: encoder, codebooks, decoder, discriminator
```

After training, the codebook entries specialize. Codebook 0 entry #948 might encode "a male voice saying a vowel at ~120Hz." The decoder learns that this vector (plus corrections from cb1-7) should produce an oscillation at 120Hz with the right formant structure.

### What Silence Looks Like at the Token Level

When the codec encodes a silent audio frame, it produces tokens like SILENCE_TOKENS `[948, 243, 1178, 546, 1736, 1030, 1978, 2008]`. Those codebook vectors sum to a latent that the decoder maps to near-zero amplitude audio. The LM doesn't "know" it's producing silence in any abstract sense — it just learned that these token IDs tend to co-occur when nobody is speaking.

The codec is a faithful translator: whatever tokens go in, a waveform comes out. The codec doesn't judge or filter — it reconstructs. All the "decisions" about what to say (or not say) happen upstream in the LM.

**Key insight**: cb0 captures ~60-70% of signal energy (pitch, phoneme identity). Each subsequent codebook captures finer detail (breathiness, room tone, high-frequency texture). This is why `first_codebook_weight` matters so much in training — cb0 is the semantic backbone.

### Codebook Hierarchy: What Each Level Captures

| Codebook | Captures | Importance |
|----------|----------|------------|
| **cb0** (semantic) | Pitch contour, phoneme identity, speaker F0 | **Critical** — removing = unintelligible |
| cb1 | Formant structure, vowel quality | High |
| cb2 | Spectral envelope detail | Medium-high |
| cb3 | Fine harmonic structure | Medium |
| cb4 | Consonant transients, plosives | Medium |
| cb5 | Room acoustics, reverb tail | Low-medium |
| cb6 | Noise floor, breathing | Low |
| **cb7** (acoustic) | High-frequency texture, sibilance | Low — removing = slightly muffled |

### The Full Moshi/PersonaPlex Architecture

**One inference step = 80ms of audio. Everything below happens every step.**

```
INPUT SIDE                          OUTPUT SIDE
----------                          -----------
User Microphone                     Agent Speaker
  |                                   ^
  v                                   |
Mimi Encoder                        Mimi Decoder
  |                                   ^
  v                                   |
user_codes [8 tokens]               agent_codes [8 tokens]
(channels 9-16)                     (channels 1-8)
  |                                   ^
  |    +--- prev agent_codes ---------+----+
  |    |    (autoregressive             |
  |    |     feedback loop)             |
  v    v                                |
+---------------------------+           |
| TEMPORAL TRANSFORMER      |           |
| (7B params, 32 layers)    |           |
|                           |           |
| Input = sum of embeddings:|           |
|   text_emb(text_t-1)      |           |
|   + emb(agent_codes_t-1)  |           |
|   + emb(user_codes_t-1)   |           |
|                           |           |
| Causal attention + KV cache|          |
+-----+-----+---------------+          |
      |     |                           |
      v     v                           |
  text    transformer_out               |
  logits    |                           |
      |     |                           |
      v     v                           |
  +------+ +------------------+         |
  | Text | |    DEPFORMER     |         |
  | Sample | (6 layers, small)|         |
  | temp,| |                  |         |
  | topk | | text_tok --> cb0 |         |
  +--+---+ |   cb0 --> cb1    |         |
     |     |   cb1 --> cb2    |         |
     |     |   ...            |         |
     |     |   cb14 --> cb15  |         |
     |     +--------+---------+         |
     |              |                   |
     |        16 audio tokens           |
     |        (8 agent + 8 user)        |
     |              |                   |
     v              v                   |
  +-----------------------------+       |
  | TOKEN CACHE [B, 17, CT]     |       |
  |                             |       |
  | ch 0:    text (delay[0])    |       |
  | ch 1-8:  agent audio        +-------+
  | ch 9-16: user audio         |  agent_codes
  |                             |  extracted with
  | Circular buffer with        |  delay compensation
  | delay-compensated output    |
  +-----------------------------+
```

**Key flow**: text token is sampled first (Inner Monologue), then fed as the *first input* to the depformer, which generates audio codebooks sequentially cb0->cb1->...->cb15. Each codebook is conditioned on the previous one.

### The Delay Pattern Trick

The temporal transformer needs to predict all 17 channels, but they have sequential dependencies (can't know cb3 without cb0-2). The delay pattern shifts higher codebooks rightward in time:

```
  Time steps:        t=0   t=1   t=2   t=3   t=4   t=5   t=6
  ─────────────────────────────────────────────────────────────
  text (delay=0):    T0    T1    T2    T3    T4    T5    T6
  agent_cb0 (d=1):   .     A0₀   A1₀   A2₀   A3₀   A4₀   A5₀
  agent_cb1 (d=2):   .     .     A0₁   A1₁   A2₁   A3₁   A4₁
  ...
  agent_cb7 (d=8):   .     .     .     .     .     .     .    A0₇ ...
  user_cb0 (d=9):    .     .     .     .     .     .     .    .   U0₀ ...
  ...

  At t=5, the transformer can attend to:
  - text T0..T5
  - agent_cb0 A0..A4  (shifted right by 1)
  - agent_cb1 A0..A3  (shifted right by 2)
  - etc.

  This means at t=5, agent_cb1 at t=3 (i.e. A3₁) was predicted
  AFTER seeing A3₀, because A3₀ was at position t=4 and cb1
  at frame 3 is at position t=5. The delay creates causal ordering!
```

The depformer then refines within each frame — it sees the temporal transformer's output and generates cb0→cb1→...→cb15 sequentially, each conditioned on the previous.

### Moshi vs Parakeet vs Traditional TTS

**Traditional Pipeline (Half-Duplex):**
```
Mic -> VAD -> ASR -> LLM -> TTS -> Speaker
       (sequential, one direction at a time)
       Latency: 500ms - 2s+
```

**Parakeet (Autoregressive TTS, not duplex):**
```
Text + speaker tags     Encoder-Decoder 3B      DAC
"[S1] Hello (laughs)" --> predict 9 codebooks --> Decoder --> Audio
                          w/ delay pattern
                          86 tok/sec x 9 levels
                          CFG-filter for alignment
                          (one-shot, no streaming input)
```

**Moshi / PersonaPlex (True Full-Duplex):**
```
Mic --> Mimi Enc --> Temporal Transformer (7B) --> Mimi Dec --> Speaker
                     + Depformer (6-layer)
                     ALL SIMULTANEOUS, every 80ms
                     Bidirectional: listens while speaking

                     text_token  = "what to say"
                     audio_tokens = "how to say it"
                     user_tokens  = "what I hear"

                     Latency: ~200ms, continuous
```

### Duplex Model Comparison Table

| Feature | Moshi | PersonaPlex | SALM-Duplex | Parakeet |
|---------|-------|-------------|-------------|----------|
| Direction | Full-duplex | Full-duplex | Full-duplex | One-shot TTS |
| Backbone | Helium 7B | Helium 7B | TinyLlama | 3B enc-dec |
| Codec | Mimi 8cb | Mimi 8cb | Nano/Mimi | DAC 9cb |
| Frame rate | 12.5 Hz | 12.5 Hz | 12.5 Hz | ~86 Hz |
| Bitrate | 1.1 kbps | 1.1 kbps | 0.6 kbps | ~9.5 kbps |
| dep_q | 8 | 16 | N/A | N/A (delay pattern) |
| User input | Codec tokens | Codec tokens | Pretrained ASR | N/A |
| Turn-taking | Implicit (learned) | Implicit (learned) | Implicit (silence injection) | N/A |
| Voice cloning | No | Yes (3-5s prompt) | Yes (21K utterances) | Yes (zero-shot) |
| Latency | ~200ms | ~200ms | 720ms (1st resp) | N/A (offline) |

### How "Silence" Works in Duplex Models

Unlike traditional TTS where silence = not generating, duplex models must **actively predict silence** every 80ms:

```
  Frame t: Agent is listening, user is speaking

  Temporal transformer input:
    text:       PAD (3)          ← "I have nothing to say"
    agent_cb0:  948 (silence)    ← silence codec token
    agent_cb1:  243 (silence)    ← silence codec token
    ...
    user_cb0:   [actual speech tokens from mic]
    user_cb1:   [actual speech tokens from mic]

  Temporal transformer output:
    text_logits → samples to PAD (3) again
    depformer → generates silence tokens for agent audio
    (user audio comes from mic, not generated)

  This continues until the model's attention over user tokens
  and its own text history triggers it to START speaking:
    text_logits → samples to "H" (start of "Hello")
    depformer → generates speech audio tokens
```

Note that silence is an **active prediction**, not the absence of output. The model must decide every 80ms: "should I speak or stay quiet?" This is fundamentally different from traditional TTS where silence simply means the system isn't running.

---

## Part 1: Duplex Architecture Fundamentals

### Moshi Core Architecture

- **Inner Monologue**: Generates time-aligned text token as *prefix* before audio tokens in each 80ms frame. Per frame: `text_token -> [audio_token_1..8]`. Text provides semantic constraint that guides audio generation. This is the key insight — separates "what to say" from "how to say it."
- **Multi-Stream**: Models user and agent speech as independent simultaneous parallel streams. No explicit turn-taking — silence/speech patterns are learned from data via cross-stream attention.
- **Codes tensor**: `[text, agent_audio_0..7, user_audio_0..7]` shape `[B, 17, T]`
- **Frame rate**: 12.5 Hz (80ms frames), sample rate 24kHz
- **Mimi codec**: 8 codebooks per stream via RVQ. First codebook = semantic, rest = acoustic detail. ~1.1 kbps.
- **Latency**: Theoretical 160ms, practical ~200ms. Within human response timing (~230ms).
- **Backbone**: Helium — 7B decoder-only transformer (LLaMA-style, RoPE, RMSNorm).

### PersonaPlex Additions

- **dep_q=16** vs Moshi dep_q=8 — depformer handles all 16 audio codebooks (8 agent + 8 user)
- **Voice conditioning**: 3-5s audio sample → speaker identity. Speaker similarity 0.57 SSIM (0.65 with ChatterboxTTS).
- **Role control**: Text prompt wrapped in `<system>` tags, concatenated as prefix tokens.
- **Training**: 2,250h synthetic dialogs, Adam LR 4e-6 (depth) / 2e-6 (temporal), cosine annealing, 24,576 steps, 8xA100 ~6h.
- **Released checkpoint**: +1,217h Fisher English real data with multi-level prompt annotation.
- **Turn-taking latency**: 0.17s. Interruption latency: 0.24s.

### SALM-Duplex (For Comparison)

- Asymmetric: pretrained streaming ASR for user input, neural codec only for agent output.
- Much more efficient — doesn't require the LLM to learn speech representation for user side.
- Uses TinyLlama (much smaller than 7B), 0.6 kbps personalized codec.
- **Key technique**: 0.64s silence injection after user turns in training data to teach turn boundaries.
- Barge-in success 94.5% vs Moshi 55.1%.

### Duplex Taxonomy (Survey)

| Level | Description | Example |
|-------|-------------|---------|
| Half-Duplex | Sequential listen→process→speak | Traditional VAD→ASR→LLM→TTS |
| Pseudo-Full-Duplex | Time-division multiplexing illusion | Many "full-duplex" claims |
| True Full-Duplex | Genuine cognitive parallelism | Moshi, PersonaPlex |

### The "When NOT to Speak" Problem

This is the core challenge for duplex models:
- Must learn silence as a **valid output** — not just absence of generation
- Moshi learns this implicitly via parallel stream modeling with cross-stream attention
- No explicit turn-taking tokens (unlike Neural FSM which uses `[C.SPEAK]`, `[S.LISTEN]`, etc.)
- The model must allocate cognitive resources dynamically — like humans who listen while talking

---

## Part 2: Our LoRA Finetuning Pipeline

### Architecture

```
Entry: train.py → get_fsdp_model() → LoRA init → FSDP wrapping → training loop
```

- **LoRA targets**: All layers with "lora" in name (attention projections in transformer)
- **LoRA init**: Kaiming uniform for `lora_A`, zeros for `lora_B` (standard)
- **skip_depformer=True** (all our configs): Depformer LoRA is FROZEN — only temporal transformer is finetuned
- **ft_embed=False** (all our configs): Embedding layer is FROZEN
- **FSDP**: Two-level wrapping separates trainable (LoRA) and frozen params

### Training Configs (All Our Runs)

| Parameter | pharma-v2 | pharma-demo-v0-1 | companion_v3 | outbound_ins |
|-----------|-----------|-------------------|--------------|--------------|
| LoRA rank | 64 | 64 | 64 | 64 |
| LoRA scaling | 2.0 | 2.0 | 2.0 | 2.0 |
| skip_depformer | true | true | true | true |
| text_padding_weight | **0.5** | **0.5** | **0.5** | **0.5** |
| first_codebook_weight | 4.0 | 3.0 | 10.0 | 3.0 |
| lr | 2e-5 | 2e-5 | 2e-5 | 2e-5 |
| batch_size | 4 | 4 | 32 | 4 |
| max_steps | 1024 | 256 | 128 | 256 |
| duration_sec | 80.0 | 80.0 | 80.0 | 80.0 |

### Loss Computation (loss.py)

```
weights = target_mask.float()

if mode == "text":
    for id in text_padding_ids:
        weights[target == id] *= text_padding_weight   # 0.5 for PAD tokens

if prompt_lengths:
    weights[b, :, :pl] = 0.0                           # zero on system prompt

if context_masks and mode == "text":
    weights[b, :, context_masks[b]] = 0.0              # zero on injection frames

ce = F.cross_entropy(logits, target, reduction="none")
loss = sum(ce * weights) / sum(weights)                 # normalized by weight sum
```

**Critical detail**: The normalization by `sum(weights)` means PAD tokens contribute less to both numerator AND denominator. Effective gradient on PAD tokens = 0.5x gradient on speech tokens.

### Gen_eval vs Real-time Server Sampling

| Parameter | Gen_eval | Server (UI defaults) | Delta |
|-----------|----------|---------------------|-------|
| temp_audio | 0.45 | 0.65 | +44% hotter |
| temp_text | 0.40 | 0.60 | +50% hotter |
| topk_audio | 80 | 150 | +87% wider |
| topk_text | 15 | 20 | +33% wider |
| Input quality | Clean bot audio | Real mic + Opus + jitter | Much noisier |
| Nudge | After 2.0s silence | **None** | Missing |
| PAD penalty | None | None | Both missing |

---

## Part 3: PAD Bias Root Cause Analysis

### The Compounding Factors

The PAD/silence bias in finetuned models is NOT a single issue — it's a cascade of reinforcing factors:

#### Factor 1: Training Data Distribution + text_padding_weight=0.5

In natural conversation, ~40-60% of frames are silence (one party listening). With `text_padding_weight=0.5`, the model receives HALF the gradient signal for predicting PAD correctly vs speech tokens. This means:
- The model learns PAD as "safe default" — it's penalized less for wrong PAD predictions
- LoRA adapters shift the base model's text distribution toward more PAD
- The base PersonaPlex model had its own PAD calibration from full pretraining; LoRA disturbs this

**Why gen_eval isn't affected**: Bot-to-bot provides clean, predictable audio input. The model's PAD vs speech decision boundary works correctly with clean input. With noisy real audio, the model's confidence drops and it defaults to the higher-prior class: PAD.

#### ~~Factor 2: skip_depformer=True Creates a Distribution Mismatch~~ (DEBUNKED)

~~We freeze the depformer and only train the temporal transformer's LoRA.~~
**Why this is wrong**: The base PersonaPlex model works fine through the exact same inference path. The depformer receives a discrete token ID → embedding lookup → audio generation. The frozen embedding table maps token ID 3 (PAD) to the same vector regardless of how confident the temporal transformer was. The depformer doesn't care about the softmax distribution — it just sees the sampled token. Since the depformer is identical to base PersonaPlex, and base PersonaPlex works, this isn't the issue.

#### Factor 3: LoRA Scaling and Rank Interact with PAD Bias

- rank=64, scaling=2.0 means LoRA contribution = 2.0 * (B @ A) where A is [d_model, 64], B is [64, d_model]
- The scaling factor amplifies the adaptation signal
- For a model that's already 50% PAD in training, the LoRA might learn to amplify PAD probability because:
  - PAD has the highest prior in the training distribution
  - With limited rank, the LoRA captures dominant patterns first → PAD prediction
  - Speech tokens have more variety (32K vocab) so need more rank to model well

#### Factor 4: No Inference-Time Correction Mechanisms

The server has NO PAD countermeasures:
- `padMult` from client UI is **never applied** server-side
- No nudge mechanism (only exists in bot_to_bot gen_eval)
- No PAD logit penalty
- No silence timeout → inject_continuation
- Repetition penalty also dead code

#### Factor 5: System Prompt Processing Delay → Missed Context

The 16-second system prompt processing means:
- Early user audio is lost (consumed by `is_alive()`)
- Model starts inference with stale/missing context about user state
- Defaults to PAD because it hasn't "heard" enough to decide to speak

#### Factor 6: Voice Prompt Incompatibility (Known, Partially Fixed)

`.pt` voice prompts are skipped entirely → model has no voice conditioning → defaults to base behavior which includes heavy silence.

### The Reinforcement Loop

```
Noisy input → low confidence → PAD prediction → PAD feeds into autoregressive cache →
next step sees "I was silent" → increased PAD probability → more PAD → ...
```

Once the model enters a PAD loop, there's no mechanism to break out. In gen_eval, the nudge mechanism breaks this. In the real-time server, nothing does.

---

## Part 4: Proposed Fix Priority

### Tier 1: Inference-Time (Fast, No Retraining)

1. **Wire up `padMult` as PAD logit bias** in `process_transformer_output()`:
   ```python
   # Before sample_token(), apply pad bias to text logits
   if self.pad_mult != 0:
       text_logits[:, :, :, 3] -= self.pad_mult  # PAD token id=3
   ```
   - Client already sends this param, just needs server-side application
   - Start with padMult=2.0, tune from there

2. **Port nudge mechanism to server opus_loop**:
   - Track consecutive PAD frames in opus_loop
   - After N seconds silence (configurable, start with 2.0s), call `inject_continuation()`
   - Use same NUDGE_PHRASES as bot_to_bot: "So,", "Right,", "Now,", "Anyway,"

3. **Wire up repetition penalty**:
   - Apply penalty to recently-generated tokens in `sample_token()`
   - Client already sends `repetitionPenalty` and `repetitionPenaltyContext`

### Tier 2: Training-Time (Next Training Run)

4. **Lower text_padding_weight to 0.05-0.1**:
   - Gives the model stronger gradient signal to learn speech token patterns
   - PAD still appears in data, model still learns it, just doesn't over-index on it

5. **Consider skip_depformer=False** (or partial depformer LoRA):
   - Allows depformer to adapt to the finetuned temporal transformer's output distribution
   - Risk: depformer has many parameters, may need lower rank or learning rate
   - Compromise: LoRA on depformer at rank=16 (vs rank=64 for temporal)

6. **Increase first_codebook_weight** for pharma configs:
   - Currently 3.0-4.0, could go to 8.0-10.0
   - Gives stronger signal on semantic audio codebook → better audio quality → less screech → less PAD recovery

7. **Re-generate `.pt` voice prompts** with finetuned model to restore voice conditioning

### Tier 3: Architecture (Longer Term)

8. **Consider SALM-Duplex style asymmetric approach**: Use pretrained ASR encoder for user input instead of having the LM model it. Reduces the model's burden and may improve robustness to noisy input.

9. **Add explicit turn-taking signal** (Neural FSM style): Instead of relying on implicit learning, add control tokens that the model can produce to signal state transitions.

---

## Part 5: Key File Reference

| Component | File | Key Lines |
|-----------|------|-----------|
| Loss computation | `moshi-finetune/finetune/loss.py` | 5-73 |
| text_padding_weight flow | `moshi-finetune/train.py` | 476-488 |
| LoRA init | `moshi-finetune/finetune/wrapped_model.py` | 98-227 |
| LoRA config | `moshi-finetune/finetune/args.py` | 66-77 |
| Token sampling | `personaplex/moshi/moshi/utils/sampling.py` | 106-126 |
| process_transformer_output | `personaplex/moshi/moshi/models/lm.py` | 914-1048 |
| Greeting forcing | `personaplex/moshi/moshi/models/lm.py` | 932-943 |
| inject_continuation | `personaplex/moshi/moshi/models/lm.py` | 1087-1094 |
| Server param application | `personaplex/moshi/moshi/server.py` | 146-149 |
| Client defaults | `personaplex/client/src/pages/Conversation/hooks/useModelParams.ts` | 4-10 |
| Bot-to-bot nudge | `moshi-finetune/finetune/bot_to_bot.py` | 402-449 |
| Interleaver (data prep) | `moshi-finetune/finetune/data/interleaver.py` | 427-697 |
| Gen_eval pipeline | `moshi-finetune/finetune/gen_eval.py` | 613-714 |
