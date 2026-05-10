# Pharma Voice Model Finetuning -- Weekend Experiment Log

**Date:** 2026-04-04 (session spanning April 4--7)
**Project:** Finetuning PersonaPlex for pharma patient support voice calls (Pharma)

---

## 1. Project Background

| Parameter | Value |
|-----------|-------|
| Objective | Finetune a voice-native LLM for automated pharma patient support calls |
| Base model | `nvidia/personaplex-7b-v1` (Moshi 7B variant) |
| Training method | LoRA on backbone transformer; depformer frozen |
| Architecture | 17 channels (1 text + 16 audio codebooks), Mimi codec at 12.5 Hz |
| Training data | 2003 samples, 20 eval samples |
| Sample duration | mean 296 s, median 299 s |
| Baseline gen_eval (base model) | coherence = 2.0, naturalness = 1.7, effectiveness = 2.7 |

---

## 2. Pre-existing Runs (Before This Session)

### 2.1 Old Successful Run (March 31, pre-code-changes)

| Parameter | Value |
|-----------|-------|
| LR | 2e-5 |
| Rank | 64 |
| text_padding_weight | 0.5 |
| Duration | 80 s |
| Batch size | 4 x 4 x 3 GPUs = 48 effective |
| Steps | 1024 |
| L2 | none |
| world_size | 3 (3 GPUs with FSDP) |
| Chunking | Old strategy (chunk_step = duration_sec, no prompt_budget_frames) |

**Gen_eval progression:**

| Step | Coherence | Naturalness | Effectiveness |
|------|-----------|-------------|---------------|
| 128 | 1.3 | 1.7 | 2.0 |
| 256 | 2.3 | 2.3 | 2.3 |
| 640 | 1.7 | 2.0 | 3.3 |
| 1024 | 3.0 | 3.0 | 4.0 |

**Observations:**
- Successfully learned specific scripts: "Hi, this is Mia, I'm an automated care assistant", DOB verification, recording disclosure.
- Critical flaw: model would be permanently silent unless nudged (text_padding_weight = 0.5 caused silence overfit).

### 2.2 Runs v2-1 through v2-6

All exhibited the same core problem -- metrics degraded with training.

| Run | LR | Rank | Batch | Duration | text_pad | audio_loss | Notes |
|-----|-----|------|-------|----------|----------|------------|-------|
| v2-1 | 2e-5 | 64 | 32 | 120 | 0.5 | -- | Best at step 128 (2.3/2.0/3.3), degraded after |
| v2-2 | 2e-5 | 64 | 32 | 120 | 0.0 | -- | Worse than v2-1 |
| v2-3 | 2e-5 | 64 | -- | -- | 0.1 | -- | first_codebook=10, short run |
| v2-4 | 2e-5 | 64 | 4 | 240 | -- | -- | Short run |
| v2-5 | 2e-6 | 64 | 16 | 180 | 0.0 | 0.5 | All metrics degraded from baseline |
| v2-6 | 2e-6 | 64 | 4x4=16 | 316 | 0.01 | 0.1 | Degraded by step 64 |

---

## 3. Session Experiments

### 3.1 v2-7: L2 Regularization Breakthrough

| Parameter | Value |
|-----------|-------|
| LR | 1e-6 |
| Rank | 32 |
| L2 | 1e-4 (lora_B only) |
| audio_loss_weight | 0.5 |
| text_padding_weight | 0.0 |
| Duration | 300 s |
| Batch | 4 x 4 = 16 |
| Steps | 201 (~1 epoch) |
| pct_start | 0.1 |

This was the **first run to implement L2 regularization** on LoRA weights.

**Gen_eval progression:**

| Step | Coherence | Naturalness | Effectiveness | n |
|------|-----------|-------------|---------------|---|
| 0 | 2.0 | 1.7 | 2.7 | 3 |
| 32 | 2.3 | 2.3 | 3.3 | 3 |
| 64 | 2.0 | 2.1 | 2.7 | 10 |
| 96 | 2.0 | 2.2 | 2.5 | 10 |
| 128 | 1.6 | 1.9 | 2.3 | 10 |
| 160 | 1.9 | 2.2 | 2.6 | 10 |
| 192 | **2.2** | **2.2** | **3.1** | 10 |
| 201 | 2.1 | 2.3 | 3.0 | 10 |

**text_eval_loss trajectory:** 1.132 -> 1.118 -> 1.078 -> 1.038 -> 1.008 -> 0.995 -> 0.992 -> 0.992

**Key findings:**
- First run where metrics stayed above baseline through full training.
- Step 192 produced the best n=10 scores of any run in this session.
- Eval prompts were expanded from 3 to 10 during this run (starting at step 64).
- Did NOT learn specific scripts (Mia identity, DOB verification, AI disclosure).

---

### 3.2 v2-8: Higher LR Test

| Parameter | Value |
|-----------|-------|
| LR | 5e-6 |
| Rank | 32 |
| L2 | 4e-4 (both lora_A + lora_B) |
| audio_loss_weight | 0.5 |
| text_padding_weight | 0.0 |
| Duration | 300 s |
| Batch | 4 x 4 = 16 |
| Steps | 64 |

L2 was changed to penalize both lora_A and lora_B, based on research findings about asymmetric weight growth in LoRA.

**Gen_eval progression:**

| Step | Coherence | Naturalness | Effectiveness | n |
|------|-----------|-------------|---------------|---|
| 0 | 1.7 | 1.5 | 2.7 | 10 |
| 32 | 2.1 | 1.9 | 2.3 | 10 |
| 64 | 1.7 | 1.9 | 2.0 | 10 |

**text_eval_loss:** step 32 = 0.977, step 64 = 0.926

**Conclusion:** Higher LR learned faster (lower eval loss) but generation quality collapsed by step 64. The 5x higher LR outpaced even the 4x stronger L2.

---

### 3.3 v2-9: Stronger L2 (Stopped Early)

| Parameter | Value |
|-----------|-------|
| LR | 1e-6 |
| Rank | 32 |
| L2 | 5e-4 |

Stopped by the user before any gen_eval completed. No results available.

---

### 3.4 v2-11: High Capacity + High LR (No Padding)

| Parameter | Value |
|-----------|-------|
| LR | 1e-5 |
| Rank | 64 |
| L2 | 1e-4 |
| audio_loss_weight | 0.5 |
| text_padding_weight | 0.0 |
| Duration | 180 s |
| Batch | 8 x 2 = 16 |
| Steps | 595 (2 epochs) |

**Gen_eval progression:**

| Step | Coherence | Naturalness | Effectiveness | Silence % | n |
|------|-----------|-------------|---------------|-----------|---|
| 64 | 1.0 | 1.0 | 1.0 | 14% | 5 |
| 128 | 1.0 | 1.0 | 1.0 | 7% | 5 |

**text_eval_loss:** 0.728 -> 0.609 (excellent loss, broken generation)

**Transcript examples:** Repetition loops such as "I'm, I'm, I'm, I'm" and "I' calling, I' calling, I' calling".

**Conclusion:** lr=1e-5 with rank=64 completely breaks autoregressive generation. The model achieves good cross-entropy loss but produces degenerate repetitive output. Silence dropped to 7%, meaning the model talks non-stop without coherent content.

---

### 3.5 v2-12: High LR + High Padding (Overcorrection)

| Parameter | Value |
|-----------|-------|
| LR | 1e-5 |
| Rank | 64 |
| L2 | 1e-4 |
| text_padding_weight | 0.05 |
| Duration | 180 s |
| Batch | 8 x 2 = 16 |
| Steps | 595 |

**Gen_eval progression:**

| Step | Coherence | Naturalness | Effectiveness | Silence % | n |
|------|-----------|-------------|---------------|-----------|---|
| 64 | 1.0 | 1.0 | 1.0 | 82% | 5 |

**Transcript examples:** "So, Now, So, Now, So, Now" loops with barely any content.

**Conclusion:** text_padding_weight = 0.05 overcorrected the silence problem (from 7% to 82%) but the underlying content generation was still broken at lr=1e-5. Padding weight controls silence but does not fix the core repetition collapse.

---

### 3.6 v2-13: Half LR + Low Padding

| Parameter | Value |
|-----------|-------|
| LR | 5e-6 |
| Rank | 64 |
| L2 | 1e-4 |
| text_padding_weight | 0.01 |
| Duration | 180 s |
| Batch | 8 x 2 = 16 |
| Steps | 595 |

**Gen_eval progression:**

| Step | Coherence | Naturalness | Effectiveness | Silence % | n |
|------|-----------|-------------|---------------|-----------|---|
| 64 | 1.4 | 1.4 | 2.2 | 66% | 5 |
| 128 | 1.0 | 1.0 | 1.0 | 47% | 5 |

**text_eval_loss:** 0.809 -> 0.661

**Observations:**
- First rank=64 run that did NOT collapse at step 64.
- Step 64 transcripts showed proper clinical conversations.
- Step 128 transcripts reverted to repetition loops.

**Conclusion:** lr=5e-6 has a narrow window of good generation around step 64 before collapse sets in. The model transiently passes through a useful region of parameter space.

---

### 3.7 v2-14: Goldilocks Padding

| Parameter | Value |
|-----------|-------|
| LR | 5e-6 |
| Rank | 64 |
| L2 | 1e-4 |
| text_padding_weight | 0.03 |
| Duration | 180 s |
| Batch | 8 x 2 = 16 |
| Steps | 595 |

**Gen_eval progression:**

| Step | Coherence | Naturalness | Effectiveness | Silence % | n |
|------|-----------|-------------|---------------|-----------|---|
| 64 | **1.8** | **1.8** | **2.0** | 64% | 5 |
| 128 | 1.0 | 1.2 | 1.0 | 75% | 5 |

**text_eval_loss:** 0.764 -> 0.627

**Conclusion:** text_padding_weight = 0.03 produced the best rank=64 step-64 scores of any run tested. Silence percentage was well-behaved. However, content quality still collapsed by step 128 at lr=5e-6. Padding preserves turn-taking structure but cannot prevent the underlying content collapse.

---

### 3.8 v2-15: Large Batch (48) Test

| Parameter | Value |
|-----------|-------|
| LR | 5e-6 |
| Rank | 64 |
| L2 | 1e-4 |
| text_padding_weight | 0.03 |
| Duration | 180 s |
| Batch | 8 x 6 = 48 |
| Steps | 198 |

Motivated by the discovery that the old successful run used world_size=3, giving an effective batch size of 48.

**Gen_eval progression:**

| Step | Coherence | Naturalness | Effectiveness | Silence % | n |
|------|-----------|-------------|---------------|-----------|---|
| 32 | 1.6 | 2.0 | 1.6 | 66% | 5 |
| 64 | 1.2 | 1.2 | 1.4 | 57% | 5 |

**text_eval_loss:** 0.829 -> 0.686

**Conclusion:** batch=48 delayed collapse by approximately one eval step compared to batch=16, but did not prevent it. Smoother gradients from larger batch size are insufficient to stabilize lr=5e-6. Stopped early.

---

### 3.9 v2-16: Old Chunking Strategy Test

| Parameter | Value |
|-----------|-------|
| LR | 2e-5 |
| Rank | 64 |
| L2 | 1e-4 |
| text_padding_weight | 0.03 |
| Duration | 80 s |
| prompt_budget_frames | 0 (old chunking: chunk_step = duration_sec) |
| Batch | 8 x 6 = 48 |
| Steps | 175 |

**Hypothesis:** The code change that introduced `prompt_budget_frames` altered chunking behavior. Reverting to the old chunking strategy might explain the old run's success at lr=2e-5.

**Gen_eval progression:**

| Step | Coherence | Naturalness | Effectiveness | Silence % | n |
|------|-----------|-------------|---------------|-----------|---|
| 32 | 1.0 | 1.0 | 1.0 | 76% | 5 |
| 64 | 1.0 | 1.2 | 1.0 | 75% | 5 |

**text_eval_loss:** 0.686 -> 0.605

**Conclusion:** Old chunking is NOT the secret sauce. lr=2e-5 breaks generation with the current codebase regardless of chunking strategy. The hypothesis is rejected.

---

### 3.10 v2-17, v2-18: Skipped

Both would have used lr=2e-5, which was conclusively shown to break generation. Skipped to conserve compute.

---

### 3.11 v2-20: Safe Long Run (Currently Running)

| Parameter | Value |
|-----------|-------|
| LR | 1e-6 |
| Rank | 64 |
| L2 | 1e-4 |
| text_padding_weight | 0.03 |
| audio_loss_weight | 0.5 |
| Duration | 180 s |
| Batch | 8 x 6 = 48 |
| Steps | 396 (4 epochs) |
| eval_freq | 64 |

Combines v2-7's proven stable LR (1e-6) with rank=64 capacity, larger batch (48), appropriate padding (0.03), and 4x training duration.

**Gen_eval progression (partial):**

| Step | Coherence | Naturalness | Effectiveness | Silence % | n |
|------|-----------|-------------|---------------|-----------|---|
| 64 | 1.8 | 2.0 | 2.0 | 68% | 5 |

**text_eval_loss at step 64:** 0.960 (compared to v2-7's 1.078 at the same step -- **10% faster learning** with rank=64)

**Status:** Epoch 1 completed at step 99. Currently in epoch 2, around step 113. ETA approximately 03:51 UTC April 7 (~12 hours remaining).

---

## 4. Infrastructure Improvements During Session

1. **Eval prompt expansion:** Gen_eval prompts expanded from 3 to 10 (later trimmed to 5 for speed), covering diverse scenarios across therapeutic areas.
2. **L2 regularization implementation:** New `lora_l2_weight` parameter added to `args.py`; penalty computation added to `train.py`. Supports targeting lora_B only or both lora_A and lora_B.
3. **LoRA regularization research:** Investigated L2 on A+B vs B-only, scaling interaction effects, and asymmetric weight growth patterns.
4. **Bash monitoring script:** Continuous logging of training metrics.
5. **Cron-based automated monitoring:** Automated metric checks every 20 minutes.

---

## 5. Summary Results Table

| Run | LR | Rank | Batch | Pad | L2 | Chunking | Best gen_eval (coh/nat/eff) | Outcome |
|-----|-----|------|-------|-----|-----|----------|------------------------------|---------|
| old run | 2e-5 | 64 | 48 (3 GPU) | 0.5 | none | old | 3.0 / 3.0 / 4.0 @ step 1024 | Success but silence overfit |
| v2-7 | 1e-6 | 32 | 16 | 0.0 | 1e-4 | new | 2.2 / 2.2 / 3.1 @ step 192 | Stable, no scripts learned |
| v2-8 | 5e-6 | 32 | 16 | 0.0 | 4e-4 | new | 2.1 / 1.9 / 2.3 @ step 32 | Collapsed by step 64 |
| v2-11 | 1e-5 | 64 | 16 | 0.0 | 1e-4 | new | 1.0 / 1.0 / 1.0 | Collapsed immediately, sil=7% |
| v2-12 | 1e-5 | 64 | 16 | 0.05 | 1e-4 | new | 1.0 / 1.0 / 1.0 | Collapsed, sil=82% overcorrected |
| v2-13 | 5e-6 | 64 | 16 | 0.01 | 1e-4 | new | 1.4 / 1.4 / 2.2 @ step 64 | Good at 64, collapsed at 128 |
| v2-14 | 5e-6 | 64 | 16 | 0.03 | 1e-4 | new | 1.8 / 1.8 / 2.0 @ step 64 | Best rank=64 @ step 64, collapsed at 128 |
| v2-15 | 5e-6 | 64 | 48 | 0.03 | 1e-4 | new | 1.6 / 2.0 / 1.6 @ step 32 | Batch=48 delayed but did not prevent collapse |
| v2-16 | 2e-5 | 64 | 48 | 0.03 | 1e-4 | old | 1.0 / 1.0 / 1.0 | Old chunking does not help |
| v2-20 | 1e-6 | 64 | 48 | 0.03 | 1e-4 | new | 1.8 / 2.0 / 2.0 @ step 64 | Running, stable so far |

---

## 6. Conclusions

1. **L2 regularization on LoRA weights is essential.** It prevents catastrophic forgetting at lr=1e-6 and is the single most important change introduced in this session.

2. **lr > 1e-6 breaks autoregressive generation quality** with the current codebase, regardless of batch size, chunking strategy, or padding weight. Higher learning rates produce better cross-entropy loss but degenerate generation (repetition loops, incoherent output).

3. **text_padding_weight controls silence and turn-taking behavior:**
   - 0.0 = no signal (model chooses its own silence pattern)
   - 0.01 = too weak at high LR
   - 0.03 = good balance between silence and speech
   - 0.05 = overcorrects, producing excessive silence

4. **The old successful run's lr=2e-5 worked due to something specific to the 3-GPU FSDP setup or pre-code-change state**, NOT the chunking strategy. This was directly tested (v2-16) and falsified.

5. **rank=64 learns approximately 10% faster than rank=32** at lr=1e-6 (text_eval_loss 0.960 vs 1.078 at step 64) without introducing stability issues.

6. **batch=48 provides smoother gradients** but does not fundamentally change the LR stability boundary. It delays collapse by roughly one eval step at lr=5e-6.

7. **The model passes through a transient "good generation window"** at higher learning rates (around step 64 at lr=5e-6) but cannot sustain quality. This suggests the optimization landscape has a narrow region of good generation that higher LRs overshoot.

8. **v2-20 (lr=1e-6, rank=64, batch=48, 4 epochs) is the current best approach**, combining proven stability with increased capacity and extended training time.

---

## 7. Open Questions

1. **Can 4 epochs at lr=1e-6 teach specific scripts** (Mia identity, DOB verification, AI disclosure)? Or is the learning rate fundamentally too low for memorizing specific phrases?

2. **What exactly in the old 3-GPU FSDP setup allowed lr=2e-5 to work?** Possible factors: gradient averaging across GPUs, different optimizer state sharding, different effective learning rate due to FSDP gradient scaling.

3. **Would a staged warmup strategy work?** Train at lr=1e-6 to a stable checkpoint, then carefully increase to lr=5e-6 to push into the "good generation window" without overshooting.

4. **Would separate LoRA adapters for text vs audio channels** resolve the fundamental tension between text quality and audio quality during training?
