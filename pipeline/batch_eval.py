"""Batch generation evaluation: run multiple seeded bot-to-bot dialogues per scenario.

Generates 10 conversations (random seeds) × 3 scenarios = 30 dialogues,
evaluates them (silence metrics, conversation profile, LLM review), and
saves aggregated results.  Two dialogues run in parallel, each on its own
GPU pair (4 GPUs total).

Usage (with a LoRA checkpoint):
    python pipeline/batch_eval.py \
        --checkpoint runs/.../checkpoints/checkpoint_003000/consolidated \
        --output-dir runs/.../batch_eval \
        --gpus 1,2,3,4

Usage (base PersonaPlex model):
    python pipeline/batch_eval.py \
        --base \
        --output-dir runs/base_eval \
        --gpus 1,2,3,4

Options:
    --num-seeds N          Number of seeds per scenario (default: 10)
    --duration D           Dialogue duration in seconds (default: 60)
    --eval-prompts PATH    Path to eval prompts JSON (default: auto-detect)
    --hf-repo REPO         HuggingFace repo for base model (default: nvidia/personaplex-7b-v1)
    --nudge-after SEC      Seconds of mutual silence before nudging (default: 5.0, 0=disabled)
    --max-nudges N         Max nudge injections per dialogue (default: 5)
    --timeout SEC          Timeout per dialogue subprocess (default: 300)
    --no-llm-review        Skip LLM transcript review
    --no-profile           Skip WhisperX conversation profile eval
    --llm-model MODEL      Claude model for review (default: claude-sonnet-4-20250514)
    --puppeteer PATH       Puppeteer knowledge base JSON: {scenario_id: [{frame: int, text: str}, ...]}
    --gpus G               4 comma-separated GPU ids: two pairs for parallel dialogues (default: 1,2,3,4)
"""

import argparse
import json
import logging
import math
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("batch_eval")

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # voice-training/
BOT_TO_BOT_CWD = PROJECT_ROOT / "personaplex" / "moshi"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
EVAL_PROFILE_SCRIPT = PROJECT_ROOT / "pipeline" / "eval_conversation_profile.py"
DEFAULT_HF_REPO = "nvidia/personaplex-7b-v1"
DEFAULT_EVAL_PROMPTS = PROJECT_ROOT / "moshi-finetune" / "finetune" / "eval_prompts.json"


def load_eval_prompts(path: str | None) -> list[dict]:
    if path:
        p = Path(path)
    else:
        p = DEFAULT_EVAL_PROMPTS
    with open(p) as f:
        return json.load(f)


def merge_lora(ckpt_dir: Path, output_path: Path, hf_repo: str) -> Path:
    sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))
    from merge_lora import merge
    return merge(ckpt_dir=ckpt_dir, output_path=output_path, hf_repo=hf_repo)


def run_single_dialogue(
    prompt: dict,
    seed: int,
    output_dir: Path,
    merged_weight: Path | None,
    gpu_a: int,
    gpu_b: int,
    duration: float,
    timeout: int,
    nudge_after: float,
    max_nudges: int,
    context_injections: list[dict] | None = None,
) -> dict:
    """Run one bot-to-bot dialogue. Returns result dict."""
    dial_id = f"{prompt['id']}_seed{seed}"
    wav_path = output_dir / f"{dial_id}.wav"
    text_a_path = output_dir / f"{dial_id}_text_a.json"
    text_b_path = output_dir / f"{dial_id}_text_b.json"

    cmd = [
        str(PYTHON), "-m", "moshi.bot_to_bot",
        "--output-wav", str(wav_path),
        "--output-text-a", str(text_a_path),
        "--output-text-b", str(text_b_path),
        "--duration", str(duration),
        "--gpu-a", str(gpu_a),
        "--gpu-b", str(gpu_b),
        "--text-prompt-a", prompt["broker_prompt"],
        "--text-prompt-b", prompt["client_prompt"],
        "--voice-prompt-a", prompt["voice_broker"],
        "--voice-prompt-b", prompt["voice_client"],
        "--seed", str(seed),
    ]
    if merged_weight is not None:
        cmd.extend(["--moshi-weight-a", str(merged_weight)])
    # If merged_weight is None (base model), Bot A also loads from HF default
    if prompt.get("greeting"):
        cmd.extend(["--greeting-a", prompt["greeting"]])
    if nudge_after > 0:
        cmd.extend(["--nudge-after", str(nudge_after), "--max-nudges", str(max_nudges)])

    # Write scheduled context injections to a temp JSON file for this dialogue
    ctx_inj_path = None
    if context_injections:
        ctx_inj_path = output_dir / f"{dial_id}_injections.json"
        with open(ctx_inj_path, "w") as f:
            json.dump(context_injections, f, ensure_ascii=False)
        cmd.extend(["--context-injections", str(ctx_inj_path)])

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(BOT_TO_BOT_CWD),
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = time.time() - t0

        if proc.returncode != 0:
            logger.error(f"Dialogue {dial_id} failed (rc={proc.returncode}):\n{proc.stderr[-500:]}")
            return {"id": dial_id, "scenario_id": prompt["id"], "seed": seed,
                    "status": "error", "error": proc.stderr[-500:]}
        if not wav_path.exists():
            logger.error(f"Dialogue {dial_id}: no WAV produced")
            return {"id": dial_id, "scenario_id": prompt["id"], "seed": seed,
                    "status": "error", "error": "no WAV file"}

        logger.info(f"Dialogue {dial_id} completed in {elapsed:.0f}s")
        result = {
            "id": dial_id, "scenario_id": prompt["id"], "seed": seed,
            "status": "ok",
            "wav_path": str(wav_path),
            "text_a_path": str(text_a_path),
            "text_b_path": str(text_b_path),
            "elapsed_s": elapsed,
            "system_prompt": prompt.get("broker_prompt", ""),
        }
        # Provenance: record context injections used during this dialogue
        if prompt.get("context_injections"):
            result["context_injections"] = prompt["context_injections"]
        return result

    except subprocess.TimeoutExpired:
        logger.error(f"Dialogue {dial_id} timed out after {timeout}s")
        return {"id": dial_id, "scenario_id": prompt["id"], "seed": seed,
                "status": "timeout"}
    except Exception as e:
        logger.error(f"Dialogue {dial_id} exception: {e}")
        return {"id": dial_id, "scenario_id": prompt["id"], "seed": seed,
                "status": "error", "error": str(e)}


def compute_silence_metrics(dialogue_results: list[dict], frame_rate: float = 12.5) -> dict:
    """Compute mutual silence metrics from text transcripts."""
    total_frames = 0
    total_mutual_silence = 0
    max_longest_silence = 0.0
    silence_tokens = {"PAD", "EPAD", "<CTX>"}

    per_dialogue = []
    for d in dialogue_results:
        if d["status"] != "ok":
            continue
        try:
            with open(d["text_a_path"]) as f:
                tokens_a = json.load(f)
            with open(d["text_b_path"]) as f:
                tokens_b = json.load(f)
        except Exception:
            continue

        n = min(len(tokens_a), len(tokens_b))
        if n == 0:
            continue

        mutual_silence = 0
        longest_streak = 0
        streak = 0
        for i in range(n):
            if tokens_a[i] in silence_tokens and tokens_b[i] in silence_tokens:
                mutual_silence += 1
                streak += 1
                longest_streak = max(longest_streak, streak)
            else:
                streak = 0

        total_frames += n
        total_mutual_silence += mutual_silence
        max_longest_silence = max(max_longest_silence, longest_streak / frame_rate)
        per_dialogue.append({
            "id": d["id"],
            "silence_pct": 100.0 * mutual_silence / n,
            "longest_silence_s": longest_streak / frame_rate,
        })

    if total_frames == 0:
        return {"aggregate": {}, "per_dialogue": []}

    return {
        "aggregate": {
            "silence_pct": 100.0 * total_mutual_silence / total_frames,
            "longest_silence_s": max_longest_silence,
        },
        "per_dialogue": per_dialogue,
    }


def evaluate_profiles(wav_paths: list[str], gpu_id: int) -> dict:
    """Run eval_conversation_profile.py on all WAVs."""
    if not wav_paths:
        return {}
    if not EVAL_PROFILE_SCRIPT.exists():
        logger.warning(f"Profile eval script not found: {EVAL_PROFILE_SCRIPT}")
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for wp in wav_paths:
            shutil.copy2(wp, tmpdir)

        out_json = Path(tmpdir) / "profile_results.json"
        env = dict(__import__("os").environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        script_text = EVAL_PROFILE_SCRIPT.read_text()
        script_text = script_text.replace(
            'os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
            '# CUDA_VISIBLE_DEVICES set via subprocess env',
        )
        patched_script = Path(tmpdir) / "eval_conversation_profile.py"
        patched_script.write_text(script_text)

        cmd = [str(PYTHON), str(patched_script), tmpdir, "--out", str(out_json)]

        logger.info(f"Running conversation profile eval on GPU {gpu_id} ({len(wav_paths)} files)")
        try:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)
            if proc.returncode != 0:
                logger.error(f"Profile eval failed:\n{proc.stderr[-500:]}")
                return {}
            if not out_json.exists():
                logger.error("Profile eval produced no output JSON")
                return {}
            with open(out_json) as f:
                return json.load(f)
        except subprocess.TimeoutExpired:
            logger.error("Profile eval timed out")
            return {}
        except Exception as e:
            logger.error(f"Profile eval exception: {e}")
            return {}


def _read_transcript(text_path: str) -> str:
    with open(text_path) as f:
        tokens = json.load(f)
    return "".join(t for t in tokens if t not in ("EPAD", "BOS", "EOS", "PAD", "<CTX>")).strip()


def _review_single(dialogue: dict, prompt: dict, model: str) -> dict:
    import anthropic

    text_a = _read_transcript(dialogue["text_a_path"])
    text_b = _read_transcript(dialogue["text_b_path"])

    # Build reference material for grounding check
    broker_prompt_full = prompt.get('broker_prompt', '')
    context_injections = dialogue.get('context_injections', [])
    reference_block = f"BROKER SYSTEM PROMPT (full):\n{broker_prompt_full}\n"
    if context_injections:
        ctx_texts = "\n".join(f"  - {c['text']}" for c in context_injections if isinstance(c, dict))
        reference_block += f"\nINJECTED CONTEXT (provided to broker mid-conversation):\n{ctx_texts}\n"

    review_prompt = f"""You are evaluating a simulated insurance broker phone conversation.

SCENARIO: {prompt.get('scenario', 'Unknown')}

{reference_block}
BROKER SAID:
{text_a[:2000]}

CLIENT SAID:
{text_b[:2000]}

Rate this conversation on four dimensions (1-5 each):
1. COHERENCE: Does the conversation flow logically? Do responses relate to what was said? Are there non-sequiturs or repetitions?
2. NATURALNESS: Does it sound like a real phone conversation? Appropriate turn-taking, backchannels, filler words?
3. EFFECTIVENESS: Does the broker accomplish their goal? Do they ask relevant questions, provide useful information, build rapport?
4. GROUNDING: Does the broker only cite facts present in the system prompt or injected context above? Score 1 = fabricated specifics (invented dollar amounts, policy numbers, dates not in the reference material), 5 = all claims traceable to provided information. If the broker hedges on unknown items rather than fabricating, that counts positively.

Respond in exactly this JSON format:
{{"notes": "<1-2 sentence summary>", "coherence": <int>, "naturalness": <int>, "effectiveness": <int>, "grounding": <int>}}"""

    client = anthropic.Anthropic()
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model, max_tokens=512,
                messages=[{"role": "user", "content": review_prompt}],
            )
            text = response.content[0].text.strip()
            if "{" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]
                return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response for {dialogue['id']}, attempt {attempt+1}")
        except Exception as e:
            logger.warning(f"LLM review error for {dialogue['id']}, attempt {attempt+1}: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    return {"coherence": float("nan"), "naturalness": float("nan"),
            "effectiveness": float("nan"), "grounding": float("nan"),
            "notes": "review failed"}


def review_transcripts(dialogue_results: list[dict], eval_prompts: list[dict], model: str) -> list[dict]:
    ok_results = [d for d in dialogue_results if d["status"] == "ok"]
    if not ok_results:
        return []

    prompts_by_id = {p["id"]: p for p in eval_prompts}
    reviews = []

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = []
        for d in ok_results:
            prompt = prompts_by_id.get(d["scenario_id"], {})
            futures.append(pool.submit(_review_single, d, prompt, model))

        for d, future in zip(ok_results, futures):
            try:
                review = future.result(timeout=60)
                review["id"] = d["id"]
                review["scenario_id"] = d["scenario_id"]
                review["seed"] = d["seed"]
                reviews.append(review)
            except Exception as e:
                logger.warning(f"Review failed for {d['id']}: {e}")
                reviews.append({
                    "id": d["id"], "scenario_id": d["scenario_id"], "seed": d["seed"],
                    "coherence": float("nan"), "naturalness": float("nan"),
                    "effectiveness": float("nan"), "grounding": float("nan"),
                    "notes": str(e),
                })

    return reviews


def aggregate_results(
    dialogue_results: list[dict],
    reviews: list[dict],
    silence: dict,
    profile: dict | list,
) -> dict:
    """Compute per-scenario and overall aggregate metrics."""

    def safe_mean(vals):
        finite = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
        return sum(finite) / len(finite) if finite else float("nan")

    n_ok = sum(1 for d in dialogue_results if d["status"] == "ok")
    n_total = len(dialogue_results)

    # Per-scenario aggregation
    scenarios = {}
    for d in dialogue_results:
        sid = d["scenario_id"]
        if sid not in scenarios:
            scenarios[sid] = {"ok": 0, "total": 0, "reviews": []}
        scenarios[sid]["total"] += 1
        if d["status"] == "ok":
            scenarios[sid]["ok"] += 1

    for r in reviews:
        sid = r.get("scenario_id")
        if sid in scenarios:
            scenarios[sid]["reviews"].append(r)

    per_scenario = {}
    for sid, data in scenarios.items():
        sc_reviews = data["reviews"]
        per_scenario[sid] = {
            "n_ok": data["ok"],
            "n_total": data["total"],
            "coherence_mean": safe_mean([r["coherence"] for r in sc_reviews]),
            "naturalness_mean": safe_mean([r["naturalness"] for r in sc_reviews]),
            "effectiveness_mean": safe_mean([r["effectiveness"] for r in sc_reviews]),
            "grounding_mean": safe_mean([r.get("grounding", float("nan")) for r in sc_reviews]),
        }

    overall = {
        "n_ok": n_ok,
        "n_total": n_total,
        "coherence_mean": safe_mean([r["coherence"] for r in reviews]),
        "naturalness_mean": safe_mean([r["naturalness"] for r in reviews]),
        "effectiveness_mean": safe_mean([r["effectiveness"] for r in reviews]),
        "grounding_mean": safe_mean([r.get("grounding", float("nan")) for r in reviews]),
    }

    if silence.get("aggregate"):
        overall.update(silence["aggregate"])

    # Extract profile summary if available
    profile_summary = {}
    if isinstance(profile, list) and profile:
        profile_summary = profile[0] if len(profile) == 1 else profile
    elif isinstance(profile, dict) and profile:
        profile_summary = profile

    return {
        "overall": overall,
        "per_scenario": per_scenario,
        "profile_summary": profile_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch generation evaluation for PersonaPlex")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str,
                       help="Path to LoRA checkpoint consolidated dir")
    group.add_argument("--merged-weight", type=str,
                       help="Path to already-merged model.safetensors")
    group.add_argument("--base", action="store_true",
                       help="Evaluate the base PersonaPlex model (no LoRA)")

    parser.add_argument("--output-dir", required=True, type=str, help="Output directory")
    parser.add_argument("--gpus", type=str, default="1,2,3,4",
                        help="4 comma-separated GPU ids: first pair + second pair for parallel dialogues")
    parser.add_argument("--num-seeds", type=int, default=10, help="Seeds per scenario")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seed values (overrides --num-seeds). Use to match seeds across runs.")
    parser.add_argument("--duration", type=float, default=60.0, help="Dialogue duration (seconds)")
    parser.add_argument("--eval-prompts", type=str, default=None, help="Path to eval prompts JSON")
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO)
    parser.add_argument("--nudge-after", type=float, default=5.0, help="Silence seconds before nudge (0=off)")
    parser.add_argument("--max-nudges", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per dialogue")
    parser.add_argument("--no-llm-review", action="store_true", help="Skip LLM transcript review")
    parser.add_argument("--no-profile", action="store_true", help="Skip WhisperX profile eval")
    parser.add_argument("--llm-model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--puppeteer", type=str, default=None,
                        help="Path to puppeteer knowledge base JSON for context injection. "
                             "Format: {scenario_id: [{frame: int, text: str}, ...]}")

    args = parser.parse_args()

    # Parse GPU ids → two (gpu_a, gpu_b) pairs
    gpu_ids = [int(g) for g in args.gpus.split(",")]
    if len(gpu_ids) != 4:
        parser.error("--gpus must be exactly 4 comma-separated GPU ids (two pairs)")
    gpu_pairs = [(gpu_ids[0], gpu_ids[1]), (gpu_ids[2], gpu_ids[3])]
    logger.info(f"GPU pairs: slot0={gpu_pairs[0]}, slot1={gpu_pairs[1]}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dialogues_dir = output_dir / "dialogues"
    dialogues_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolve model weight ---
    merged_weight = None  # None = base model from HF
    temp_merged = None

    if args.checkpoint:
        ckpt_dir = Path(args.checkpoint)
        merged_path = output_dir / "merged_model.safetensors"
        logger.info(f"Merging LoRA from {ckpt_dir} → {merged_path}")
        merge_lora(ckpt_dir, merged_path, args.hf_repo)
        merged_weight = merged_path
        temp_merged = merged_path
    elif args.merged_weight:
        merged_weight = Path(args.merged_weight)
        if not merged_weight.exists():
            logger.error(f"Merged weight not found: {merged_weight}")
            sys.exit(1)
    else:
        # --base: both bots use default HF model
        logger.info("Evaluating base PersonaPlex model (no LoRA)")

    # --- Load eval prompts ---
    eval_prompts = load_eval_prompts(args.eval_prompts)
    logger.info(f"Loaded {len(eval_prompts)} scenarios from eval prompts")

    # --- Load puppeteer knowledge base ---
    puppeteer_kb: dict[str, list[dict]] = {}
    if args.puppeteer:
        with open(args.puppeteer) as f:
            puppeteer_kb = json.load(f)
        logger.info(f"Loaded puppeteer knowledge base: {len(puppeteer_kb)} scenarios with injections")

    # Resolve seeds: explicit list or random
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        rng = random.Random()
        seeds = [rng.randint(0, 2**31 - 1) for _ in range(args.num_seeds)]
    total = len(eval_prompts) * len(seeds)
    logger.info(f"Will run {total} dialogues: {len(eval_prompts)} scenarios × {len(seeds)} seeds")
    logger.info(f"Seeds: {seeds}")

    # Build job list
    jobs = []
    for prompt in eval_prompts:
        for seed in seeds:
            jobs.append((prompt, seed))

    # --- Run dialogues (2 in parallel via GPU-pair slots) ---
    gpu_slot_q = queue.Queue()
    for pair in gpu_pairs:
        gpu_slot_q.put(pair)

    all_results = []
    results_lock = threading.Lock()
    counter = [0]  # mutable counter for thread-safe incrementing

    def _run_job(job):
        prompt, seed = job
        gpu_a, gpu_b = gpu_slot_q.get()
        try:
            with results_lock:
                counter[0] += 1
                idx = counter[0]
            # Look up puppeteer injections for this scenario
            ctx_inj = puppeteer_kb.get(prompt["id"])
            if ctx_inj:
                # Store on prompt for provenance logging in results
                prompt = {**prompt, "context_injections": ctx_inj}
            logger.info(f"[{idx}/{total}] Scenario={prompt['id']} Seed={seed} GPUs=({gpu_a},{gpu_b})"
                         + (f" +{len(ctx_inj)} injections" if ctx_inj else ""))
            return run_single_dialogue(
                prompt=prompt,
                seed=seed,
                output_dir=dialogues_dir,
                merged_weight=merged_weight,
                gpu_a=gpu_a,
                gpu_b=gpu_b,
                duration=args.duration,
                timeout=args.timeout,
                nudge_after=args.nudge_after,
                max_nudges=args.max_nudges,
                context_injections=ctx_inj,
            )
        finally:
            gpu_slot_q.put((gpu_a, gpu_b))

    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(_run_job, jobs):
            all_results.append(result)
            # Save incremental progress (thread-safe: pool.map returns in order)
            with open(output_dir / "dialogues_progress.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

    n_ok = sum(1 for d in all_results if d["status"] == "ok")
    logger.info(f"{n_ok}/{total} dialogues succeeded")

    # Clean up temp merged weights
    if temp_merged and temp_merged.exists():
        logger.info(f"Removing merged weights: {temp_merged}")
        temp_merged.unlink()

    if n_ok == 0:
        logger.error("All dialogues failed — skipping evaluation")
        final = {"dialogues": all_results, "metrics": {}}
        with open(output_dir / "results.json", "w") as f:
            json.dump(final, f, indent=2, default=str)
        sys.exit(1)

    # --- Silence metrics ---
    logger.info("Computing silence metrics...")
    silence = compute_silence_metrics(all_results)

    # --- Profile eval ---
    profile = {}
    if not args.no_profile:
        wav_paths = [d["wav_path"] for d in all_results if d["status"] == "ok"]
        try:
            profile = evaluate_profiles(wav_paths, gpu_ids[0])
        except Exception as e:
            logger.error(f"Profile eval failed: {e}")

    # --- LLM review ---
    reviews = []
    if not args.no_llm_review:
        logger.info("Running LLM transcript review...")
        try:
            reviews = review_transcripts(all_results, eval_prompts, args.llm_model)
        except Exception as e:
            logger.error(f"LLM review failed: {e}")

    # --- Aggregate ---
    metrics = aggregate_results(all_results, reviews, silence, profile)

    # --- Save ---
    final = {
        "config": {
            "checkpoint": args.checkpoint,
            "merged_weight": str(args.merged_weight) if args.merged_weight else None,
            "base": args.base if hasattr(args, "base") else False,
            "num_seeds": args.num_seeds,
            "seeds": seeds,
            "gpus": gpu_ids,
            "duration": args.duration,
            "hf_repo": args.hf_repo,
            "nudge_after": args.nudge_after,
            "max_nudges": args.max_nudges,
            "llm_model": args.llm_model if not args.no_llm_review else None,
            "puppeteer": args.puppeteer,
        },
        "metrics": metrics,
        "silence": silence,
        "reviews": reviews,
        "dialogues": all_results,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 60)
    overall = metrics["overall"]
    print(f"Dialogues: {overall['n_ok']}/{overall['n_total']} succeeded")
    if not math.isnan(overall.get("coherence_mean", float("nan"))):
        print(f"Coherence:     {overall['coherence_mean']:.2f}/5")
        print(f"Naturalness:   {overall['naturalness_mean']:.2f}/5")
        print(f"Effectiveness: {overall['effectiveness_mean']:.2f}/5")
        grounding = overall.get("grounding_mean", float("nan"))
        if not math.isnan(grounding):
            print(f"Grounding:     {grounding:.2f}/5")
    if "silence_pct" in overall:
        print(f"Silence:       {overall['silence_pct']:.1f}% mutual, longest={overall['longest_silence_s']:.1f}s")

    print("\nPer-scenario:")
    for sid, sc in metrics["per_scenario"].items():
        coh = sc["coherence_mean"]
        nat = sc["naturalness_mean"]
        eff = sc["effectiveness_mean"]
        gnd = sc.get("grounding_mean", float("nan"))
        coh_s = f"{coh:.2f}" if not math.isnan(coh) else "N/A"
        nat_s = f"{nat:.2f}" if not math.isnan(nat) else "N/A"
        eff_s = f"{eff:.2f}" if not math.isnan(eff) else "N/A"
        gnd_s = f"{gnd:.2f}" if not math.isnan(gnd) else "N/A"
        print(f"  {sid}: {sc['n_ok']}/{sc['n_total']} ok | C={coh_s} N={nat_s} E={eff_s} G={gnd_s}")

    print(f"\nFull results: {results_path}")


if __name__ == "__main__":
    main()
