"""Run gen_eval multiple times with different seeds for statistical comparison.

Usage:
    cd moshi-finetune
    .venv/bin/python run_multi_seed_eval.py \
        --checkpoint /mnt/data/runs/companion_training-v0-16/checkpoints/checkpoint_000096/consolidated \
        --output-dir /mnt/data/runs/multi_seed_eval_v0-16-step96 \
        --num-seeds 10

    # For base model (no checkpoint):
    .venv/bin/python run_multi_seed_eval.py \
        --base \
        --output-dir /mnt/data/runs/multi_seed_eval_base \
        --num-seeds 10
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("multi_seed_eval")

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
MOSHI_DIR = Path(__file__).resolve().parent.parent / "personaplex" / "moshi"
sys.path.insert(0, str(MOSHI_DIR))

from finetune.gen_eval import (
    merge_lora,
    run_dialogues,
    _compute_silence_metrics,
    review_transcripts,
    _load_eval_prompts,
)

DEFAULT_HF_REPO = "nvidia/personaplex-7b-v1"


def main():
    parser = argparse.ArgumentParser(description="Multi-seed gen_eval for statistical comparison")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint consolidated dir")
    parser.add_argument("--base", action="store_true", help="Evaluate base model (no LoRA)")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--eval-prompts", type=str, default="finetune/eval_prompts_companion.json")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_HF_REPO)
    parser.add_argument("--llm-model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--no-llm-review", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_prompts = _load_eval_prompts(args.eval_prompts)
    logger.info(f"Loaded {len(eval_prompts)} eval prompts")

    # Get merged weight path
    if args.base:
        # Download base model weight
        from huggingface_hub import hf_hub_download
        merged_weight = Path(hf_hub_download(args.hf_repo, "model.safetensors"))
        logger.info(f"Using base model: {merged_weight}")
    else:
        assert args.checkpoint, "--checkpoint required unless --base"
        merged_weight = output_dir / "merged_model.safetensors"
        if not merged_weight.exists():
            logger.info(f"Merging LoRA from {args.checkpoint}")
            merge_lora(Path(args.checkpoint), merged_weight, args.hf_repo)
        else:
            logger.info(f"Using existing merged weights: {merged_weight}")

    # Run dialogues for each seed
    all_dialogues = []
    for seed in range(args.num_seeds):
        logger.info(f"=== Seed {seed}/{args.num_seeds} ===")
        seed_dir = output_dir / f"seed_{seed:02d}"

        dialogue_results = run_dialogues(
            merged_weight=merged_weight,
            eval_prompts=eval_prompts,
            output_dir=seed_dir / "dialogues",
            gpu_a=args.gpu,
            gpu_b=args.gpu,
            duration=args.duration,
            seed=seed,
            timeout=args.timeout,
        )

        n_ok = sum(1 for d in dialogue_results if d["status"] == "ok")
        logger.info(f"Seed {seed}: {n_ok}/{len(eval_prompts)} dialogues succeeded")

        for d in dialogue_results:
            d["seed"] = seed
        all_dialogues.extend(dialogue_results)

    # LLM review all dialogues
    reviews = []
    if not args.no_llm_review:
        logger.info(f"Running LLM review on {len(all_dialogues)} dialogues...")
        reviews = review_transcripts(all_dialogues, eval_prompts, args.llm_model)

    # Aggregate results
    results = {
        "n_seeds": args.num_seeds,
        "n_prompts": len(eval_prompts),
        "n_dialogues": len(all_dialogues),
        "n_ok": sum(1 for d in all_dialogues if d["status"] == "ok"),
        "reviews": reviews,
        "per_scenario": {},
    }

    # Group reviews by scenario
    for r in reviews:
        sid = r["id"]
        if sid not in results["per_scenario"]:
            results["per_scenario"][sid] = []
        results["per_scenario"][sid].append(r)

    # Compute stats per scenario
    summary = {}
    for sid, scenario_reviews in results["per_scenario"].items():
        scores = {
            "coherence": [r["coherence"] for r in scenario_reviews if r.get("coherence") is not None],
            "naturalness": [r["naturalness"] for r in scenario_reviews if r.get("naturalness") is not None],
            "effectiveness": [r["effectiveness"] for r in scenario_reviews if r.get("effectiveness") is not None],
        }
        summary[sid] = {}
        for metric, vals in scores.items():
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                summary[sid][metric] = {"mean": round(mean, 2), "std": round(std, 2), "n": len(vals), "values": vals}

    results["summary"] = summary

    # Overall means
    all_coh = [r["coherence"] for r in reviews if r.get("coherence") is not None]
    all_nat = [r["naturalness"] for r in reviews if r.get("naturalness") is not None]
    all_eff = [r["effectiveness"] for r in reviews if r.get("effectiveness") is not None]

    if all_coh:
        results["overall"] = {
            "coherence": {"mean": round(sum(all_coh) / len(all_coh), 2), "std": round((sum((v - sum(all_coh)/len(all_coh))**2 for v in all_coh) / len(all_coh))**0.5, 2)},
            "naturalness": {"mean": round(sum(all_nat) / len(all_nat), 2), "std": round((sum((v - sum(all_nat)/len(all_nat))**2 for v in all_nat) / len(all_nat))**0.5, 2)},
            "effectiveness": {"mean": round(sum(all_eff) / len(all_eff), 2), "std": round((sum((v - sum(all_eff)/len(all_eff))**2 for v in all_eff) / len(all_eff))**0.5, 2)},
        }
        logger.info(f"Overall: coh={results['overall']['coherence']}, nat={results['overall']['naturalness']}, eff={results['overall']['effectiveness']}")

    # Save
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_dir / 'results.json'}")

    # Print summary table
    print("\n=== Per-Scenario Summary ===")
    print(f"{'scenario':30s}  {'coh':>12s}  {'nat':>12s}  {'eff':>12s}")
    print("-" * 70)
    for sid in sorted(summary):
        s = summary[sid]
        coh = f"{s['coherence']['mean']:.1f}±{s['coherence']['std']:.1f}" if 'coherence' in s else "N/A"
        nat = f"{s['naturalness']['mean']:.1f}±{s['naturalness']['std']:.1f}" if 'naturalness' in s else "N/A"
        eff = f"{s['effectiveness']['mean']:.1f}±{s['effectiveness']['std']:.1f}" if 'effectiveness' in s else "N/A"
        print(f"{sid:30s}  {coh:>12s}  {nat:>12s}  {eff:>12s}")

    if "overall" in results:
        o = results["overall"]
        print("-" * 70)
        print(f"{'OVERALL':30s}  {o['coherence']['mean']:.1f}±{o['coherence']['std']:.1f}:>12s  {o['naturalness']['mean']:.1f}±{o['naturalness']['std']:.1f}:>12s  {o['effectiveness']['mean']:.1f}±{o['effectiveness']['std']:.1f}:>12s")


if __name__ == "__main__":
    main()
