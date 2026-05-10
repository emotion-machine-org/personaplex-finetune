"""Test different sampling parameters on base and finetuned models.

Runs each eval prompt once per sampling config, then LLM-reviews the transcripts.
"""

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("sampling_test")

sys.path.insert(0, str(Path(__file__).parent))
MOSHI_DIR = Path(__file__).resolve().parent.parent / "personaplex" / "moshi"
sys.path.insert(0, str(MOSHI_DIR))
PYTHON = str(Path(__file__).resolve().parent / ".venv" / "bin" / "python")

from finetune.gen_eval import (
    review_transcripts,
    _load_eval_prompts,
    merge_lora,
)

# Sampling configs to test
SAMPLING_CONFIGS = {
    "warm_balanced":      {"temp_text": 0.7, "topk_text": 30,  "temp_audio": 0.55, "topk_audio": 100},
    "warm_rep1.2":        {"temp_text": 0.7, "topk_text": 30,  "temp_audio": 0.55, "topk_audio": 100, "rep_penalty": 1.2},
    "warm_rep1.3":        {"temp_text": 0.7, "topk_text": 30,  "temp_audio": 0.55, "topk_audio": 100, "rep_penalty": 1.3},
    "warm_rep1.5":        {"temp_text": 0.7, "topk_text": 30,  "temp_audio": 0.55, "topk_audio": 100, "rep_penalty": 1.5},
    "warm_rep2.0":        {"temp_text": 0.7, "topk_text": 30,  "temp_audio": 0.55, "topk_audio": 100, "rep_penalty": 2.0},
}


def run_dialogue(model_weight, prompt, output_dir, sampling, gpu=0, duration=120, seed=42, timeout=600):
    """Run a single bot-to-bot dialogue."""
    dial_id = prompt["id"]
    wav_path = output_dir / f"{dial_id}.wav"
    text_a_path = output_dir / f"{dial_id}_text_a.json"
    text_b_path = output_dir / f"{dial_id}_text_b.json"

    cmd = [
        PYTHON, "-m", "moshi.bot_to_bot",
        "--output-wav", str(wav_path),
        "--output-text-a", str(text_a_path),
        "--output-text-b", str(text_b_path),
        "--duration", str(duration),
        "--gpu-a", str(gpu), "--gpu-b", str(gpu),
        "--text-prompt-a", prompt["broker_prompt"],
        "--text-prompt-b", prompt["client_prompt"],
        "--voice-prompt-a", prompt["voice_broker"],
        "--voice-prompt-b", prompt["voice_client"],
        "--seed", str(seed),
        "--temp-audio", str(sampling["temp_audio"]),
        "--temp-text", str(sampling["temp_text"]),
        "--topk-audio", str(sampling["topk_audio"]),
        "--topk-text", str(sampling["topk_text"]),
    ]
    if sampling.get("rep_penalty", 0) > 0:
        cmd.extend(["--rep-penalty", str(sampling["rep_penalty"])])
        if "rep_penalty_window" in sampling:
            cmd.extend(["--rep-penalty-window", str(sampling["rep_penalty_window"])])
    if model_weight:
        cmd.extend(["--moshi-weight-a", str(model_weight)])
    if prompt.get("greeting"):
        cmd.extend(["--greeting-a", prompt["greeting"]])

    env = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    import os
    full_env = {**os.environ, **env}

    try:
        t0 = time.time()
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=str(MOSHI_DIR), env=full_env,
        )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            return {
                "id": dial_id, "status": "ok",
                "text_a_path": str(text_a_path),
                "text_b_path": str(text_b_path),
                "wav_path": str(wav_path),
                "system_prompt": prompt.get("broker_prompt", ""),
                "elapsed_s": elapsed,
            }
        else:
            logger.error(f"{dial_id} failed: {proc.stderr[-200:]}")
            return {"id": dial_id, "status": "error"}
    except subprocess.TimeoutExpired:
        logger.error(f"{dial_id} timed out")
        return {"id": dial_id, "status": "timeout"}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--output-dir", type=str, default="/mnt/data/runs/sampling_test")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="LoRA checkpoint consolidated dir (omit for base only)")
    parser.add_argument("--eval-prompts", type=str, default="finetune/eval_prompts_companion.json")
    parser.add_argument("--configs", type=str, default=None,
                       help="Comma-separated config names to test (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_prompts = _load_eval_prompts(args.eval_prompts)

    configs_to_test = SAMPLING_CONFIGS
    if args.configs:
        names = args.configs.split(",")
        configs_to_test = {k: v for k, v in SAMPLING_CONFIGS.items() if k in names}

    # Prepare models
    models = {}
    # Base model (no weight = downloads from HF)
    models["base"] = None

    if args.checkpoint:
        merged_path = output_dir / "merged_model.safetensors"
        if not merged_path.exists():
            logger.info(f"Merging LoRA from {args.checkpoint}")
            merge_lora(Path(args.checkpoint), merged_path, "nvidia/personaplex-7b-v1")
        models["finetuned"] = str(merged_path)

    all_results = {}

    for model_name, model_weight in models.items():
        for config_name, sampling in configs_to_test.items():
            key = f"{model_name}__{config_name}"
            logger.info(f"\n=== {key} ===")
            logger.info(f"  sampling: {sampling}")

            dial_dir = output_dir / key / "dialogues"
            dial_dir.mkdir(parents=True, exist_ok=True)

            dialogue_results = []
            for prompt in eval_prompts:
                logger.info(f"  Running {prompt['id']}...")
                result = run_dialogue(
                    model_weight, prompt, dial_dir, sampling,
                    gpu=args.gpu, duration=args.duration,
                )
                dialogue_results.append(result)

            n_ok = sum(1 for d in dialogue_results if d["status"] == "ok")
            logger.info(f"  {n_ok}/{len(eval_prompts)} succeeded")

            # LLM review
            reviews = review_transcripts(dialogue_results, eval_prompts, "claude-sonnet-4-20250514")
            for r in reviews:
                r["model"] = model_name
                r["sampling"] = config_name

            all_results[key] = {
                "model": model_name,
                "sampling": config_name,
                "params": sampling,
                "reviews": reviews,
            }

            # Print scores
            coh = [r["coherence"] for r in reviews if r.get("coherence") is not None]
            nat = [r["naturalness"] for r in reviews if r.get("naturalness") is not None]
            eff = [r["effectiveness"] for r in reviews if r.get("effectiveness") is not None]
            if coh:
                avg = (sum(coh)/len(coh) + sum(nat)/len(nat) + sum(eff)/len(eff)) / 3
                logger.info(f"  SCORES: coh={sum(coh)/len(coh):.1f} nat={sum(nat)/len(nat):.1f} eff={sum(eff)/len(eff):.1f} avg={avg:.1f}")

    # Save all results
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n=== SAMPLING PARAMETER SWEEP RESULTS ===")
    print(f"{'config':30s}  {'model':12s}  {'coh':>5s}  {'nat':>5s}  {'eff':>5s}  {'avg':>5s}")
    print("-" * 72)
    for key, data in sorted(all_results.items()):
        reviews = data["reviews"]
        coh = [r["coherence"] for r in reviews if r.get("coherence") is not None]
        nat = [r["naturalness"] for r in reviews if r.get("naturalness") is not None]
        eff = [r["effectiveness"] for r in reviews if r.get("effectiveness") is not None]
        if coh:
            mc, mn, me = sum(coh)/len(coh), sum(nat)/len(nat), sum(eff)/len(eff)
            print(f"{data['sampling']:30s}  {data['model']:12s}  {mc:5.1f}  {mn:5.1f}  {me:5.1f}  {(mc+mn+me)/3:5.1f}")


if __name__ == "__main__":
    main()
