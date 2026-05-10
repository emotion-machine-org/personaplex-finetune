"""Merge LoRA adapter weights into the base PersonaPlex model and save a full checkpoint.

Usage:
    python merge_lora.py \
        --checkpoint runs/pharma_demo_ft-lora_r128_v2/checkpoints/checkpoint_003000/consolidated \
        --output runs/pharma_demo_ft-lora_r128_v2/merged/model.safetensors \
        --hf-repo nvidia/personaplex-7b-v1

Then run inference with the merged model:
    cd personaplex/moshi && python -m moshi.offline \
        --moshi-weight ../../runs/pharma_demo_ft-lora_r128_v2/merged/model.safetensors \
        --input-wav input.wav --output-wav output.wav --output-text output.json \
        --voice-prompt NATM1.pt --text-prompt "Your system prompt here"
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import safetensors.torch
from huggingface_hub import hf_hub_download


def merge(
    ckpt_dir: Path,
    output_path: Path,
    hf_repo: str = "nvidia/personaplex-7b-v1",
    base_weights: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Path:
    """Merge LoRA adapter into base model weights and save.

    Args:
        ckpt_dir: Path to checkpoint consolidated dir (containing lora.safetensors + config.json)
        output_path: Path to write merged model.safetensors
        hf_repo: HuggingFace repo for base model weights
        base_weights: Path to local base model.safetensors (skips HF download)
        dtype: Data type for merged weights

    Returns:
        output_path
    """
    ckpt_dir = Path(ckpt_dir)
    output_path = Path(output_path)

    lora_path = ckpt_dir / "lora.safetensors"
    config_path = ckpt_dir / "config.json"

    if not lora_path.exists():
        raise FileNotFoundError(f"{lora_path} not found")
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")

    # Read LoRA config
    with open(config_path) as f:
        config = json.load(f)
    lora_rank = config.get("lora_rank", 128)
    lora_scaling = config.get("lora_scaling", 2.0)
    print(f"LoRA config: rank={lora_rank}, scaling={lora_scaling}")

    # Load base model weights
    if base_weights:
        base_path = base_weights
    else:
        print(f"Downloading base model from {hf_repo}...")
        base_path = hf_hub_download(hf_repo, "model.safetensors")
    print(f"Loading base weights from {base_path}")
    base_state = safetensors.torch.load_file(base_path, device="cpu")

    # Load LoRA weights
    print(f"Loading LoRA weights from {lora_path}")
    lora_state = safetensors.torch.load_file(str(lora_path), device="cpu")

    # Classify LoRA keys into two categories:
    # 1. Per-step attention projections: *.self_attn.{in,out}_projs.{i}.lora_{A,B}.weight
    #    These need to be reassembled into monolithic weights before merging.
    # 2. Regular LoRA layers: *.lora_{A,B}.weight / *.frozen_W.weight
    #    These map directly to base keys via parent + ".weight"

    # Pattern for per-step attention projections
    attn_proj_pattern = re.compile(
        r'^(.*\.self_attn)\.(in_projs|out_projs)\.(\d+)\.(lora_A\.weight|lora_B\.weight|frozen_W\.weight)$'
    )

    # Group LoRA weights
    # attn_groups: {(prefix, proj_type): {step_idx: {part: tensor}}}
    attn_groups = defaultdict(lambda: defaultdict(dict))
    # regular_groups: {parent: {part: tensor}}
    regular_groups = defaultdict(dict)

    for key, tensor in lora_state.items():
        m = attn_proj_pattern.match(key)
        if m:
            attn_prefix = m.group(1)       # e.g. "transformer.layers.0.self_attn"
            proj_type = m.group(2)          # "in_projs" or "out_projs"
            step_idx = int(m.group(3))      # 0..15
            part = m.group(4)              # "lora_A.weight" etc.
            attn_groups[(attn_prefix, proj_type)][step_idx][part] = tensor
        else:
            # Regular LoRA key
            for suffix in [".frozen_W.weight", ".lora_A.weight", ".lora_B.weight"]:
                if key.endswith(suffix):
                    parent = key[: -len(suffix)]
                    short = suffix.lstrip(".")
                    regular_groups[parent][short] = tensor
                    break

    merged_count = 0

    # --- Merge regular (non-attention-proj) LoRA layers ---
    for parent, tensors in regular_groups.items():
        if "lora_A.weight" not in tensors or "lora_B.weight" not in tensors:
            print(f"WARNING: Incomplete LoRA group for {parent}, skipping")
            continue

        A = tensors["lora_A.weight"].to(dtype=torch.float32)
        B = tensors["lora_B.weight"].to(dtype=torch.float32)
        delta = (B @ A) * lora_scaling

        base_key = parent + ".weight"
        if base_key in base_state:
            base_w = base_state[base_key].to(dtype=torch.float32)
            base_state[base_key] = (base_w + delta).to(dtype=dtype)
            merged_count += 1
        elif "frozen_W.weight" in tensors:
            base_w = tensors["frozen_W.weight"].to(dtype=torch.float32)
            base_state[base_key] = (base_w + delta).to(dtype=dtype)
            merged_count += 1
        else:
            print(f"WARNING: No base weight found for {base_key}, skipping")

    # --- Merge per-step attention projection LoRA layers ---
    # During training, StreamingMultiheadAttention._load_hook splits monolithic weights:
    #   in_proj_weight  [mult * qkv_dim, embed_dim] → in_projs.{i}.weight  [qkv_dim, embed_dim]
    #   out_proj.weight [mult * embed_dim, embed_dim] → out_projs.{i}.weight [embed_dim, embed_dim]
    # We reverse this: compute per-step delta, concatenate, add to monolithic base weight.

    for (attn_prefix, proj_type), steps in attn_groups.items():
        num_steps = max(steps.keys()) + 1

        # Compute per-step deltas
        per_step_deltas = []
        all_ok = True
        for i in range(num_steps):
            if i not in steps:
                print(f"WARNING: Missing step {i} for {attn_prefix}.{proj_type}, skipping group")
                all_ok = False
                break
            s = steps[i]
            if "lora_A.weight" not in s or "lora_B.weight" not in s:
                print(f"WARNING: Incomplete LoRA at {attn_prefix}.{proj_type}.{i}, skipping group")
                all_ok = False
                break
            A = s["lora_A.weight"].to(dtype=torch.float32)
            B = s["lora_B.weight"].to(dtype=torch.float32)
            per_step_deltas.append((B @ A) * lora_scaling)

        if not all_ok:
            continue

        # Concatenate deltas along dim 0 to match monolithic layout
        full_delta = torch.cat(per_step_deltas, dim=0)

        # Map to base key name
        if proj_type == "in_projs":
            base_key = f"{attn_prefix}.in_proj_weight"
        else:  # out_projs
            base_key = f"{attn_prefix}.out_proj.weight"

        if base_key in base_state:
            base_w = base_state[base_key].to(dtype=torch.float32)
            if base_w.shape != full_delta.shape:
                print(f"WARNING: Shape mismatch for {base_key}: "
                      f"base={base_w.shape}, delta={full_delta.shape}, skipping")
                continue
            base_state[base_key] = (base_w + full_delta).to(dtype=dtype)
            merged_count += 1
        else:
            print(f"WARNING: No base weight found for {base_key}, skipping")

    print(f"Merged {merged_count} LoRA layers/groups into base weights")

    # Verify no LoRA keys were missed
    total_lora_groups = len(regular_groups) + len(attn_groups)
    if merged_count < total_lora_groups:
        print(f"WARNING: {total_lora_groups - merged_count} groups were not merged")

    # Cast all weights to target dtype
    for key in base_state:
        if base_state[key].is_floating_point():
            base_state[key] = base_state[key].to(dtype=dtype)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_path}")
    safetensors.torch.save_file(base_state, str(output_path))
    print(f"Done! Merged model size: {output_path.stat().st_size / 1e9:.2f} GB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base PersonaPlex model")
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Path to checkpoint consolidated dir (containing lora.safetensors + config.json)"
    )
    parser.add_argument(
        "--output", required=True, type=str,
        help="Path to write merged model.safetensors"
    )
    parser.add_argument(
        "--hf-repo", type=str, default="nvidia/personaplex-7b-v1",
        help="HuggingFace repo for base model weights"
    )
    parser.add_argument(
        "--base-weights", type=str, default=None,
        help="Path to local base model.safetensors (skips HF download)"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"],
        help="Data type for merged weights"
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    if not (ckpt_dir / "lora.safetensors").exists():
        print(f"ERROR: {ckpt_dir / 'lora.safetensors'} not found")
        sys.exit(1)
    if not (ckpt_dir / "config.json").exists():
        print(f"ERROR: {ckpt_dir / 'config.json'} not found")
        sys.exit(1)

    merge(
        ckpt_dir=ckpt_dir,
        output_path=Path(args.output),
        hf_repo=args.hf_repo,
        base_weights=args.base_weights,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
