"""Step 5: Create train/eval manifest JSONL files for moshi-finetune.

Each line: {"path": "/absolute/path/to/stereo.wav", "duration": 123.45}
Companion .json alignment files must exist alongside each .wav.
"""

import argparse
import json
import random
from pathlib import Path

import soundfile as sf


def get_duration(wav_path: Path) -> float | None:
    """Get audio duration in seconds using soundfile."""
    try:
        info = sf.info(str(wav_path))
        return info.duration
    except Exception as e:
        print(f"  SKIP {wav_path.name}: {e}")
        return None


def parse_csv_args(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        items.extend(part.strip() for part in value.split(",") if part.strip())
    return items


def stem_allowed(stem: str, include_prefixes: list[str], exclude_prefixes: list[str]) -> bool:
    if include_prefixes and not any(stem.startswith(prefix) for prefix in include_prefixes):
        return False
    if exclude_prefixes and any(stem.startswith(prefix) for prefix in exclude_prefixes):
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Create train/eval manifests for moshi-finetune")
    parser.add_argument("--stereo-dir", type=Path, default=Path("data/stereo_wav"))
    parser.add_argument("--train-output", type=Path, default=Path("data/dataset/train.jsonl"))
    parser.add_argument("--eval-output", type=Path, default=Path("data/dataset/eval.jsonl"))
    parser.add_argument("--eval-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resolve-symlinks",
        action="store_true",
        help=(
            "Resolve WAV symlinks in manifest paths. By default symlink paths are "
            "preserved so staged sidecar JSON files next to symlinked WAVs are used."
        ),
    )
    parser.add_argument(
        "--include-prefix",
        action="append",
        default=[],
        help="Only include WAV stems starting with this prefix. Repeat or comma-separate values.",
    )
    parser.add_argument(
        "--exclude-prefix",
        action="append",
        default=[],
        help="Exclude WAV stems starting with this prefix. Repeat or comma-separate values.",
    )
    args = parser.parse_args()
    include_prefixes = parse_csv_args(args.include_prefix)
    exclude_prefixes = parse_csv_args(args.exclude_prefix)

    # Find all stereo WAVs that have companion alignment JSONs
    wav_files = [
        wav_path
        for wav_path in sorted(args.stereo_dir.glob("*.wav"))
        if stem_allowed(wav_path.stem, include_prefixes, exclude_prefixes)
    ]
    print(f"Found {len(wav_files)} WAV files in {args.stereo_dir}")
    if include_prefixes:
        print(f"  include prefixes: {', '.join(include_prefixes)}")
    if exclude_prefixes:
        print(f"  exclude prefixes: {', '.join(exclude_prefixes)}")

    records = []
    for wav_path in wav_files:
        json_path = wav_path.with_suffix(".json")
        if not json_path.exists():
            print(f"  SKIP {wav_path.name}: no companion .json alignment file")
            continue

        duration = get_duration(wav_path)
        if duration is None or duration <= 0:
            continue

        manifest_path = wav_path.resolve() if args.resolve_symlinks else wav_path.absolute()
        records.append({
            "path": str(manifest_path),
            "duration": duration,
        })

    print(f"Valid records: {len(records)}")
    if not records:
        print("No valid records found. Exiting.")
        return

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(records)

    n_eval = max(1, int(len(records) * args.eval_fraction))
    # With very few files, put at least 1 in eval and rest in train
    if len(records) <= 2:
        n_eval = 0  # too few to split, put all in train
        print("Too few files for eval split — all going to train")

    eval_records = records[:n_eval]
    train_records = records[n_eval:]

    # Write manifests
    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.eval_output.parent.mkdir(parents=True, exist_ok=True)

    for path, recs in [(args.train_output, train_records), (args.eval_output, eval_records)]:
        with open(path, "w") as f:
            for rec in recs:
                json.dump(rec, f)
                f.write("\n")
        print(f"Wrote {len(recs)} records to {path}")

    # Summary
    total_dur = sum(r["duration"] for r in records)
    train_dur = sum(r["duration"] for r in train_records)
    eval_dur = sum(r["duration"] for r in eval_records)
    print(f"\nTotal audio: {total_dur:.1f}s ({total_dur/3600:.2f}h)")
    print(f"  Train: {train_dur:.1f}s ({len(train_records)} files)")
    print(f"  Eval:  {eval_dur:.1f}s ({len(eval_records)} files)")


if __name__ == "__main__":
    main()
