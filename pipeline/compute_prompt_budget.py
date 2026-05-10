#!/usr/bin/env python3
"""Compute the prompt_budget_frames value for a dataset.

Scans all stereo JSON files and computes the maximum system prompt frame
count (text tokens + silence padding).  This value should be set as
system_prompt.prompt_budget_frames in the training config so that audio
chunks tile without gaps and no context injections are lost.

Usage:
    python pipeline/compute_prompt_budget.py \
        --stereo-dir data/<dataset>/stereo_wav \
        --tokenizer-path /path/to/tokenizer_spm_32k_3.model \
        [--audio-silence-frames 6]
"""

import argparse
import json
import glob
import sentencepiece


def main():
    parser = argparse.ArgumentParser(
        description="Compute prompt_budget_frames for a dataset"
    )
    parser.add_argument("--stereo-dir", required=True, help="Directory with stereo JSON files")
    parser.add_argument("--tokenizer-path", required=True, help="Path to sentencepiece tokenizer model")
    parser.add_argument("--audio-silence-frames", type=int, default=6, help="Silence frames between prompt sections (default: 6)")
    args = parser.parse_args()

    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)

    sil = args.audio_silence_frames
    files = sorted(glob.glob(f"{args.stereo_dir}/*.json"))

    if not files:
        print(f"No JSON files found in {args.stereo_dir}")
        return

    max_frames = 0
    total = 0
    frame_sum = 0

    for f in files:
        d = json.load(open(f))
        tp = d.get("text_prompt", "")
        if not tp:
            continue
        total += 1
        wrapped = f"<system> {tp} </system>"
        tokens = sp.Encode(wrapped)
        # Layout: [voice_frames] [silence] [text_frames] [silence]
        # No voice prompt: voice_frames = 0
        prompt_frames = 0 + sil + len(tokens) + sil
        frame_sum += prompt_frames
        if prompt_frames > max_frames:
            max_frames = prompt_frames

    if total == 0:
        print("No samples with text_prompt found.")
        return

    avg_frames = frame_sum / total
    print(f"Scanned {total} samples with text_prompt")
    print(f"  min prompt frames: (not tracked, use max)")
    print(f"  max prompt frames: {max_frames} ({max_frames / 12.5:.1f}s at 12.5Hz)")
    print(f"  avg prompt frames: {avg_frames:.0f} ({avg_frames / 12.5:.1f}s at 12.5Hz)")
    print()
    print(f"Recommended setting:")
    print(f"  system_prompt:")
    print(f"    prompt_budget_frames: {max_frames}")


if __name__ == "__main__":
    main()
