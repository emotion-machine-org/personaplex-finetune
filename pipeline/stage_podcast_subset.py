#!/usr/bin/env python3
"""Stage curated podcast stereo data for companion training.

The inspected podcast disk already contains stereo WAVs and JSON sidecars for
the broad podcast corpus.  This script stages only ids selected by
``podcast_filtered_index.json``, symlinks their WAV files, and writes normalized
JSON sidecars that use the same speaker labels as the synthetic companion data.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


LABEL_MAP = {
    "SPEAKER_AGENT": "SPEAKER_BROKER",
    "SPEAKER_HUMAN": "SPEAKER_CLIENT",
    "SPEAKER_BROKER": "SPEAKER_BROKER",
    "SPEAKER_CLIENT": "SPEAKER_CLIENT",
}


def load_filtered_index(path: Path) -> dict[str, dict]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object keyed by podcast id in {path}")
    return data


def normalize_label(label: str) -> str:
    try:
        return LABEL_MAP[label]
    except KeyError as exc:
        raise ValueError(f"Unsupported speaker label: {label}") from exc


def normalize_sidecar(sidecar: dict, filtered_record: dict) -> dict:
    out = dict(sidecar)
    out["text_prompt"] = filtered_record.get("system_prompt") or sidecar.get("text_prompt", "")
    out["source"] = "podcast_filtered"
    out["assistant_name"] = filtered_record.get("assistant_name")
    out["podcast_classification"] = filtered_record.get("classification")
    out["source_metadata"] = filtered_record.get("source_metadata")

    alignments = []
    for item in sidecar.get("alignments", []):
        if len(item) != 3:
            raise ValueError(f"Bad alignment item: {item!r}")
        word, ts, speaker = item
        alignments.append([word, ts, normalize_label(speaker)])
    out["alignments"] = alignments

    turns = []
    for turn in sidecar.get("turns", []):
        next_turn = dict(turn)
        if "speaker" in next_turn:
            next_turn["speaker"] = normalize_label(next_turn["speaker"])
        turns.append(next_turn)
    out["turns"] = turns

    return out


def link_or_copy_wav(src: Path, dst: Path, copy_wav: bool, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()

    if copy_wav:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage curated podcast subset for companion training")
    parser.add_argument("--filtered-index", type=Path, required=True, help="podcast_filtered_index.json")
    parser.add_argument("--source-stereo-dir", type=Path, required=True, help="Existing podcast stereo_wav dir")
    parser.add_argument("--output-stereo-dir", type=Path, required=True, help="Staged stereo_wav dir")
    parser.add_argument("--limit", type=int, default=None, help="Stage only the first N ids")
    parser.add_argument("--copy-wav", action="store_true", help="Copy WAV files instead of symlinking")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing staged files")
    args = parser.parse_args()

    filtered = load_filtered_index(args.filtered_index)
    args.output_stereo_dir.mkdir(parents=True, exist_ok=True)

    staged = 0
    missing_wav: list[str] = []
    missing_json: list[str] = []
    failed: list[str] = []

    for pod_id in sorted(filtered):
        if args.limit is not None and staged >= args.limit:
            break

        wav_src = args.source_stereo_dir / f"{pod_id}.wav"
        json_src = args.source_stereo_dir / f"{pod_id}.json"
        wav_dst = args.output_stereo_dir / f"{pod_id}.wav"
        json_dst = args.output_stereo_dir / f"{pod_id}.json"

        if not wav_src.exists():
            missing_wav.append(pod_id)
            continue
        if not json_src.exists():
            missing_json.append(pod_id)
            continue

        try:
            with json_src.open() as f:
                sidecar = json.load(f)
            normalized = normalize_sidecar(sidecar, filtered[pod_id])
            if not normalized.get("text_prompt"):
                raise ValueError("missing normalized text_prompt")

            link_or_copy_wav(wav_src, wav_dst, args.copy_wav, args.overwrite)
            if json_dst.exists() and not args.overwrite:
                pass
            else:
                with json_dst.open("w") as f:
                    json.dump(normalized, f, ensure_ascii=False)
                    f.write("\n")
            staged += 1
        except Exception as exc:  # noqa: BLE001 - report all staging failures
            failed.append(f"{pod_id}: {exc}")

    print(f"Filtered ids: {len(filtered)}")
    print(f"Staged: {staged}")
    print(f"Missing WAV: {len(missing_wav)}")
    print(f"Missing JSON: {len(missing_json)}")
    print(f"Failed: {len(failed)}")

    for label, items in [
        ("missing_wav", missing_wav),
        ("missing_json", missing_json),
        ("failed", failed),
    ]:
        if items:
            print(f"{label}:")
            for item in items[:20]:
                print(f"  {item}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

