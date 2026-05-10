#!/usr/bin/env python3
"""Prepare companion v6 JSONL for the audio/training pipeline.

Raw v6 records store the PersonaPlex prompt as ``system_prompt``.  The
downstream stereo/alignment step propagates only ``text_prompt`` into sidecar
JSON files, so this script copies ``system_prompt`` into ``text_prompt`` and
performs basic dialogue validation before audio generation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

try:
    from parse_dialogues import parse_dialogue_to_vibevoice
except ImportError:
    from pipeline.parse_dialogues import parse_dialogue_to_vibevoice


SPEAKER_RE = re.compile(r"^(USER|[A-Z][A-Z0-9_ -]{0,30}):\s+")


def iter_jsonl(path: Path):
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            yield line_no, json.loads(line)


def count_script_words(dialogue: str) -> int:
    script = parse_dialogue_to_vibevoice(dialogue)
    if script is None:
        return 0
    return len(script.split())


def count_parseable_turns(dialogue: str) -> tuple[int, list[str]]:
    n_turns = 0
    bad_lines: list[str] = []
    for raw_line in dialogue.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if SPEAKER_RE.match(line):
            n_turns += 1
        else:
            bad_lines.append(line)
    return n_turns, bad_lines


def prepare_record(record: dict) -> dict:
    out = dict(record)
    if not out.get("text_prompt"):
        out["text_prompt"] = out.get("system_prompt", "")
    out["source"] = out.get("source", "companion_v6")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare companion v6 JSONL for training data generation")
    parser.add_argument("--input", type=Path, required=True, help="Raw companion_training_v6.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Prepared JSONL output")
    parser.add_argument(
        "--max-words",
        type=int,
        default=3000,
        help="Report records over this VibeVoice script word count; generation may skip them",
    )
    parser.add_argument(
        "--drop-over-max-words",
        action="store_true",
        help="Drop records with dialogue word count greater than --max-words",
    )
    parser.add_argument(
        "--drop-invalid",
        action="store_true",
        help="Drop invalid records and exit successfully after reporting them",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    over_max: list[tuple[str, int]] = []
    bad: list[str] = []

    with args.output.open("w") as out_f:
        for line_no, record in iter_jsonl(args.input):
            counts["input"] += 1
            rec_id = record.get("id", f"line-{line_no}")
            dialogue = record.get("dialogue") or ""
            text_prompt = record.get("system_prompt") or record.get("text_prompt") or ""

            if not dialogue:
                bad.append(f"{rec_id}: missing dialogue")
                continue
            if not text_prompt:
                bad.append(f"{rec_id}: missing system/text prompt")
                continue

            n_turns, bad_lines = count_parseable_turns(dialogue)
            if n_turns == 0 or bad_lines:
                preview = "; ".join(line[:80] for line in bad_lines[:3])
                bad.append(f"{rec_id}: bad dialogue lines: {preview}")
                continue

            n_words = count_script_words(dialogue)
            if n_words == 0:
                bad.append(f"{rec_id}: dialogue did not convert to VibeVoice script")
                continue
            if n_words > args.max_words:
                over_max.append((rec_id, n_words))
                if args.drop_over_max_words:
                    counts["dropped_over_max"] += 1
                    continue

            prepared = prepare_record(record)
            json.dump(prepared, out_f, ensure_ascii=False)
            out_f.write("\n")
            counts["output"] += 1

    print(f"Input records: {counts['input']}")
    print(f"Output records: {counts['output']}")
    if args.drop_over_max_words:
        print(f"Dropped over max words: {counts['dropped_over_max']}")
    print(f"Records over {args.max_words} VibeVoice script words: {len(over_max)}")
    if over_max:
        heaviest = sorted(over_max, key=lambda x: x[1], reverse=True)[:10]
        for rec_id, n_words in heaviest:
            print(f"  over_max {rec_id}: {n_words} words")

    if bad:
        print(f"Invalid records: {len(bad)}")
        for item in bad[:20]:
            print(f"  {item}")
        if not args.drop_invalid:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
