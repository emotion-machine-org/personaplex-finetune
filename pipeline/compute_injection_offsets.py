#!/usr/bin/env python3
"""Convert context_injections from after_turn to frame_offset using turn/alignment data.

Reads stereo JSON files (preferably with explicit turn metadata from
create_stereo.py) and converts turn-based injection placement to frame offsets
at 12.5 Hz.

Placement strategy (determined automatically from transcript structure):
  - Proactive (~70%): turn before marker is CLIENT → frame_offset = start of that client utterance
  - Reactive (~30%): turn before marker is BROKER hedge → frame_offset = end of that broker utterance

Usage:
    python pipeline/compute_injection_offsets.py --stereo-dir data/stereo_wav
    python pipeline/compute_injection_offsets.py --stereo-dir data/stereo_wav --frame-rate 12.5
"""

import argparse
import json
import logging
import math
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FRAME_RATE = 12.5  # Mimi codec frame rate


def normalize_turns(turns: list[dict]) -> list[dict]:
    """Normalize persisted turn metadata into the shape used by this script."""
    normalized = []
    for i, turn in enumerate(turns):
        speaker = turn.get("speaker")
        start = turn.get("start")
        end = turn.get("end")
        if speaker is None or start is None or end is None:
            continue
        normalized.append({
            "speaker": speaker,
            "start": float(start),
            "end": float(end),
            "text": turn.get("text", ""),
            "index": int(turn.get("index", i)),
        })
    return normalized


def reconstruct_turns(alignments: list) -> list[dict]:
    """Reconstruct speaker turns from word-level alignments.

    Groups consecutive same-speaker words into turns.
    This is a fallback for older JSON files that do not preserve raw turn order.
    Returns list of {"speaker": str, "start": float, "end": float, "words": list}.
    """
    if not alignments:
        return []

    turns = []
    current_speaker = alignments[0][2]
    current_start = alignments[0][1][0]
    current_end = alignments[0][1][1]
    current_words = [alignments[0][0]]

    for word, (start, end), speaker in alignments[1:]:
        if speaker == current_speaker:
            current_end = end
            current_words.append(word)
        else:
            turns.append({
                "speaker": current_speaker,
                "start": current_start,
                "end": current_end,
                "words": current_words,
            })
            current_speaker = speaker
            current_start = start
            current_end = end
            current_words = [word]

    turns.append({
        "speaker": current_speaker,
        "start": current_start,
        "end": current_end,
        "words": current_words,
    })
    return turns


def compute_frame_offset(
    after_turn: int,
    turns: list[dict],
    frame_rate: float = FRAME_RATE,
) -> int | None:
    """Compute frame_offset for an injection placed after the given turn index.

    Proactive (turn is CLIENT): frame_offset = start of that client turn.
    Reactive (turn is BROKER): frame_offset = end of that broker turn.
    """
    if after_turn < 0 or after_turn >= len(turns):
        return None

    turn = turns[after_turn]

    if turn["speaker"] == "SPEAKER_CLIENT":
        # Proactive: inject at START of client utterance (context loads while client speaks)
        return int(turn["start"] * frame_rate)
    else:
        # Reactive: inject right AFTER broker hedge (context loads during brief pause)
        return math.ceil(turn["end"] * frame_rate)


def process_json(json_path: Path, frame_rate: float) -> tuple[bool, str]:
    """Process a single stereo JSON file, adding frame_offset to context_injections.

    Returns (changed, message).
    """
    with open(json_path) as f:
        data = json.load(f)

    injections = data.get("context_injections")
    if not injections:
        return False, "no context_injections"

    alignments = data.get("alignments", [])
    if not alignments:
        return False, "no alignments"

    turns = normalize_turns(data.get("turns", []))
    if not turns:
        turns = reconstruct_turns(alignments)
    if not turns:
        return False, "no turns reconstructed"

    changed = False
    for inj in injections:
        after_turn = inj.get("after_turn")
        if after_turn is None:
            continue

        fo = compute_frame_offset(after_turn, turns, frame_rate)
        if fo is None:
            logger.warning(
                "%s: after_turn=%d out of range (have %d turns), skipping",
                json_path.stem, after_turn, len(turns),
            )
            continue

        inj["frame_offset"] = fo
        changed = True

    if changed:
        with open(json_path, "w") as f:
            json.dump(data, f, ensure_ascii=False)

    return changed, f"{sum(1 for i in injections if 'frame_offset' in i)}/{len(injections)} offsets computed"


def main():
    parser = argparse.ArgumentParser(
        description="Convert context_injections from after_turn to frame_offset"
    )
    parser.add_argument("--stereo-dir", type=Path, required=True,
                        help="Directory containing stereo JSON files")
    parser.add_argument("--frame-rate", type=float, default=FRAME_RATE,
                        help=f"Audio frame rate in Hz (default: {FRAME_RATE})")
    args = parser.parse_args()

    json_files = sorted(args.stereo_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", args.stereo_dir)
        return

    logger.info("Processing %d JSON files in %s", len(json_files), args.stereo_dir)

    processed = 0
    skipped = 0
    for jf in json_files:
        changed, msg = process_json(jf, args.frame_rate)
        if changed:
            processed += 1
            logger.info("%s: %s", jf.stem, msg)
        else:
            skipped += 1

    logger.info("Done: %d updated, %d skipped (no injections or no alignments)", processed, skipped)


if __name__ == "__main__":
    main()
