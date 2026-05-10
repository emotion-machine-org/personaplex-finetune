#!/usr/bin/env python3
"""Podcast pipeline step 1: Convert source podcast audio to 24kHz mono WAV.

Reads the podcast transcript JSONL + audio mapping to locate source audio files,
then converts each to 24kHz mono WAV for downstream stereo creation.

Usage:
    python pipeline/podcast_to_mono.py \
        --transcripts /path/to/podcast_conversations_all.jsonl \
        --audio-mapping /path/to/podcast_audio_mapping.json \
        --audio-dir /path/to/podcast_audio \
        --output-dir data/podcast/mono_wav \
        --workers 8
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 24000


def convert_one(
    src_path: Path, dst_path: Path, target_sr: int = TARGET_SR
) -> tuple[str, bool, str]:
    """Convert a single audio file to mono WAV at target sample rate via ffmpeg."""
    pod_id = dst_path.stem
    if dst_path.exists() and dst_path.stat().st_size > 1000:
        return pod_id, True, "skipped (exists)"

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(src_path),
            "-ac", "1",                    # mono
            "-ar", str(target_sr),         # 24kHz
            "-sample_fmt", "s16",          # 16-bit PCM
            "-f", "wav",
            str(dst_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            return pod_id, False, f"ffmpeg error: {result.stderr[-200:]}"

        # Verify output
        if not dst_path.exists() or dst_path.stat().st_size < 1000:
            return pod_id, False, "output file too small or missing"

        return pod_id, True, "ok"
    except subprocess.TimeoutExpired:
        return pod_id, False, "ffmpeg timeout (>300s)"
    except Exception as e:
        return pod_id, False, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Convert podcast source audio to 24kHz mono WAV"
    )
    parser.add_argument(
        "--transcripts", type=Path, required=True,
        help="Path to podcast_conversations_all.jsonl"
    )
    parser.add_argument(
        "--audio-mapping", type=Path, required=True,
        help="Path to podcast_audio_mapping.json (pod_id -> audio filename)"
    )
    parser.add_argument(
        "--audio-dir", type=Path, required=True,
        help="Directory containing source podcast audio files"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for 24kHz mono WAVs"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip files that already exist in output dir"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process first N records (for testing)"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load mapping
    with open(args.audio_mapping) as f:
        mapping = json.load(f)
    logger.info("Loaded %d audio mappings", len(mapping))

    # Load transcript IDs
    pod_ids = []
    with open(args.transcripts) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            pod_ids.append(rec["id"])

    if args.limit:
        pod_ids = pod_ids[:args.limit]

    # Build work items
    work = []
    skipped_no_mapping = 0
    skipped_no_audio = 0
    for pod_id in pod_ids:
        audio_fname = mapping.get(pod_id)
        if not audio_fname:
            skipped_no_mapping += 1
            continue

        src_path = args.audio_dir / audio_fname
        if not src_path.exists():
            skipped_no_audio += 1
            continue

        dst_path = args.output_dir / f"{pod_id}.wav"
        if args.resume and dst_path.exists() and dst_path.stat().st_size > 1000:
            continue

        work.append((src_path, dst_path))

    logger.info(
        "Work: %d to convert, %d no mapping, %d no audio file",
        len(work), skipped_no_mapping, skipped_no_audio,
    )

    if not work:
        logger.info("Nothing to do.")
        return

    # Process in parallel
    success = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(convert_one, src, dst): dst.stem
            for src, dst in work
        }
        for i, future in enumerate(as_completed(futures), 1):
            pod_id, ok, msg = future.result()
            if ok:
                success += 1
            else:
                failed += 1
                logger.warning("%s: FAILED - %s", pod_id, msg)

            if i % 50 == 0:
                logger.info("Progress: %d/%d (success=%d, failed=%d)", i, len(work), success, failed)

    logger.info("Done: %d success, %d failed out of %d", success, failed, len(work))


if __name__ == "__main__":
    main()
