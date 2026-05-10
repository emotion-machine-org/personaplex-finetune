#!/usr/bin/env python3
"""Podcast pipeline step 2: Create stereo WAVs from mono podcast audio using
pre-existing word-level timestamps from WhisperX transcription.

Unlike the synthetic pipeline's create_stereo.py (which re-transcribes mono
TTS output with WhisperX), this script uses the word-level timestamps that
already exist in the podcast transcript JSONL. This is both faster and more
accurate since the timestamps were produced during the original diarization.

For each podcast conversation:
1. Load 24kHz mono WAV (from podcast_to_mono.py)
2. Read word-level timestamps + speaker labels from transcript JSONL
3. Build speaker segments and route to stereo channels
   (left = speaker_1, right = speaker_2)
4. Write stereo WAV + alignment JSON in voice-training format

Output format matches create_stereo.py exactly:
  - stereo_wav/{pod_id}.wav  (24kHz, 2ch)
  - stereo_wav/{pod_id}.json (alignments + turns + text_prompt)

Usage:
    python pipeline/podcast_create_stereo.py \
        --transcripts /path/to/podcast_conversations_all.jsonl \
        --mono-dir data/podcast/mono_wav \
        --output-dir data/podcast/stereo_wav \
        --workers 8

    # With duration limits for training:
    python pipeline/podcast_create_stereo.py \
        --transcripts /path/to/podcast_conversations_all.jsonl \
        --mono-dir data/podcast/mono_wav \
        --output-dir data/podcast/stereo_wav \
        --max-duration 600 \
        --workers 8
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FADE_MS = 10          # crossfade at segment boundaries (ms)
SPEAKER_LEFT = "speaker_1"    # left channel
SPEAKER_RIGHT = "speaker_2"   # right channel

# Labels for alignment JSON (matches voice-training convention)
LABEL_LEFT = "SPEAKER_AGENT"
LABEL_RIGHT = "SPEAKER_HUMAN"


def load_transcripts(path: Path) -> dict[str, dict]:
    """Load podcast transcripts JSONL into a dict keyed by pod_id."""
    lookup = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            lookup[rec["id"]] = rec
    return lookup


def build_word_list(messages: list[dict]) -> list[dict]:
    """Extract flat word list with timestamps and speaker labels from messages.

    Each message has:
        role: "speaker_1" | "speaker_2"
        words: [{word, speaker, start, end}, ...]
    """
    words = []
    for msg_idx, msg in enumerate(messages):
        role = msg["role"]
        for w in msg.get("words", []):
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None:
                continue
            words.append({
                "word": w["word"],
                "start": float(start),
                "end": float(end),
                "role": role,
                "msg_index": msg_idx,
            })
    return words


def build_speaker_segments(
    words: list[dict], audio_duration: float
) -> list[dict]:
    """Group consecutive same-speaker words into segments, filling gaps."""
    if not words:
        return []

    segments = []
    current = {
        "role": words[0]["role"],
        "start": words[0]["start"],
        "end": words[0]["end"],
    }

    for w in words[1:]:
        if w["role"] == current["role"]:
            current["end"] = max(current["end"], w["end"])
        else:
            segments.append(current)
            current = {
                "role": w["role"],
                "start": w["start"],
                "end": w["end"],
            }
    segments.append(current)

    # Fill gaps between segments using midpoints
    for i in range(len(segments) - 1):
        gap_start = segments[i]["end"]
        gap_end = segments[i + 1]["start"]
        if gap_end > gap_start:
            mid = (gap_start + gap_end) / 2
            segments[i]["end"] = mid
            segments[i + 1]["start"] = mid

    # Extend to audio boundaries
    segments[0]["start"] = 0.0
    segments[-1]["end"] = audio_duration

    return segments


def build_turn_metadata(messages: list[dict]) -> list[dict]:
    """Build turn-level metadata from the message array."""
    turns = []
    for i, msg in enumerate(messages):
        role = msg["role"]
        label = LABEL_LEFT if role == SPEAKER_LEFT else LABEL_RIGHT
        start = msg.get("start")
        end = msg.get("end")

        # Fallback: derive from word timestamps
        if (start is None or end is None) and msg.get("words"):
            word_starts = [w["start"] for w in msg["words"] if "start" in w]
            word_ends = [w["end"] for w in msg["words"] if "end" in w]
            if word_starts:
                start = min(word_starts)
            if word_ends:
                end = max(word_ends)

        turns.append({
            "index": i,
            "speaker": label,
            "start": float(start) if start is not None else 0.0,
            "end": float(end) if end is not None else 0.0,
            "text": msg["content"],
        })
    return turns


def create_stereo_wav(
    mono_path: Path,
    segments: list[dict],
    output_path: Path,
    sr: int,
    max_duration: Optional[float] = None,
) -> tuple[bool, float]:
    """Create stereo WAV with speaker_1 on left, speaker_2 on right.

    Returns (success, actual_duration).
    """
    import soundfile as sf

    mono, file_sr = sf.read(mono_path)
    if file_sr != sr:
        logger.warning(
            "%s: expected %dHz, got %dHz — proceeding anyway",
            mono_path.name, sr, file_sr,
        )
        sr = file_sr

    n_samples = len(mono)
    actual_duration = n_samples / sr

    # Truncate if max_duration is set
    if max_duration and actual_duration > max_duration:
        n_samples = int(max_duration * sr)
        mono = mono[:n_samples]
        actual_duration = max_duration

    left = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)
    fade_samples = int(FADE_MS / 1000 * sr)

    for seg in segments:
        s = max(0, int(seg["start"] * sr))
        e = min(n_samples, int(seg["end"] * sr))
        if e <= s:
            continue

        chunk = mono[s:e].copy().astype(np.float32)

        # Apply fade in/out to avoid clicks
        fade_len = min(fade_samples, len(chunk) // 2)
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
            fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
            chunk[:fade_len] *= fade_in
            chunk[-fade_len:] *= fade_out

        if seg["role"] == SPEAKER_LEFT:
            left[s:e] = chunk
        else:
            right[s:e] = chunk

    stereo = np.stack([left, right], axis=-1)
    sf.write(str(output_path), stereo, sr)
    return True, actual_duration


def build_alignment_list(words: list[dict]) -> list[list]:
    """Build alignment list in voice-training format:
    [[word, [start, end], SPEAKER_LABEL], ...]
    """
    alignments = []
    for w in words:
        label = LABEL_LEFT if w["role"] == SPEAKER_LEFT else LABEL_RIGHT
        alignments.append([w["word"], [w["start"], w["end"]], label])
    return alignments


def build_text_prompt(rec: dict) -> str:
    """Build a text_prompt from podcast metadata for the alignment JSON.

    This serves as the system prompt equivalent for podcast data during training.
    """
    meta = rec.get("source_metadata", {})
    show = meta.get("show_name", "Unknown Show")
    episode = meta.get("episode_name", "")
    category = meta.get("show_category", "")
    duration = meta.get("duration_min", 0)

    lines = [
        f"SOURCE: Organic podcast conversation",
        f"SHOW: {show}",
    ]
    if episode:
        lines.append(f"EPISODE: {episode}")
    if category:
        lines.append(f"CATEGORY: {category}")
    lines.append(f"DURATION: {duration:.0f} minutes")
    lines.append(f"SPEAKERS: 2 (speaker_1=left/agent, speaker_2=right/human)")
    lines.append(
        "CONTEXT: Real conversational audio with natural turn-taking, "
        "backchannels, overlaps, and disfluencies."
    )
    return "\n".join(lines)


def quality_check(
    mono_path: Path, stereo_path: Path, words: list[dict], pod_id: str
) -> list[str]:
    """Run quality checks on the stereo output."""
    import soundfile as sf

    warnings = []

    mono, sr = sf.read(mono_path)
    stereo, _ = sf.read(stereo_path)

    # Handle truncation — compare up to stereo length
    n = len(stereo)
    mono_trimmed = mono[:n]

    left, right = stereo[:, 0], stereo[:, 1]

    # Check both channels have energy
    left_energy = np.sum(left ** 2)
    right_energy = np.sum(right ** 2)
    total_energy = left_energy + right_energy

    if left_energy < 1e-6:
        warnings.append(f"{pod_id}: left channel (speaker_1) near-zero energy")
    if right_energy < 1e-6:
        warnings.append(f"{pod_id}: right channel (speaker_2) near-zero energy")

    # Check coverage
    mono_energy = np.sum(mono_trimmed.astype(np.float32) ** 2)
    if mono_energy > 0:
        coverage = total_energy / mono_energy
        if coverage < 0.7:
            warnings.append(f"{pod_id}: stereo covers only {coverage:.0%} of mono energy")

    # Check speaker balance (at least 10% each)
    if total_energy > 0:
        balance = min(left_energy, right_energy) / total_energy
        if balance < 0.05:
            warnings.append(
                f"{pod_id}: extreme speaker imbalance "
                f"(L={left_energy / total_energy:.0%}, R={right_energy / total_energy:.0%})"
            )

    # Check word count
    if len(words) < 100:
        warnings.append(f"{pod_id}: only {len(words)} aligned words")

    return warnings


def process_one(
    pod_id: str,
    rec: dict,
    mono_dir: Path,
    output_dir: Path,
    max_duration: Optional[float] = None,
) -> tuple[str, bool, list[str], float]:
    """Process a single podcast record → stereo WAV + alignment JSON.

    Returns (pod_id, success, warnings, duration).
    """
    mono_path = mono_dir / f"{pod_id}.wav"
    stereo_path = output_dir / f"{pod_id}.wav"
    json_path = output_dir / f"{pod_id}.json"

    if not mono_path.exists():
        return pod_id, False, [f"{pod_id}: mono WAV not found"], 0.0

    messages = rec.get("messages", [])
    if not messages:
        return pod_id, False, [f"{pod_id}: no messages"], 0.0

    # Build word list from pre-existing timestamps
    words = build_word_list(messages)
    if len(words) < 10:
        return pod_id, False, [f"{pod_id}: too few words with timestamps ({len(words)})"], 0.0

    # Get audio duration from the mono file
    import soundfile as sf
    mono_info = sf.info(str(mono_path))
    audio_duration = mono_info.duration
    sr = mono_info.samplerate

    effective_duration = min(audio_duration, max_duration) if max_duration else audio_duration

    # Filter words within duration limit
    if max_duration:
        words = [w for w in words if w["start"] < max_duration]

    # Build speaker segments
    segments = build_speaker_segments(words, effective_duration)

    # Create stereo WAV
    ok, actual_dur = create_stereo_wav(mono_path, segments, stereo_path, sr, max_duration)
    if not ok:
        return pod_id, False, [f"{pod_id}: stereo creation failed"], 0.0

    # Build alignment JSON
    alignments = build_alignment_list(words)

    # Build turn metadata (filter by duration if needed)
    turns = build_turn_metadata(messages)
    if max_duration:
        turns = [t for t in turns if t["start"] < max_duration]

    text_prompt = build_text_prompt(rec)

    out_json = {
        "alignments": alignments,
        "turns": turns,
        "text_prompt": text_prompt,
    }
    with open(json_path, "w") as f:
        json.dump(out_json, f, ensure_ascii=False)

    # Quality checks
    warnings = quality_check(mono_path, stereo_path, words, pod_id)

    return pod_id, True, warnings, actual_dur


def main():
    parser = argparse.ArgumentParser(
        description="Create stereo WAVs from podcast mono audio using pre-existing timestamps"
    )
    parser.add_argument(
        "--transcripts", type=Path, required=True,
        help="Path to podcast_conversations_all.jsonl",
    )
    parser.add_argument(
        "--mono-dir", type=Path, required=True,
        help="Directory with 24kHz mono WAVs from podcast_to_mono.py",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for stereo WAVs + alignment JSONs",
    )
    parser.add_argument(
        "--max-duration", type=float, default=None,
        help="Truncate conversations longer than this many seconds",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip files that already have stereo output",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process first N records (for testing)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load transcripts
    logger.info("Loading transcripts from %s...", args.transcripts)
    transcripts = load_transcripts(args.transcripts)
    logger.info("Loaded %d transcripts", len(transcripts))

    # Build work list
    work = []
    for pod_id, rec in transcripts.items():
        if args.limit and len(work) >= args.limit:
            break

        mono_path = args.mono_dir / f"{pod_id}.wav"
        if not mono_path.exists():
            continue

        if args.resume:
            stereo_path = args.output_dir / f"{pod_id}.wav"
            json_path = args.output_dir / f"{pod_id}.json"
            if stereo_path.exists() and json_path.exists():
                continue

        work.append((pod_id, rec))

    logger.info("Work: %d conversations to process", len(work))
    if not work:
        logger.info("Nothing to do.")
        return

    # Process
    success = 0
    failed = 0
    all_warnings = []
    total_duration = 0.0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one, pod_id, rec,
                args.mono_dir, args.output_dir, args.max_duration,
            ): pod_id
            for pod_id, rec in work
        }

        for i, future in enumerate(as_completed(futures), 1):
            pod_id, ok, warnings, duration = future.result()
            all_warnings.extend(warnings)

            if ok:
                success += 1
                total_duration += duration
            else:
                failed += 1
                for w in warnings:
                    logger.warning(w)

            if i % 50 == 0:
                logger.info(
                    "Progress: %d/%d (success=%d, failed=%d, total_audio=%.1fh)",
                    i, len(work), success, failed, total_duration / 3600,
                )

    # Summary
    logger.info("=" * 60)
    logger.info("DONE: %d success, %d failed", success, failed)
    logger.info("Total stereo audio: %.1f hours", total_duration / 3600)
    if all_warnings:
        logger.info("Warnings (%d):", len(all_warnings))
        for w in all_warnings[:20]:
            logger.info("  %s", w)
        if len(all_warnings) > 20:
            logger.info("  ... and %d more", len(all_warnings) - 20)


if __name__ == "__main__":
    main()
