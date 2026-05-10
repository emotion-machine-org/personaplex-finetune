#!/usr/bin/env python3
"""Step 3+4: Create stereo WAVs from mono VibeVoice output using WhisperX alignment.

For each mono WAV:
1. Transcribe with WhisperX to get word-level timestamps (wav2vec2 forced alignment)
2. Match words to ground-truth script for speaker assignment
3. Create stereo WAV (left=broker/Speaker1, right=client/Speaker2)
4. Write alignment JSON for broker words (Moshi finetune format)

Usage:
    python pipeline/create_stereo.py \
        --mono-dir data/mono_wav \
        --scripts-dir data/scripts \
        --output-dir data/stereo_wav \
        --device cuda:0
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisperx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FADE_MS = 10  # crossfade duration at segment boundaries


def parse_script(script_path: Path) -> list[dict]:
    """Parse a VibeVoice script into turns with speaker labels.

    Returns list of {"role": "BROKER"|"CLIENT", "text": str, "speaker_num": int}
    """
    turns = []
    with open(script_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^Speaker (\d+): (.+)$", line)
            if m:
                num = int(m.group(1))
                turns.append({
                    "role": "BROKER" if num == 1 else "CLIENT",
                    "text": m.group(2),
                    "speaker_num": num,
                })
    return turns


def normalize_word(word: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    return re.sub(r"[^a-z0-9']", "", word.lower())


def flatten_turns(turns: list[dict]) -> list[dict]:
    """Flatten turns into ordered word list with speaker labels."""
    words = []
    for turn_idx, turn in enumerate(turns):
        for w in turn["text"].split():
            n = normalize_word(w)
            if n:
                words.append({
                    "norm": n,
                    "raw": w,
                    "role": turn["role"],
                    "turn_index": turn_idx,
                })
    return words


def match_words_to_speakers(
    wx_words: list[dict], gt_turns: list[dict]
) -> tuple[list[dict], float]:
    """Match WhisperX words to ground-truth turns for speaker assignment.

    Uses sequential greedy matching with lookahead.
    Returns (matched_words, match_rate).
    """
    gt_flat = flatten_turns(gt_turns)
    gt_idx = 0
    results = []
    matched_count = 0

    for wx in wx_words:
        wx_norm = normalize_word(wx["word"])
        if not wx_norm:
            continue
        if "start" not in wx or "end" not in wx:
            continue

        matched = False
        remaining = len(gt_flat) - gt_idx
        lookahead = min(15, remaining)

        for la in range(lookahead):
            if gt_flat[gt_idx + la]["norm"] == wx_norm:
                gt_word = gt_flat[gt_idx + la]
                results.append({
                    "word": wx["word"],
                    "start": wx["start"],
                    "end": wx["end"],
                    "role": gt_word["role"],
                    "turn_index": gt_word["turn_index"],
                })
                gt_idx = gt_idx + la + 1
                matched = True
                matched_count += 1
                break

        if not matched:
            # Assign to current speaker context
            if gt_idx < len(gt_flat):
                role = gt_flat[gt_idx]["role"]
                turn_index = gt_flat[gt_idx]["turn_index"]
            elif results:
                role = results[-1]["role"]
                turn_index = results[-1]["turn_index"]
            else:
                role = "BROKER"
                turn_index = 0
            results.append({
                "word": wx["word"],
                "start": wx["start"],
                "end": wx["end"],
                "role": role,
                "turn_index": turn_index,
            })

    total = len([w for w in wx_words if normalize_word(w.get("word", "")) and "start" in w])
    match_rate = matched_count / total if total else 0
    return results, match_rate


def build_speaker_segments(matched_words: list[dict], audio_duration: float) -> list[dict]:
    """Group consecutive same-speaker words into segments, filling gaps."""
    if not matched_words:
        return []

    # Group consecutive words by speaker
    segments = []
    current = {"role": matched_words[0]["role"], "start": matched_words[0]["start"], "end": matched_words[0]["end"]}

    for w in matched_words[1:]:
        if w["role"] == current["role"]:
            current["end"] = w["end"]
        else:
            segments.append(current)
            current = {"role": w["role"], "start": w["start"], "end": w["end"]}
    segments.append(current)

    # Fill gaps between segments using midpoints
    for i in range(len(segments) - 1):
        gap_start = segments[i]["end"]
        gap_end = segments[i + 1]["start"]
        if gap_end > gap_start:
            mid = (gap_start + gap_end) / 2
            segments[i]["end"] = mid
            segments[i + 1]["start"] = mid

    # Extend first/last to audio boundaries
    segments[0]["start"] = 0.0
    segments[-1]["end"] = audio_duration

    return segments


def build_turn_metadata(matched_words: list[dict], gt_turns: list[dict]) -> list[dict]:
    """Build explicit turn-level metadata preserving the original transcript order."""
    words_by_turn: list[list[dict]] = [[] for _ in gt_turns]
    for word in matched_words:
        turn_index = word.get("turn_index")
        if isinstance(turn_index, int) and 0 <= turn_index < len(words_by_turn):
            words_by_turn[turn_index].append(word)

    next_starts: list[float | None] = [None] * len(gt_turns)
    next_start: float | None = None
    for i in range(len(gt_turns) - 1, -1, -1):
        if words_by_turn[i]:
            next_start = float(words_by_turn[i][0]["start"])
        next_starts[i] = next_start

    turns_out = []
    prev_end: float | None = None
    for i, turn in enumerate(gt_turns):
        speaker = "SPEAKER_BROKER" if turn["role"] == "BROKER" else "SPEAKER_CLIENT"
        if words_by_turn[i]:
            start = float(words_by_turn[i][0]["start"])
            end = float(words_by_turn[i][-1]["end"])
        else:
            start = prev_end if prev_end is not None else next_starts[i]
            if start is None:
                start = 0.0
            end = next_starts[i] if next_starts[i] is not None else start
            if end < start:
                end = start
        turns_out.append({
            "index": i,
            "speaker": speaker,
            "start": start,
            "end": end,
            "text": turn["text"],
        })
        prev_end = end
    return turns_out


def create_stereo_wav(
    mono_path: Path, segments: list[dict], output_path: Path, sr: int
):
    """Create stereo WAV with broker on left, client on right."""
    mono, file_sr = sf.read(mono_path)
    assert file_sr == sr, f"Expected {sr}Hz, got {file_sr}Hz"

    n_samples = len(mono)
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

        if seg["role"] == "BROKER":
            left[s:e] = chunk
        else:
            right[s:e] = chunk

    stereo = np.stack([left, right], axis=-1)
    sf.write(str(output_path), stereo, sr)


def write_alignment_json(
    matched_words: list[dict],
    gt_turns: list[dict],
    output_path: Path,
    jsonl_record: dict | None = None,
):
    """Write alignment JSON with both-speaker timings and optional JSONL metadata.

    Alignments use SPEAKER_BROKER / SPEAKER_CLIENT labels for both speakers.
    If jsonl_record is provided, text_prompt and context_injections are propagated.
    """
    alignments = []
    for w in matched_words:
        label = "SPEAKER_BROKER" if w["role"] == "BROKER" else "SPEAKER_CLIENT"
        alignments.append([w["word"], [w["start"], w["end"]], label])

    out: dict = {
        "alignments": alignments,
        "turns": build_turn_metadata(matched_words, gt_turns),
    }

    if jsonl_record is not None:
        if jsonl_record.get("text_prompt"):
            out["text_prompt"] = jsonl_record["text_prompt"]
        if jsonl_record.get("context_injections"):
            out["context_injections"] = jsonl_record["context_injections"]

    with open(output_path, "w") as f:
        json.dump(out, f, ensure_ascii=False)


def quality_check(
    mono_path: Path, stereo_path: Path, match_rate: float, dialogue_id: str
) -> list[str]:
    """Run quality checks, return list of warnings."""
    warnings = []

    mono, sr = sf.read(mono_path)
    stereo, _ = sf.read(stereo_path)
    left, right = stereo[:, 0], stereo[:, 1]

    # Check both channels have energy
    left_energy = np.sum(left ** 2)
    right_energy = np.sum(right ** 2)
    if left_energy < 1e-6:
        warnings.append(f"{dialogue_id}: left channel (broker) has near-zero energy")
    if right_energy < 1e-6:
        warnings.append(f"{dialogue_id}: right channel (client) has near-zero energy")

    # Check coverage
    mono_energy = np.sum(mono ** 2)
    stereo_energy = left_energy + right_energy
    if mono_energy > 0:
        coverage = stereo_energy / mono_energy
        if coverage < 0.8:
            warnings.append(f"{dialogue_id}: stereo covers only {coverage:.0%} of mono energy")

    # Check match rate
    if match_rate < 0.7:
        warnings.append(f"{dialogue_id}: low word match rate {match_rate:.0%}")

    return warnings


def process_one(
    mono_path: Path,
    script_path: Path,
    output_dir: Path,
    whisper_model,
    align_model,
    align_metadata,
    device: str,
    language: str = "en",
    jsonl_record: dict | None = None,
) -> tuple[bool, list[str]]:
    """Process a single mono WAV → stereo WAV + alignment JSON."""
    dialogue_id = mono_path.stem
    stereo_path = output_dir / f"{dialogue_id}.wav"
    json_path = output_dir / f"{dialogue_id}.json"

    # Parse ground-truth script
    turns = parse_script(script_path)
    if not turns:
        return False, [f"{dialogue_id}: no turns parsed from script"]

    # Load audio for WhisperX (needs 16kHz)
    audio = whisperx.load_audio(str(mono_path))

    # Transcribe
    result = whisper_model.transcribe(audio, batch_size=16, language=language)

    # Align to get word-level timestamps
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Extract word-level timestamps
    wx_words = result.get("word_segments", [])
    if not wx_words:
        return False, [f"{dialogue_id}: WhisperX produced no word segments"]

    # Match to ground-truth speakers
    matched_words, match_rate = match_words_to_speakers(wx_words, turns)
    logger.info(
        "%s: %d words aligned, match rate %.1f%%",
        dialogue_id, len(matched_words), match_rate * 100,
    )

    # Get audio info for stereo creation
    mono_info = sf.info(str(mono_path))
    sr = mono_info.samplerate
    audio_duration = mono_info.duration

    # Build speaker segments and create stereo
    segments = build_speaker_segments(matched_words, audio_duration)
    create_stereo_wav(mono_path, segments, stereo_path, sr)

    # Write alignment JSON (step 4)
    write_alignment_json(matched_words, turns, json_path, jsonl_record=jsonl_record)

    # Quality checks
    warnings = quality_check(mono_path, stereo_path, match_rate, dialogue_id)

    return True, warnings


def run_shard(args, pairs, jsonl_lookup: dict | None = None):
    """Process a shard of pairs on a single GPU."""
    torch_device = f"{args.device}:{args.device_index}" if args.device == "cuda" else args.device

    logger.info("Loading WhisperX model (%s) on %s...", args.whisper_model, torch_device)
    whisper_model = whisperx.load_model(
        args.whisper_model, args.device,
        device_index=args.device_index,
        compute_type=args.compute_type,
    )

    logger.info("Loading alignment model on %s...", torch_device)
    align_model, align_metadata = whisperx.load_align_model(
        language_code=args.language, device=torch_device,
    )

    all_warnings = []
    success = 0
    for mono_path, script_path in pairs:
        logger.info("Processing %s...", mono_path.name)
        record = jsonl_lookup.get(mono_path.stem) if jsonl_lookup else None
        try:
            ok, warnings = process_one(
                mono_path, script_path, args.output_dir,
                whisper_model, align_model, align_metadata, torch_device,
                language=args.language,
                jsonl_record=record,
            )
            all_warnings.extend(warnings)
            if ok:
                success += 1
        except Exception:
            logger.exception("Failed to process %s", mono_path.name)

    logger.info("Done: %d/%d succeeded", success, len(pairs))
    if all_warnings:
        logger.warning("Quality warnings:")
        for w in all_warnings:
            logger.warning("  %s", w)


def load_jsonl_lookup(path: Path) -> dict[str, dict]:
    """Load a dialogues JSONL file into a dict keyed by dialogue ID."""
    lookup = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            dial_id = record.get("id")
            if dial_id:
                lookup[dial_id] = record
    logger.info("Loaded %d records from %s", len(lookup), path)
    return lookup


def main():
    import subprocess

    parser = argparse.ArgumentParser(description="Create stereo WAVs from mono VibeVoice output")
    parser.add_argument("--mono-dir", type=Path, required=True)
    parser.add_argument("--scripts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dialogues", type=Path, default=None,
                        help="Dialogues JSONL file — propagates text_prompt and context_injections into stereo JSON")
    parser.add_argument("--whisper-model", default="medium", help="Whisper model size")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs for parallel processing, e.g. '0,1,2,3,4'")
    parser.add_argument("--shard", type=int, default=None,
                        help="Shard index (used internally by --gpus launcher)")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards (used internally by --gpus launcher)")
    parser.add_argument("--language", default="en",
                        help="Language code for WhisperX (default: en)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of parallel worker processes per GPU")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files that already have stereo WAV output")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load JSONL lookup if provided
    jsonl_lookup = load_jsonl_lookup(args.dialogues) if args.dialogues else None

    # Multi-GPU launcher mode (also handles --batch-size for parallel workers per GPU)
    if args.gpus and args.shard is None:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        num_workers = len(gpu_ids) * args.batch_size
        logger.info("Launching %d workers across GPUs %s (%d per GPU)",
                     num_workers, gpu_ids, args.batch_size)
        procs = []
        shard_idx = 0
        for gpu_id in gpu_ids:
            for _ in range(args.batch_size):
                cmd = [
                    sys.executable, __file__,
                    "--mono-dir", str(args.mono_dir),
                    "--scripts-dir", str(args.scripts_dir),
                    "--output-dir", str(args.output_dir),
                    "--whisper-model", args.whisper_model,
                    "--device", "cuda",
                    "--device-index", "0",
                    "--compute-type", args.compute_type,
                    "--language", args.language,
                    "--shard", str(shard_idx),
                    "--num-shards", str(num_workers),
                ]
                if args.dialogues:
                    cmd.extend(["--dialogues", str(args.dialogues)])
                if args.resume:
                    cmd.append("--resume")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
                logger.info("Worker %d/%d on GPU %d", shard_idx, num_workers, gpu_id)
                procs.append(subprocess.Popen(cmd, env=env))
                shard_idx += 1

        failed = []
        for i, p in enumerate(procs):
            p.wait()
            if p.returncode != 0:
                failed.append(i)
        if failed:
            logger.error("Workers failed: %s", failed)
            sys.exit(1)
        logger.info("All %d workers complete.", num_workers)
        return

    # Find all mono WAVs with matching scripts
    mono_files = sorted(args.mono_dir.glob("*.wav"))
    pairs = []
    for mf in mono_files:
        sf_path = args.scripts_dir / f"{mf.stem}.txt"
        if sf_path.exists():
            pairs.append((mf, sf_path))
        else:
            logger.warning("No script for %s, skipping", mf.name)

    # Shard selection
    if args.shard is not None and args.num_shards is not None:
        pairs = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard]
        logger.info("Shard %d/%d: %d files", args.shard, args.num_shards, len(pairs))

    # Resume: skip already-processed files
    if args.resume:
        before = len(pairs)
        pairs = [
            (m, s) for m, s in pairs
            if not (args.output_dir / f"{m.stem}.wav").exists()
        ]
        logger.info("Resume: skipping %d already done, %d remaining", before - len(pairs), len(pairs))

    logger.info("Found %d mono WAVs to process", len(pairs))
    if not pairs:
        return

    run_shard(args, pairs, jsonl_lookup=jsonl_lookup)


if __name__ == "__main__":
    main()
