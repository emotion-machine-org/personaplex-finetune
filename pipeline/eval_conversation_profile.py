#!/usr/bin/env python3
"""
Evaluate conversational profile of stereo dialogue files.

Measures per the CANDOR-derived targets in what-makes-a-good-conversationalist.md:
  1. Turn gap (floor-transfer offset) distribution
  2. Speech rate (words per second)
  3. Loudness variation (per-turn dB standard deviation)
  4. Turn duration distribution
  5. Backchannel rate and type

Usage:
    python eval_conversation_profile.py elevenlabs-data/stereo_wav vibevoice-data/stereo_wav --compare
    python eval_conversation_profile.py elevenlabs-data/stereo_wav --max-files 5
"""

import argparse
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Config ──────────────────────────────────────────────────────────────────

TARGET_SR = 16000

# VAD parameters (energy-based, tuned for clean TTS stereo)
VAD_FRAME_MS = 30
VAD_HOP_MS = 10
VAD_ENERGY_THRESHOLD_DB = -40  # dB below peak
VAD_MIN_SPEECH_S = 0.08
VAD_MIN_SILENCE_S = 0.20

# Turn segmentation
MIN_TURN_GAP_S = 0.05

# Backchannel detection
BACKCHANNEL_MAX_WORDS = 3
BACKCHANNEL_MAX_DURATION_S = 2.0

GENERIC_BC = {"yeah", "yes", "yep", "yup", "ya", "mhm", "mm", "mmm", "hmm",
              "uh", "huh", "uh-huh", "uhuh", "right", "okay", "ok", "sure"}
SPECIFIC_BC = {"exactly", "absolutely", "totally", "definitely",
               "oh", "ah", "wow", "really", "no",
               "nice", "cool", "great"}
ALL_BC = GENERIC_BC | SPECIFIC_BC


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Segment:
    start: float
    end: float
    channel: int

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ConversationMetrics:
    file: str = ""
    duration_s: float = 0.0
    turn_gaps_ms: list = field(default_factory=list)
    turn_gap_median_ms: float = 0.0
    turn_gap_iqr_ms: tuple = (0.0, 0.0)
    turn_gap_pct_over_1s: float = 0.0
    speech_rate_wps_user: float = 0.0
    speech_rate_wps_system: float = 0.0
    speech_rate_wps_overall: float = 0.0
    speech_rate_sd_across_turns: float = 0.0
    loudness_turn_sd_mean: float = 0.0
    loudness_turn_sd_median: float = 0.0
    loudness_interturn_sd: float = 0.0
    turn_duration_mean_s: float = 0.0
    turn_duration_median_s: float = 0.0
    n_turns: int = 0
    backchannel_count: int = 0
    backchannel_rate_per_hour: float = 0.0
    backchannel_generic_count: int = 0
    backchannel_specific_count: int = 0


# ── VAD ──────────────────────────────────────────────────────────────────────

def energy_vad(audio: np.ndarray, sr: int) -> list[tuple[float, float]]:
    """Energy-based VAD for a mono channel. Returns list of (start, end) in seconds."""
    frame_len = int(sr * VAD_FRAME_MS / 1000)
    hop_len = int(sr * VAD_HOP_MS / 1000)
    n_frames = max(1, (len(audio) - frame_len) // hop_len + 1)

    # Vectorized frame energy
    indices = np.arange(n_frames)[:, None] * hop_len + np.arange(frame_len)
    indices = np.clip(indices, 0, len(audio) - 1)
    frames = audio[indices]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    energies = 20 * np.log10(rms + 1e-10)

    threshold = np.max(energies) + VAD_ENERGY_THRESHOLD_DB
    is_speech = energies > threshold

    # Extract contiguous speech regions
    diff = np.diff(is_speech.astype(int), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    segments = [(s * hop_len / sr, e * hop_len / sr) for s, e in zip(starts, ends)]

    # Merge short silences
    merged = []
    for s, e in segments:
        if merged and s - merged[-1][1] < VAD_MIN_SILENCE_S:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    return [(s, e) for s, e in merged if e - s >= VAD_MIN_SPEECH_S]


def get_turns(audio_stereo: np.ndarray, sr: int) -> list[Segment]:
    """Run VAD on both channels, merge close same-channel segments into turns."""
    segments = []
    for ch in range(2):
        for s, e in energy_vad(audio_stereo[:, ch], sr):
            segments.append(Segment(start=s, end=e, channel=ch))
    segments.sort(key=lambda s: s.start)

    # Merge close same-channel segments
    turns = []
    for seg in segments:
        if turns and turns[-1].channel == seg.channel and (seg.start - turns[-1].end) < MIN_TURN_GAP_S:
            turns[-1] = Segment(turns[-1].start, seg.end, seg.channel)
        else:
            turns.append(seg)
    return turns


# ── Turn gaps ────────────────────────────────────────────────────────────────

def compute_turn_gaps(turns: list[Segment]) -> list[float]:
    """Floor-transfer offsets in ms. Positive = gap, negative = overlap."""
    gaps = []
    for i, seg in enumerate(turns):
        for j in range(i + 1, len(turns)):
            other = turns[j]
            if other.channel != seg.channel:
                fto_ms = (other.start - seg.end) * 1000
                if -2000 < fto_ms < 5000:
                    gaps.append(fto_ms)
                break
    return gaps


# ── Loudness ─────────────────────────────────────────────────────────────────

def compute_loudness(audio_stereo: np.ndarray, sr: int, turns: list[Segment]) -> dict:
    """Per-turn loudness statistics."""
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    turn_means, turn_sds = [], []

    for seg in turns:
        mono = audio_stereo[int(seg.start * sr):int(seg.end * sr), seg.channel]
        if len(mono) < frame_len:
            continue
        n = max(1, (len(mono) - frame_len) // hop_len + 1)
        idx = np.arange(n)[:, None] * hop_len + np.arange(frame_len)
        idx = np.clip(idx, 0, len(mono) - 1)
        rms = np.sqrt(np.mean(mono[idx] ** 2, axis=1) + 1e-10)
        db = 20 * np.log10(rms + 1e-10)
        turn_means.append(np.mean(db))
        turn_sds.append(np.std(db))

    return {
        "turn_sd_mean": float(np.mean(turn_sds)) if turn_sds else 0.0,
        "turn_sd_median": float(np.median(turn_sds)) if turn_sds else 0.0,
        "interturn_sd": float(np.std(turn_means)) if turn_means else 0.0,
    }


# ── WhisperX transcription ──────────────────────────────────────────────────

_whisperx_model = None
_align_model = None
_align_metadata = None


def load_whisperx():
    global _whisperx_model, _align_model, _align_metadata
    if _whisperx_model is None:
        import whisperx
        print("  Loading WhisperX large-v3 on cuda:0...", flush=True)
        _whisperx_model = whisperx.load_model(
            "large-v3", device="cuda", device_index=0, compute_type="float16",
            language="en",
        )
        print("  Loading alignment model on cuda:0...", flush=True)
        _align_model, _align_metadata = whisperx.load_align_model(
            language_code="en", device="cuda:0"
        )
    return _whisperx_model, _align_model, _align_metadata


def transcribe_channel(audio_mono: np.ndarray, sr: int) -> list[dict]:
    """Transcribe a mono channel and return word-level timestamps via WhisperX."""
    import whisperx

    model, align_model, align_metadata = load_whisperx()

    # Resample to 16kHz
    if sr != TARGET_SR:
        t = torch.from_numpy(audio_mono).float().unsqueeze(0)
        audio_16k = torchaudio.transforms.Resample(sr, TARGET_SR)(t).squeeze(0).numpy()
    else:
        audio_16k = audio_mono

    # Transcribe
    result = model.transcribe(audio_16k, batch_size=16, language="en", task="transcribe")

    # Align for word-level timestamps
    if result["segments"]:
        aligned = whisperx.align(
            result["segments"], align_model, align_metadata, audio_16k, "cuda:0"
        )
        # Flatten all words
        words = []
        for seg in aligned["segments"]:
            for w in seg.get("words", []):
                if "start" in w and "end" in w:
                    words.append({
                        "word": w["word"].lower().strip(".,!?;:\"'"),
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                    })
        return words
    return []


# ── Speech rate ──────────────────────────────────────────────────────────────

def compute_speech_rate(words_by_ch: dict, turns: list[Segment]) -> dict:
    """Words-per-second per turn, per channel, and overall."""
    turn_rates = []
    for seg in turns:
        ch_words = words_by_ch.get(seg.channel, [])
        tw = [w for w in ch_words if w["start"] >= seg.start - 0.15 and w["end"] <= seg.end + 0.15]
        if seg.duration > 0.5 and len(tw) >= 2:
            turn_rates.append((seg.channel, len(tw) / seg.duration))

    user = [r for ch, r in turn_rates if ch == 0]
    system = [r for ch, r in turn_rates if ch == 1]
    all_r = [r for _, r in turn_rates]
    return {
        "user_wps": float(np.mean(user)) if user else 0.0,
        "system_wps": float(np.mean(system)) if system else 0.0,
        "overall_wps": float(np.mean(all_r)) if all_r else 0.0,
        "sd_across_turns": float(np.std(all_r)) if all_r else 0.0,
    }


# ── Backchannels ─────────────────────────────────────────────────────────────

def detect_backchannels(words_by_ch: dict, turns: list[Segment]) -> dict:
    """Detect backchannels from short turns with backchannel vocabulary."""
    counts = {"generic": 0, "specific": 0, "total": 0}
    for seg in turns:
        if seg.duration > BACKCHANNEL_MAX_DURATION_S:
            continue
        ch_words = words_by_ch.get(seg.channel, [])
        tw = [w for w in ch_words if w["start"] >= seg.start - 0.15 and w["end"] <= seg.end + 0.15]
        if not tw or len(tw) > BACKCHANNEL_MAX_WORDS:
            continue
        bc_count = sum(1 for w in tw if w["word"] in ALL_BC)
        if bc_count / len(tw) < 0.5:
            continue
        counts["total"] += 1
        if any(w["word"] in SPECIFIC_BC for w in tw):
            counts["specific"] += 1
        else:
            counts["generic"] += 1
    return counts


# ── Main analysis ────────────────────────────────────────────────────────────

def analyze_file(filepath: str) -> ConversationMetrics:
    # Use parent dir name if file is conversation.wav inside a subdirectory
    basename = os.path.basename(filepath)
    if basename == "conversation.wav":
        label = os.path.basename(os.path.dirname(filepath))
    else:
        label = basename
    print(f"  {label}", flush=True)
    audio, sr = sf.read(filepath)
    assert audio.ndim == 2 and audio.shape[1] == 2

    m = ConversationMetrics(file=label, duration_s=len(audio) / sr)

    # VAD → turns
    turns = get_turns(audio, sr)
    if not turns:
        print("    WARNING: no speech detected")
        return m
    m.n_turns = len(turns)

    # Turn gaps
    gaps = compute_turn_gaps(turns)
    if gaps:
        m.turn_gaps_ms = gaps
        m.turn_gap_median_ms = float(np.median(gaps))
        q25, q75 = np.percentile(gaps, [25, 75])
        m.turn_gap_iqr_ms = (float(q25), float(q75))
        m.turn_gap_pct_over_1s = float(np.mean([g > 1000 for g in gaps]) * 100)

    # Turn duration
    durs = [t.duration for t in turns if t.duration > 0.1]
    if durs:
        m.turn_duration_mean_s = float(np.mean(durs))
        m.turn_duration_median_s = float(np.median(durs))

    # Loudness
    ld = compute_loudness(audio, sr, turns)
    m.loudness_turn_sd_mean = ld["turn_sd_mean"]
    m.loudness_turn_sd_median = ld["turn_sd_median"]
    m.loudness_interturn_sd = ld["interturn_sd"]

    # WhisperX transcription per channel
    words_by_ch = {}
    for ch in range(2):
        label = "user" if ch == 0 else "system"
        print(f"    transcribing {label}...", flush=True)
        words = transcribe_channel(audio[:, ch].copy(), sr)
        words_by_ch[ch] = words
        print(f"      {len(words)} words", flush=True)

    # Speech rate
    rate = compute_speech_rate(words_by_ch, turns)
    m.speech_rate_wps_user = rate["user_wps"]
    m.speech_rate_wps_system = rate["system_wps"]
    m.speech_rate_wps_overall = rate["overall_wps"]
    m.speech_rate_sd_across_turns = rate["sd_across_turns"]

    # Backchannels
    bc = detect_backchannels(words_by_ch, turns)
    m.backchannel_count = bc["total"]
    hours = m.duration_s / 3600
    m.backchannel_rate_per_hour = bc["total"] / hours if hours > 0 else 0.0
    m.backchannel_generic_count = bc["generic"]
    m.backchannel_specific_count = bc["specific"]

    return m


def analyze_directory(dirpath: str, max_files: int = 0) -> list[ConversationMetrics]:
    # Support both flat layout (dir/*.wav) and nested layout (dir/dial-*/conversation.wav)
    wavs = sorted(f for f in os.listdir(dirpath) if f.endswith(".wav"))
    if wavs:
        # Flat layout: WAV files directly in directory
        wav_paths = [os.path.join(dirpath, f) for f in wavs]
    else:
        # Nested layout: subdirectories containing conversation.wav
        subdirs = sorted(d for d in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, d)))
        wav_paths = []
        for sd in subdirs:
            candidate = os.path.join(dirpath, sd, "conversation.wav")
            if os.path.exists(candidate):
                wav_paths.append(candidate)
    if max_files > 0:
        wav_paths = wav_paths[:max_files]
    print(f"\nAnalyzing {len(wav_paths)} files in {dirpath}")
    return [analyze_file(p) for p in wav_paths]


def summarize(results: list[ConversationMetrics], label: str) -> dict:
    if not results:
        return {}
    all_gaps = [g for r in results for g in r.turn_gaps_ms]
    all_dur = [r.turn_duration_median_s for r in results if r.turn_duration_median_s > 0]

    def safe_mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    return {
        "label": label,
        "n_files": len(results),
        "total_duration_min": sum(r.duration_s for r in results) / 60,
        "turn_gap": {
            "median_ms": float(np.median(all_gaps)) if all_gaps else 0,
            "iqr_ms": [float(np.percentile(all_gaps, 25)), float(np.percentile(all_gaps, 75))] if all_gaps else [0, 0],
            "pct_over_1s": float(np.mean([g > 1000 for g in all_gaps]) * 100) if all_gaps else 0,
            "pct_overlap": float(np.mean([g < 0 for g in all_gaps]) * 100) if all_gaps else 0,
            "n_transitions": len(all_gaps),
        },
        "speech_rate": {
            "mean_wps": safe_mean([r.speech_rate_wps_overall for r in results if r.speech_rate_wps_overall > 0]),
            "user_wps": safe_mean([r.speech_rate_wps_user for r in results if r.speech_rate_wps_user > 0]),
            "system_wps": safe_mean([r.speech_rate_wps_system for r in results if r.speech_rate_wps_system > 0]),
            "sd_across_turns": safe_mean([r.speech_rate_sd_across_turns for r in results]),
        },
        "loudness": {
            "intra_turn_sd_mean_dB": safe_mean([r.loudness_turn_sd_mean for r in results]),
            "intra_turn_sd_median_dB": safe_mean([r.loudness_turn_sd_median for r in results]),
            "inter_turn_sd_dB": safe_mean([r.loudness_interturn_sd for r in results]),
        },
        "turn_duration": {
            "mean_s": safe_mean(all_dur),
            "median_s": float(np.median(all_dur)) if all_dur else 0,
            "mean_n_turns": safe_mean([r.n_turns for r in results]),
        },
        "backchannels": {
            "mean_rate_per_hour": safe_mean([r.backchannel_rate_per_hour for r in results]),
            "mean_count": safe_mean([r.backchannel_count for r in results]),
            "mean_generic": safe_mean([r.backchannel_generic_count for r in results]),
            "mean_specific": safe_mean([r.backchannel_specific_count for r in results]),
        },
    }


def print_summary(s: dict):
    label = s["label"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {s['n_files']} files, {s['total_duration_min']:.1f} min total")
    print(f"{'='*60}")
    tg = s["turn_gap"]
    print(f"\n  Turn Gaps (CANDOR target: median 200-400 ms)")
    print(f"    Median:       {tg['median_ms']:+.0f} ms")
    print(f"    IQR:          [{tg['iqr_ms'][0]:.0f}, {tg['iqr_ms'][1]:.0f}] ms")
    print(f"    > 1 s:        {tg['pct_over_1s']:.1f}% (target: < 5%)")
    print(f"    Overlaps:     {tg['pct_overlap']:.1f}%")
    print(f"    Transitions:  {tg['n_transitions']}")
    sr = s["speech_rate"]
    print(f"\n  Speech Rate (CANDOR target: 3.2-3.5 WPS)")
    print(f"    Overall:      {sr['mean_wps']:.2f} WPS")
    print(f"    User ch:      {sr['user_wps']:.2f} WPS")
    print(f"    System ch:    {sr['system_wps']:.2f} WPS")
    print(f"    SD across turns: {sr['sd_across_turns']:.2f}")
    ld = s["loudness"]
    print(f"\n  Loudness Variation (higher = more dynamic)")
    print(f"    Intra-turn SD (mean):   {ld['intra_turn_sd_mean_dB']:.2f} dB")
    print(f"    Intra-turn SD (median): {ld['intra_turn_sd_median_dB']:.2f} dB")
    print(f"    Inter-turn SD:          {ld['inter_turn_sd_dB']:.2f} dB")
    td = s["turn_duration"]
    print(f"\n  Turn Duration (CANDOR target: median 5-8 s)")
    print(f"    Mean:         {td['mean_s']:.2f} s")
    print(f"    Median:       {td['median_s']:.2f} s")
    print(f"    Avg turns/dial: {td['mean_n_turns']:.0f}")
    bc = s["backchannels"]
    print(f"\n  Backchannels (CANDOR target: ~1000/hour)")
    print(f"    Rate:         {bc['mean_rate_per_hour']:.0f} /hour")
    print(f"    Avg count:    {bc['mean_count']:.1f} per dialogue")
    print(f"    Generic:      {bc['mean_generic']:.1f}")
    print(f"    Specific:     {bc['mean_specific']:.1f}")


def print_comparison(s1: dict, s2: dict):
    l1, l2 = s1["label"], s2["label"]
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {l1} vs {l2}")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<35} {l1:>14} {l2:>14}")
    print(f"  {'-'*63}")
    rows = [
        ("Turn gap median (ms)", f"{s1['turn_gap']['median_ms']:+.0f}", f"{s2['turn_gap']['median_ms']:+.0f}"),
        ("Turn gap > 1s (%)", f"{s1['turn_gap']['pct_over_1s']:.1f}", f"{s2['turn_gap']['pct_over_1s']:.1f}"),
        ("Turn gap overlaps (%)", f"{s1['turn_gap']['pct_overlap']:.1f}", f"{s2['turn_gap']['pct_overlap']:.1f}"),
        ("Speech rate (WPS)", f"{s1['speech_rate']['mean_wps']:.2f}", f"{s2['speech_rate']['mean_wps']:.2f}"),
        ("Speech rate SD", f"{s1['speech_rate']['sd_across_turns']:.2f}", f"{s2['speech_rate']['sd_across_turns']:.2f}"),
        ("Loudness intra-turn SD (dB)", f"{s1['loudness']['intra_turn_sd_mean_dB']:.2f}", f"{s2['loudness']['intra_turn_sd_mean_dB']:.2f}"),
        ("Loudness inter-turn SD (dB)", f"{s1['loudness']['inter_turn_sd_dB']:.2f}", f"{s2['loudness']['inter_turn_sd_dB']:.2f}"),
        ("Turn duration median (s)", f"{s1['turn_duration']['median_s']:.2f}", f"{s2['turn_duration']['median_s']:.2f}"),
        ("Turns per dialogue", f"{s1['turn_duration']['mean_n_turns']:.0f}", f"{s2['turn_duration']['mean_n_turns']:.0f}"),
        ("Backchannels / hour", f"{s1['backchannels']['mean_rate_per_hour']:.0f}", f"{s2['backchannels']['mean_rate_per_hour']:.0f}"),
    ]
    for label, v1, v2 in rows:
        print(f"  {label:<35} {v1:>14} {v2:>14}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate conversational profile of stereo dialogue files")
    parser.add_argument("dirs", nargs="+", help="Directories of stereo WAV files")
    parser.add_argument("--compare", action="store_true", help="Side-by-side comparison (requires 2 dirs)")
    parser.add_argument("--max-files", type=int, default=0, help="Max files per directory (0=all)")
    parser.add_argument("--out", type=str, default="eval_results.json", help="Output JSON file")
    args = parser.parse_args()

    summaries = []
    for dirpath in args.dirs:
        label = Path(dirpath).parts[-2] if "stereo_wav" in dirpath else Path(dirpath).stem
        results = analyze_directory(dirpath, max_files=args.max_files)
        s = summarize(results, label)
        print_summary(s)
        summaries.append(s)

    if args.compare and len(summaries) == 2:
        print_comparison(summaries[0], summaries[1])

    with open(args.out, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
