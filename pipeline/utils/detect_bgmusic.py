"""Scan all files for background music using noise floor as primary detector."""

import os
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

AUDIO_DIR = "/mnt/data/personaplex_data/companion_training_v3/mono_wav"


def analyze(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    sf = librosa.feature.spectral_flatness(y=y)[0]

    # Noise floor: bottom 10% of frames
    sorted_rms = np.sort(rms_db)
    noise_floor = np.mean(sorted_rms[:max(1, len(sorted_rms) // 10)])

    # Quiet segment analysis
    median_rms_db = np.median(rms_db)
    quiet_mask = rms_db < (median_rms_db - 10)
    if np.any(quiet_mask):
        quiet_S = S[:, quiet_mask]
        quiet_hf = np.sum(quiet_S[freqs >= 2000] ** 2) / (np.sum(quiet_S ** 2) + 1e-10)
    else:
        quiet_hf = 0.0

    return {
        "noise_floor_db": noise_floor,
        "quiet_hf_ratio": quiet_hf,
        "sf_mean": np.mean(sf),
    }


wav_files = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
print(f"Scanning {len(wav_files)} files...\n")

results = []
for i, fname in enumerate(wav_files):
    m = analyze(os.path.join(AUDIO_DIR, fname))
    results.append((fname, m))
    if (i + 1) % 40 == 0:
        print(f"  {i + 1}/{len(wav_files)}...")

# Sort by noise floor descending (highest = most suspect)
results.sort(key=lambda x: -x[1]["noise_floor_db"])

print(f"\n{'File':<20} {'NoiseFloor':>11} {'QuietHF':>8} {'SpecFlat':>9}")
print("-" * 55)
for fname, m in results:
    flag = " ***" if m["noise_floor_db"] > -50 else ""
    print(f"{fname:<20} {m['noise_floor_db']:>10.1f}dB {m['quiet_hf_ratio']:>8.4f} {m['sf_mean']:>9.4f}{flag}")

suspects = [(f, m) for f, m in results if m["noise_floor_db"] > -50]
print(f"\nSUSPECT (noise floor > -50 dB): {len(suspects)} / {len(results)}")
for f, m in suspects:
    print(f"  {f}  floor={m['noise_floor_db']:.1f}dB")
