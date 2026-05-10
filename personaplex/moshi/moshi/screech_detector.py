"""Screech detection for Moshi/PersonaPlex inference.

Detects degenerate audio generation (screeching) that occurs when the model
gets stuck emitting PAD text tokens while the depformer generates garbage
audio. Two complementary spectral signals are tracked (OR logic):

1. **Mid-band (1-3kHz) energy ratio** — catches the early onset of screeching
   before it reaches full high-frequency intensity. Triggers when 8 consecutive
   PAD frames exceed 0.25 mid-band ratio (~0.5s after onset starts).

2. **High-frequency (>4kHz) energy ratio** — catches full-blown screeching.
   Triggers when 3 consecutive PAD frames exceed 0.40 HF ratio.

Either signal firing triggers recovery. The caller should inject a recovery
token (e.g. "OK") via inject_continuation().

Also tracks text repetition to detect degenerate loops.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# Default PAD token id in Moshi's text vocabulary.
PAD_TOKEN_ID = 3
EPAD_TOKEN_ID = 0

# Defaults tuned from eval_venting_work.wav analysis.
DEFAULT_MID_THRESHOLD = 0.25       # 1-3kHz band ratio — catches early onset
DEFAULT_MID_CONSEC_FRAMES = 8      # ~0.64s of elevated mid-band
DEFAULT_HF_THRESHOLD = 0.40       # >4kHz ratio — catches full screech
DEFAULT_HF_CONSEC_FRAMES = 3      # ~0.24s of high-frequency
DEFAULT_COOLDOWN_FRAMES = 6       # ~0.5s — just enough for "OK" to play out


class ScreechDetector:
    """Stateful per-bot screech detector for live inference.

    Usage::

        detector = ScreechDetector()

        # In your frame loop:
        if detector.step(text_token_id, pcm_frame, sample_rate):
            # Screeching detected — inject recovery token
            lm_gen.inject_continuation(tokenizer.encode("OK"))
            detector.reset()
    """

    def __init__(
        self,
        mid_threshold: float = DEFAULT_MID_THRESHOLD,
        mid_consec_frames: int = DEFAULT_MID_CONSEC_FRAMES,
        hf_threshold: float = DEFAULT_HF_THRESHOLD,
        hf_consec_frames: int = DEFAULT_HF_CONSEC_FRAMES,
        cooldown_frames: int = DEFAULT_COOLDOWN_FRAMES,
    ):
        self.mid_threshold = mid_threshold
        self.mid_consec_frames = mid_consec_frames
        self.hf_threshold = hf_threshold
        self.hf_consec_frames = hf_consec_frames
        self.cooldown_frames = cooldown_frames

        self._mid_consec = 0
        self._hf_consec = 0
        self._cooldown_remaining = 0
        self._total_triggers = 0
        self._recent_tokens: list[str] = []

    def step(
        self,
        text_token_id: int,
        pcm_frame: Optional[np.ndarray] = None,
        sample_rate: int = 24000,
    ) -> bool:
        """Process one frame and return True if screeching is detected.

        Dual spectral detection: triggers when EITHER mid-band or HF ratio
        exceeds its threshold for the required number of consecutive PAD frames.

        Args:
            text_token_id: The text token id for this frame (3 = PAD).
            pcm_frame: Raw PCM audio for this frame, shape (N,) or (1, 1, N).
            sample_rate: Audio sample rate (for FFT frequency mapping).

        Returns:
            True if screeching is detected and recovery should be injected.
        """
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False

        is_pad = text_token_id in (PAD_TOKEN_ID, EPAD_TOKEN_ID)

        if not is_pad:
            self._mid_consec = 0
            self._hf_consec = 0
            return False

        if pcm_frame is None:
            return False

        # Compute both band ratios in one FFT pass
        mid_ratio, hf_ratio = self._compute_band_ratios(pcm_frame, sample_rate)

        if mid_ratio > self.mid_threshold:
            self._mid_consec += 1
        else:
            self._mid_consec = 0

        if hf_ratio > self.hf_threshold:
            self._hf_consec += 1
        else:
            self._hf_consec = 0

        if (self._mid_consec >= self.mid_consec_frames
                or self._hf_consec >= self.hf_consec_frames):
            self._trigger()
            return True

        return False

    def step_text_repetition(self, text_token: str) -> bool:
        """Track text tokens for repetition loop detection.

        Returns True if a degenerate repetition loop is detected
        (e.g., same phrase repeated 5+ times).
        """
        if text_token in ("PAD", "EPAD", "BOS", "EOS"):
            return False

        self._recent_tokens.append(text_token)
        if len(self._recent_tokens) > 200:
            self._recent_tokens = self._recent_tokens[-200:]

        tokens = self._recent_tokens
        if len(tokens) < 40:
            return False

        ngram_size = 8
        last_ngram = tuple(tokens[-ngram_size:])
        count = 0
        for i in range(len(tokens) - ngram_size):
            if tuple(tokens[i:i + ngram_size]) == last_ngram:
                count += 1

        return count >= 5

    def reset(self):
        """Reset detector state after injection."""
        self._mid_consec = 0
        self._hf_consec = 0
        self._cooldown_remaining = self.cooldown_frames

    @property
    def total_triggers(self) -> int:
        return self._total_triggers

    def _trigger(self):
        self._total_triggers += 1
        self._cooldown_remaining = self.cooldown_frames
        self._mid_consec = 0
        self._hf_consec = 0

    def _compute_band_ratios(self, pcm: np.ndarray, sr: int) -> tuple[float, float]:
        """Compute mid-band (1-3kHz) and HF (>4kHz) energy ratios in one FFT."""
        pcm = np.asarray(pcm).flatten()
        if len(pcm) == 0:
            return (0.0, 0.0)
        fft_mag = np.abs(np.fft.rfft(pcm))
        freqs = np.fft.rfftfreq(len(pcm), 1.0 / sr)
        total = fft_mag.sum()
        if total < 1e-12:
            return (0.0, 0.0)
        mid_energy = fft_mag[(freqs > 1000) & (freqs <= 3000)].sum()
        hf_energy = fft_mag[freqs > 4000].sum()
        return (float(mid_energy / total), float(hf_energy / total))
