"""Chromagram extraction for polyphonic (chord) recognition.

A strummed chord has several fundamentals sounding at once, so period-based pitch
detection doesn't apply. Instead we fold the spectrum onto the twelve pitch classes,
discarding octave information — which is exactly the right thing to discard, because a
chord is defined by its pitch classes regardless of how it's voiced on the neck.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.signal import find_peaks

CHROMA_WINDOW = 8192
CHROMA_HOP = 2048
#: Zero-padding sharpens the interpolation onto pitch classes at the low end, where
#: adjacent semitones are only a few hertz apart.
CHROMA_FFT_SIZE = 16384

MIN_FREQ = 75.0
MAX_FREQ = 2000.0


def _build_pitch_class_matrix(
    fft_size: int, sample_rate: int, min_freq: float, max_freq: float
) -> np.ndarray:
    """Map FFT bins onto pitch classes, as a ``(12, n_bins)`` matrix.

    Each bin's energy is split between the two nearest pitch classes in proportion to
    where it falls between them, so a slightly detuned string still lands mostly on the
    right class instead of being snapped to a neighbour.
    """
    n_bins = fft_size // 2 + 1
    freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
    matrix = np.zeros((12, n_bins))

    usable = (freqs >= min_freq) & (freqs <= max_freq)
    indices = np.flatnonzero(usable)
    if indices.size == 0:
        return matrix

    midi = 69.0 + 12.0 * np.log2(freqs[indices] / 440.0)
    lower = np.floor(midi).astype(int)
    frac = midi - lower

    matrix[lower % 12, indices] += 1.0 - frac
    matrix[(lower + 1) % 12, indices] += frac
    return matrix


class ChromaAnalyser:
    """Produces smoothed, normalised chroma vectors from audio frames."""

    def __init__(
        self,
        sample_rate: int,
        *,
        window: int = CHROMA_WINDOW,
        fft_size: int = CHROMA_FFT_SIZE,
        min_freq: float = MIN_FREQ,
        max_freq: float = MAX_FREQ,
        smoothing_frames: int = 5,
        rms_threshold: float = 0.005,
    ) -> None:
        self.sample_rate = sample_rate
        self.window = window
        self.fft_size = fft_size
        self.rms_threshold = rms_threshold
        self._hann = np.hanning(window)
        self._matrix = _build_pitch_class_matrix(fft_size, sample_rate, min_freq, max_freq)
        self._history: deque[np.ndarray] = deque(maxlen=smoothing_frames)

    def reset(self) -> None:
        self._history.clear()

    def analyse(self, frame: np.ndarray) -> np.ndarray | None:
        """Return a smoothed unit-norm chroma vector, or ``None`` if the input is quiet.

        ``frame`` may be longer than the analysis window, in which case the most recent
        window's worth is used — the caller shares one buffer read with pitch detection.
        """
        if len(frame) < self.window:
            return None
        segment = frame[-self.window :].astype(np.float64)

        rms = float(np.sqrt(np.mean(segment**2)))
        if rms < self.rms_threshold:
            self._history.clear()
            return None

        spectrum = np.abs(np.fft.rfft(segment * self._hann, self.fft_size))
        # Work with energy rather than amplitude: it weights the strong partials that
        # actually define the chord over the wash of low-level noise between them.
        chroma = self._matrix @ (spectrum**2)

        norm = np.linalg.norm(chroma)
        if norm <= 0:
            return None
        self._history.append(chroma / norm)

        # Median over recent frames rides out the noisy attack of a strum without
        # smearing across an actual chord change the way a long average would.
        smoothed = np.median(np.array(self._history), axis=0)
        norm = np.linalg.norm(smoothed)
        return smoothed / norm if norm > 0 else None

    def bass_pitch_class(
        self, frame: np.ndarray, *, max_freq: float = 400.0, relative_floor: float = 0.25
    ) -> int | None:
        """Pitch class of the lowest clearly-sounding partial.

        Chords that share most of their notes — C major and A minor differ by one of
        three — are separated mainly by which note is in the bass, so this breaks ties
        that chroma alone cannot.

        The *lowest* significant peak is what's wanted, not the loudest: in a strummed
        G the open B string is often louder than the low G beneath it, so taking the
        maximum would report the wrong root. Anything within ``relative_floor`` of the
        strongest peak in the band counts as sounding.
        """
        if len(frame) < self.window:
            return None
        segment = frame[-self.window :].astype(np.float64)
        if float(np.sqrt(np.mean(segment**2))) < self.rms_threshold:
            return None

        spectrum = np.abs(np.fft.rfft(segment * self._hann, self.fft_size))
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        band = np.flatnonzero((freqs >= MIN_FREQ) & (freqs <= max_freq))
        if band.size == 0:
            return None

        magnitudes = spectrum[band]
        peak = magnitudes.max()
        if peak <= 0:
            return None

        # Look for local maxima rather than any bin above the floor: a strong partial
        # has skirts spreading several bins either side, and the lower skirt would
        # otherwise be picked as a "lower note" a semitone or two flat of the real one.
        peaks, _ = find_peaks(magnitudes, height=relative_floor * peak)
        if peaks.size == 0:
            return None
        lowest = band[peaks[0]]
        return int(round(69.0 + 12.0 * np.log2(freqs[lowest] / 440.0))) % 12
