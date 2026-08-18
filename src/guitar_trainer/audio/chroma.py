"""Chromagram extraction for polyphonic (chord) recognition.

A strummed chord has several fundamentals sounding at once, so period-based pitch
detection doesn't apply. Instead we fold the spectrum onto the twelve pitch classes,
discarding octave information — which is exactly the right thing to discard, because a
chord is defined by its pitch classes regardless of how it's voiced on the neck.

What gets folded is spectral *peaks* with compressed magnitudes, not raw energy.
Folding energy (``|X|**2``) fails badly on a real guitar: string loudness spans well
over 20dB within one chord (an open low E's fundamental measured at 0.3-9% of the
loudest partial on real recordings), and squaring turns that spread into a chroma
vector that is effectively one-hot on whichever partial dominates — every other
chord tone's evidence is numerically destroyed before matching ever sees it. The
peaks themselves, in magnitude, still spell out the chord (measured on real
recordings: an E major's sustain peaks are literally B2/E3/G#3/B3), so the honest
vector keeps exactly those: peak bins only (the noise floor between them excluded),
each weighted by a compressed relative magnitude so weak-but-real strings count.
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

#: Peaks are taken from [MIN_FREQ, this] — the band where guitar chord voicings put
#: their string fundamentals (up to F#4/G4 for open D/Dsus4 shapes). Above it, higher
#: harmonics add spill classes (a 7th harmonic lands a minor 7th up, feeding X7
#: templates) without adding information the fundamentals don't already carry.
PEAK_MAX_FREQ = 500.0
#: A peak must reach this fraction of the band's strongest peak to count at all.
#: Below it is the noise floor / FFT leakage skirt, not a string.
PEAK_FLOOR = 0.04
#: Peak weight = ((rel - floor)/(1 - floor)) ** this. Subtracting the floor first
#: means dust that barely clears it compresses to ~0 instead of being inflated; the
#: exponent then lifts genuinely weak strings toward the strong ones. Tuned against
#: the real chord fixtures: full 9/9 recognition holds across 0.35-0.40 (and most of
#: 0.33-0.45); energy folding (effectively power 2 with no floor) recognised 0/9.
PEAK_COMPRESSION = 0.35

#: single_series_pitch_class: how far below MIN_FREQ an implied fundamental may sit
#: (drop-D low string is 73.4Hz; anything below ~62Hz is not this instrument).
SERIES_MIN_F0 = 62.0
#: Relative mistuning allowed between a peak and an exact harmonic. Real strings run
#: slightly sharp in their upper partials (inharmonicity), well inside 3%.
SERIES_TOLERANCE = 0.03
#: Fraction of total strong-peak weight one series must explain to claim the frame.
SERIES_MIN_FRACTION = 0.93
#: Only peaks at least this loud (raw, relative to the band peak) participate in the
#: series test — floor-level dust must not be able to break an otherwise clean fit.
SERIES_STRONG_PEAK = 0.10
#: When the claimed fundamental itself is missing (see the method docstring), weight
#: at chord-tone harmonic numbers (5, 7, ...) above this share vetoes the claim.
SERIES_CHORD_TONE_SHARE = 0.12


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
        freqs = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        self._peak_band = np.flatnonzero((freqs >= min_freq) & (freqs <= PEAK_MAX_FREQ))
        self._peak_band_freqs = freqs[self._peak_band]

    def reset(self) -> None:
        self._history.clear()

    def _band_peaks(self, segment: np.ndarray):
        """Spectral peaks in the fundamental band: (bin indices, freqs, weights).

        Weights are soft-floored compressed relative magnitudes — see PEAK_FLOOR /
        PEAK_COMPRESSION. Returns ``(None, None, None)`` when nothing peaks.
        """
        magnitude = np.abs(np.fft.rfft(segment * self._hann, self.fft_size))
        band_mag = magnitude[self._peak_band]
        top = band_mag.max() if band_mag.size else 0.0
        if top <= 0:
            return None, None, None
        peaks, _ = find_peaks(band_mag, height=PEAK_FLOOR * top)
        if peaks.size == 0:
            return None, None, None
        rel = band_mag[peaks] / top
        weights = ((rel - PEAK_FLOOR) / (1.0 - PEAK_FLOOR)) ** PEAK_COMPRESSION
        return self._peak_band[peaks], self._peak_band_freqs[peaks], weights

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

        bins, _, weights = self._band_peaks(segment)
        if bins is None:
            return None
        masked = np.zeros(self.fft_size // 2 + 1)
        masked[bins] = weights
        # Fold only the peak bins; the proportional split in the matrix still
        # absorbs detuning, but the floor between peaks contributes nothing.
        chroma = self._matrix @ masked

        norm = np.linalg.norm(chroma)
        if norm <= 0:
            return None
        self._history.append(chroma / norm)

        # Median over recent frames rides out the noisy attack of a strum without
        # smearing across an actual chord change the way a long average would.
        smoothed = np.median(np.array(self._history), axis=0)
        norm = np.linalg.norm(smoothed)
        return smoothed / norm if norm > 0 else None

    def single_series_pitch_class(self, frame: np.ndarray) -> tuple[int, float] | None:
        """``(pitch_class, explained_fraction)`` if one harmonic series explains the
        frame, else ``None``.

        If every strong peak fits one harmonic series with a fundamental in the
        instrument's range, this is a single ringing note, not a chord — regardless
        of how chord-like the folded pitch classes look. That distinction cannot be
        made after folding: a low string whose fundamental is buried (the real E2
        sustain reads as one huge B — its own 3rd harmonic) folds to the same
        classes as genuine chords do. The frequency-domain series structure is what
        still tells them apart, so it is checked here, pre-fold.

        Only strong peaks participate (``SERIES_STRONG_PEAK``): floor dust must not
        break the fit. And when the fundamental itself is not observed — the weak
        low strings that motivate this — the claim is only believed if the implied
        harmonic numbers stay in the root/fifth family {1,2,3,4,6,8,12}: a major
        triad voiced as harmonics 2,3,4,5 of a sub-octave root is otherwise
        indistinguishable from a single note, and the tell is that a real played
        third (k=5) carries far more weight than a decayed 5th harmonic ever does.
        """
        if len(frame) < self.window:
            return None
        segment = frame[-self.window :].astype(np.float64)
        if float(np.sqrt(np.mean(segment**2))) < self.rms_threshold:
            return None
        bins, freqs, weights = self._band_peaks(segment)
        if bins is None:
            return None

        rel = weights ** (1.0 / PEAK_COMPRESSION) * (1.0 - PEAK_FLOOR) + PEAK_FLOOR
        strong = rel >= SERIES_STRONG_PEAK
        freqs, weights = freqs[strong], weights[strong]
        if freqs.size == 0:
            return None

        total = weights.sum()
        lowest = freqs.min()
        for divisor in (1, 2, 3, 4):
            f0 = lowest / divisor
            if f0 < SERIES_MIN_F0:
                break
            k = np.round(freqs / f0)
            fits = (k >= 1) & (np.abs(freqs - k * f0) <= SERIES_TOLERANCE * k * f0)
            explained = weights[fits].sum() / total
            if explained < SERIES_MIN_FRACTION:
                continue
            if divisor > 1:
                chord_tone_k = fits & ~np.isin(k, (1, 2, 3, 4, 6, 8, 12))
                if weights[chord_tone_k].sum() > SERIES_CHORD_TONE_SHARE * total:
                    continue
            pitch_class = int(round(69.0 + 12.0 * np.log2(f0 / 440.0))) % 12
            return pitch_class, float(explained)
        return None

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
