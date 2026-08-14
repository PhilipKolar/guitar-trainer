"""Monophonic pitch detection using the YIN algorithm.

YIN (de Cheveigné & Kawahara, 2002) tracks the fundamental even when it is weaker than
its harmonics, which matters on guitar: a plucked low E often has more energy in the
second harmonic than in the fundamental, and a plain autocorrelation peak-picker would
report the octave above.

The difference function is computed via FFT autocorrelation rather than the textbook
O(N^2) double loop, which keeps a 4096-sample window well inside a millisecond.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..core.notes import cents_off, freq_to_midi, nearest_note, note_name

#: Guitar range with headroom: below open low E, up to the 24th fret of the high E.
MIN_FREQ = 70.0
MAX_FREQ = 1400.0

DEFAULT_WINDOW = 4096
DEFAULT_HOP = 1024
#: Below this, YIN's normalised difference indicates a periodic signal.
DEFAULT_THRESHOLD = 0.12


@dataclass(frozen=True)
class PitchResult:
    """A single frame of pitch analysis."""

    freq: float
    midi_float: float
    midi: int
    name: str
    cents: float
    confidence: float
    rms: float

    @property
    def pitch_class(self) -> int:
        return self.midi % 12

    @property
    def note_only(self) -> str:
        """Name without the octave, e.g. ``"A"``."""
        return note_name(self.midi, with_octave=False)

    @classmethod
    def from_freq(cls, freq: float, confidence: float, rms: float) -> PitchResult:
        midi, cents = nearest_note(freq)
        return cls(
            freq=freq,
            midi_float=freq_to_midi(freq),
            midi=midi,
            name=note_name(midi),
            cents=cents,
            confidence=confidence,
            rms=rms,
        )


def _difference_function(x: np.ndarray, max_tau: int) -> np.ndarray:
    """YIN's squared-difference function d(tau), computed through the FFT.

    d(tau) = sum_j (x_j - x_{j+tau})^2
           = r(0) + r_tau(0) - 2 r(tau)

    where r(tau) is the autocorrelation and the two power terms are running sums of
    squares over the leading and trailing windows.
    """
    n = len(x)
    x = x.astype(np.float64, copy=False)

    # Autocorrelation via the Wiener-Khinchin theorem, zero-padded to avoid wraparound.
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, size)
    autocorr = np.fft.irfft(spectrum * np.conjugate(spectrum), size)[:max_tau]

    # Running power sums. cumsum[k] is the energy of the first k samples.
    power = np.concatenate(([0.0], np.cumsum(x * x)))
    taus = np.arange(max_tau)
    # Leading window x[0 : n-tau], trailing window x[tau : n].
    leading = power[n - taus] - power[0]
    trailing = power[n] - power[taus]

    return leading + trailing - 2.0 * autocorr


def _cumulative_mean_normalised(diff: np.ndarray) -> np.ndarray:
    """YIN step 3: normalise d(tau) by its running mean.

    This is what removes the "tau=0 is always best" problem and makes a single absolute
    threshold meaningful across signals of any loudness.
    """
    cmnd = np.ones_like(diff)
    running = np.cumsum(diff[1:])
    taus = np.arange(1, len(diff))
    with np.errstate(divide="ignore", invalid="ignore"):
        cmnd[1:] = diff[1:] * taus / running
    cmnd[~np.isfinite(cmnd)] = 1.0
    return cmnd


def _parabolic_refine(values: np.ndarray, tau: int) -> float:
    """Fit a parabola through the minimum and its neighbours for sub-sample precision.

    Without this, resolution at the top of the guitar range is roughly 30 cents per
    sample of period — far too coarse for a tuner.
    """
    if tau <= 0 or tau >= len(values) - 1:
        return float(tau)
    a, b, c = values[tau - 1], values[tau], values[tau + 1]
    denom = a + c - 2.0 * b
    if denom == 0.0:
        return float(tau)
    return tau + 0.5 * (a - c) / denom


class YinDetector:
    """Stateless-per-call YIN detector configured for a fixed sample rate."""

    def __init__(
        self,
        sample_rate: int,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        min_freq: float = MIN_FREQ,
        max_freq: float = MAX_FREQ,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_tau = max(2, int(sample_rate / max_freq))
        self.max_tau = int(sample_rate / min_freq) + 1

    def detect(self, frame: np.ndarray) -> PitchResult | None:
        """Analyse one window. Returns ``None`` when nothing periodic is found."""
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        if rms < 1e-6:
            return None

        max_tau = min(self.max_tau, len(frame) - 1)
        if max_tau <= self.min_tau:
            return None

        diff = _difference_function(frame, max_tau)
        cmnd = _cumulative_mean_normalised(diff)

        tau = self._pick_tau(cmnd)
        if tau is None:
            return None

        refined = _parabolic_refine(cmnd, tau)
        if refined <= 0:
            return None

        freq = self.sample_rate / refined
        if not self.min_freq <= freq <= self.max_freq:
            return None

        if not self._has_energy_at(frame, freq):
            return None

        confidence = float(np.clip(1.0 - cmnd[tau], 0.0, 1.0))
        return PitchResult.from_freq(freq, confidence, rms)

    def _has_energy_at(self, frame: np.ndarray, freq: float, *, floor: float = 0.02) -> bool:
        """Reject periods that are a subharmonic of the real fundamental.

        A signal is periodic at every subharmonic of its true pitch, so YIN can settle
        on a period two or three times too long and report a note an octave or a
        twelfth below what was played. The tell is spectral: at a genuine fundamental
        there is a peak, whereas at a spurious subharmonic there is nothing. The floor
        is deliberately lenient — a guitar's fundamental is often much quieter than its
        harmonics, and rejecting those would be worse than the error being guarded
        against.
        """
        windowed = frame.astype(np.float64) * np.hanning(len(frame))
        magnitude = np.abs(np.fft.rfft(windowed))
        bin_width = self.sample_rate / len(frame)

        # Compare against the whole spectrum, not just the guitar band: content that
        # lives entirely above the band (hiss, a stray harmonic-only signal) must not
        # be able to set a low reference and let its own leakage pass as a fundamental.
        peak = magnitude[1:].max() if len(magnitude) > 1 else 0.0
        if peak <= 0.0:
            return False

        # Widen by a couple of bins to absorb leakage and the coarse low-end resolution.
        target = int(round(freq / bin_width))
        lo = max(0, target - 2)
        hi = min(len(magnitude), target + 3)
        return bool(magnitude[lo:hi].max() >= floor * peak)

    def _pick_tau(self, cmnd: np.ndarray) -> int | None:
        """First dip below the threshold, descended to its local minimum.

        Taking the *first* qualifying dip rather than the global minimum is what avoids
        octave-down errors; walking to the bottom of that dip avoids landing on its
        leading edge.
        """
        search = cmnd[self.min_tau :]
        below = np.flatnonzero(search < self.threshold)

        if below.size:
            tau = int(below[0]) + self.min_tau
            # Descend to the bottom of this dip.
            while tau + 1 < len(cmnd) and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            return tau

        # Nothing crossed the threshold. Fall back to the global minimum, which lets
        # borderline-noisy signals still register, gated by the confidence value.
        tau = int(np.argmin(search)) + self.min_tau
        return tau if cmnd[tau] < 0.6 else None


class NoteGate:
    """Turns a stream of per-frame results into stable note events.

    A guitar pluck starts with a burst of inharmonic attack noise, and fretting hand
    movement produces short bogus pitches. Requiring the pitch to hold steady for a few
    consecutive frames before accepting it is what makes practice scoring feel correct
    rather than twitchy.
    """

    def __init__(
        self,
        *,
        rms_threshold: float = 0.01,
        confidence_threshold: float = 0.5,
        stable_frames: int = 3,
        tolerance_cents: float = 40.0,
    ) -> None:
        self.rms_threshold = rms_threshold
        self.confidence_threshold = confidence_threshold
        self.stable_frames = stable_frames
        self.tolerance_cents = tolerance_cents
        self._history: deque[PitchResult] = deque(maxlen=stable_frames)
        self._current: PitchResult | None = None

    @property
    def current(self) -> PitchResult | None:
        """The note currently held, or ``None`` if the gate is closed."""
        return self._current

    def reset(self) -> None:
        self._history.clear()
        self._current = None

    def push(self, result: PitchResult | None) -> PitchResult | None:
        """Feed one frame. Returns the stable note, or ``None`` while unsettled."""
        if (
            result is None
            or result.rms < self.rms_threshold
            or result.confidence < self.confidence_threshold
        ):
            self._history.clear()
            self._current = None
            return None

        self._history.append(result)
        if len(self._history) < self.stable_frames:
            return None

        first = self._history[0]
        if any(
            abs(cents_off(r.freq, first.midi_float)) > self.tolerance_cents
            for r in self._history
        ):
            # Still moving — keep only the newest frame and start counting again.
            newest = self._history[-1]
            self._history.clear()
            self._history.append(newest)
            self._current = None
            return None

        # Report the median frame, which rejects a single outlier within the window.
        stable = sorted(self._history, key=lambda r: r.midi_float)[len(self._history) // 2]
        self._current = stable
        return stable
