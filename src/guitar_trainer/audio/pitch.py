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

from ..core.notes import cents_off, freq_to_midi, midi_to_freq, nearest_note, note_name

#: Guitar range with headroom: below open low E, up to the 24th fret of the high E.
MIN_FREQ = 70.0
MAX_FREQ = 1400.0

DEFAULT_WINDOW = 4096
DEFAULT_HOP = 1024
#: Below this, YIN's normalised difference indicates a periodic signal.
DEFAULT_THRESHOLD = 0.12

#: A longer-period (lower-frequency) candidate is preferred over the one first picked
#: only if its CMND value is at least this much better — see _prefer_lower_octave.
#: Tuned against real recordings (tests/fixtures/audio/notes/): genuine fundamentals
#: won by margins of ~3-10x there, while a near-tie late in a note's decay (where the
#: true fundamental had genuinely faded, not just been mis-picked) sat around 0.9x.
#: 0.75 catches the former without forcing the latter.
OCTAVE_PREFERENCE_RATIO = 0.75
#: How many integer multiples of the picked tau to check for a better-fitting
#: fundamental. 6 matches the harmonic count used elsewhere (chroma, chord templates).
OCTAVE_SEARCH_MULTIPLES = 6

#: How much of a frame's spectral energy must sit at multiples of the detected
#: fundamental for it to count as an instrument tone. A clean plucked string
#: concentrates almost all its energy there (ratio > 0.98 across the neck, even at
#: 10dB SNR); a keyboard click or a door slam spreads it across the spectrum instead
#: (ratio < 0.52 for every synthesised click that fools YIN at all). 0.6 sits well
#: clear of both populations. Tuned in tests/test_pitch.py::TestPurityFilter — see
#: that suite before changing it.
DEFAULT_MIN_HARMONIC_RATIO = 0.6
#: Harmonics above this are not counted — negligible on a guitar, and counting them
#: just adds room for noise to be miscounted as "explained".
HARMONIC_CHECK_MAX_FREQ = 6000.0
HARMONIC_CHECK_HARMONICS = 12


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
    harmonic_ratio: float = 1.0
    """Fraction of spectral energy explained by this pitch's harmonic series.
    High for a clean instrument tone, low for noise — see DEFAULT_MIN_HARMONIC_RATIO."""

    @property
    def pitch_class(self) -> int:
        return self.midi % 12

    @property
    def note_only(self) -> str:
        """Name without the octave, e.g. ``"A"``."""
        return note_name(self.midi, with_octave=False)

    @classmethod
    def from_freq(
        cls, freq: float, confidence: float, rms: float, harmonic_ratio: float = 1.0
    ) -> PitchResult:
        midi, cents = nearest_note(freq)
        return cls(
            freq=freq,
            midi_float=freq_to_midi(freq),
            midi=midi,
            name=note_name(midi),
            cents=cents,
            confidence=confidence,
            rms=rms,
            harmonic_ratio=harmonic_ratio,
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
        min_harmonic_ratio: float = DEFAULT_MIN_HARMONIC_RATIO,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.min_harmonic_ratio = min_harmonic_ratio
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

        tau = self._pick_tau(cmnd, frame)
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

        harmonic_ratio = self._harmonic_energy_ratio(frame, freq)
        if harmonic_ratio < self.min_harmonic_ratio:
            return None

        confidence = float(np.clip(1.0 - cmnd[tau], 0.0, 1.0))
        return PitchResult.from_freq(freq, confidence, rms, harmonic_ratio)

    def _harmonic_energy_ratio(
        self, frame: np.ndarray, freq: float, *, n_harmonics: int = HARMONIC_CHECK_HARMONICS
    ) -> float:
        """Fraction of spectral energy that sits at multiples of ``freq``.

        This is what distinguishes "an instrument was played" from "something loud
        happened near the microphone". A plucked string concentrates nearly all of its
        energy at the fundamental and its harmonics; a keyboard click, a knock, or a
        cough spreads energy across the spectrum instead, and scores low here even
        though it might still be loud enough to pass the RMS gate and even periodic
        enough to give YIN a confident-looking dip.

        A sustained sung or spoken vowel is a genuine exception: voice is also a
        harmonic source, so this ratio does not reliably tell it apart from a guitar
        note. It targets percussive and noise-like false positives specifically.
        """
        windowed = frame.astype(np.float64) * np.hanning(len(frame))
        power = np.abs(np.fft.rfft(windowed)) ** 2
        bin_width = self.sample_rate / len(frame)

        upper = min(len(power), int(HARMONIC_CHECK_MAX_FREQ / bin_width) + 1)
        lower = max(1, int(self.min_freq / bin_width))
        if upper <= lower:
            return 0.0
        total = power[lower:upper].sum()
        if total <= 0:
            return 0.0

        harmonic_energy = 0.0
        nyquist = self.sample_rate / 2.0
        for k in range(1, n_harmonics + 1):
            target = freq * k
            if target >= HARMONIC_CHECK_MAX_FREQ or target >= nyquist:
                break
            center = target / bin_width
            lo = max(0, int(center) - 2)
            hi = min(len(power), int(center) + 3)
            harmonic_energy += power[lo:hi].sum()

        return float(min(1.0, harmonic_energy / total))

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

    def _pick_tau(self, cmnd: np.ndarray, frame: np.ndarray) -> int | None:
        """First dip below the threshold, descended to its local minimum, then
        checked against longer-period multiples in case one of them is actually the
        true fundamental.

        Taking the *first* qualifying dip rather than the global minimum is what
        avoids octave-down errors (locking onto a subharmonic); walking to the bottom
        of that dip avoids landing on its leading edge. But scanning short-tau-first
        has the opposite failure mode too: right after a pick attack, a strong upper
        partial can genuinely look more periodic within one short analysis window
        than the true, lower fundamental, and get accepted as the first dip before
        the fundamental's own (better) dip is ever reached. `_prefer_lower_octave`
        catches that — real recordings, not just theory, showed a fundamental's CMND
        several times lower than the harmonic's at the exact frame this happened
        (tests/fixtures/audio/notes/, see AGENTS.md).
        """
        search = cmnd[self.min_tau :]
        below = np.flatnonzero(search < self.threshold)

        if below.size:
            tau = int(below[0]) + self.min_tau
            # Descend to the bottom of this dip.
            while tau + 1 < len(cmnd) and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            return self._prefer_lower_octave(cmnd, tau, frame)

        # Nothing crossed the threshold. Fall back to the global minimum, which lets
        # borderline-noisy signals still register, gated by the confidence value.
        tau = int(np.argmin(search)) + self.min_tau
        return tau if cmnd[tau] < 0.6 else None

    def _prefer_lower_octave(
        self,
        cmnd: np.ndarray,
        tau: int,
        frame: np.ndarray,
        *,
        max_multiple: int = OCTAVE_SEARCH_MULTIPLES,
    ) -> int:
        """Check integer multiples of ``tau`` (longer periods = lower frequencies)
        for a noticeably better-fitting fundamental than the one just picked.

        A genuine fundamental that's actually present wins this comparison by a wide
        margin (3-10x lower CMND in the recordings that motivated this); noise-level
        differences between nearby candidates don't clear `OCTAVE_PREFERENCE_RATIO`,
        so this doesn't destabilise an already-correct pick.

        Two guards, both found by letting this run against real recordings and the
        full synthetic suite rather than reasoning it through on paper alone:

        - The candidate must clear the absolute periodicity threshold on its own,
          not just be relatively better than the original — in a quiet decay tail
          where neither candidate is a clean periodic signal, "relatively less bad"
          can win out over noise.
        - The candidate must have genuine spectral energy at its own frequency
          (`_has_energy_at`), not just a low CMND value. CMND's normalisation (it
          divides by a *running mean* that grows with tau) can produce deceptively
          low values at large tau with no real periodicity behind them at all —
          without this check, the search would occasionally "win" with a spurious
          low-frequency candidate that then fails every check downstream in
          `detect()`, turning a correct detection into no detection whatsoever
          (worse than doing nothing). This is what the first version of this method
          did, caught by the synthetic suite going from 0 to 43 failures.
        """
        best_tau, best_value = tau, cmnd[tau]
        for k in range(2, max_multiple + 1):
            candidate = tau * k
            if candidate >= len(cmnd) - 1:
                break
            # The true dip may sit a sample or two either side of the exact multiple.
            c = candidate
            while c + 1 < len(cmnd) and cmnd[c + 1] < cmnd[c]:
                c += 1
            while c - 1 > 0 and cmnd[c - 1] < cmnd[c]:
                c -= 1
            if cmnd[c] >= self.threshold or cmnd[c] >= best_value * OCTAVE_PREFERENCE_RATIO:
                continue
            refined = _parabolic_refine(cmnd, c)
            if refined <= 0:
                continue
            candidate_freq = self.sample_rate / refined
            if not self.min_freq <= candidate_freq <= self.max_freq:
                continue
            if self._has_energy_at(frame, candidate_freq):
                best_tau, best_value = c, cmnd[c]
        return best_tau


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


#: How much a new (median-filtered) estimate moves the smoothed value each frame.
#: Tuned in tests/test_pitch.py::TestPitchSmoother against synthetic jitter: this
#: cuts steady-state noise from ~15 cents std down to ~3, without adding more than a
#: couple of frames of lag on real pitch movement.
SMOOTHER_ALPHA = 0.2
#: A jump bigger than this snaps instantly instead of sliding through it — the
#: difference between "this note bent" and "a different string is now ringing".
SMOOTHER_RESET_CENTS = 50.0
SMOOTHER_MEDIAN_WINDOW = 3


class PitchSmoother:
    """Smooths a stream of per-frame pitch estimates for live display.

    Raw single-frame YIN output has real frame-to-frame jitter even on a clean,
    steady note — from windowing/estimation noise, from the string's own decay and
    micro-vibrato, and occasionally a single-frame outlier. None of that is something
    NoteGate helps with here: NoteGate exists to decide *whether* a note qualifies as
    a scored answer, not to steady what's continuously shown on a live meter, and it
    only reports once a note has already settled rather than every frame.

    A median-of-3 pre-filter kills isolated single-frame outliers almost entirely
    (spikes that would otherwise show as a wild swing on the needle); an exponential
    moving average on top removes the residual smaller jitter. Both operate on the
    MIDI (log-frequency) axis, since cents — and human pitch perception — are linear
    there, not in Hz. A jump past ``reset_threshold_cents`` snaps immediately rather
    than sliding through intervening pitches, so switching strings feels instant
    rather than gliding.
    """

    def __init__(
        self,
        *,
        alpha: float = SMOOTHER_ALPHA,
        reset_threshold_cents: float = SMOOTHER_RESET_CENTS,
        median_window: int = SMOOTHER_MEDIAN_WINDOW,
    ) -> None:
        self.alpha = alpha
        self.reset_threshold_cents = reset_threshold_cents
        self._median_window = median_window
        self._buffer: deque[float] = deque(maxlen=median_window)
        self._state: float | None = None
        self._last_extras: PitchResult | None = None

    def reset(self) -> None:
        self._buffer.clear()
        self._state = None
        self._last_extras = None

    def push(self, result: PitchResult | None) -> PitchResult | None:
        """Feed one raw frame. Returns a smoothed result, or ``None`` on silence."""
        if result is None:
            self.reset()
            return None

        self._buffer.append(result.midi_float)
        self._last_extras = result
        if len(self._buffer) < self._median_window:
            # Not enough history yet to median-filter; pass the raw estimate through
            # rather than waiting, so there's no dead silence at the very start of a
            # note before smoothing kicks in a couple of frames later.
            candidate = result.midi_float
        else:
            candidate = float(np.median(self._buffer))

        if self._state is None or abs((candidate - self._state) * 100) > self.reset_threshold_cents:
            self._state = candidate
        else:
            self._state += self.alpha * (candidate - self._state)

        return PitchResult.from_freq(
            midi_to_freq(self._state), result.confidence, result.rms, result.harmonic_ratio
        )
