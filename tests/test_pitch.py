import numpy as np
import pytest

import synth
from guitar_trainer.audio.pitch import NoteGate, PitchResult, PitchSmoother, YinDetector
from guitar_trainer.core.notes import STANDARD, midi_to_freq, name_to_midi, note_name

SR = synth.SAMPLE_RATE
WINDOW = 4096

# Every note on a 22-fret standard-tuned neck, low E2 to D6.
ALL_GUITAR_MIDI = list(range(name_to_midi("E2"), name_to_midi("E4") + 22 + 1))


@pytest.fixture(scope="module")
def detector():
    return YinDetector(SR)


def error_cents(result: PitchResult, expected_freq: float) -> float:
    return abs(1200 * np.log2(result.freq / expected_freq))


class TestYinAccuracy:
    @pytest.mark.parametrize("midi", ALL_GUITAR_MIDI)
    def test_pure_sine_across_the_neck(self, detector, midi):
        freq = midi_to_freq(midi)
        result = detector.detect(synth.sine(freq, duration=0.2)[:WINDOW])
        assert result is not None, f"no detection for {note_name(midi)}"
        assert result.midi == midi, f"{note_name(midi)}: got {result.name}"
        assert error_cents(result, freq) < 5.0

    @pytest.mark.parametrize("midi", ALL_GUITAR_MIDI[::3])
    def test_sawtooth(self, detector, midi):
        freq = midi_to_freq(midi)
        result = detector.detect(synth.sawtooth(freq, duration=0.2)[:WINDOW])
        assert result is not None
        assert result.midi == midi
        assert error_cents(result, freq) < 5.0

    @pytest.mark.parametrize("midi", ALL_GUITAR_MIDI[::2])
    def test_plucked_string(self, detector, midi):
        freq = midi_to_freq(midi)
        signal = synth.plucked_string(freq, duration=0.4, attack_noise=0.05)
        # Skip the attack, as the note gate does in practice.
        result = detector.detect(signal[2048 : 2048 + WINDOW])
        assert result is not None
        assert result.midi == midi
        assert error_cents(result, freq) < 10.0

    @pytest.mark.parametrize("string", range(6))
    def test_open_strings(self, detector, string):
        midi = STANDARD.open_midi[string]
        signal = synth.plucked_string(midi_to_freq(midi), duration=0.4, attack_noise=0.05)
        result = detector.detect(signal[2048 : 2048 + WINDOW])
        assert result is not None
        assert result.name == note_name(midi)


class TestOctaveErrors:
    """The failure mode that matters most: reporting the octave above or below."""

    @pytest.mark.parametrize("note", ["E2", "F2", "A2", "D3", "G3", "B3", "E4"])
    def test_weak_fundamental_does_not_cause_octave_error(self, detector, note):
        midi = name_to_midi(note)
        signal = synth.plucked_string(
            midi_to_freq(midi), duration=0.4, weak_fundamental=True, attack_noise=0.05
        )
        result = detector.detect(signal[2048 : 2048 + WINDOW])
        assert result is not None
        assert result.midi == midi, f"{note}: octave error, got {result.name}"

    @pytest.mark.parametrize("note", ["E2", "A2", "D3"])
    def test_inharmonic_low_strings(self, detector, note):
        """Wound low strings have stretched partials; the fundamental must still win."""
        midi = name_to_midi(note)
        signal = synth.plucked_string(
            midi_to_freq(midi), duration=0.4, inharmonicity=2e-4, attack_noise=0.05
        )
        result = detector.detect(signal[2048 : 2048 + WINDOW])
        assert result is not None
        assert result.midi == midi


class TestNoise:
    @pytest.mark.parametrize("midi", [name_to_midi(n) for n in ["E2", "A2", "G3", "E4", "A4"]])
    @pytest.mark.parametrize("snr_db", [30, 20, 10])
    def test_survives_noise(self, detector, midi, snr_db):
        freq = midi_to_freq(midi)
        clean = synth.plucked_string(freq, duration=0.4)
        noisy = synth.add_noise(clean, snr_db, seed=midi)
        result = detector.detect(noisy[2048 : 2048 + WINDOW])
        assert result is not None
        assert result.midi == midi, f"{note_name(midi)} at {snr_db}dB SNR -> {result.name}"

    def test_pure_noise_is_low_confidence_or_rejected(self, detector):
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 0.1, WINDOW).astype(np.float32)
        result = detector.detect(noise)
        assert result is None or result.confidence < 0.6

    def test_silence_returns_none(self, detector):
        assert detector.detect(np.zeros(WINDOW, dtype=np.float32)) is None


class TestRange:
    def test_below_range_rejected(self, detector):
        # 40 Hz is well under the low E; nothing should be reported.
        assert detector.detect(synth.sine(40.0, duration=0.2)[:WINDOW]) is None

    def test_above_range_rejected(self, detector):
        assert detector.detect(synth.sine(3000.0, duration=0.2)[:WINDOW]) is None


class TestOctavePreference:
    """`_prefer_lower_octave`: found by a real recorded fixture, not reasoned out on
    paper. Scanning short-tau-first (needed to avoid octave-*down* errors) means a
    strong upper partial right after a pick attack can be accepted as the first dip
    before the true, lower fundamental's own — much better — dip is ever reached.
    See AGENTS.md and tests/fixtures/audio/notes/E2.wav for the motivating case
    (E2 briefly read as B3, its 3rd harmonic).

    The first version of this fix (comparing CMND values alone, no spectral check)
    caused a real regression — 0 to 43 synthetic test failures — because CMND's
    running-mean normalisation can produce a deceptively low value at a large tau
    with no genuine periodicity behind it at all. Every test here that expects *no*
    switch is guarding against a version of that regression recurring.
    """

    def test_prefers_a_much_better_longer_period_candidate_with_real_energy(self):
        sr = 48000
        detector = YinDetector(sr)
        freq_true = midi_to_freq(name_to_midi("E2"))
        frame = synth.sine(freq_true, duration=4096 / sr, sr=sr)[:4096].astype(np.float64)

        cmnd = np.ones(700)
        tau_first, tau_true = 194, 582  # true fundamental sits at ~3x the first dip
        cmnd[tau_first] = 0.10  # crosses the default 0.12 threshold — picked first
        cmnd[tau_true] = 0.03  # much better, and the frame genuinely contains it

        assert detector._prefer_lower_octave(cmnd, tau_first, frame) == tau_true

    def test_does_not_switch_without_real_spectral_support(self):
        """The regression guard: a great-looking CMND value alone isn't enough."""
        sr = 48000
        detector = YinDetector(sr)
        frame = np.zeros(4096)  # nothing in the spectrum anywhere

        cmnd = np.ones(700)
        cmnd[194] = 0.10
        cmnd[582] = 0.03  # looks decisive on paper, but nothing backs it up

        assert detector._prefer_lower_octave(cmnd, 194, frame) == 194

    def test_does_not_switch_for_a_marginal_difference(self):
        sr = 48000
        detector = YinDetector(sr)
        freq = midi_to_freq(name_to_midi("E2"))
        frame = synth.sine(freq, duration=4096 / sr, sr=sr)[:4096].astype(np.float64)

        cmnd = np.ones(700)
        cmnd[194] = 0.10
        cmnd[582] = 0.09  # better, but not by OCTAVE_PREFERENCE_RATIO's margin

        assert detector._prefer_lower_octave(cmnd, 194, frame) == 194

    def test_candidate_must_also_clear_the_absolute_threshold(self):
        sr = 48000
        detector = YinDetector(sr)
        freq = midi_to_freq(name_to_midi("E2"))
        frame = synth.sine(freq, duration=4096 / sr, sr=sr)[:4096].astype(np.float64)

        cmnd = np.ones(700)
        cmnd[194] = 0.20
        cmnd[582] = 0.13  # comfortably ratio-better (0.13 < 0.20*0.75), but >= 0.12
        assert detector.threshold == pytest.approx(0.12)

        assert detector._prefer_lower_octave(cmnd, 194, frame) == 194

    def test_picks_the_best_among_several_multiples(self):
        sr = 48000
        detector = YinDetector(sr)
        freq = midi_to_freq(name_to_midi("E2"))
        frame = synth.sine(freq, duration=4096 / sr, sr=sr)[:4096].astype(np.float64)

        cmnd = np.ones(700)
        cmnd[97] = 0.10  # first dip (1x)
        cmnd[194] = 0.08  # 2x: better, but not the best
        cmnd[582] = 0.03  # 6x: the actual best fit, and matches the real frame

        assert detector._prefer_lower_octave(cmnd, 97, frame) == 582

    def test_stays_within_array_bounds_for_a_short_cmnd(self):
        sr = 48000
        detector = YinDetector(sr)
        frame = synth.sine(200.0, duration=4096 / sr, sr=sr)[:4096].astype(np.float64)
        cmnd = np.ones(50)
        cmnd[40] = 0.05
        # Must not raise even though every multiple of 40 falls outside the array.
        assert detector._prefer_lower_octave(cmnd, 40, frame) == 40


class TestPurityFilter:
    """Rejecting noise that happens to be loud and periodic enough to fool YIN.

    This is what guards against keyboard clicks and other percussive sounds near the
    mic being reported as played notes. It works because a plucked string concentrates
    energy at its harmonic series while noise spreads it across the spectrum — see
    audio/pitch.py's DEFAULT_MIN_HARMONIC_RATIO for the measured separation between the
    two populations that this threshold was tuned against.
    """

    @pytest.mark.parametrize("seed", range(10))
    def test_keyboard_clicks_are_rejected(self, detector, seed):
        click = synth.keyboard_click(seed=seed)
        padded = np.zeros(WINDOW, dtype=np.float32)
        n = min(len(click), WINDOW)
        padded[:n] = click[:n]
        result = detector.detect(padded)
        assert result is None, f"click (seed {seed}) was reported as {result.name if result else None}"

    @pytest.mark.parametrize("offset", [0, 512, 1024, 2048])
    def test_clicks_rejected_at_any_window_offset(self, detector, offset):
        click = synth.keyboard_click(seed=1)
        padded = np.zeros(WINDOW, dtype=np.float32)
        n = min(len(click), WINDOW - offset)
        padded[offset : offset + n] = click[:n]
        assert detector.detect(padded) is None

    @pytest.mark.parametrize("note", ["E2", "A2", "D3", "G3", "B3", "E4"])
    def test_clean_guitar_notes_still_pass(self, detector, note):
        """The filter must not cost accuracy on the thing it's meant to let through."""
        midi = name_to_midi(note)
        signal = synth.plucked_string(midi_to_freq(midi), duration=0.4, attack_noise=0.05)
        result = detector.detect(signal[2048 : 2048 + WINDOW])
        assert result is not None
        assert result.midi == midi
        assert result.harmonic_ratio > 0.9

    def test_noisy_guitar_notes_still_pass(self, detector):
        """A guitar note at a realistic SNR shouldn't be caught by a filter meant for
        noise that has no note underneath it at all."""
        midi = name_to_midi("A2")
        signal = synth.add_noise(synth.plucked_string(midi_to_freq(midi), duration=0.4), 15)
        result = detector.detect(signal[2048 : 2048 + WINDOW])
        assert result is not None
        assert result.midi == midi

    def test_pure_white_noise_is_rejected(self, detector):
        rng = np.random.default_rng(3)
        noise = rng.normal(0, 0.2, WINDOW).astype(np.float32)
        assert detector.detect(noise) is None

    def test_ratio_field_reflects_signal_quality(self, detector):
        clean = detector.detect(synth.sine(220.0, duration=0.2)[:WINDOW])
        assert clean is not None
        assert clean.harmonic_ratio > 0.99  # a pure tone is essentially all harmonic

    def test_threshold_is_configurable(self):
        """A caller that wants the old, more permissive behaviour still can."""
        lenient = YinDetector(SR, min_harmonic_ratio=0.0)
        click = synth.keyboard_click(seed=1)
        padded = np.zeros(WINDOW, dtype=np.float32)
        padded[: min(len(click), WINDOW)] = click[:WINDOW]
        # Not asserting it detects something — only that the gate is the thing that
        # was disabled, i.e. this must not raise and must respect the override.
        lenient.detect(padded)
        assert lenient.min_harmonic_ratio == 0.0


class TestPitchSmoother:
    """Live-display smoothing for the Tuner and Free Detect meters.

    This is what fixes the needle jumping around on a single steady note: raw
    per-frame YIN output has real frame-to-frame jitter even on clean audio, and
    NoteGate doesn't help here since it exists to decide whether a note qualifies as a
    scored answer, not to steady a continuously-updating meter. Thresholds below were
    picked by simulating realistic jitter and outliers — see the tuning notes in
    audio/pitch.py's SMOOTHER_* constants.
    """

    def result(self, midi_float: float, **kwargs) -> PitchResult:
        return PitchResult.from_freq(
            midi_to_freq(midi_float), kwargs.get("confidence", 0.9), kwargs.get("rms", 0.1)
        )

    def test_first_frame_passes_through_immediately(self):
        """No dead silence at note onset while the median filter fills up."""
        smoother = PitchSmoother()
        out = smoother.push(self.result(69.0))
        assert out is not None
        assert out.midi == 69

    def test_steady_jitter_is_reduced(self):
        rng = np.random.default_rng(0)
        smoother = PitchSmoother()
        raw_cents_error = []
        smoothed_cents_error = []
        for _ in range(100):
            jittered = 45.0 + rng.normal(0, 0.08)  # ~8 cents std, realistic frame noise
            out = smoother.push(self.result(jittered))
            raw_cents_error.append((jittered - 45.0) * 100)
            smoothed_cents_error.append((out.midi_float - 45.0) * 100)

        # Skip the first few frames while the median window and EMA are still filling.
        # ~8 cents of input noise should come down to a few cents, not eliminated
        # outright — this is a live meter, not a scoring gate.
        assert np.std(smoothed_cents_error[10:]) < 5.0
        assert np.std(smoothed_cents_error[10:]) < np.std(raw_cents_error[10:]) * 0.6

    def test_single_frame_outlier_is_rejected(self):
        """A lone wild estimate must not make the needle jump — the median-of-3
        pre-filter is what an EMA alone cannot do."""
        smoother = PitchSmoother()
        for _ in range(5):
            smoother.push(self.result(45.0))
        out = smoother.push(self.result(45.0 + 1.0))  # one frame, 100 cents off
        assert abs((out.midi_float - 45.0) * 100) < 20.0

    def test_genuine_note_change_snaps_quickly(self):
        """A real string/fret change must not glide through the notes in between."""
        smoother = PitchSmoother()
        for _ in range(10):
            smoother.push(self.result(45.0))
        out = None
        for _ in range(6):
            out = smoother.push(self.result(50.0))  # a real 5-semitone jump
        assert abs(out.midi_float - 50.0) < 0.1

    def test_small_bend_is_smoothed_not_snapped(self):
        """A bend/vibrato-sized movement (under the reset threshold) should ease
        rather than jump, since that's expressive pitch movement, not a new note.

        The median-of-3 pre-filter needs the new pitch to hold for two consecutive
        frames before it becomes the majority in its window — a single frame's worth
        of movement is exactly the kind of blip the filter is designed to absorb, by
        the same mechanism that rejects true outliers.
        """
        smoother = PitchSmoother(reset_threshold_cents=50.0)
        for _ in range(10):
            smoother.push(self.result(45.0))
        smoother.push(self.result(45.0 + 0.3))  # first frame of the bend
        out = smoother.push(self.result(45.0 + 0.3))  # second: now the majority
        moved = abs((out.midi_float - 45.0) * 100)
        assert 0 < moved < 30  # eased partway there, not instantly at the target

    def test_large_jump_resets_instead_of_easing(self):
        smoother = PitchSmoother(reset_threshold_cents=50.0)
        for _ in range(10):
            smoother.push(self.result(45.0))
        smoother.push(self.result(46.0))  # first frame of the new note
        out = smoother.push(self.result(46.0))  # second: the median now sees it
        assert out.midi_float == pytest.approx(46.0)

    def test_silence_resets_immediately(self):
        smoother = PitchSmoother()
        smoother.push(self.result(45.0))
        assert smoother.push(None) is None
        # No lingering state — the very next note should not ease in from the old one.
        out = smoother.push(self.result(50.0))
        assert out.midi_float == pytest.approx(50.0)

    def test_reset_clears_state(self):
        smoother = PitchSmoother()
        smoother.push(self.result(45.0))
        smoother.reset()
        out = smoother.push(self.result(50.0))
        assert out.midi_float == pytest.approx(50.0)

    def test_preserves_confidence_rms_and_harmonic_ratio(self):
        smoother = PitchSmoother()
        result = PitchResult.from_freq(midi_to_freq(45.0), 0.77, 0.05, 0.91)
        out = smoother.push(result)
        assert out.confidence == pytest.approx(0.77)
        assert out.rms == pytest.approx(0.05)
        assert out.harmonic_ratio == pytest.approx(0.91)

    def test_smooths_a_real_detector_stream(self, detector):
        """End-to-end sanity check against actual YIN output, not just synthetic
        midi-float jitter."""
        freq = midi_to_freq(name_to_midi("A2"))
        signal = synth.plucked_string(freq, duration=1.0, attack_noise=0.05)
        smoother = PitchSmoother()

        raw_cents, smoothed_cents = [], []
        for start in range(4096, len(signal) - WINDOW, 1024):
            result = detector.detect(signal[start : start + WINDOW])
            if result is None:
                continue
            raw_cents.append(result.cents)
            out = smoother.push(result)
            smoothed_cents.append(out.cents)

        assert len(smoothed_cents) > 10
        assert np.std(smoothed_cents[5:]) <= np.std(raw_cents[5:])


class TestPitchResult:
    def test_fields_are_consistent(self):
        result = PitchResult.from_freq(440.0, confidence=0.9, rms=0.1)
        assert result.midi == 69
        assert result.name == "A4"
        assert result.note_only == "A"
        assert result.pitch_class == 9
        assert result.cents == pytest.approx(0.0)

    def test_detected_result_reports_cents(self, detector):
        # 20 cents sharp of A4.
        freq = midi_to_freq(69 + 0.2)
        result = detector.detect(synth.sine(freq, duration=0.2)[:WINDOW])
        assert result is not None
        assert result.midi == 69
        assert result.cents == pytest.approx(20.0, abs=5.0)

    def test_confidence_is_higher_for_clean_signals(self, detector):
        clean = detector.detect(synth.sine(220.0, duration=0.2)[:WINDOW])
        noisy = detector.detect(
            synth.add_noise(synth.sine(220.0, duration=0.2), 5)[:WINDOW]
        )
        assert clean is not None and noisy is not None
        assert clean.confidence > noisy.confidence


class TestNoteGate:
    def make(self, **kwargs):
        return NoteGate(rms_threshold=0.01, confidence_threshold=0.5, **kwargs)

    def result(self, midi, *, confidence=0.9, rms=0.1):
        return PitchResult.from_freq(midi_to_freq(midi), confidence, rms)

    def test_requires_consecutive_stable_frames(self):
        gate = self.make(stable_frames=3)
        assert gate.push(self.result(69)) is None
        assert gate.push(self.result(69)) is None
        stable = gate.push(self.result(69))
        assert stable is not None and stable.midi == 69

    def test_quiet_frames_rejected(self):
        gate = self.make(stable_frames=2)
        assert gate.push(self.result(69, rms=0.001)) is None
        assert gate.push(self.result(69, rms=0.001)) is None
        assert gate.current is None

    def test_low_confidence_rejected(self):
        gate = self.make(stable_frames=2)
        gate.push(self.result(69, confidence=0.2))
        gate.push(self.result(69, confidence=0.2))
        assert gate.current is None

    def test_unstable_pitch_does_not_settle(self):
        gate = self.make(stable_frames=3)
        for midi in [69, 71, 67, 72, 65]:
            assert gate.push(self.result(midi)) is None

    def test_settles_after_a_change(self):
        gate = self.make(stable_frames=3)
        for _ in range(3):
            gate.push(self.result(69))
        assert gate.current.midi == 69
        # Move to a new note; the gate must re-settle rather than report immediately.
        assert gate.push(self.result(74)) is None
        gate.push(self.result(74))
        stable = gate.push(self.result(74))
        assert stable is not None and stable.midi == 74

    def test_silence_closes_the_gate(self):
        gate = self.make(stable_frames=2)
        gate.push(self.result(69))
        gate.push(self.result(69))
        assert gate.current is not None
        assert gate.push(None) is None
        assert gate.current is None

    def test_single_outlier_within_tolerance_is_absorbed(self):
        gate = self.make(stable_frames=3, tolerance_cents=60)
        gate.push(self.result(69))
        # 30 cents sharp — inside tolerance, so it should not reset the run.
        gate.push(PitchResult.from_freq(midi_to_freq(69.3), 0.9, 0.1))
        stable = gate.push(self.result(69))
        assert stable is not None and stable.midi == 69

    def test_reset_clears_state(self):
        gate = self.make(stable_frames=2)
        gate.push(self.result(69))
        gate.push(self.result(69))
        gate.reset()
        assert gate.current is None
        assert gate.push(self.result(69)) is None


class TestEndToEnd:
    def test_gated_detection_over_a_played_note(self, detector):
        """Frame-by-frame over a synthesised pluck, as the worker thread does it."""
        midi = name_to_midi("G3")
        signal = synth.plucked_string(midi_to_freq(midi), duration=0.6, attack_noise=0.1)
        gate = NoteGate(rms_threshold=0.005, stable_frames=3)

        detected = []
        for start in range(0, len(signal) - WINDOW, 1024):
            stable = gate.push(detector.detect(signal[start : start + WINDOW]))
            if stable is not None:
                detected.append(stable.midi)

        assert detected, "note never settled"
        assert all(m == midi for m in detected), f"unstable: {set(detected)}"
