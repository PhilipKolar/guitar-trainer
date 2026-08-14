import numpy as np
import pytest

import synth
from guitar_trainer.audio.pitch import NoteGate, PitchResult, YinDetector
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
