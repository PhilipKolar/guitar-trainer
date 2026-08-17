"""Tests for the fixture-recording script's non-interactive logic.

The interactive recording loop itself (input(), live capture) isn't covered here —
there's no audio device in CI and no one to press Enter. What's covered is everything
around it: the note plan, the WAV round-trip, and the sanity-check scoring, since
those are what would silently corrupt a recording session if broken.
"""

import numpy as np
import pytest

import synth
from guitar_trainer.core.notes import STANDARD, midi_to_freq, name_to_midi, note_name
from guitar_trainer.scripts.record_fixtures import (
    NOTE_PLAN,
    _level_bar,
    sanity_check,
    save_wav,
)


class TestNotePlan:
    def test_covers_one_full_chromatic_octave(self):
        assert len(NOTE_PLAN) == 12
        assert len({midi for _, midi in NOTE_PLAN}) == 12

    def test_starts_on_the_open_low_e(self):
        fret, midi = NOTE_PLAN[0]
        assert fret == 0
        assert midi == STANDARD.open_midi[0]
        assert note_name(midi) == "E2"

    def test_frets_are_consecutive_semitones_on_one_string(self):
        for fret, midi in NOTE_PLAN:
            assert midi == STANDARD.midi_at(0, fret)
        assert [fret for fret, _ in NOTE_PLAN] == list(range(12))


class TestWavRoundTrip:
    def test_save_and_load_preserves_the_signal(self, tmp_path):
        original = synth.sine(440.0, duration=0.5, amplitude=0.4)
        path = tmp_path / "test.wav"
        save_wav(path, original, 48000)

        sr, loaded = synth.load_wav(path)
        assert sr == 48000
        assert len(loaded) == len(original)
        # 16-bit quantisation, not lossless, but should be very close.
        np.testing.assert_allclose(loaded, original, atol=1e-3)

    def test_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "test.wav"
        save_wav(path, synth.sine(440.0, duration=0.1), 48000)
        assert path.exists()

    def test_clips_out_of_range_samples_rather_than_wrapping(self, tmp_path):
        loud = np.full(1000, 1.5, dtype=np.float32)  # well over full scale
        path = tmp_path / "loud.wav"
        save_wav(path, loud, 48000)

        _, loaded = synth.load_wav(path)
        assert np.all(loaded <= 1.0)
        assert np.all(loaded > 0)  # clipped high, not wrapped to negative

    def test_filenames_round_trip_through_note_name(self, tmp_path):
        """The whole fixture scheme relies on filename == note_name(midi)."""
        for _, midi in NOTE_PLAN:
            name = note_name(midi)
            assert name_to_midi(name) == midi


class TestSanityCheck:
    def test_correct_note_reads_as_looking_right(self):
        midi = name_to_midi("E2")
        samples = synth.plucked_string(midi_to_freq(midi), duration=2.0, attack_noise=0.05)
        message = sanity_check(samples, midi, 48000)
        assert "looks right" in message

    def test_wrong_note_is_flagged(self):
        actual = name_to_midi("F2")
        expected = name_to_midi("E2")
        samples = synth.plucked_string(midi_to_freq(actual), duration=2.0, attack_noise=0.05)
        message = sanity_check(samples, expected, 48000)
        assert "expected E2" in message
        assert "looks right" not in message

    def test_silence_is_flagged(self):
        message = sanity_check(np.zeros(48000 * 2, dtype=np.float32), name_to_midi("E2"), 48000)
        assert "no stable note" in message


class TestLevelBar:
    def test_silence_is_empty(self):
        assert _level_bar(0.0) == "." * 30

    def test_loud_signal_fills_the_bar(self):
        assert _level_bar(1.0) == "#" * 30

    def test_length_is_constant_regardless_of_level(self):
        for rms in [0.0, 0.05, 0.1, 0.5, 5.0]:
            assert len(_level_bar(rms)) == 30
