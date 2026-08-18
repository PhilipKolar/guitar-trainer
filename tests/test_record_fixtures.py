"""Tests for the fixture-recording script's non-interactive logic.

The interactive recording loop itself (input(), live capture) isn't covered here —
there's no audio device in CI and no one to press Enter. What's covered is everything
around it: the note plan, the WAV round-trip, and the sanity-check scoring, since
those are what would silently corrupt a recording session if broken.
"""

import threading
import time

import numpy as np
import pytest

import synth
from guitar_trainer.audio.capture import AudioCapture
from guitar_trainer.core.chords import Chord
from guitar_trainer.core.notes import STANDARD, midi_to_freq, name_to_midi, note_name
from guitar_trainer.scripts.record_fixtures import (
    CHORD_PLAN,
    NOTE_PLAN,
    _level_bar,
    record_clip,
    sanity_check,
    sanity_check_chord,
    save_wav,
)


class TestRecordClipBufferSizing:
    """Regression coverage for a real crash: AudioCapture's default ring buffer
    (~1.4s, sized for live detection, not for recording a whole take) is far shorter
    than a typical multi-second clip, and RingBuffer.read_latest() raises rather than
    truncating when asked for more than it holds. main() must size the buffer to fit
    --duration; these exercise that directly against the real AudioCapture/RingBuffer,
    not a mock, so a future change that drops the sizing fails loudly here instead of
    mid-recording-session.
    """

    def test_buffer_sized_for_duration_holds_a_full_clip(self):
        duration, sample_rate = 4.0, 48000
        capture = AudioCapture(sample_rate=sample_rate, buffer_seconds=duration + 1.0)
        samples_needed = int(duration * sample_rate)
        capture.buffer.write(np.zeros(samples_needed, dtype=np.float32))

        # Must not raise — this is exactly what crashed with the default 1s buffer.
        result = capture.buffer.read_latest(samples_needed)
        assert len(result) == samples_needed

    def test_default_buffer_is_too_small_for_a_multi_second_clip(self):
        """Confirms the bug this is guarding against is real, not hypothetical —
        the default buffer alone does raise for a normal recording duration."""
        duration, sample_rate = 4.0, 48000
        capture = AudioCapture(sample_rate=sample_rate)  # default buffer_seconds
        with pytest.raises(ValueError):
            capture.buffer.read_latest(int(duration * sample_rate))

    def test_record_clip_does_not_crash_on_a_multi_second_duration(self, monkeypatch):
        """record_clip() waits out PRE_ROLL_SECONDS and *then* clears the buffer, so
        — unlike the other two tests here — the samples have to arrive after that,
        from a separate thread, the way a real capture callback would deliver them."""
        import guitar_trainer.scripts.record_fixtures as record_fixtures

        monkeypatch.setattr(record_fixtures, "PRE_ROLL_SECONDS", 0.01)

        duration, sample_rate = 4.0, 48000
        capture = AudioCapture(sample_rate=sample_rate, buffer_seconds=duration + 1.0)
        samples_needed = int(duration * sample_rate)

        def feed_after_clear() -> None:
            time.sleep(0.05)
            capture.buffer.write(np.zeros(samples_needed, dtype=np.float32))

        threading.Thread(target=feed_after_clear, daemon=True).start()
        result = record_clip(capture, duration, sample_rate)
        assert len(result) == samples_needed


class TestPreRoll:
    """Regression coverage for a real contamination bug: the acoustic tap of the
    Enter keypress that starts recording — picked up by a mic sitting right next to
    the keyboard — was ending up as the first ~0.3-0.4s of every clip (confirmed via
    a clear amplitude spike at t=0 in real recordings, followed by a quiet gap before
    the actual note began). record_clip() now waits out the tap before the timed
    window starts.
    """

    def test_pre_tap_audio_is_discarded_not_recorded(self, monkeypatch):
        import guitar_trainer.scripts.record_fixtures as record_fixtures

        monkeypatch.setattr(record_fixtures, "PRE_ROLL_SECONDS", 0.05)

        duration, sample_rate = 0.2, 48000
        capture = AudioCapture(sample_rate=sample_rate, buffer_seconds=duration + 1.0)
        # A loud "tap" already sitting in the buffer before record_clip() is even
        # called — simulates the keyboard-tap transient a mic would have already
        # picked up by the time Enter is pressed.
        capture.buffer.write(np.full(1000, 0.9, dtype=np.float32))

        def feed() -> None:
            time.sleep(0.1)  # after the (shortened) pre-roll
            capture.buffer.write(np.zeros(int(duration * sample_rate), dtype=np.float32))

        threading.Thread(target=feed, daemon=True).start()
        result = record_clip(capture, duration, sample_rate)

        # The loud pre-roll tap must not appear anywhere in the recorded clip.
        assert not np.any(result > 0.5)

    def test_default_is_long_enough_for_a_key_tap_to_decay(self):
        import guitar_trainer.scripts.record_fixtures as record_fixtures

        # Not a precise acoustic claim — just a sanity floor. Too short and this
        # regresses back to capturing the tap; the exact value was picked from what a
        # real recording showed (spike gone by ~0.4s).
        assert record_fixtures.PRE_ROLL_SECONDS >= 0.4


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


class TestChordPlan:
    def test_covers_e_a_d_major_minor_and_dominant_seventh(self):
        assert len(CHORD_PLAN) == 9
        symbols = {chord.name() for chord in CHORD_PLAN}
        assert symbols == {"E", "Em", "E7", "A", "Am", "A7", "D", "Dm", "D7"}

    def test_filenames_round_trip_through_chord_parse(self):
        """The whole fixture scheme relies on filename == chord.name()."""
        for chord in CHORD_PLAN:
            assert Chord.parse(chord.name()) == chord


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


#: Standard open-position voicings for the nine chords CHORD_PLAN covers — the same
#: shapes a player would actually record. Matches the real voicings used to validate
#: chord recognition itself in test_chords.py::VOICINGS (plus D7, absent there).
_CHORD_VOICINGS = {
    "E": ["E2", "B2", "E3", "G#3", "B3", "E4"],
    "Em": ["E2", "B2", "E3", "G3", "B3", "E4"],
    "E7": ["E2", "B2", "D3", "G#3", "B3", "E4"],
    "A": ["A2", "E3", "A3", "C#4", "E4"],
    "Am": ["A2", "E3", "A3", "C4", "E4"],
    "A7": ["A2", "E3", "G3", "C#4", "E4"],
    "D": ["D3", "A3", "D4", "F#4"],
    "Dm": ["D3", "A3", "D4", "F4"],
    "D7": ["D3", "A3", "C4", "F#4"],
}


def _strummed_chord(symbol: str, **kwargs) -> np.ndarray:
    midi = [name_to_midi(n) for n in _CHORD_VOICINGS[symbol]]
    return synth.strum(midi, duration=1.5, **kwargs)


class TestSanityCheckChord:
    def test_correct_chord_reads_as_looking_right(self):
        message = sanity_check_chord(_strummed_chord("E"), Chord.parse("E"), 48000)
        assert "looks right" in message

    def test_wrong_chord_is_flagged(self):
        message = sanity_check_chord(_strummed_chord("A"), Chord.parse("E"), 48000)
        assert "expected E" in message
        assert "looks right" not in message

    def test_silence_is_flagged(self):
        message = sanity_check_chord(np.zeros(48000 * 3, dtype=np.float32), Chord.parse("E"), 48000)
        assert "no stable chord" in message

    @pytest.mark.parametrize("symbol", sorted(_CHORD_VOICINGS))
    def test_every_planned_chord_reads_correctly_in_its_standard_voicing(self, symbol):
        """Each of the nine chords CHORD_PLAN covers, played as its standard open
        voicing, should read as itself — this is what the recorder leans on to flag
        a bad take on the spot rather than after the whole session."""
        message = sanity_check_chord(_strummed_chord(symbol), Chord.parse(symbol), 48000)
        assert "looks right" in message, message


class TestLevelBar:
    def test_silence_is_empty(self):
        assert _level_bar(0.0) == "." * 30

    def test_loud_signal_fills_the_bar(self):
        assert _level_bar(1.0) == "#" * 30

    def test_length_is_constant_regardless_of_level(self):
        for rms in [0.0, 0.05, 0.1, 0.5, 5.0]:
            assert len(_level_bar(rms)) == 30
