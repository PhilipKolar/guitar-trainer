import math

import pytest

from guitar_trainer.core import notes as N


class TestFrequencyConversion:
    def test_a440_is_midi_69(self):
        assert N.freq_to_midi(440.0) == pytest.approx(69.0)
        assert N.midi_to_freq(69) == pytest.approx(440.0)

    def test_octave_doubles_frequency(self):
        assert N.midi_to_freq(81) == pytest.approx(880.0)
        assert N.freq_to_midi(220.0) == pytest.approx(57.0)

    @pytest.mark.parametrize("midi", range(28, 90))
    def test_round_trip(self, midi):
        assert N.freq_to_midi(N.midi_to_freq(midi)) == pytest.approx(midi)

    def test_known_guitar_frequencies(self):
        # Standard-tuning open strings, to the usual published values.
        for name, freq in [
            ("E2", 82.41),
            ("A2", 110.00),
            ("D3", 146.83),
            ("G3", 196.00),
            ("B3", 246.94),
            ("E4", 329.63),
        ]:
            assert N.midi_to_freq(N.name_to_midi(name)) == pytest.approx(freq, abs=0.01)

    def test_rejects_non_positive_frequency(self):
        with pytest.raises(ValueError):
            N.freq_to_midi(0.0)
        with pytest.raises(ValueError):
            N.freq_to_midi(-100.0)


class TestNoteNames:
    def test_middle_c(self):
        assert N.note_name(60) == "C4"
        assert N.name_to_midi("C4") == 60

    def test_sharps_and_flats(self):
        assert N.note_name(61) == "C#4"
        assert N.note_name(61, use_flats=True) == "Db4"
        assert N.name_to_midi("C#4") == N.name_to_midi("Db4") == 61

    def test_without_octave(self):
        assert N.note_name(69, with_octave=False) == "A"

    def test_negative_octave(self):
        assert N.note_name(0) == "C-1"
        assert N.name_to_midi("C-1") == 0

    @pytest.mark.parametrize("midi", range(0, 128))
    def test_name_round_trip(self, midi):
        assert N.name_to_midi(N.note_name(midi)) == midi

    def test_unicode_accidentals(self):
        assert N.name_to_pitch_class("C♯") == 1
        assert N.name_to_pitch_class("D♭") == 1

    def test_lowercase_accepted(self):
        assert N.name_to_pitch_class("bb") == 10

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            N.name_to_pitch_class("H")
        with pytest.raises(ValueError):
            N.name_to_midi("A")  # no octave


class TestCents:
    def test_in_tune_is_zero_cents(self):
        assert N.cents_off(440.0, 69) == pytest.approx(0.0)

    def test_semitone_is_100_cents(self):
        assert N.cents_off(N.midi_to_freq(70), 69) == pytest.approx(100.0)

    def test_flat_is_negative(self):
        assert N.cents_off(435.0, 69) < 0

    def test_nearest_note_rounds_to_closest(self):
        midi, cents = N.nearest_note(437.0)
        assert midi == 69
        assert -20 < cents < 0

        # Just past the midpoint between A4 and A#4 should round up.
        midpoint = N.midi_to_freq(69.5)
        midi, cents = N.nearest_note(midpoint * 1.001)
        assert midi == 70


class TestTuning:
    def test_standard_open_strings(self):
        assert STANDARD_NAMES == ["E2", "A2", "D3", "G3", "B3", "E4"]

    def test_string_indexing_is_low_to_high(self):
        assert N.STANDARD.string_label(0) == "E2"
        assert N.STANDARD.string_label(5) == "E4"
        assert N.STANDARD.string_count == 6

    def test_midi_at_fret(self):
        # 5th fret of the low E is A2, matching the 5th string open.
        assert N.STANDARD.midi_at(0, 5) == N.STANDARD.open_midi[1]
        # 12th fret is an octave above open.
        assert N.STANDARD.midi_at(0, 12) == N.STANDARD.open_midi[0] + 12

    def test_a440_positions(self):
        # A4 is the high-E string 5th fret and the B string 10th fret.
        positions = N.STANDARD.positions_for_midi(N.name_to_midi("A4"))
        assert (5, 5) in positions
        assert (4, 10) in positions
        # The A string only reaches A4 at the 24th fret, beyond this 22-fret neck.
        assert not any(string == 1 for string, _ in positions)

    def test_a220_positions(self):
        # A3 is the A string 12th fret and the low E string 17th fret.
        positions = N.STANDARD.positions_for_midi(N.name_to_midi("A3"))
        assert (1, 12) in positions
        assert (0, 17) in positions

    def test_pitch_class_positions_span_all_strings(self):
        positions = N.STANDARD.positions_for_pitch_class(N.name_to_pitch_class("A"))
        # With 22 frets every string reaches at least one A.
        assert {s for s, _ in positions} == set(range(6))
        for string, fret in positions:
            assert N.STANDARD.midi_at(string, fret) % 12 == 9

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError):
            N.STANDARD.midi_at(6, 0)
        with pytest.raises(IndexError):
            N.STANDARD.midi_at(0, 23)

    def test_drop_d_lowers_only_sixth_string(self):
        assert N.DROP_D.open_midi[0] == N.STANDARD.open_midi[0] - 2
        assert N.DROP_D.open_midi[1:] == N.STANDARD.open_midi[1:]

    def test_half_step_down_is_uniform(self):
        for lowered, standard in zip(N.HALF_STEP_DOWN.open_midi, N.STANDARD.open_midi):
            assert lowered == standard - 1

    def test_nearest_string(self):
        # A slightly flat low E should still identify string 0.
        string, cents = N.STANDARD.nearest_string(80.0)
        assert string == 0
        assert cents < 0

        string, cents = N.STANDARD.nearest_string(330.0)
        assert string == 5
        assert abs(cents) < 10

    def test_midi_range(self):
        low, high = N.STANDARD.midi_range()
        assert low == N.name_to_midi("E2")
        assert high == N.name_to_midi("E4") + 22

    def test_preset_lookup(self):
        assert N.preset_by_name("Drop D") is N.DROP_D
        assert N.preset_by_name("nonexistent") is None


class TestFretGeometry:
    def test_nut_at_zero(self):
        assert N.fret_x_positions(22)[0] == 0.0

    def test_twelfth_fret_at_half_scale(self):
        assert N.fret_x_positions(22)[12] == pytest.approx(0.5)

    def test_monotonically_increasing_with_shrinking_gaps(self):
        xs = N.fret_x_positions(22)
        gaps = [b - a for a, b in zip(xs, xs[1:])]
        assert all(g > 0 for g in gaps)
        assert all(later < earlier for earlier, later in zip(gaps, gaps[1:]))

    def test_scales_with_scale_length(self):
        xs = N.fret_x_positions(12, scale_length=648.0)
        assert xs[12] == pytest.approx(324.0)

    def test_returns_one_entry_per_fret_plus_nut(self):
        assert len(N.fret_x_positions(24)) == 25


class TestNoteSet:
    def test_all_notes(self):
        assert N.ALL_NOTES.pitch_classes == tuple(range(12))

    def test_naturals_have_no_accidentals(self):
        assert N.NATURAL_NOTES.labels() == ["C", "D", "E", "F", "G", "A", "B"]

    def test_deduplicates_and_sorts(self):
        s = N.NoteSet("custom", (9, 0, 9, 12))
        assert s.pitch_classes == (0, 9)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            N.NoteSet("empty", ())


STANDARD_NAMES = [N.note_name(m) for m in N.STANDARD.open_midi]
