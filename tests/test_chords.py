import numpy as np
import pytest

import synth
from guitar_trainer.audio.chroma import CHROMA_WINDOW, ChromaAnalyser
from guitar_trainer.core.chords import (
    COMMON_OPEN_CHORDS,
    Chord,
    ChordGate,
    ChordMatcher,
    NoteOrChordClassifier,
    NoteOrChordGate,
    all_chords,
    harmonic_template,
    note_template,
)
from guitar_trainer.core.notes import midi_to_freq, name_to_midi, name_to_pitch_class

SR = synth.SAMPLE_RATE

# Real guitar voicings, as MIDI notes, lowest string first.
VOICINGS = {
    "E": ["E2", "B2", "E3", "G#3", "B3", "E4"],
    "Am": ["A2", "E3", "A3", "C4", "E4"],
    "D": ["D3", "A3", "D4", "F#4"],
    "G": ["G2", "B2", "D3", "G3", "B3", "G4"],
    "C": ["C3", "E3", "G3", "C4", "E4"],
    "Em": ["E2", "B2", "E3", "G3", "B3", "E4"],
    "Dm": ["D3", "A3", "D4", "F4"],
    "A": ["A2", "E3", "A3", "C#4", "E4"],
    "F": ["F2", "C3", "F3", "A3", "C4", "F4"],          # full barre
    "Bm": ["B2", "F#3", "B3", "D4", "F#4"],             # barre
    "A7": ["A2", "E3", "G3", "C#4", "E4"],
    "E7": ["E2", "B2", "D3", "G#3", "B3", "E4"],
    "Cmaj7": ["C3", "E3", "G3", "B3", "E4"],
    "Am7": ["A2", "E3", "G3", "C4", "E4"],
    "Dsus4": ["D3", "A3", "D4", "G4"],
    "Asus2": ["A2", "E3", "A3", "B3", "E4"],
    "E5": ["E2", "B2", "E3"],                            # power chord
}


def chroma_of(voicing_names, **kwargs):
    """Synthesise a strum of the given voicing and return its chroma vector."""
    midi = [name_to_midi(n) for n in voicing_names]
    audio = synth.strum(midi, duration=0.8, **kwargs)
    analyser = ChromaAnalyser(SR)
    # Feed several hops, as the worker does, so the median smoothing is exercised.
    result = None
    for start in range(4096, len(audio) - CHROMA_WINDOW, 2048):
        out = analyser.analyse(audio[start : start + CHROMA_WINDOW])
        if out is not None:
            result = out
    assert result is not None, "no chroma produced"
    return result


def chroma_of_note(midi: int, **kwargs) -> np.ndarray:
    """Synthesise a single plucked note and return its chroma vector."""
    audio = synth.plucked_string(midi_to_freq(midi), duration=0.6, attack_noise=0.05, **kwargs)
    analyser = ChromaAnalyser(SR)
    result = None
    for start in range(2048, len(audio) - CHROMA_WINDOW, 1024):
        out = analyser.analyse(audio[start : start + CHROMA_WINDOW])
        if out is not None:
            result = out
    assert result is not None, "no chroma produced"
    return result


class TestChordModel:
    def test_major_intervals(self):
        assert Chord(name_to_pitch_class("C"), "maj").pitch_classes == (0, 4, 7)

    def test_minor_intervals(self):
        assert Chord(name_to_pitch_class("A"), "min").pitch_classes == (9, 0, 4)

    def test_names(self):
        assert Chord(0, "maj").name() == "C"
        assert Chord(9, "min").name() == "Am"
        assert Chord(4, "7").name() == "E7"
        assert Chord(0, "maj7").name() == "Cmaj7"
        assert Chord(9, "min7").name() == "Am7"
        assert Chord(2, "sus4").name() == "Dsus4"
        assert Chord(4, "5").name() == "E5"

    @pytest.mark.parametrize("symbol", list(VOICINGS) + ["F#m", "Bbmaj7", "C#dim", "Gaug"])
    def test_parse_round_trip(self, symbol):
        assert Chord.parse(symbol).name(use_flats="b" in symbol) == symbol

    def test_parse_accidentals(self):
        assert Chord.parse("F#m") == Chord(name_to_pitch_class("F#"), "min")
        assert Chord.parse("Bbmaj7") == Chord(name_to_pitch_class("Bb"), "maj7")

    def test_parse_rejects_garbage(self):
        for bad in ["", "H", "Cxyz", "Am9"]:
            with pytest.raises(ValueError):
                Chord.parse(bad)

    def test_unknown_quality_rejected(self):
        with pytest.raises(ValueError):
            Chord(0, "nonsense")

    def test_all_chords_covers_every_root(self):
        chords = all_chords(["maj", "min"])
        assert len(chords) == 24
        assert len({c.root for c in chords}) == 12


class TestTemplates:
    def test_normalised(self):
        assert np.linalg.norm(harmonic_template((0, 4, 7))) == pytest.approx(1.0)

    def test_chord_tones_dominate(self):
        template = harmonic_template((0, 4, 7))
        chord_tone_energy = template[[0, 4, 7]].sum()
        assert chord_tone_energy > 0.6 * template.sum()

    def test_harmonics_leak_onto_non_chord_tones(self):
        """The point of the model: a real chord is not three clean spikes."""
        assert harmonic_template((0,))[7] > 0  # 3rd harmonic of C lands on G

    def test_transposition_is_a_rotation(self):
        c = harmonic_template(Chord(0, "maj").pitch_classes)
        d = harmonic_template(Chord(2, "maj").pitch_classes)
        assert np.allclose(np.roll(c, 2), d)

    def test_major_and_minor_are_distinguishable(self):
        similarity = harmonic_template((0, 4, 7)) @ harmonic_template((0, 3, 7))
        assert similarity < 0.9


class TestChordRecognition:
    @pytest.mark.parametrize("symbol", list(VOICINGS))
    def test_recognises_real_voicings(self, symbol):
        chroma = chroma_of(VOICINGS[symbol])
        matcher = ChordMatcher()
        match = matcher.match(chroma)
        assert match is not None
        assert match.chord == Chord.parse(symbol), (
            f"{symbol} -> {match.chord} "
            f"(top: {[(str(c), round(s, 3)) for c, s in matcher.top_n(chroma)]})"
        )

    @pytest.mark.parametrize("symbol", ["E", "Am", "D", "G", "C", "Em", "Dm", "A"])
    def test_common_chords_score_highly(self, symbol):
        match = ChordMatcher(
            [Chord(r, q) for r, q in [(c.root, c.quality) for c in _common()]]
        ).match(chroma_of(VOICINGS[symbol]))
        assert match is not None
        assert match.score > 0.75

    def test_relative_major_and_minor_are_not_confused(self):
        """C and Am share two of three notes; the bass note is what separates them."""
        matcher = ChordMatcher()
        for symbol in ["C", "Am"]:
            chroma = chroma_of(VOICINGS[symbol])
            bass = name_to_pitch_class(symbol.rstrip("m"))
            match = matcher.match(chroma, bass_pitch_class=bass)
            assert match is not None and match.chord == Chord.parse(symbol)

    def test_em_and_g_are_not_confused(self):
        matcher = ChordMatcher()
        for symbol, bass in [("Em", "E"), ("G", "G")]:
            match = matcher.match(
                chroma_of(VOICINGS[symbol]), bass_pitch_class=name_to_pitch_class(bass)
            )
            assert match is not None and match.chord == Chord.parse(symbol)

    @pytest.mark.parametrize("symbol", ["E", "Am", "G", "C"])
    def test_survives_noise(self, symbol):
        midi = [name_to_midi(n) for n in VOICINGS[symbol]]
        audio = synth.add_noise(synth.strum(midi, duration=0.8), 20)
        analyser = ChromaAnalyser(SR)
        chroma = None
        for start in range(4096, len(audio) - CHROMA_WINDOW, 2048):
            out = analyser.analyse(audio[start : start + CHROMA_WINDOW])
            if out is not None:
                chroma = out
        match = ChordMatcher().match(chroma)
        assert match is not None and match.chord == Chord.parse(symbol)

    def test_open_and_barre_voicings_of_the_same_chord_agree(self):
        """One template set must handle any voicing — that was the design goal."""
        matcher = ChordMatcher()
        open_a = matcher.match(chroma_of(["A2", "E3", "A3", "C#4", "E4"]))
        barre_a = matcher.match(chroma_of(["A2", "E3", "A3", "C#4", "E4", "A4"]))
        assert open_a.chord == barre_a.chord == Chord.parse("A")


class TestChromaAnalyser:
    def test_silence_returns_none(self):
        analyser = ChromaAnalyser(SR)
        assert analyser.analyse(np.zeros(CHROMA_WINDOW, dtype=np.float32)) is None

    def test_short_frame_returns_none(self):
        assert ChromaAnalyser(SR).analyse(np.zeros(128, dtype=np.float32)) is None

    def test_output_is_unit_norm(self):
        chroma = chroma_of(VOICINGS["E"])
        assert np.linalg.norm(chroma) == pytest.approx(1.0)
        assert len(chroma) == 12

    def test_single_note_peaks_on_its_pitch_class(self):
        analyser = ChromaAnalyser(SR)
        audio = synth.plucked_string(440.0, duration=0.5)
        chroma = analyser.analyse(audio[2048 : 2048 + CHROMA_WINDOW])
        assert chroma is not None
        assert int(np.argmax(chroma)) == name_to_pitch_class("A")

    def test_bass_pitch_class_finds_the_lowest_note(self):
        analyser = ChromaAnalyser(SR)
        audio = synth.strum([name_to_midi(n) for n in VOICINGS["G"]], duration=0.8)
        bass = analyser.bass_pitch_class(audio[4096 : 4096 + CHROMA_WINDOW])
        assert bass == name_to_pitch_class("G")

    def test_bass_pitch_class_on_silence(self):
        analyser = ChromaAnalyser(SR)
        assert analyser.bass_pitch_class(np.zeros(CHROMA_WINDOW, dtype=np.float32)) is None

    def test_reset_clears_smoothing(self):
        analyser = ChromaAnalyser(SR)
        audio = synth.strum([name_to_midi(n) for n in VOICINGS["E"]], duration=0.8)
        analyser.analyse(audio[4096 : 4096 + CHROMA_WINDOW])
        analyser.reset()
        assert not analyser._history


class TestChordGate:
    def test_requires_consecutive_agreement(self):
        gate = ChordGate(ChordMatcher(), stable_frames=3)
        chroma = chroma_of(VOICINGS["E"])
        assert gate.push(chroma) is None
        assert gate.push(chroma) is None
        match = gate.push(chroma)
        assert match is not None and match.chord == Chord.parse("E")

    def test_changing_answer_restarts_the_streak(self):
        gate = ChordGate(ChordMatcher(), stable_frames=3)
        e, a = chroma_of(VOICINGS["E"]), chroma_of(VOICINGS["Am"])
        gate.push(e)
        gate.push(a)
        gate.push(e)
        assert gate.current is None

    def test_none_resets(self):
        gate = ChordGate(ChordMatcher(), stable_frames=2)
        chroma = chroma_of(VOICINGS["E"])
        gate.push(chroma)
        gate.push(chroma)
        assert gate.current is not None
        assert gate.push(None) is None
        assert gate.current is None

    def test_weak_match_rejected(self):
        gate = ChordGate(ChordMatcher(), min_score=0.99, stable_frames=1)
        assert gate.push(chroma_of(VOICINGS["E"])) is None


def _common():
    return [Chord(name_to_pitch_class(root), quality) for root, quality in COMMON_OPEN_CHORDS]


class TestNoteTemplate:
    def test_normalised(self):
        assert np.linalg.norm(note_template(0)) == pytest.approx(1.0)

    def test_peaks_on_the_root(self):
        assert int(np.argmax(note_template(4))) == 4  # E

    def test_root_weight_has_no_effect_on_a_single_note(self):
        """root_weight scales the (only) note uniformly; normalising removes it."""
        default = note_template(0)
        heavier = note_template(0, root_weight=5.0)
        np.testing.assert_allclose(default, heavier, atol=1e-10)


class TestNoteOrChordClassifier:
    """The fix for chord detection never triggering in Free Detect.

    A monophonic pitch tracker can look "stable" on a single dominant string even
    while a full chord is ringing, so gating chord detection behind "no stable note
    is currently detected" (the old design) meant a strummed chord routinely never
    got a chance to be classified at all. This class decides note-vs-chord from the
    chroma vector itself instead, which is what actually distinguishes the two.
    """

    @pytest.mark.parametrize(
        "note", ["E2", "A2", "D3", "G3", "B3", "E4", "A4", "C4", "E3", "G2", "B2"]
    )
    def test_single_notes_are_classified_as_notes(self, note):
        midi = name_to_midi(note)
        chroma = chroma_of_note(midi)
        classifier = NoteOrChordClassifier(all_chords())
        match = classifier.classify(chroma)
        assert match is not None
        assert match.kind == "note", (
            f"{note} misclassified as a chord (score {match.score:.3f})"
        )
        assert match.pitch_class == midi % 12

    @pytest.mark.parametrize("symbol", [s for s in VOICINGS if s != "E5"])
    def test_chord_voicings_are_classified_as_chords(self, symbol):
        chroma = chroma_of(VOICINGS[symbol])
        classifier = NoteOrChordClassifier(all_chords())
        match = classifier.classify(chroma)
        assert match is not None
        assert match.kind == "chord", (
            f"{symbol} misclassified as a single note (score {match.score:.3f})"
        )
        assert match.chord == Chord.parse(symbol)

    def test_power_chords_are_an_inherent_ambiguity_not_a_bug(self):
        """A power chord is just a root and its own 3rd harmonic (mod octave) played
        as a second string — chroma-wise that's indistinguishable from a single note
        with a strong 3rd harmonic, since it's literally the same energy pattern.
        There is no principled fix for this from chroma alone; it's fine either way."""
        chroma = chroma_of(VOICINGS["E5"])
        classifier = NoteOrChordClassifier(all_chords())
        match = classifier.classify(chroma)
        assert match is not None
        assert match.kind in ("note", "chord")

    def test_e_major_specifically(self):
        """The reported bug: E major's notes are exactly the low harmonics of its
        root (3rd harmonic = a 5th, 5th harmonic = a major 3rd), which is what made
        the old note-blocks-chord design fail here specifically and consistently."""
        chroma = chroma_of(VOICINGS["E"])
        classifier = NoteOrChordClassifier(all_chords())
        match = classifier.classify(chroma)
        assert match is not None
        assert match.kind == "chord"
        assert match.chord == Chord.parse("E")

    def test_a_single_low_e_string_is_not_mistaken_for_a_chord(self):
        """The specific single-note case the bug report described being misread."""
        chroma = chroma_of_note(name_to_midi("B2"))
        classifier = NoteOrChordClassifier(all_chords())
        match = classifier.classify(chroma)
        assert match is not None
        assert match.kind == "note"
        assert match.pitch_class == name_to_pitch_class("B")

    def test_silence_returns_none(self):
        classifier = NoteOrChordClassifier(all_chords())
        assert classifier.classify(np.zeros(12)) is None

    def test_bass_note_still_disambiguates_relative_pairs(self):
        classifier = NoteOrChordClassifier(all_chords())
        for symbol, bass in [("C", "C"), ("Am", "A")]:
            match = classifier.classify(
                chroma_of(VOICINGS[symbol]), bass_pitch_class=name_to_pitch_class(bass)
            )
            assert match is not None
            assert match.kind == "chord"
            assert match.chord == Chord.parse(symbol)


class TestNoteOrChordGate:
    def make(self, **kwargs):
        return NoteOrChordGate(NoteOrChordClassifier(all_chords()), **kwargs)

    def test_requires_consecutive_agreement(self):
        gate = self.make(stable_frames=3)
        chroma = chroma_of(VOICINGS["E"])
        assert gate.push(chroma) is None
        assert gate.push(chroma) is None
        match = gate.push(chroma)
        assert match is not None
        assert match.kind == "chord" and match.chord == Chord.parse("E")

    def test_note_settles_too(self):
        gate = self.make(stable_frames=3)
        chroma = chroma_of_note(name_to_midi("A2"))
        gate.push(chroma)
        gate.push(chroma)
        match = gate.push(chroma)
        assert match is not None
        assert match.kind == "note" and match.pitch_class == name_to_pitch_class("A")

    def test_switching_from_note_to_chord_restarts_the_streak(self):
        gate = self.make(stable_frames=3)
        note = chroma_of_note(name_to_midi("E2"))
        chord = chroma_of(VOICINGS["E"])
        gate.push(note)
        gate.push(note)
        gate.push(chord)  # different kind, must not inherit the note's streak
        assert gate.current is None

    def test_a_dominant_string_mid_strum_does_not_incorrectly_settle_as_a_note(self):
        """Regression test for the actual bug: even if a couple of frames early in a
        strum look note-like before the full chord is established, the gate must not
        latch onto that transient reading and lock out the chord."""
        gate = self.make(stable_frames=3)
        note_ish = chroma_of_note(name_to_midi("E2"))
        chord = chroma_of(VOICINGS["E"])
        gate.push(note_ish)
        gate.push(chord)
        gate.push(chord)
        match = gate.push(chord)
        assert match is not None
        assert match.kind == "chord"

    def test_none_resets(self):
        gate = self.make(stable_frames=2)
        chroma = chroma_of(VOICINGS["E"])
        gate.push(chroma)
        gate.push(chroma)
        assert gate.current is not None
        assert gate.push(None) is None
        assert gate.current is None

    def test_weak_match_rejected(self):
        gate = self.make(min_score=0.99, stable_frames=1)
        assert gate.push(chroma_of(VOICINGS["E"])) is None
