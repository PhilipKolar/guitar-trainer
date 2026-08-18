import numpy as np
import pytest

import synth
from guitar_trainer.audio.chroma import CHROMA_WINDOW, ChromaAnalyser
from guitar_trainer.core.chords import (
    COMMON_OPEN_CHORDS,
    Chord,
    ChordMatcher,
    NoteOrChordClassifier,
    NoteOrChordGate,
    NoteOrChordMatch,
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
    """Synthesise a single plucked note and return its chroma vector.

    Only meaningful for notes whose 2nd harmonic still falls inside the
    chord-tuned fold band (below ~250Hz fundamentals) — the fold is deliberately
    narrow (see PEAK_MAX_FREQ), and higher notes rely on single_series_pitch_class
    instead. Use classify_note_like_the_app_would for a note anywhere on the neck.
    """
    audio = synth.plucked_string(midi_to_freq(midi), duration=0.6, attack_noise=0.05, **kwargs)
    analyser = ChromaAnalyser(SR)
    result = None
    for start in range(2048, len(audio) - CHROMA_WINDOW, 1024):
        out = analyser.analyse(audio[start : start + CHROMA_WINDOW])
        if out is not None:
            result = out
    assert result is not None, "no chroma produced"
    return result


def classify_note_like_the_app_would(midi: int, **kwargs) -> NoteOrChordMatch:
    """Synthesise a single plucked note and classify it the way NoteOrChordGate
    actually does: single_series_pitch_class first (the mechanism that covers the
    whole neck), classify() as the fallback (the chord-tuned fold, which only has
    real content for low notes) — not classify() alone, which is no longer a
    complete guarantee on its own since the presence-chroma redesign.
    """
    audio = synth.plucked_string(midi_to_freq(midi), duration=0.6, attack_noise=0.05, **kwargs)
    analyser = ChromaAnalyser(SR)
    classifier = NoteOrChordClassifier(all_chords())
    result = None
    for start in range(2048, len(audio) - CHROMA_WINDOW, 1024):
        frame = audio[start : start + CHROMA_WINDOW]
        series = analyser.single_series_pitch_class(frame)
        if series is not None:
            pitch_class, fraction = series
            result = NoteOrChordMatch("note", pitch_class, None, fraction)
            continue
        chroma = analyser.analyse(frame)
        if chroma is not None:
            result = classifier.classify(chroma)
    assert result is not None, "never classified as anything"
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
        """220Hz (A3), not 440Hz (A4): the fold band is deliberately narrow (see
        PEAK_MAX_FREQ) and only has room for both a note's fundamental AND its
        2nd harmonic — which analyse()'s MIN_STRONG_PEAKS requires — when the
        fundamental itself is low enough. Higher notes are covered instead by
        TestNoteOrChordClassifier's classify_note_like_the_app_would cases."""
        analyser = ChromaAnalyser(SR)
        audio = synth.plucked_string(220.0, duration=0.5)
        chroma = analyser.analyse(audio[2048 : 2048 + CHROMA_WINDOW])
        assert chroma is not None
        assert int(np.argmax(chroma)) == name_to_pitch_class("A")

    def test_bass_pitch_class_finds_the_lowest_note(self):
        analyser = ChromaAnalyser(SR)
        audio = synth.strum([name_to_midi(n) for n in VOICINGS["G"]], duration=0.8)
        bass = analyser.bass_pitch_class(audio[4096 : 4096 + CHROMA_WINDOW])
        assert bass == name_to_pitch_class("G")

    def test_a_lone_tone_produces_no_chroma_at_all(self):
        """MIN_STRONG_PEAKS: a signal with only one real spectral peak is treated
        as no content, same as silence — this is what a real live report turned
        out to be (an incidental environmental tone confidently read as a note)."""
        analyser = ChromaAnalyser(SR, rms_threshold=0.01)
        t = np.arange(CHROMA_WINDOW) / SR
        pure = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        assert analyser.analyse(pure) is None
        assert analyser.single_series_pitch_class(pure) is None

    def test_a_real_note_with_two_peaks_still_produces_chroma(self):
        analyser = ChromaAnalyser(SR, rms_threshold=0.01)
        audio = synth.plucked_string(220.0, duration=0.5)
        assert analyser.analyse(audio[2048 : 2048 + CHROMA_WINDOW]) is not None

    def test_bass_pitch_class_on_silence(self):
        analyser = ChromaAnalyser(SR)
        assert analyser.bass_pitch_class(np.zeros(CHROMA_WINDOW, dtype=np.float32)) is None

    def test_reset_clears_smoothing(self):
        analyser = ChromaAnalyser(SR)
        audio = synth.strum([name_to_midi(n) for n in VOICINGS["E"]], duration=0.8)
        analyser.analyse(audio[4096 : 4096 + CHROMA_WINDOW])
        analyser.reset()
        assert not analyser._history


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
        """Through the same two-mechanism path the app actually uses (series
        first, the chord-tuned fold as fallback) — classify() alone is no longer
        a complete guarantee for every note on the neck, only single_series_
        pitch_class is (see PEAK_MAX_FREQ vs SERIES_MAX_FREQ)."""
        midi = name_to_midi(note)
        match = classify_note_like_the_app_would(midi)
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


class TestSingleSeriesDetection:
    """ChromaAnalyser.single_series_pitch_class — the pre-fold check that tells a
    single ringing string apart from a chord, which pitch-class space cannot do
    (a weak-fundamental E2 folds to the same classes as chords do).
    """

    def analyser(self):
        return ChromaAnalyser(SR)

    def tone(self, freqs_amps, duration=0.4):
        t = np.arange(int(SR * duration)) / SR
        out = np.zeros_like(t)
        for freq, amp in freqs_amps:
            out += amp * np.sin(2 * np.pi * freq * t)
        return (0.3 * out / np.max(np.abs(out))).astype(np.float32)

    def test_a_plain_harmonic_series_is_that_note(self):
        e2 = midi_to_freq(name_to_midi("E2"))
        audio = self.tone([(e2, 1.0), (2 * e2, 0.8), (3 * e2, 0.6), (4 * e2, 0.3)])
        result = self.analyser().single_series_pitch_class(audio[: CHROMA_WINDOW])
        assert result is not None
        pc, fraction = result
        assert pc == name_to_pitch_class("E")
        assert fraction > 0.9

    def test_a_series_with_a_buried_fundamental_is_still_that_note(self):
        """The real E2 case: the fundamental is far below the peak floor, only
        harmonics 2/3/4 are visible — still one string, and still an E."""
        e2 = midi_to_freq(name_to_midi("E2"))
        audio = self.tone([(2 * e2, 0.5), (3 * e2, 1.0), (4 * e2, 0.3)])
        result = self.analyser().single_series_pitch_class(audio[: CHROMA_WINDOW])
        assert result is not None
        assert result[0] == name_to_pitch_class("E")

    def test_a_major_triad_voiced_as_harmonics_is_not_claimed_as_a_note(self):
        """The trap: open D major (D3 A3 D4 F#4) is exactly harmonics 2,3,4,5 of a
        sub-octave D2. The played third at k=5 is far too strong to be a decayed
        harmonic — the claim must be refused."""
        d2 = midi_to_freq(name_to_midi("D2"))
        audio = self.tone([(2 * d2, 1.0), (3 * d2, 0.8), (4 * d2, 0.7), (5 * d2, 0.7)])
        assert self.analyser().single_series_pitch_class(audio[: CHROMA_WINDOW]) is None

    def test_a_real_seventh_harmonic_does_not_veto_a_genuine_single_note(self):
        """The chord-tone veto's first version excluded every harmonic number
        outside a hand-picked "safe" set {1,2,3,4,6,8,12} — which wrongly caught
        k=7, a perfectly ordinary string partial, and broke a real recorded E2
        (measured: its 7th harmonic legitimately carries ~20% of its strong-peak
        weight). Only k%5==0 (5, 10, 15...) aliases to a major third, which is the
        one interval a natural harmonic series cannot otherwise produce — that is
        the actual danger this veto exists to catch, not general richness."""
        e2 = midi_to_freq(name_to_midi("E2"))
        audio = self.tone([(2 * e2, 0.6), (3 * e2, 1.0), (7 * e2, 0.4)])
        result = self.analyser().single_series_pitch_class(audio[:CHROMA_WINDOW])
        assert result is not None
        assert result[0] == name_to_pitch_class("E")

    def test_a_power_chord_is_not_a_series(self):
        """Root + fifth (ratio 1.5) never fits one integer-harmonic series."""
        e2 = midi_to_freq(name_to_midi("E2"))
        b2 = midi_to_freq(name_to_midi("B2"))
        audio = self.tone([(e2, 1.0), (b2, 0.9), (2 * e2, 0.6), (2 * b2, 0.5)])
        assert self.analyser().single_series_pitch_class(audio[: CHROMA_WINDOW]) is None

    def test_floor_dust_cannot_break_the_fit(self):
        """Peaks below SERIES_STRONG_PEAK are excluded from the test — quiet
        sympathetic resonances (real recordings show them at 4-8%) must not stop a
        clearly single note from being recognised as one."""
        a2 = midi_to_freq(name_to_midi("A2"))
        audio = self.tone(
            [(a2, 1.0), (2 * a2, 0.7), (3 * a2, 0.5), (207.0, 0.06), (261.0, 0.05)]
        )
        result = self.analyser().single_series_pitch_class(audio[: CHROMA_WINDOW])
        assert result is not None
        assert result[0] == name_to_pitch_class("A")

    def test_quiet_input_is_none(self):
        silent = np.zeros(CHROMA_WINDOW, dtype=np.float32)
        assert self.analyser().single_series_pitch_class(silent) is None

    def test_finds_a_high_note_the_chord_fold_band_cannot_represent_at_all(self):
        """The real bug: A4 (440Hz)'s 2nd harmonic (880Hz) sits above the
        chord-tuned fold band (PEAK_MAX_FREQ=500) entirely, so analyse() legitimately
        has nothing to fold for it — but single_series_pitch_class scans its own,
        much wider band (SERIES_MAX_FREQ) and must still find it. Regression test
        for every synthesised note above ~B4 going undetected."""
        analyser = self.analyser()
        audio = self.tone([(440.0, 1.0), (880.0, 0.5), (1320.0, 0.3)])
        frame = audio[:CHROMA_WINDOW]
        assert analyser.analyse(frame) is None  # confirms the fold really is empty
        result = analyser.single_series_pitch_class(frame)
        assert result is not None
        assert result[0] == name_to_pitch_class("A")


class TestNoteOrChordGate:
    def make(self, **kwargs):
        return NoteOrChordGate(NoteOrChordClassifier(all_chords()), **kwargs)

    def test_a_fresh_chord_needs_five_agreeing_frames(self):
        """A strum physically takes ~100ms for all strings to sound, and half-formed
        strums read as neighbouring chords — so a brand-new chord identity needs
        FRESH_CHORD_FRAMES (5) agreeing frames, not the ordinary 3."""
        gate = self.make()
        chroma = chroma_of(VOICINGS["E"])
        for _ in range(4):
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
        gate = self.make()
        note_ish = chroma_of_note(name_to_midi("E2"))
        chord = chroma_of(VOICINGS["E"])
        gate.push(note_ish)
        for _ in range(4):
            gate.push(chord)
        match = gate.push(chord)
        assert match is not None
        assert match.kind == "chord"

    def test_none_clears_the_reading_but_not_instantly_the_memory(self):
        gate = self.make(stable_frames=2)
        chroma = chroma_of(VOICINGS["E"])
        for _ in range(5):
            gate.push(chroma)
        assert gate.current is not None
        assert gate.push(None) is None
        assert gate.current is None

    def test_weak_match_rejected(self):
        gate = self.make(min_score=0.99, confirm_score=0.99, stable_frames=1,
                         fresh_chord_frames=1)
        assert gate.push(chroma_of(VOICINGS["E"])) is None

    def test_identity_change_mid_decay_needs_an_onset(self):
        """The physics rule: once E major is established, a decaying signal that
        starts template-matching some other identity (real E recordings drift to
        Bsus4, then E5, then a bare B as strings fade) must NOT be believed —
        nothing new was played. A genuine re-pluck (RMS rise) unlocks it."""
        gate = self.make()
        e_major = chroma_of(VOICINGS["E"])
        e5 = chroma_of(VOICINGS["E5"])
        for _ in range(5):
            gate.push(e_major, rms=0.1)
        assert gate.current is not None and gate.current.chord == Chord.parse("E")
        # Decay: quieter and quieter frames that now read as E5 — all suppressed.
        for rms in [0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04]:
            assert gate.push(e5, rms=rms) is None
        # A re-strum (clear RMS rise over the decay floor) is believed.
        confirmed = None
        for _ in range(5):
            confirmed = gate.push(e5, rms=0.2)
        assert confirmed is not None and confirmed.chord == Chord.parse("E5")

    def test_power_chord_fills_in_to_the_full_chord_without_an_onset(self):
        """Root+fifth speak first in a strum; the third emerging a few frames later
        is the same sound event, not a new one."""
        gate = self.make()
        e5 = chroma_of(VOICINGS["E5"])
        e_major = chroma_of(VOICINGS["E"])
        for _ in range(5):
            gate.push(e5, rms=0.1)
        assert gate.current is not None and gate.current.chord == Chord.parse("E5")
        confirmed = None
        for rms in [0.09, 0.09, 0.08, 0.08, 0.08]:
            confirmed = gate.push(e_major, rms=rms)  # decaying — no onset
        assert confirmed is not None and confirmed.chord == Chord.parse("E")

    def test_series_frames_classify_as_that_note_with_the_fit_as_confidence(self):
        """A single-series frame is a proven single note even when the folded
        chroma is too skewed for the note template to score well — the fit fraction
        itself is the confidence (this is the real E2 weak-fundamental case)."""
        gate = self.make()
        chroma = chroma_of_note(name_to_midi("E2"))
        confirmed = None
        for _ in range(3):
            confirmed = gate.push(chroma, series=(name_to_pitch_class("E"), 0.97))
        assert confirmed is not None
        assert confirmed.kind == "note"
        assert confirmed.pitch_class == name_to_pitch_class("E")
        assert confirmed.score == pytest.approx(0.97)

    def test_series_is_used_even_when_chroma_is_none(self):
        """A note fretted high on the neck has nowhere to appear in the
        deliberately narrow chord-tuned fold (analyse() legitimately returns None
        for it), but single_series_pitch_class's own wider scan still finds it —
        the gate must use that, not treat a None chroma as no evidence at all."""
        gate = self.make()
        confirmed = None
        for _ in range(3):
            confirmed = gate.push(None, series=(name_to_pitch_class("A"), 0.95))
        assert confirmed is not None
        assert confirmed.kind == "note"
        assert confirmed.pitch_class == name_to_pitch_class("A")

    def test_memory_survives_loud_but_unnameable_frames(self):
        """Frames that are loud but momentarily unclassifiable must not erode the
        held identity's memory — only genuine quiet does."""
        gate = self.make(forget_after=3)
        e_major = chroma_of(VOICINGS["E"])
        e5 = chroma_of(VOICINGS["E5"])
        for _ in range(5):
            gate.push(e_major, rms=0.1)
        # Many loud-but-unnameable frames: memory must hold, so a decaying E5
        # takeover attempt right after is still refused.
        for _ in range(10):
            gate.push(None, rms=0.05)
        for _ in range(6):
            assert gate.push(e5, rms=0.04) is None

    def test_sustained_quiet_forgets_the_held_identity(self):
        gate = self.make(forget_after=3)
        e_major = chroma_of(VOICINGS["E"])
        e5 = chroma_of(VOICINGS["E5"])
        for _ in range(5):
            gate.push(e_major, rms=0.1)
        for _ in range(3):
            gate.push(None, rms=0.001)  # genuine quiet
        confirmed = None
        for _ in range(5):
            confirmed = gate.push(e5, rms=0.02)  # soft playing, fresh context
        assert confirmed is not None and confirmed.chord == Chord.parse("E5")
