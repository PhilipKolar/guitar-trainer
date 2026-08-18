"""Tests against real recorded guitar audio through the chord-recognition path.

The chord counterpart to test_real_recordings.py: run
scripts/record_fixtures.py --kind chords once to capture real chord strums on your
actual guitar/pickup/mic, and every one of them gets checked here on every test run
after that. Recording nothing is fine — pytest's empty-parametrize behaviour turns
each test into a single visible "skipped" entry rather than an error.

This exercises the exact same objects Free Detect uses live
(NoteOrChordClassifier(all_chords()) + NoteOrChordGate — see
ui/modes.py::FreeDetectPanel._CLASSIFIER) over a sliding window sized and hopped the
way AnalysisWorker actually reads the buffer (see ui/worker.py::AnalysisWorker), not
a hand-picked convenient size — that distinction is what let the window-size bug
(chroma silently never firing live) go undetected by every other chord test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import synth
from guitar_trainer.audio.chroma import ChromaAnalyser
from guitar_trainer.audio.pitch import DEFAULT_HOP, DEFAULT_WINDOW
from guitar_trainer.core.chords import Chord, NoteOrChordClassifier, NoteOrChordGate, all_chords
from guitar_trainer.core.notes import name_to_midi

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio" / "chords"
CHORD_FILES = sorted(FIXTURES_DIR.glob("*.wav")) if FIXTURES_DIR.exists() else []


def _stable_chord_readings(samples: np.ndarray, sample_rate: int) -> list:
    """Run a real recording through the exact chroma extraction and note-or-chord
    classification the app uses for Free Detect, frame by frame the way
    AnalysisWorker._process does it — RMS and single-series analysis included, since
    the gate's onset/decay physics run on those — and return every stable
    classification the gate accepted, in order.

    ``rms_threshold=0.01`` mirrors the app's default noise gate
    (``Config.noise_gate``), which AnalysisWorker.set_noise_gate applies to the
    chroma path at startup; the analyser's own bare default is lower."""
    analyser = ChromaAnalyser(sample_rate, rms_threshold=0.01)
    gate = NoteOrChordGate(NoteOrChordClassifier(all_chords()))
    window = max(DEFAULT_WINDOW, analyser.window)
    readings = []
    for start in range(0, len(samples) - window, DEFAULT_HOP):
        frame = samples[start : start + window]
        rms = float(np.sqrt(np.mean(frame[-analyser.window :].astype(np.float64) ** 2)))
        chroma = analyser.analyse(frame)
        bass = analyser.bass_pitch_class(frame) if chroma is not None else None
        series = analyser.single_series_pitch_class(frame) if chroma is not None else None
        stable = gate.push(chroma, bass, rms=rms, series=series)
        if stable is not None:
            readings.append(stable)
    return readings


class TestRecordedChords:
    @pytest.mark.parametrize("path", CHORD_FILES, ids=lambda p: p.stem)
    def test_settles_on_the_correct_chord(self, path):
        expected = Chord.parse(path.stem)
        sample_rate, samples = synth.load_wav(path)
        readings = _stable_chord_readings(samples, sample_rate)
        chord_readings = [r for r in readings if r.kind == "chord"]

        assert chord_readings, (
            f"{path.name}: never settled on a stable chord — either it was "
            f"misclassified as a single note throughout, or nothing cleared the gate"
        )
        wrong = sorted({r.chord.name() for r in chord_readings if r.chord != expected})
        assert not wrong, (
            f"{path.name}: expected only {expected.name()}, also settled on "
            f"{wrong} at some point — this is exactly the 'jumps around between "
            f"chords' bug"
        )


NOTE_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio" / "notes"
NOTE_FILES = sorted(NOTE_FIXTURES_DIR.glob("*.wav")) if NOTE_FIXTURES_DIR.exists() else []


class TestRecordedNotesThroughFreeDetect:
    """The note fixtures through the same chord-recognition pipeline.

    Free Detect suppresses the YIN note display whenever this path claims a chord
    is sounding, so a single note misread here doesn't just show a wrong label — it
    blocks the correct note from ever being shown. That is exactly what made "it
    can't detect an E note" happen live: before the presence-chroma redesign, every
    one of these twelve fixtures produced wrong chord or wrong-note stables (E2
    read as a stable B note plus E5/B7/C7 chords, because its 3rd harmonic dwarfs
    its fundamental on this instrument).
    """

    @pytest.mark.parametrize("path", NOTE_FILES, ids=lambda p: p.stem)
    def test_a_single_note_is_never_reported_as_a_chord_or_wrong_note(self, path):
        expected_pc = name_to_midi(path.stem) % 12
        sample_rate, samples = synth.load_wav(path)
        readings = _stable_chord_readings(samples, sample_rate)

        chord_stables = sorted({r.chord.name() for r in readings if r.kind == "chord"})
        assert not chord_stables, (
            f"{path.name}: a single note produced stable chord readings "
            f"{chord_stables} — Free Detect would show a phantom chord and "
            f"suppress the real note"
        )
        wrong_notes = sorted(
            {r.pitch_class for r in readings if r.kind == "note" and r.pitch_class != expected_pc}
        )
        assert not wrong_notes, (
            f"{path.name}: expected only pitch class {expected_pc}, "
            f"also settled on {wrong_notes}"
        )


def test_fixture_coverage():
    """Not a correctness check — just makes the current recorded coverage visible
    in test output (run with -s to see it) rather than something you have to go
    check the filesystem for."""
    if not CHORD_FILES:
        pytest.skip("no recorded chord fixtures yet — see scripts/record_fixtures.py --kind chords")
    print(f"\n{len(CHORD_FILES)} recorded chord fixture(s): {[p.stem for p in CHORD_FILES]}")
