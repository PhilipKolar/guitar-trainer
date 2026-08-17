"""Tests against real recorded guitar audio.

This is what closes the loop on "detection feels erratic" reports: run
scripts/record_fixtures.py once to capture real single notes on your actual guitar,
pickup and mic, and every one of them gets checked here on every test run after that.
Recording nothing is fine — pytest's empty-parametrize behaviour turns each test into
a single visible "skipped" entry (reason: empty parameter set) rather than an error,
so an unrecorded fixture set doesn't break the suite for anyone who hasn't recorded
anything yet.

Everything else in the DSP suite runs against synthesised audio, which validates the
*algorithm* but can't catch anything specific to one particular instrument/room/player
— exactly the class of bug a vague "it feels jumpy" report points at. This is the
complement to that, not a replacement for it: a fixed set of recordings is one
room/session/mic, not a model of every possible input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import synth
from guitar_trainer.audio.pitch import DEFAULT_WINDOW, NoteGate, YinDetector
from guitar_trainer.core.notes import name_to_midi, note_name

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio" / "notes"
NOTE_FILES = sorted(FIXTURES_DIR.glob("*.wav")) if FIXTURES_DIR.exists() else []


def _stable_readings(samples: np.ndarray, sample_rate: int) -> list:
    """Run a real recording through the exact detector + stability gate the app
    uses, frame by frame the way the worker thread does it, and return every
    reading the gate accepted, in order."""
    detector = YinDetector(sample_rate)
    gate = NoteGate()
    hop = DEFAULT_WINDOW // 4
    readings = []
    for start in range(0, len(samples) - DEFAULT_WINDOW, hop):
        stable = gate.push(detector.detect(samples[start : start + DEFAULT_WINDOW]))
        if stable is not None:
            readings.append(stable)
    return readings


class TestRecordedNotes:
    @pytest.mark.parametrize("path", NOTE_FILES, ids=lambda p: p.stem)
    def test_settles_on_the_correct_note(self, path):
        expected_midi = name_to_midi(path.stem)
        sample_rate, samples = synth.load_wav(path)
        readings = _stable_readings(samples, sample_rate)

        assert readings, f"{path.name}: never settled on a stable note"
        wrong = [r.name for r in readings if r.midi != expected_midi]
        assert not wrong, (
            f"{path.name}: expected only {note_name(expected_midi)}, also settled on "
            f"{sorted(set(wrong))} at some point ({len(wrong)}/{len(readings)} "
            f"readings wrong) — this is exactly the 'jumps around between notes' bug"
        )

    @pytest.mark.parametrize("path", NOTE_FILES, ids=lambda p: p.stem)
    def test_settles_reasonably_in_tune(self, path):
        """A loose tolerance — this is meant to catch gross detuning/octave-adjacent
        bugs in the detector, not grade your playing technique. A guitar tuned with a
        hardware tuner should land well inside this."""
        expected_midi = name_to_midi(path.stem)
        sample_rate, samples = synth.load_wav(path)
        readings = [r for r in _stable_readings(samples, sample_rate) if r.midi == expected_midi]
        if not readings:
            pytest.skip("covered by test_settles_on_the_correct_note")

        median_cents = float(np.median([r.cents for r in readings]))
        assert abs(median_cents) < 30, f"{path.name}: median {median_cents:+.0f} cents off"


def test_fixture_coverage():
    """Not a correctness check — just makes the current recorded coverage visible
    in test output (run with -s to see it) rather than something you have to go
    check the filesystem for."""
    if not NOTE_FILES:
        pytest.skip("no recorded fixtures yet — see scripts/record_fixtures.py")
    print(f"\n{len(NOTE_FILES)} recorded note fixture(s): {[p.stem for p in NOTE_FILES]}")
