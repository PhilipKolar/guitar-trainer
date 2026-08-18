"""AnalysisWorker tests.

These call ``_process()`` directly with synthetic frames rather than running the
worker's thread loop, so they exercise the gating logic without opening a real audio
stream.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

import synth  # noqa: E402
from guitar_trainer.audio.capture import AudioCapture  # noqa: E402
from guitar_trainer.core.notes import midi_to_freq, name_to_midi  # noqa: E402
from guitar_trainer.ui.worker import AnalysisWorker  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def make_worker(gate: float = 0.02) -> AnalysisWorker:
    worker = AnalysisWorker(AudioCapture())
    worker.set_noise_gate(gate)
    return worker


class TestGateWiring:
    """Regression coverage for the Gate control silently not gating the live readout.

    frame_analysed feeds Tuner.on_pitch and FreeDetectPanel's meter/detail text
    directly; only the practice-mode "stable note" path went through the configured
    RMS gate. That let quiet-but-audible sounds (talking, keyboard clicks) light up
    the tuner needle and free-detect readout no matter how the gate was set.
    """

    def test_quiet_frame_below_gate_yields_no_result(self, qapp):
        worker = make_worker(gate=0.05)
        seen = []
        worker.frame_analysed.connect(seen.append)

        freq = midi_to_freq(name_to_midi("A2"))
        quiet = synth.sine(freq, duration=0.1, amplitude=0.01)[:4096]
        worker._process(quiet, now=0.0)

        assert seen == [None]

    def test_loud_frame_above_gate_yields_a_result(self, qapp):
        worker = make_worker(gate=0.01)
        seen = []
        worker.frame_analysed.connect(seen.append)

        freq = midi_to_freq(name_to_midi("A2"))
        loud = synth.sine(freq, duration=0.1, amplitude=0.3)[:4096]
        worker._process(loud, now=0.0)

        assert len(seen) == 1 and seen[0] is not None
        assert seen[0].midi == name_to_midi("A2")

    def test_raising_the_gate_silences_a_previously_audible_frame(self, qapp):
        freq = midi_to_freq(name_to_midi("E2"))
        frame = synth.sine(freq, duration=0.1, amplitude=0.03)[:4096]

        low_gate = make_worker(gate=0.005)
        seen_low = []
        low_gate.frame_analysed.connect(seen_low.append)
        low_gate._process(frame, now=0.0)
        assert seen_low[0] is not None

        high_gate = make_worker(gate=0.5)
        seen_high = []
        high_gate.frame_analysed.connect(seen_high.append)
        high_gate._process(frame, now=0.0)
        assert seen_high == [None]

    def test_gate_change_takes_effect_on_the_next_frame(self, qapp):
        worker = make_worker(gate=0.005)
        freq = midi_to_freq(name_to_midi("E2"))
        frame = synth.sine(freq, duration=0.1, amplitude=0.03)[:4096]

        seen = []
        worker.frame_analysed.connect(seen.append)
        worker._process(frame, now=0.0)
        assert seen[-1] is not None

        worker.set_noise_gate(0.5)
        worker._process(frame, now=0.0)
        assert seen[-1] is None

    def test_chroma_threshold_tracks_the_same_gate(self, qapp):
        worker = make_worker(gate=0.2)
        assert worker.chroma.rms_threshold == pytest.approx(0.2)

    def test_level_meter_still_receives_every_frame(self, qapp):
        """The gate must not silence the level meter — the whole point of the meter
        is to show what's happening even while it's being filtered out."""
        worker = make_worker(gate=0.5)
        levels = []
        worker.level_changed.connect(levels.append)
        quiet = synth.sine(220.0, duration=0.1, amplitude=0.01)[:4096]
        worker._process(quiet, now=0.0)
        assert levels and levels[0] > 0


class TestReadWindowSizing:
    """Regression coverage for a real, previously-undetected bug: run()'s buffer
    read used to be sized only for the pitch detector (worker.window, 4096
    samples), but ChromaAnalyser needs at least its own window (8192) or it
    silently returns None. Every real frame in the running app was starved this
    way — chroma_updated/bass_changed never fired at all, live, no matter how
    clean the input was. TestBassOrdering below didn't catch it because it feeds
    a hand-built 8192-sample frame straight into _process(), bypassing the exact
    buffer read that was too short. This test performs that same read.
    """

    def test_read_window_covers_both_analysers(self, qapp):
        worker = make_worker()
        assert worker._read_window >= worker.window
        assert worker._read_window >= worker.chroma.window

    def test_the_actual_run_loop_buffer_read_is_large_enough_for_chroma(self, qapp):
        worker = make_worker(gate=0.005)
        midi_notes = [name_to_midi(n) for n in ["E2", "B2", "E3", "G#3", "B3", "E4"]]
        strum = synth.strum(midi_notes, duration=1.0)
        worker.capture.buffer.write(strum.astype(np.float32))

        # Exactly what run()'s loop does each iteration — not a hand-picked size.
        frame = worker.capture.buffer.read_latest(worker._read_window)

        chroma_events = []
        worker.chroma_updated.connect(chroma_events.append)
        worker._process(frame, now=0.0)

        assert chroma_events, "chroma never fired from a real run()-style buffer read"

    def test_pitch_detection_is_unaffected_by_chroma_needing_more_history(self, qapp):
        """The extra history read in for chroma must not change what the pitch
        detector sees — it should still analyse exactly worker.window samples."""
        worker = make_worker(gate=0.005)
        freq = midi_to_freq(name_to_midi("A2"))
        tone = synth.sine(freq, duration=1.0, amplitude=0.3)
        worker.capture.buffer.write(tone.astype(np.float32))

        frame = worker.capture.buffer.read_latest(worker._read_window)
        assert len(frame) > worker.window  # otherwise this test proves nothing

        seen = []
        worker.frame_analysed.connect(seen.append)
        worker._process(frame, now=0.0)

        assert seen[0] is not None and seen[0].midi == name_to_midi("A2")


class TestBassOrdering:
    def test_bass_emitted_before_chroma_for_the_same_frame(self, qapp):
        """FreeDetectPanel and ChordPracticePanel rely on this ordering to read a
        fresh bass note when a chord's chroma vector arrives."""
        worker = make_worker(gate=0.005)
        order = []
        worker.bass_changed.connect(lambda pc: order.append("bass"))
        worker.chroma_updated.connect(lambda c: order.append("chroma"))

        midi_notes = [name_to_midi(n) for n in ["G2", "B2", "D3", "G3", "B3", "G4"]]
        strum = synth.strum(midi_notes, duration=0.8)
        worker._process(strum[4096 : 4096 + 8192], now=0.0)

        assert order == ["bass", "chroma"]


class TestSnapshot:
    """snapshot_ready is what feeds the physics-aware note-or-chord gate: it must
    fire for EVERY processed frame — quiet ones included, since sustained quiet is
    how the gate's identity memory expires — and carry the frame's rms."""

    def test_fires_with_chroma_bass_and_series_on_a_loud_frame(self, qapp):
        worker = make_worker(gate=0.005)
        seen = []
        worker.snapshot_ready.connect(seen.append)
        midi_notes = [name_to_midi(n) for n in ["G2", "B2", "D3", "G3", "B3", "G4"]]
        strum = synth.strum(midi_notes, duration=0.8)
        worker._process(strum[4096 : 4096 + 8192], now=0.0)

        assert len(seen) == 1
        snapshot = seen[0]
        assert snapshot.chroma is not None
        assert snapshot.rms > 0.005
        assert snapshot.series is None  # a strummed chord is not one series

    def test_fires_on_quiet_frames_too_with_no_chroma(self, qapp):
        worker = make_worker(gate=0.05)
        seen = []
        worker.snapshot_ready.connect(seen.append)
        quiet = synth.sine(220.0, duration=0.3, amplitude=0.01)[:8192]
        worker._process(quiet, now=0.0)

        assert len(seen) == 1
        assert seen[0].chroma is None
        assert seen[0].rms > 0.0

    def test_a_single_note_carries_its_series_analysis(self, qapp):
        worker = make_worker(gate=0.005)
        seen = []
        worker.snapshot_ready.connect(seen.append)
        note = synth.plucked_string(midi_to_freq(name_to_midi("A2")), duration=0.5)
        worker._process(note[2048 : 2048 + 8192], now=0.0)

        assert len(seen) == 1
        assert seen[0].series is not None
        pc, fraction = seen[0].series
        assert pc == name_to_midi("A2") % 12
        assert fraction > 0.9


class TestSuppression:
    def test_suppression_window_blanks_everything(self, qapp):
        worker = make_worker(gate=0.001)
        worker.suppress_until(10.0)
        seen = []
        worker.frame_analysed.connect(seen.append)
        freq = midi_to_freq(name_to_midi("A2"))
        loud = synth.sine(freq, duration=0.1, amplitude=0.3)[:4096]
        worker._process(loud, now=5.0)
        assert seen == []
