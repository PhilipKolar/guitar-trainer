"""AnalysisWorker tests.

These call ``_process()`` directly with synthetic frames rather than running the
worker's thread loop, so they exercise the gating logic without opening a real audio
stream.
"""

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
