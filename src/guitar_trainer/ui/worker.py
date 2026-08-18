"""The bridge between the audio thread and the GUI.

Qt widgets may only be touched from the GUI thread, and DSP may not run on it without
making the interface stutter. So analysis lives on its own thread and reports results
through signals, which Qt delivers as queued calls on the GUI thread.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from ..audio.capture import DEFAULT_SAMPLE_RATE, AudioCapture
from ..audio.chroma import ChromaAnalyser
from ..audio.pitch import DEFAULT_HOP, DEFAULT_WINDOW, NoteGate, PitchResult, YinDetector


class AnalysisWorker(QObject):
    """Runs the detection loop on a worker thread."""

    #: Every analysed frame, gated or not — drives the live readout and level meter.
    frame_analysed = Signal(object)  # PitchResult | None
    #: A note that has held steady long enough to be trusted.
    note_stable = Signal(object)  # PitchResult
    #: The stable note ended.
    note_released = Signal()
    #: Normalised chroma vector, for chord matching.
    chroma_updated = Signal(object)  # np.ndarray
    #: Pitch class of the lowest clearly-sounding partial, or None. Always emitted
    #: immediately before chroma_updated for the same frame, so a slot that stores it
    #: and one that consumes it can rely on the value being current.
    bass_changed = Signal(object)  # int | None
    #: Input level, for the meter.
    level_changed = Signal(float)
    #: Something went wrong opening or reading the stream.
    error = Signal(str)

    def __init__(
        self,
        capture: AudioCapture,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        window: int = DEFAULT_WINDOW,
        hop: int = DEFAULT_HOP,
    ) -> None:
        super().__init__()
        self.capture = capture
        self.window = window
        self.hop = hop
        self.detector = YinDetector(sample_rate)
        self.gate = NoteGate()
        self.chroma = ChromaAnalyser(sample_rate)
        # ChromaAnalyser needs more history than the pitch detector does (its window
        # is 8192 samples, for low-end frequency resolution on a chord, vs pitch's
        # 4096) — read enough for whichever analyser needs more, and let _process()
        # slice back down to `self.window` for the pitch path. Reading only
        # `self.window` here used to starve chroma.analyse() of samples on every
        # single frame, so it silently returned None forever: chroma_updated and
        # bass_changed never fired in the running app, which meant Free Detect's
        # note-vs-chord classification and chord practice scoring never worked live,
        # even though both were fully covered by tests that fed hand-built
        # correctly-sized frames straight into _process() and never exercised this
        # read. See test_worker.py::TestBassOrdering and AGENTS.md.
        self._read_window = max(self.window, self.chroma.window)
        self._running = False
        self._suppress_until = 0.0
        self._had_stable_note = False

    # ------------------------------------------------------------- controls

    def set_noise_gate(self, rms_threshold: float) -> None:
        self.gate.rms_threshold = rms_threshold
        # Chord recognition (chroma) had its own fixed, much lower threshold; tying it
        # to the same user-configurable gate means calibrating once covers both paths.
        self.chroma.rms_threshold = rms_threshold

    def set_min_harmonic_ratio(self, ratio: float) -> None:
        self.detector.min_harmonic_ratio = ratio

    def suppress_until(self, monotonic_time: float) -> None:
        """Ignore input up to a point in time.

        Used to blank out the metronome click so speaker bleed into the microphone
        can't be scored as a played note.
        """
        self._suppress_until = max(self._suppress_until, monotonic_time)

    def stop(self) -> None:
        self._running = False

    # ----------------------------------------------------------------- loop

    def run(self) -> None:
        """Entry point, connected to the thread's ``started`` signal."""
        import time

        self._running = True
        try:
            self.capture.start()
        except Exception as exc:
            self.error.emit(f"Could not open the audio input: {exc}")
            return

        interval = self.hop / self.detector.sample_rate
        next_frame = time.monotonic()

        try:
            while self._running:
                now = time.monotonic()
                sleep_for = next_frame - now
                if sleep_for > 0:
                    QThread.usleep(int(sleep_for * 1_000_000))
                next_frame += interval
                if next_frame < time.monotonic() - interval:
                    # Fell behind (a slow frame, or the machine was suspended); resync
                    # rather than spinning through a backlog of stale windows.
                    next_frame = time.monotonic() + interval

                frame = self.capture.buffer.read_latest(self._read_window)
                self._process(frame, time.monotonic())
        finally:
            self.capture.stop()

    def _process(self, frame: np.ndarray, now: float) -> None:
        # `frame` may carry more history than pitch detection wants (see
        # _read_window) — trim back down to the tuned pitch window so YIN's behaviour
        # doesn't silently change with however much extra chroma needs.
        pitch_frame = frame[-self.window :]
        rms = float(np.sqrt(np.mean(pitch_frame.astype(np.float64) ** 2)))
        self.level_changed.emit(rms)

        if now < self._suppress_until:
            return

        # Below the configured gate, don't even run detection — this is what makes
        # the Gate control actually mean something for the Tuner and Free Detect live
        # readouts. Previously only the practice modes' "stable note" scoring checked
        # it; the raw per-frame result shown here bypassed it entirely, so quiet
        # keyboard clicks and talking still lit up the tuner needle and note readout
        # no matter how the gate was set.
        result: PitchResult | None = (
            self.detector.detect(pitch_frame) if rms >= self.gate.rms_threshold else None
        )
        self.frame_analysed.emit(result)

        stable = self.gate.push(result)
        if stable is not None:
            self._had_stable_note = True
            self.note_stable.emit(stable)
        elif self.gate.current is None and self._had_stable_note:
            self._had_stable_note = False
            self.note_released.emit()

        chroma = self.chroma.analyse(frame)
        if chroma is not None:
            self.bass_changed.emit(self.chroma.bass_pitch_class(frame))
            self.chroma_updated.emit(chroma)


class AnalysisThread:
    """Owns the worker and its thread, and shuts both down cleanly.

    Qt requires the thread be stopped before the objects living on it are destroyed;
    doing that in one place avoids the usual "QThread destroyed while still running"
    crash on exit.
    """

    def __init__(self, capture: AudioCapture, **kwargs) -> None:
        self.capture = capture
        self.thread = QThread()
        self.worker = AnalysisWorker(capture, **kwargs)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout_ms: int = 2000) -> None:
        self.worker.stop()
        self.thread.quit()
        if not self.thread.wait(timeout_ms):
            self.thread.terminate()
            self.thread.wait()

    @property
    def is_running(self) -> bool:
        return self.thread.isRunning()
