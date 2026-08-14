"""The four mode panels: tuner, free detect, note practice, chord practice.

Each panel receives detections from the analysis worker and decides what to show. The
two practice panels share :class:`PracticePanel`, which owns the session engine, the
timing controls and the stats plumbing — the only differences between drilling notes
and drilling chords are what a challenge is and which detection stream answers it.
"""

from __future__ import annotations

import random

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..audio.metronome import MAX_BPM, MIN_BPM, BeatCounter, Metronome
from ..core.chords import CHORD_QUALITIES, QUALITY_SUFFIX, Chord, ChordMatcher, all_chords
from ..core.notes import (
    ALL_NOTES,
    NATURAL_NOTES,
    NoteSet,
    Tuning,
    pitch_class_name,
)
from ..core.session import (
    Attempt,
    ChallengePicker,
    ChordChallenge,
    NoteChallenge,
    Outcome,
    SessionEngine,
    chord_challenges,
    note_challenges,
)
from . import theme
from .fretboard import FretboardWidget
from .widgets import BigNoteLabel, CentsMeter


class ModePanel(QWidget):
    """Base for all mode panels.

    Panels are told when they become visible so only the active one reacts to
    detections; a background panel updating its widgets is wasted work and, in the
    practice modes, would score answers the user can't see.
    """

    #: Ask the main window to highlight a pitch class on the shared fretboard.
    highlight_requested = Signal(object)  # int | None

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.active = False

    def on_activated(self) -> None:
        self.active = True

    def on_deactivated(self) -> None:
        self.active = False

    def on_pitch(self, result) -> None:
        """A raw analysis frame (may be ``None``)."""

    def on_note_stable(self, result) -> None:
        """A note that held steady long enough to be trusted."""

    def on_note_released(self) -> None:
        """The stable note stopped sounding."""

    def on_chroma(self, chroma) -> None:
        """A chroma vector, for chord recognition."""


class TunerPanel(ModePanel):
    """Shows the nearest open string and how far off it is."""

    def __init__(self, tuning: Tuning, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tuning = tuning

        self.note_label = BigNoteLabel("—")
        self.meter = CentsMeter()

        self.detail = QLabel("Play a string")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")

        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont(self.status.font())
        font.setPointSize(14)
        font.setBold(True)
        self.status.setFont(font)

        self.strings = QLabel("")
        self.strings.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.strings.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        self._refresh_string_list()

        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(self.note_label)
        layout.addWidget(self.meter)
        layout.addWidget(self.status)
        layout.addWidget(self.detail)
        layout.addStretch(1)
        layout.addWidget(self.strings)

    def set_tuning(self, tuning: Tuning) -> None:
        self._tuning = tuning
        self._refresh_string_list()

    def _refresh_string_list(self) -> None:
        names = " · ".join(
            self._tuning.string_label(s) for s in reversed(range(self._tuning.string_count))
        )
        self.strings.setText(f"Target: {names}")

    def on_pitch(self, result) -> None:
        if not self.active:
            return
        if result is None:
            self.note_label.show_text(None)
            self.meter.set_cents(None)
            self.status.setText("")
            self.detail.setText("Play a string")
            return

        string, cents = self._tuning.nearest_string(result.freq)
        target = self._tuning.string_label(string)

        self.note_label.show_text(target)
        self.meter.set_cents(cents)

        if abs(cents) <= CentsMeter.IN_TUNE_CENTS:
            self.status.setText("In tune")
            self.status.setStyleSheet(f"color: {theme.IN_TUNE.name()};")
        else:
            direction = "Too sharp — loosen" if cents > 0 else "Too flat — tighten"
            self.status.setText(direction)
            self.status.setStyleSheet(f"color: {theme.OUT_OF_TUNE.name()};")

        self.detail.setText(
            f"string {self._tuning.string_count - string} · "
            f"{result.freq:.2f} Hz · {cents:+.1f} cents"
        )


class FreeDetectPanel(ModePanel):
    """Live readout of whatever is being played."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.note_label = BigNoteLabel("—")
        self.meter = CentsMeter()

        self.detail = QLabel("Play a note")
        self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")

        self.history = QLabel("")
        self.history.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont(self.history.font())
        font.setPointSize(16)
        self.history.setFont(font)
        self.history.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")

        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(self.note_label)
        layout.addWidget(self.meter)
        layout.addWidget(self.detail)
        layout.addStretch(1)
        layout.addWidget(QLabel("Recent:", alignment=Qt.AlignmentFlag.AlignCenter))
        layout.addWidget(self.history)

        self._recent: list[str] = []

    def on_pitch(self, result) -> None:
        if not self.active:
            return
        if result is None:
            self.meter.set_cents(None)
            self.detail.setText("Play a note")
            return
        self.meter.set_cents(result.cents)
        self.detail.setText(
            f"{result.freq:.2f} Hz · {result.cents:+.1f} cents · "
            f"confidence {result.confidence:.0%}"
        )

    def on_note_stable(self, result) -> None:
        if not self.active:
            return
        self.note_label.show_text(result.name, theme.DETECTED)
        self.highlight_requested.emit(result.pitch_class)

        if not self._recent or self._recent[-1] != result.note_only:
            self._recent.append(result.note_only)
            del self._recent[:-12]
            self.history.setText("  ".join(self._recent))

    def on_note_released(self) -> None:
        if not self.active:
            return
        self.note_label.show_text(None)
        self.highlight_requested.emit(None)

    def on_deactivated(self) -> None:
        super().on_deactivated()
        self.highlight_requested.emit(None)


class PracticePanel(ModePanel):
    """Shared practice loop for both notes and chords.

    Subclasses supply the challenge set and consume the detection stream that answers
    it; everything else — timing mode, metronome, scoring, stats — lives here.
    """

    #: An attempt was completed, for the stats store.
    attempt_recorded = Signal(object)  # Attempt
    #: Ask the worker to ignore input until a monotonic timestamp (metronome bleed).
    suppress_requested = Signal(float)

    mode_name = "practice"
    stats_prefix = "note:"

    def __init__(self, config, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.engine: SessionEngine | None = None
        self.weight_fn = None

        self.metronome = Metronome(
            bpm=config.bpm,
            beats_per_bar=config.beats_per_bar,
        )
        self.metronome.muted = config.metronome_muted
        self.beat_counter = BeatCounter(config.beats_per_challenge)

        self._build_ui()

        # One timer drives the countdown bar, the beat pulse and timeout checks.
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._on_tick)

    # ------------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        self.prompt_label = BigNoteLabel("Ready")
        self.feedback = QLabel("Press Start")
        self.feedback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont(self.feedback.font())
        font.setPointSize(14)
        self.feedback.setFont(font)

        self.countdown = QProgressBar()
        self.countdown.setTextVisible(False)
        self.countdown.setFixedHeight(6)
        self.countdown.setRange(0, 1000)
        self.countdown.hide()

        self.beat_label = QLabel("")
        self.beat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont(self.beat_label.font())
        font.setPointSize(18)
        self.beat_label.setFont(font)

        self.score_label = QLabel("")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_session)
        self.skip_button = QPushButton("Skip")
        self.skip_button.clicked.connect(self._skip)
        self.skip_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.skip_button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self._timing_controls())
        layout.addWidget(self.challenge_selector())
        layout.addStretch(1)
        layout.addWidget(self.prompt_label)
        layout.addWidget(self.countdown)
        layout.addWidget(self.feedback)
        layout.addWidget(self.beat_label)
        layout.addStretch(1)
        layout.addWidget(self.score_label)
        layout.addLayout(buttons)

    def _timing_controls(self) -> QWidget:
        box = QGroupBox("Timing")

        self.free_radio = QRadioButton("Free — take your time")
        self.rhythm_radio = QRadioButton("Rhythm — play on the beat")
        self.free_radio.setChecked(not self.config.rhythm_mode)
        self.rhythm_radio.setChecked(self.config.rhythm_mode)

        group = QButtonGroup(self)
        group.addButton(self.free_radio)
        group.addButton(self.rhythm_radio)
        self.rhythm_radio.toggled.connect(self._on_timing_changed)

        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(MIN_BPM, MAX_BPM)
        self.bpm_spin.setValue(self.config.bpm)
        self.bpm_spin.setSuffix(" BPM")
        self.bpm_spin.valueChanged.connect(self._on_bpm_changed)

        self.beats_spin = QSpinBox()
        self.beats_spin.setRange(1, 16)
        self.beats_spin.setValue(self.config.beats_per_challenge)
        self.beats_spin.setSuffix(" beats each")
        self.beats_spin.valueChanged.connect(self._on_beats_changed)

        self.mute_check = QCheckBox("Mute click")
        self.mute_check.setChecked(self.config.metronome_muted)
        self.mute_check.toggled.connect(self._on_mute_changed)

        row = QHBoxLayout()
        row.addWidget(self.free_radio)
        row.addWidget(self.rhythm_radio)
        row.addSpacing(12)
        row.addWidget(self.bpm_spin)
        row.addWidget(self.beats_spin)
        row.addWidget(self.mute_check)
        row.addStretch(1)
        box.setLayout(row)

        self._update_rhythm_controls()
        return box

    def _update_rhythm_controls(self) -> None:
        enabled = self.rhythm_radio.isChecked()
        for widget in (self.bpm_spin, self.beats_spin, self.mute_check):
            widget.setEnabled(enabled)

    def challenge_selector(self) -> QWidget:
        raise NotImplementedError

    def build_challenges(self) -> list:
        raise NotImplementedError

    # -------------------------------------------------------------- session

    @property
    def running(self) -> bool:
        return self.engine is not None

    def toggle_session(self) -> None:
        self.stop_session() if self.running else self.start_session()

    def start_session(self) -> None:
        try:
            challenges = self.build_challenges()
        except ValueError as exc:
            self.feedback.setText(str(exc))
            self.feedback.setStyleSheet(f"color: {theme.INCORRECT.name()};")
            return

        if not challenges:
            self.feedback.setText("Select at least one to practise")
            self.feedback.setStyleSheet(f"color: {theme.INCORRECT.name()};")
            return

        rhythm = self.rhythm_radio.isChecked()
        timeout = None
        if rhythm:
            timeout = self.metronome.seconds_per_beat * self.beats_spin.value()

        self.engine = SessionEngine(
            picker=ChallengePicker(
                challenges, rng=random.Random(), weight_fn=self.weight_fn
            ),
            timeout_seconds=timeout,
            on_challenge=self._on_challenge,
            on_result=self._on_result,
        )

        if rhythm:
            self.beat_counter = BeatCounter(self.beats_spin.value())
            try:
                self.metronome.start()
            except Exception as exc:
                self.feedback.setText(f"Metronome unavailable: {exc}")
            self.beat_counter.reset(self.metronome.beat_index)

        self.engine.start()
        self._timer.start()
        self.start_button.setText("Stop")
        self.skip_button.setEnabled(True)
        self.countdown.setVisible(rhythm)

    def stop_session(self) -> None:
        self._timer.stop()
        self.metronome.stop()
        if self.engine:
            self.engine.stop()
        self.engine = None
        self.start_button.setText("Start")
        self.skip_button.setEnabled(False)
        self.countdown.hide()
        self.beat_label.setText("")
        self.prompt_label.show_text("Ready")
        self.feedback.setText("Press Start")
        self.feedback.setStyleSheet(f"color: {theme.TEXT.name()};")
        self.highlight_requested.emit(None)

    def _skip(self) -> None:
        if self.engine:
            self.engine.skip()

    def _on_challenge(self, challenge) -> None:
        self.prompt_label.show_text(challenge.prompt, theme.TARGET)
        self.on_new_challenge(challenge)

    def on_new_challenge(self, challenge) -> None:
        """Hook for subclasses to update the fretboard or other hints."""

    def _on_result(self, attempt: Attempt) -> None:
        if attempt.outcome is Outcome.CORRECT:
            seconds = (attempt.response_ms or 0) / 1000
            self.feedback.setText(f"Correct — {seconds:.1f}s")
            self.feedback.setStyleSheet(f"color: {theme.CORRECT.name()};")
        elif attempt.outcome is Outcome.TIMEOUT:
            self.feedback.setText(f"Missed {attempt.prompt}")
            self.feedback.setStyleSheet(f"color: {theme.INCORRECT.name()};")
        else:
            self.feedback.setText(f"Skipped {attempt.prompt}")
            self.feedback.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")

        self.attempt_recorded.emit(attempt)
        self._update_score()

    def _update_score(self) -> None:
        if not self.engine or not self.engine.total:
            self.score_label.setText("")
            return
        median = self.engine.median_response_ms
        median_text = f" · median {median / 1000:.1f}s" if median else ""
        self.score_label.setText(
            f"{self.engine.correct_count}/{self.engine.total} correct "
            f"({self.engine.accuracy:.0%}){median_text}"
        )

    # ----------------------------------------------------------------- tick

    def _on_tick(self) -> None:
        if not self.engine:
            return

        if self.rhythm_radio.isChecked() and self.metronome.is_running:
            beat = self.metronome.beat_index
            elapsed, _ = self.beat_counter.update(beat)
            if elapsed:
                # Blank detection across the click so speaker bleed into the mic is
                # never scored as a played note.
                import time

                self.suppress_requested.emit(time.monotonic() + 0.05)
            self._update_beat_display(beat)

        remaining = self.engine.time_remaining
        if remaining is not None and self.engine.timeout_seconds:
            self.countdown.setValue(int(1000 * remaining / self.engine.timeout_seconds))

        self.engine.tick()

    def _update_beat_display(self, beat: int) -> None:
        per_bar = self.metronome.beats_per_bar
        position = beat % per_bar
        dots = ["●" if i == position else "○" for i in range(per_bar)]
        self.beat_label.setText(" ".join(dots))

    # ------------------------------------------------------------ lifecycle

    def on_deactivated(self) -> None:
        super().on_deactivated()
        if self.running:
            self.stop_session()

    def _on_timing_changed(self) -> None:
        self.config.rhythm_mode = self.rhythm_radio.isChecked()
        self._update_rhythm_controls()
        if self.running:
            # Restart so the new timing applies immediately rather than next session.
            self.stop_session()
            self.start_session()

    def _on_bpm_changed(self, value: int) -> None:
        self.config.bpm = value
        self.metronome.bpm = value
        if self.engine and self.engine.timeout_seconds is not None:
            self.engine.timeout_seconds = (
                self.metronome.seconds_per_beat * self.beats_spin.value()
            )

    def _on_beats_changed(self, value: int) -> None:
        self.config.beats_per_challenge = value
        self.beat_counter = BeatCounter(value)
        if self.engine and self.engine.timeout_seconds is not None:
            self.engine.timeout_seconds = self.metronome.seconds_per_beat * value

    def _on_mute_changed(self, muted: bool) -> None:
        self.config.metronome_muted = muted
        self.metronome.muted = muted


class NotePracticePanel(PracticePanel):
    """Drill single notes, answerable in any octave."""

    mode_name = "note"
    stats_prefix = "note:"

    def challenge_selector(self) -> QWidget:
        box = QGroupBox("Notes to practise")

        self.note_set_combo = QComboBox()
        self.note_set_combo.addItems(["All 12 notes", "Naturals only", "Custom"])
        index = self.note_set_combo.findText(self.config.note_set)
        self.note_set_combo.setCurrentIndex(max(0, index))
        self.note_set_combo.currentTextChanged.connect(self._on_set_changed)

        self.note_checks: dict[int, QCheckBox] = {}
        grid = QGridLayout()
        selected = set(self.config.custom_note_set or range(12))
        for pitch_class in range(12):
            check = QCheckBox(pitch_class_name(pitch_class, use_flats=self.config.use_flats))
            check.setChecked(pitch_class in selected)
            check.toggled.connect(self._on_custom_toggled)
            self.note_checks[pitch_class] = check
            grid.addWidget(check, pitch_class // 6, pitch_class % 6)

        layout = QVBoxLayout()
        layout.addWidget(self.note_set_combo)
        layout.addLayout(grid)
        box.setLayout(layout)

        self._sync_checks()
        return box

    def _on_set_changed(self, text: str) -> None:
        self.config.note_set = text
        self._sync_checks()

    def _sync_checks(self) -> None:
        """Checkboxes double as a display of the preset sets, and as the custom editor."""
        text = self.note_set_combo.currentText()
        custom = text == "Custom"
        for pitch_class, check in self.note_checks.items():
            check.setEnabled(custom)
            if not custom:
                preset = ALL_NOTES if text == "All 12 notes" else NATURAL_NOTES
                check.blockSignals(True)
                check.setChecked(pitch_class in preset.pitch_classes)
                check.blockSignals(False)

    def _on_custom_toggled(self) -> None:
        self.config.custom_note_set = [
            pc for pc, check in self.note_checks.items() if check.isChecked()
        ]

    def build_challenges(self) -> list[NoteChallenge]:
        text = self.note_set_combo.currentText()
        if text == "All 12 notes":
            note_set = ALL_NOTES
        elif text == "Naturals only":
            note_set = NATURAL_NOTES
        else:
            selected = [pc for pc, c in self.note_checks.items() if c.isChecked()]
            if not selected:
                raise ValueError("Select at least one note to practise")
            note_set = NoteSet("Custom", tuple(selected))
        return note_challenges(note_set, use_flats=self.config.use_flats)

    def on_new_challenge(self, challenge) -> None:
        self.highlight_requested.emit(None)

    def on_note_stable(self, result) -> None:
        if self.active and self.engine:
            self.engine.submit(result, label=result.name)


class ChordPracticePanel(PracticePanel):
    """Drill chords, in any voicing."""

    mode_name = "chord"
    stats_prefix = "chord:"

    def __init__(self, config, parent: QWidget | None = None) -> None:
        super().__init__(config, parent)
        self._matcher = ChordMatcher()
        self._last_bass: int | None = None
        self._streak = 0
        self._candidate: Chord | None = None

    def challenge_selector(self) -> QWidget:
        box = QGroupBox("Chords to practise")

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(
            [QUALITY_SUFFIX[q] or "major" for q in CHORD_QUALITIES]
        )
        self._qualities = list(CHORD_QUALITIES)
        self.quality_combo.currentIndexChanged.connect(self._refresh_chord_grid)

        self.chord_checks: dict[str, QCheckBox] = {}
        self._grid = QGridLayout()

        add_common = QPushButton("Common open chords")
        add_common.clicked.connect(self._select_common)
        clear = QPushButton("Clear all")
        clear.clicked.connect(self._clear_selection)

        self.selection_label = QLabel("")
        self.selection_label.setStyleSheet(f"color: {theme.TEXT_DIM.name()};")
        self.selection_label.setWordWrap(True)

        row = QHBoxLayout()
        row.addWidget(QLabel("Quality:"))
        row.addWidget(self.quality_combo)
        row.addWidget(add_common)
        row.addWidget(clear)
        row.addStretch(1)

        layout = QVBoxLayout()
        layout.addLayout(row)
        layout.addLayout(self._grid)
        layout.addWidget(self.selection_label)
        box.setLayout(layout)

        self._selected: set[str] = set(self.config.chord_symbols)
        self._refresh_chord_grid()
        return box

    def _refresh_chord_grid(self) -> None:
        """Show the twelve roots for the selected quality.

        120 checkboxes at once would be unusable, so the grid is one quality at a time
        while the selection persists across switches.
        """
        while self._grid.count():
            item = self._grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.chord_checks.clear()

        quality = self._qualities[max(0, self.quality_combo.currentIndex())]
        for root in range(12):
            symbol = Chord(root, quality).name(use_flats=self.config.use_flats)
            check = QCheckBox(symbol)
            check.setChecked(symbol in self._selected)
            check.toggled.connect(self._on_chord_toggled)
            self.chord_checks[symbol] = check
            self._grid.addWidget(check, root // 6, root % 6)

        self._update_selection_label()

    def _on_chord_toggled(self) -> None:
        for symbol, check in self.chord_checks.items():
            if check.isChecked():
                self._selected.add(symbol)
            else:
                self._selected.discard(symbol)
        self.config.chord_symbols = sorted(self._selected)
        self._update_selection_label()

    def _select_common(self) -> None:
        from ..core.config import default_chord_symbols

        self._selected.update(default_chord_symbols())
        self._refresh_chord_grid()
        self.config.chord_symbols = sorted(self._selected)

    def _clear_selection(self) -> None:
        self._selected.clear()
        self.config.chord_symbols = []
        self._refresh_chord_grid()

    def _update_selection_label(self) -> None:
        if self._selected:
            self.selection_label.setText(
                f"{len(self._selected)} selected: {', '.join(sorted(self._selected))}"
            )
        else:
            self.selection_label.setText("Nothing selected")

    def build_challenges(self) -> list[ChordChallenge]:
        chords = []
        for symbol in sorted(self._selected):
            try:
                chords.append(Chord.parse(symbol))
            except ValueError:
                continue
        if not chords:
            raise ValueError("Select at least one chord to practise")

        # Score only against the chords being drilled. Restricting the candidate set
        # removes most of the near-misses that make full 120-chord matching hard —
        # nothing can be mistaken for a chord that isn't in the exercise.
        self._matcher = ChordMatcher(chords)
        return chord_challenges(chords)

    def on_new_challenge(self, challenge) -> None:
        self._streak = 0
        self._candidate = None
        self.highlight_requested.emit(None)

    def on_chroma(self, chroma) -> None:
        if not (self.active and self.engine and chroma is not None):
            return

        match = self._matcher.match(chroma, self._last_bass)
        if match is None or match.score < 0.75:
            self._streak = 0
            self._candidate = None
            return

        if match.chord == self._candidate:
            self._streak += 1
        else:
            self._candidate = match.chord
            self._streak = 1

        # Hold for several frames so a slow strum isn't scored mid-way through.
        if self._streak >= 3:
            self.engine.submit(match, label=match.chord.name())

    def set_bass(self, pitch_class: int | None) -> None:
        self._last_bass = pitch_class
