"""The application window: toolbar, shared fretboard, and the mode tabs."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..audio.capture import AudioCapture, list_input_devices
from ..core.config import Config
from ..core.notes import TUNING_PRESETS, Tuning
from ..core.stats import StatsStore
from . import theme
from .fretboard import FretboardWidget, LabelMode
from .modes import ChordPracticePanel, FreeDetectPanel, NotePracticePanel, TunerPanel
from .stats_panel import StatsPanel
from .widgets import LevelMeter
from .worker import AnalysisThread


class MainWindow(QMainWindow):
    def __init__(self, config: Config | None = None, store: StatsStore | None = None) -> None:
        super().__init__()
        self.config = config or Config.load()
        self.store = store or StatsStore()
        self.setWindowTitle("Guitar Trainer")
        self.resize(1360, 760)

        self._session_id: int | None = None
        self._devices = list_input_devices()
        self._calibrating = False
        self._calibration_samples: list[float] = []

        # Debounced prompt-save, triggered by _mark_dirty() from every settings
        # change. Must exist before the panels are built, since wiring them up
        # connects their config_changed signal to it.
        self._dirty_timer = QTimer(self)
        self._dirty_timer.setSingleShot(True)
        self._dirty_timer.timeout.connect(self._save_config)

        self.fretboard = FretboardWidget(self.config.tuning())
        self.fretboard.set_label_mode(self._label_mode_from_config())
        self.fretboard.set_use_flats(self.config.use_flats)

        self._build_panels()
        self._build_layout()
        self._build_toolbar()

        self.analysis: AnalysisThread | None = None
        self._start_analysis()

        # Fallback safety net in case a change is ever made without going through
        # _mark_dirty(); the debounced save above is what actually covers normal use.
        self._save_timer = QTimer(self)
        self._save_timer.setInterval(30_000)
        self._save_timer.timeout.connect(self._save_config)
        self._save_timer.start()

    # -------------------------------------------------------------- building

    def _label_mode_from_config(self) -> LabelMode:
        for mode in LabelMode:
            if mode.value == self.config.label_mode:
                return mode
        return LabelMode.NONE

    def _build_panels(self) -> None:
        self.tuner = TunerPanel(self.config.tuning())
        self.free = FreeDetectPanel()
        self.note_practice = NotePracticePanel(self.config)
        self.chord_practice = ChordPracticePanel(self.config)
        self.stats_panel = StatsPanel(self.store)

        self.panels = [self.tuner, self.free, self.note_practice, self.chord_practice]
        for panel in self.panels:
            panel.highlight_requested.connect(self.fretboard.set_detected_pitch_class)
            panel.chord_highlight_requested.connect(self._on_chord_highlight)
            panel.config_changed.connect(self._mark_dirty)

        for panel in (self.note_practice, self.chord_practice):
            panel.attempt_recorded.connect(self._record_attempt)
            panel.suppress_requested.connect(self._suppress_analysis)

    def _build_layout(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.addTab(self.tuner, "Tuner")
        self.tabs.addTab(self.free, "Free detect")
        self.tabs.addTab(self.note_practice, "Note practice")
        self.tabs.addTab(self.chord_practice, "Chord practice")
        self.tabs.addTab(self.stats_panel, "Stats")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.fretboard, 2)
        layout.addWidget(self.tabs, 3)
        self.setCentralWidget(central)

        self.status_label = QLabel("")
        self.level_meter = LevelMeter()
        self.level_meter.set_threshold(self.config.noise_gate)
        self.statusBar().addPermanentWidget(self.status_label)
        self.statusBar().addPermanentWidget(self.level_meter)

        self._on_tab_changed(0)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Setup")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel(" Input: "))
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(240)
        for device in self._devices:
            self.device_combo.addItem(str(device), device.index)
        self._select_configured_device()
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        toolbar.addWidget(self.device_combo)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Tuning: "))
        self.tuning_combo = QComboBox()
        for tuning in TUNING_PRESETS:
            self.tuning_combo.addItem(tuning.name)
        index = self.tuning_combo.findText(self.config.tuning_name)
        self.tuning_combo.setCurrentIndex(max(0, index))
        self.tuning_combo.currentTextChanged.connect(self._on_tuning_changed)
        toolbar.addWidget(self.tuning_combo)

        self.fret_spin = QSpinBox()
        self.fret_spin.setRange(12, 24)
        self.fret_spin.setValue(self.config.fret_count)
        self.fret_spin.setSuffix(" frets")
        self.fret_spin.valueChanged.connect(self._on_fret_count_changed)
        toolbar.addWidget(self.fret_spin)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Show notes: "))
        self.label_combo = QComboBox()
        for mode in LabelMode:
            self.label_combo.addItem(mode.value)
        self.label_combo.setCurrentText(self._label_mode_from_config().value)
        self.label_combo.currentTextChanged.connect(self._on_label_mode_changed)
        toolbar.addWidget(self.label_combo)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Gate: "))
        self.gate_spin = QDoubleSpinBox()
        self.gate_spin.setRange(0.001, 0.2)
        self.gate_spin.setSingleStep(0.005)
        self.gate_spin.setDecimals(3)
        self.gate_spin.setValue(self.config.noise_gate)
        self.gate_spin.setToolTip(
            "Ignore audio quieter than this. Raise it if room noise triggers detection;\n"
            "lower it if quiet playing is missed. The marker on the level meter shows\n"
            "where it sits."
        )
        self.gate_spin.valueChanged.connect(self._on_gate_changed)
        toolbar.addWidget(self.gate_spin)

        self.calibrate_button = QPushButton("Calibrate")
        self.calibrate_button.setToolTip(
            "Stay quiet for a second (no playing, talking or typing) and this sets\n"
            "the gate just above whatever the microphone is picking up right now."
        )
        self.calibrate_button.clicked.connect(self._start_gate_calibration)
        toolbar.addWidget(self.calibrate_button)

        toolbar.addWidget(QLabel(" Purity: "))
        self.purity_spin = QDoubleSpinBox()
        self.purity_spin.setRange(0.0, 0.95)
        self.purity_spin.setSingleStep(0.05)
        self.purity_spin.setDecimals(2)
        self.purity_spin.setValue(self.config.min_harmonic_ratio)
        self.purity_spin.setToolTip(
            "How clean a tone has to be to count as a played note. Higher rejects\n"
            "sounds that pass the gate but aren't an instrument — keyboard clicks,\n"
            "knocks, coughs. Lower it if real notes on a noisy or distorted signal\n"
            "path are being missed. This does not filter out speech or singing —\n"
            "a voice is genuinely harmonic like a guitar string, so only distance\n"
            "and the noise gate help with that."
        )
        self.purity_spin.valueChanged.connect(self._on_purity_changed)
        toolbar.addWidget(self.purity_spin)

        flats = QAction("♭", self)
        flats.setCheckable(True)
        flats.setChecked(self.config.use_flats)
        flats.setToolTip("Name accidentals as flats instead of sharps")
        flats.toggled.connect(self._on_flats_toggled)
        toolbar.addAction(flats)

    def _select_configured_device(self) -> None:
        """Match the saved device by name; indices shift when devices come and go."""
        if self.config.device_name:
            index = self.device_combo.findText(self.config.device_name)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
                return
        for i, device in enumerate(self._devices):
            if device.is_default:
                self.device_combo.setCurrentIndex(i)
                return

    # -------------------------------------------------------------- analysis

    def _current_device_index(self) -> int | None:
        data = self.device_combo.currentData()
        return int(data) if data is not None else None

    def _start_analysis(self) -> None:
        if not self._devices:
            self.status_label.setText("No audio input found")
            return

        capture = AudioCapture(device=self._current_device_index())
        self.analysis = AnalysisThread(capture)
        worker = self.analysis.worker
        worker.set_noise_gate(self.config.noise_gate)
        worker.set_min_harmonic_ratio(self.config.min_harmonic_ratio)
        worker.frame_analysed.connect(self._on_frame)
        worker.note_stable.connect(self._on_note_stable)
        worker.note_released.connect(self._on_note_released)
        worker.bass_changed.connect(self._on_bass)
        worker.chroma_updated.connect(self._on_chroma)
        worker.snapshot_ready.connect(self._on_snapshot)
        worker.level_changed.connect(self._on_level)
        worker.error.connect(self._on_audio_error)
        self.analysis.start()
        self.status_label.setText("Listening")

    def _restart_analysis(self) -> None:
        if self.analysis:
            self.analysis.stop()
            self.analysis = None
        self._start_analysis()

    def _suppress_analysis(self, until: float) -> None:
        if self.analysis:
            self.analysis.worker.suppress_until(until)

    def _on_audio_error(self, message: str) -> None:
        self.status_label.setText("Audio error")
        QMessageBox.warning(self, "Audio error", message)

    # ---------------------------------------------------------------- routing

    def _on_frame(self, result) -> None:
        for panel in self.panels:
            if panel.active:
                panel.on_pitch(result)

    def _on_note_stable(self, result) -> None:
        for panel in self.panels:
            if panel.active:
                panel.on_note_stable(result)

    def _on_note_released(self) -> None:
        for panel in self.panels:
            if panel.active:
                panel.on_note_released()

    def _on_chord_highlight(self, pitch_classes) -> None:
        if pitch_classes:
            self.fretboard.set_target_pitch_classes(pitch_classes, color=theme.DETECTED)
        else:
            self.fretboard.clear_targets()

    def _on_bass(self, pitch_class) -> None:
        for panel in self.panels:
            if panel.active:
                panel.on_bass(pitch_class)

    def _on_chroma(self, chroma) -> None:
        for panel in self.panels:
            if panel.active:
                panel.on_chroma(chroma)

    def _on_snapshot(self, snapshot) -> None:
        for panel in self.panels:
            if panel.active:
                panel.on_snapshot(snapshot)

    def _on_level(self, rms: float) -> None:
        self.level_meter.set_level(rms)
        if self._calibrating:
            self._calibration_samples.append(rms)

    def _record_attempt(self, attempt) -> None:
        panel = self.tabs.currentWidget()
        mode = getattr(panel, "mode_name", "practice")
        if self._session_id is None:
            self._session_id = self.store.start_session(mode, bpm=self.config.bpm)
        self.store.record(self._session_id, attempt)

    # ----------------------------------------------------------------- events

    def _on_tab_changed(self, index: int) -> None:
        widget = self.tabs.widget(index)
        for panel in self.panels:
            if panel is widget:
                panel.on_activated()
            elif panel.active:
                panel.on_deactivated()

        if widget is self.stats_panel:
            self.stats_panel.refresh()

        # Practice modes weight selection towards weak spots when asked to.
        if widget in (self.note_practice, self.chord_practice):
            self._apply_weighting(widget)

    def _apply_weighting(self, panel) -> None:
        if not self.config.weight_by_weakness:
            panel.weight_fn = None
            return
        weights = self.store.weakness_weights(prefix=panel.stats_prefix)
        panel.weight_fn = lambda key: weights.get(key, 1.0)

    def _on_device_changed(self) -> None:
        self.config.device_name = self.device_combo.currentText()
        self._restart_analysis()
        self._mark_dirty()

    def _on_tuning_changed(self, name: str) -> None:
        self.config.tuning_name = name
        self._apply_tuning()
        self._mark_dirty()

    def _on_fret_count_changed(self, value: int) -> None:
        self.config.fret_count = value
        self._apply_tuning()
        self._mark_dirty()

    def _apply_tuning(self) -> None:
        tuning: Tuning = self.config.tuning()
        self.fretboard.set_tuning(tuning)
        self.tuner.set_tuning(tuning)

    def _on_label_mode_changed(self, text: str) -> None:
        self.config.label_mode = text
        self.fretboard.set_label_mode(self._label_mode_from_config())
        self._mark_dirty()

    def _on_gate_changed(self, value: float) -> None:
        self.config.noise_gate = value
        self.level_meter.set_threshold(value)
        if self.analysis:
            self.analysis.worker.set_noise_gate(value)
        self._mark_dirty()

    def _start_gate_calibration(self) -> None:
        if not self.analysis:
            return
        self._calibration_samples = []
        self._calibrating = True
        self.calibrate_button.setEnabled(False)
        self.calibrate_button.setText("Listening…")
        QTimer.singleShot(1200, self._finish_gate_calibration)

    def _finish_gate_calibration(self) -> None:
        self._calibrating = False
        self.calibrate_button.setEnabled(True)
        self.calibrate_button.setText("Calibrate")
        if not self._calibration_samples:
            return
        # A margin above the observed noise floor: played notes peak far higher than
        # idle room sound, so this rarely needs a manual nudge afterwards.
        floor = max(self._calibration_samples)
        gate = floor * 1.8 + 0.002
        gate = max(self.gate_spin.minimum(), min(self.gate_spin.maximum(), gate))
        self.gate_spin.setValue(gate)

    def _on_purity_changed(self, value: float) -> None:
        self.config.min_harmonic_ratio = value
        if self.analysis:
            self.analysis.worker.set_min_harmonic_ratio(value)
        self._mark_dirty()

    def _on_flats_toggled(self, use_flats: bool) -> None:
        self.config.use_flats = use_flats
        self.fretboard.set_use_flats(use_flats)
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        """Schedule a prompt save, debounced so a burst of changes — dragging a
        spinbox, ticking several checkboxes — writes once rather than on every tick.

        This is what makes selections survive anything short of a hard kill: closing
        the window and the periodic timer both already saved, but neither helps if the
        app is stopped seconds after a change (Ctrl-C used to skip the save entirely —
        see the SIGINT handler in __main__.py — and the old 30s timer alone left a
        window where a quick change-then-close would still lose the change).
        """
        self._dirty_timer.start(400)

    def _save_config(self) -> None:
        try:
            self.config.save()
        except OSError:
            # Not worth interrupting practice over; it will be retried on the next tick.
            pass

    def closeEvent(self, event) -> None:
        for panel in self.panels:
            if getattr(panel, "running", False):
                panel.stop_session()
        if self.analysis:
            self.analysis.stop()
        if self._session_id is not None:
            self.store.end_session(self._session_id)
        self._save_config()
        self.store.close()
        super().closeEvent(event)
