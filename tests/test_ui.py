"""Headless UI tests.

These run under the offscreen Qt platform and never open an audio stream, so they
exercise widget construction, signal wiring and the practice loop's interaction with
the UI without needing hardware.
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from guitar_trainer.audio.pitch import PitchResult  # noqa: E402
from guitar_trainer.core.chords import Chord, harmonic_template  # noqa: E402
from guitar_trainer.core.config import Config  # noqa: E402
from guitar_trainer.core.notes import (  # noqa: E402
    DROP_D,
    STANDARD,
    midi_to_freq,
    name_to_midi,
)
from guitar_trainer.core.stats import StatsStore  # noqa: E402
from guitar_trainer.ui.fretboard import FretboardWidget, LabelMode  # noqa: E402
from guitar_trainer.ui.modes import (  # noqa: E402
    ChordPracticePanel,
    FreeDetectPanel,
    NotePracticePanel,
    TunerPanel,
)
from guitar_trainer.ui.stats_panel import StatsPanel  # noqa: E402
from guitar_trainer.ui.widgets import BigNoteLabel, CentsMeter, LevelMeter  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def config():
    return Config()


def pitch(midi: int, **kwargs) -> PitchResult:
    return PitchResult.from_freq(midi_to_freq(midi), kwargs.get("confidence", 0.9), 0.1)


class TestFretboard:
    def test_constructs_and_paints(self, qapp):
        board = FretboardWidget(STANDARD)
        board.resize(900, 220)
        board.grab()  # forces a full paintEvent

    @pytest.mark.parametrize("mode", list(LabelMode))
    def test_every_label_mode_paints(self, qapp, mode):
        board = FretboardWidget(STANDARD)
        board.resize(900, 220)
        board.set_label_mode(mode)
        board.grab()

    def test_highlight_and_targets_paint(self, qapp):
        board = FretboardWidget(STANDARD)
        board.resize(900, 220)
        board.set_detected_pitch_class(9)
        board.set_target_pitch_classes([0, 4, 7])
        board.grab()

    def test_tuning_change_applies(self, qapp):
        board = FretboardWidget(STANDARD)
        board.set_tuning(DROP_D)
        assert board.tuning is DROP_D
        board.resize(900, 220)
        board.grab()

    def test_targets_resolve_to_positions(self, qapp):
        board = FretboardWidget(STANDARD)
        board.set_target_pitch_classes([9])
        assert board._target_positions == set(STANDARD.positions_for_pitch_class(9))

    def test_clear_targets(self, qapp):
        board = FretboardWidget(STANDARD)
        board.set_target_pitch_classes([0])
        board.clear_targets()
        assert board._target_positions == set()

    def test_geometry_covers_every_position(self, qapp):
        """Every fingering must map to a point inside the widget."""
        board = FretboardWidget(STANDARD)
        board.resize(900, 220)
        for string in range(STANDARD.string_count):
            for fret in range(STANDARD.fret_count + 1):
                point = board._marker_center(string, fret)
                assert -20 <= point.x() <= board.width()
                assert 0 <= point.y() <= board.height()

    def test_click_maps_back_to_a_position(self, qapp):
        board = FretboardWidget(STANDARD)
        board.resize(900, 220)
        for string, fret in [(0, 1), (3, 7), (5, 12)]:
            point = board._marker_center(string, fret)
            assert board._position_at(point.x(), point.y()) == (string, fret)

    def test_tiny_widget_does_not_crash(self, qapp):
        board = FretboardWidget(STANDARD)
        board.resize(10, 10)
        board.grab()

    def test_single_string_tuning(self, qapp):
        from guitar_trainer.core.notes import Tuning

        board = FretboardWidget(Tuning("one", (40,), 12))
        board.resize(400, 120)
        board.grab()


class TestWidgets:
    def test_level_meter_paints(self, qapp):
        meter = LevelMeter()
        meter.resize(200, 14)
        for level in [0.0, 0.05, 0.5, 2.0]:
            meter.set_level(level)
            meter.grab()

    def test_level_meter_clamps(self, qapp):
        meter = LevelMeter()
        meter.set_level(100.0)
        assert meter._level == 1.0

    def test_cents_meter_paints(self, qapp):
        meter = CentsMeter()
        meter.resize(300, 64)
        for cents in [None, 0.0, -25.0, 80.0]:
            meter.set_cents(cents)
            meter.grab()

    def test_cents_meter_in_tune_flag(self, qapp):
        meter = CentsMeter()
        meter.set_cents(2.0)
        assert meter.in_tune
        meter.set_cents(30.0)
        assert not meter.in_tune
        meter.set_cents(None)
        assert not meter.in_tune

    def test_big_note_label(self, qapp):
        label = BigNoteLabel("—")
        label.show_text("A4")
        assert label.text() == "A4"
        label.show_text(None)
        assert label.text() == "—"


class TestTunerPanel:
    def test_reports_nearest_string(self, qapp):
        panel = TunerPanel(STANDARD)
        panel.on_activated()
        panel.on_pitch(pitch(name_to_midi("A2")))
        assert panel.note_label.text() == "A2"
        assert "In tune" in panel.status.text()

    def test_detects_flat(self, qapp):
        panel = TunerPanel(STANDARD)
        panel.on_activated()
        # 30 cents flat of the low E.
        result = PitchResult.from_freq(midi_to_freq(name_to_midi("E2") - 0.3), 0.9, 0.1)
        panel.on_pitch(result)
        assert "tighten" in panel.status.text()

    def test_detects_sharp(self, qapp):
        panel = TunerPanel(STANDARD)
        panel.on_activated()
        result = PitchResult.from_freq(midi_to_freq(name_to_midi("E2") + 0.3), 0.9, 0.1)
        panel.on_pitch(result)
        assert "loosen" in panel.status.text()

    def test_silence_resets(self, qapp):
        panel = TunerPanel(STANDARD)
        panel.on_activated()
        panel.on_pitch(pitch(name_to_midi("A2")))
        panel.on_pitch(None)
        assert panel.note_label.text() == "—"

    def test_inactive_panel_ignores_input(self, qapp):
        panel = TunerPanel(STANDARD)
        panel.on_pitch(pitch(name_to_midi("A2")))
        assert panel.note_label.text() == "—"

    def test_tuning_change_updates_targets(self, qapp):
        panel = TunerPanel(STANDARD)
        panel.set_tuning(DROP_D)
        assert "D2" in panel.strings.text()


class TestFreeDetectPanel:
    def test_shows_stable_notes(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        panel.on_note_stable(pitch(name_to_midi("G3")))
        assert panel.note_label.text() == "G3"

    def test_emits_highlight(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        seen = []
        panel.highlight_requested.connect(seen.append)
        panel.on_note_stable(pitch(name_to_midi("A4")))
        assert seen == [9]

    def test_release_clears(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        panel.on_note_stable(pitch(name_to_midi("A4")))
        panel.on_note_released()
        assert panel.note_label.text() == "—"

    def test_history_deduplicates_repeats(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        for _ in range(3):
            panel.on_note_stable(pitch(name_to_midi("A4")))
        panel.on_note_stable(pitch(name_to_midi("B4")))
        assert panel._recent == ["A", "B"]

    def test_history_is_bounded(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        for i in range(40):
            panel.on_note_stable(pitch(40 + i))
        assert len(panel._recent) <= 12


class TestNotePracticePanel:
    def test_starts_and_stops(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.on_activated()
        panel.start_session()
        assert panel.running
        assert panel.engine.current is not None
        panel.stop_session()
        assert not panel.running

    def test_correct_answer_advances(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.on_activated()
        panel.start_session()
        challenge = panel.engine.current

        panel.on_note_stable(pitch(60 + challenge.pitch_class))
        assert panel.engine.current is not challenge
        assert panel.engine.correct_count == 1
        panel.stop_session()

    def test_wrong_answer_does_not_advance(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.on_activated()
        panel.start_session()
        challenge = panel.engine.current

        panel.on_note_stable(pitch(60 + (challenge.pitch_class + 1) % 12))
        assert panel.engine.current is challenge
        panel.stop_session()

    def test_any_octave_is_accepted(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.on_activated()
        panel.start_session()
        challenge = panel.engine.current
        # Two octaves above the first match.
        panel.on_note_stable(pitch(60 + challenge.pitch_class + 24))
        assert panel.engine.correct_count == 1
        panel.stop_session()

    def test_emits_attempts(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.on_activated()
        recorded = []
        panel.attempt_recorded.connect(recorded.append)
        panel.start_session()
        panel.on_note_stable(pitch(60 + panel.engine.current.pitch_class))
        assert len(recorded) == 1
        panel.stop_session()

    def test_custom_set_with_nothing_selected_is_rejected(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.note_set_combo.setCurrentText("Custom")
        for check in panel.note_checks.values():
            check.setChecked(False)
        panel.start_session()
        assert not panel.running
        assert "at least one" in panel.feedback.text()

    def test_naturals_only_set(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.note_set_combo.setCurrentText("Naturals only")
        challenges = panel.build_challenges()
        assert len(challenges) == 7
        assert all(c.pitch_class in (0, 2, 4, 5, 7, 9, 11) for c in challenges)

    def test_deactivation_stops_the_session(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.on_activated()
        panel.start_session()
        panel.on_deactivated()
        assert not panel.running

    def test_skip_records_an_attempt(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.on_activated()
        recorded = []
        panel.attempt_recorded.connect(recorded.append)
        panel.start_session()
        panel._skip()
        assert len(recorded) == 1
        assert not recorded[0].correct
        panel.stop_session()

    def test_rhythm_mode_sets_a_timeout(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.on_activated()
        panel.rhythm_radio.setChecked(True)
        panel.bpm_spin.setValue(120)
        panel.beats_spin.setValue(4)
        panel.start_session()
        # 4 beats at 120bpm is 2 seconds.
        assert panel.engine.timeout_seconds == pytest.approx(2.0)
        panel.stop_session()

    def test_bpm_change_updates_a_running_timeout(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.on_activated()
        panel.rhythm_radio.setChecked(True)
        panel.bpm_spin.setValue(60)
        panel.start_session()
        panel.bpm_spin.setValue(120)
        assert panel.engine.timeout_seconds == pytest.approx(
            panel.metronome.seconds_per_beat * panel.beats_spin.value()
        )
        panel.stop_session()


class TestChordPracticePanel:
    def test_builds_challenges_from_selection(self, qapp, config):
        panel = ChordPracticePanel(config)
        challenges = panel.build_challenges()
        assert len(challenges) == len(config.chord_symbols)

    def test_empty_selection_is_rejected(self, qapp, config):
        panel = ChordPracticePanel(config)
        panel._clear_selection()
        panel.on_activated()
        panel.start_session()
        assert not panel.running

    def test_matcher_restricted_to_selected_chords(self, qapp, config):
        """Scoring only against the drilled chords is what makes this reliable."""
        panel = ChordPracticePanel(config)
        panel._selected = {"C", "G"}
        panel.build_challenges()
        assert {str(c) for c in panel._matcher.chords} == {"C", "G"}

    def test_recognises_its_target_chord(self, qapp, config):
        panel = ChordPracticePanel(config)
        panel._selected = {"C", "G", "Am", "E"}
        panel.on_activated()
        panel.start_session()

        target = panel.engine.current.chord
        chroma = harmonic_template(target.pitch_classes)
        for _ in range(3):
            panel.on_chroma(chroma)

        assert panel.engine.correct_count == 1
        panel.stop_session()

    def test_wrong_chord_does_not_advance(self, qapp, config):
        panel = ChordPracticePanel(config)
        panel._selected = {"C", "G"}
        panel.on_activated()
        panel.start_session()
        challenge = panel.engine.current

        other = Chord.parse("G" if challenge.chord == Chord.parse("C") else "C")
        for _ in range(5):
            panel.on_chroma(harmonic_template(other.pitch_classes))

        assert panel.engine.current is challenge
        panel.stop_session()

    def test_requires_a_sustained_match(self, qapp, config):
        panel = ChordPracticePanel(config)
        panel._selected = {"C", "G", "Am", "E"}
        panel.on_activated()
        panel.start_session()

        chroma = harmonic_template(panel.engine.current.chord.pitch_classes)
        panel.on_chroma(chroma)
        assert panel.engine.correct_count == 0  # one frame is not enough
        panel.stop_session()

    def test_noise_does_not_score(self, qapp, config):
        panel = ChordPracticePanel(config)
        panel._selected = {"C", "G"}
        panel.on_activated()
        panel.start_session()
        for _ in range(10):
            panel.on_chroma(np.full(12, 1 / np.sqrt(12)))
        assert panel.engine.correct_count == 0
        panel.stop_session()

    def test_quality_switching_preserves_selection(self, qapp, config):
        panel = ChordPracticePanel(config)
        panel._selected = {"C", "Am"}
        panel.quality_combo.setCurrentIndex(1)  # minor
        panel.quality_combo.setCurrentIndex(0)  # back to major
        assert "C" in panel._selected and "Am" in panel._selected

    def test_common_chords_button(self, qapp, config):
        panel = ChordPracticePanel(config)
        panel._clear_selection()
        panel._select_common()
        assert len(panel._selected) == 8


class TestStatsPanel:
    def test_empty_history(self, qapp, tmp_path):
        with StatsStore(tmp_path / "s.db") as store:
            panel = StatsPanel(store)
            assert "No practice history" in panel.summary.text()
            assert panel.table.rowCount() == 0

    def test_populated(self, qapp, tmp_path):
        from guitar_trainer.core.session import Attempt, Outcome

        with StatsStore(tmp_path / "s.db") as store:
            session = store.start_session("note")
            store.record(session, Attempt("note:9", "A", Outcome.CORRECT, 1200))
            store.record(session, Attempt("note:0", "C", Outcome.TIMEOUT, 4000))

            panel = StatsPanel(store)
            assert panel.table.rowCount() == 2
            # Weakest first: the timed-out C should lead.
            assert panel.table.item(0, 0).text() == "C"


class TestMainWindow:
    def test_constructs_without_audio(self, qapp, tmp_path, monkeypatch):
        """The window must come up even with no input device present."""
        import guitar_trainer.ui.main_window as mw

        monkeypatch.setattr(mw, "list_input_devices", lambda: [])
        with StatsStore(tmp_path / "s.db") as store:
            window = mw.MainWindow(Config(), store)
            assert window.tabs.count() == 5
            assert "No audio input" in window.status_label.text()
            window.analysis = None
            window.close()

    def test_tab_switching_activates_one_panel(self, qapp, tmp_path, monkeypatch):
        import guitar_trainer.ui.main_window as mw

        monkeypatch.setattr(mw, "list_input_devices", lambda: [])
        with StatsStore(tmp_path / "s.db") as store:
            window = mw.MainWindow(Config(), store)
            for index in range(window.tabs.count()):
                window.tabs.setCurrentIndex(index)
                active = [p for p in window.panels if p.active]
                assert len(active) <= 1
            window.analysis = None
            window.close()

    def test_tuning_change_propagates(self, qapp, tmp_path, monkeypatch):
        import guitar_trainer.ui.main_window as mw

        monkeypatch.setattr(mw, "list_input_devices", lambda: [])
        with StatsStore(tmp_path / "s.db") as store:
            window = mw.MainWindow(Config(), store)
            window.tuning_combo.setCurrentText("Drop D")
            assert window.fretboard.tuning.open_midi == DROP_D.open_midi
            assert "D2" in window.tuner.strings.text()
            window.analysis = None
            window.close()

    def test_attempts_are_persisted(self, qapp, tmp_path, monkeypatch):
        import guitar_trainer.ui.main_window as mw

        monkeypatch.setattr(mw, "list_input_devices", lambda: [])
        with StatsStore(tmp_path / "s.db") as store:
            window = mw.MainWindow(Config(), store)
            window.tabs.setCurrentWidget(window.note_practice)
            window.note_practice.start_session()
            challenge = window.note_practice.engine.current
            window.note_practice.on_note_stable(pitch(60 + challenge.pitch_class))
            window.note_practice.stop_session()

            assert store.totals() == (1, 1)
            window.analysis = None
            window.close()

    def test_config_saved_on_close(self, qapp, tmp_path, monkeypatch):
        import guitar_trainer.ui.main_window as mw

        monkeypatch.setattr(mw, "list_input_devices", lambda: [])
        import guitar_trainer.core.config as config_module

        config_path = tmp_path / "config.toml"
        monkeypatch.setattr(config_module, "config_path", lambda: config_path)

        with StatsStore(tmp_path / "s.db") as store:
            window = mw.MainWindow(Config(), store)
            window.fret_spin.setValue(24)
            window.analysis = None
            window.close()

        assert Config.load(config_path).fret_count == 24
