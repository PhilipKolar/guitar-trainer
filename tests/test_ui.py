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
    name_to_pitch_class,
    pitch_class_name,
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

    def test_silence_holds_the_last_reading_briefly(self, qapp):
        """A decaying note's sustain routinely drops below the gate before it's
        actually inaudible; the display must not blank the instant that happens."""
        panel = TunerPanel(STANDARD)
        panel.on_activated()
        panel.on_pitch(pitch(name_to_midi("A2")))
        panel.on_pitch(None)
        assert panel.note_label.text() == "A2"

    def test_display_clears_once_the_hold_expires(self, qapp):
        panel = TunerPanel(STANDARD)
        panel.on_activated()
        panel.on_pitch(pitch(name_to_midi("A2")))
        panel.on_pitch(None)
        panel._clear_display()  # simulate the hold timer firing, without the wait
        assert panel.note_label.text() == "—"

    def test_inactive_panel_ignores_input(self, qapp):
        panel = TunerPanel(STANDARD)
        panel.on_pitch(pitch(name_to_midi("A2")))
        assert panel.note_label.text() == "—"

    def test_tuning_change_updates_targets(self, qapp):
        panel = TunerPanel(STANDARD)
        panel.set_tuning(DROP_D)
        assert "D2" in panel.strings.text()

    def test_live_readout_is_smoothed(self, qapp):
        """Regression coverage for the needle jitter fix: a single wild outlier frame
        amid a steady note must not swing the displayed cents value wildly."""
        panel = TunerPanel(STANDARD)
        panel.on_activated()
        base = name_to_midi("A2")
        for _ in range(6):
            panel.on_pitch(PitchResult.from_freq(midi_to_freq(base), 0.9, 0.1))
        outlier = PitchResult.from_freq(midi_to_freq(base + 1.0), 0.9, 0.1)  # 100c off
        panel.on_pitch(outlier)
        # Cents shown are relative to the nearest string target (still A2), so a
        # rejected outlier should leave it close to in tune, not swing to ~100 cents.
        _, cents = STANDARD.nearest_string(midi_to_freq(base))
        assert abs(panel.meter._cents) < 20.0

    def test_deactivation_resets_smoothing(self, qapp):
        panel = TunerPanel(STANDARD)
        panel.on_activated()
        # Fill the median window with the old note, so a leftover buffer would bias
        # the first reading after reactivation if it weren't cleared.
        for _ in range(3):
            panel.on_pitch(pitch(name_to_midi("A2")))
        panel.on_deactivated()
        panel.on_activated()
        panel.on_pitch(pitch(name_to_midi("E4")))
        assert panel.note_label.text() == "E4"


class TestFreeDetectPanel:
    @staticmethod
    def snap(panel, chroma, bass=None, series=None, rms=0.1):
        """Deliver a chroma frame the way the worker now does — as a ChromaSnapshot
        (chroma + bass + series + rms together) via on_snapshot."""
        from guitar_trainer.ui.worker import ChromaSnapshot

        panel.on_snapshot(ChromaSnapshot(chroma, bass, series, rms))

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

    def test_release_holds_the_last_reading_briefly(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        panel.on_note_stable(pitch(name_to_midi("A4")))
        panel.on_note_released()
        assert panel.note_label.text() == "A4"

    def test_display_clears_once_the_hold_expires(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        panel.on_note_stable(pitch(name_to_midi("A4")))
        panel.on_note_released()
        panel._clear_display()  # simulate the hold timer firing, without the wait
        assert panel.note_label.text() == "—"

    def test_chord_detection_works_right_after_a_note(self, qapp):
        """Chord matching must never be gated behind note state — a chord strummed
        immediately after a single note has to be recognised, not ignored."""
        panel = FreeDetectPanel()
        panel.on_activated()
        panel.on_note_stable(pitch(name_to_midi("A4")))
        panel.on_note_released()
        chroma = harmonic_template(Chord.parse("Am").pitch_classes)
        for _ in range(5):
            self.snap(panel, chroma)
        assert panel.note_label.text() == "Am"

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

    def test_hold_timer_actually_fires_after_real_time(self, qapp):
        """End-to-end check that the timer is wired up, not just that
        _clear_display works when called directly."""
        from PySide6.QtCore import QEventLoop, QTimer as QtTimer

        panel = FreeDetectPanel()
        panel.on_activated()
        panel.on_note_stable(pitch(name_to_midi("A4")))
        assert panel.note_label.text() == "A4"

        loop = QEventLoop()
        QtTimer.singleShot(panel._hold_timer.interval() + 150, loop.quit)
        loop.exec()

        assert panel.note_label.text() == "—"

    def test_fresh_activity_cancels_a_pending_clear(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        panel.on_note_stable(pitch(name_to_midi("A4")))
        panel.on_note_released()  # display is now held, counting down to clear
        panel.on_note_stable(pitch(name_to_midi("B4")))  # restarts the hold
        assert panel.note_label.text() == "B4"

    def test_recognises_a_chord(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        chroma = harmonic_template(Chord.parse("Am").pitch_classes)
        for _ in range(5):
            self.snap(panel, chroma)
        assert panel.note_label.text() == "Am"

    def test_chord_requires_sustained_agreement(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        chroma = harmonic_template(Chord.parse("G").pitch_classes)
        self.snap(panel, chroma)
        assert panel.note_label.text() == "—"

    def test_bass_note_disambiguates_relative_pairs(self, qapp):
        """C and Am share two of three notes; the wired-through bass is what separates
        them in the live signal path (this bug existed until now — see AGENTS.md)."""
        panel = FreeDetectPanel()
        panel.on_activated()
        for symbol, bass in [("C", 0), ("Am", 9)]:
            panel._gate.reset()
            chroma = harmonic_template(Chord.parse(symbol).pitch_classes)
            for _ in range(5):
                self.snap(panel, chroma, bass=bass)
            assert panel.note_label.text() == symbol

    def test_a_note_that_genuinely_sounds_like_a_chord_is_shown_as_a_chord(self, qapp):
        """The behaviour this replaced: the old design let a YIN-stable single note
        block chord detection outright, which is exactly why a strummed E major was
        never recognised — YIN happily locks onto one dominant string mid-strum.
        If the chroma genuinely says "chord", that must win regardless of what the
        monophonic tracker reported a moment earlier."""
        panel = FreeDetectPanel()
        panel.on_activated()
        panel.on_note_stable(pitch(name_to_midi("E3")))
        chroma = harmonic_template(Chord.parse("Am").pitch_classes)
        for _ in range(5):
            self.snap(panel, chroma)
        assert panel.note_label.text() == "Am"

    def test_a_single_notes_own_chroma_does_not_get_reclassified_as_a_chord(self, qapp):
        """The flip side: real chroma evidence that agrees it's just one note must
        not spuriously override the note display."""
        panel = FreeDetectPanel()
        panel.on_activated()
        panel.on_note_stable(pitch(name_to_midi("E3")))
        chroma = harmonic_template((name_to_pitch_class("E"),))
        for _ in range(3):
            self.snap(panel, chroma)
        assert panel.note_label.text() == "E3"

    def test_chord_path_resumes_once_the_note_is_released(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        panel.on_note_stable(pitch(name_to_midi("E3")))
        panel.on_note_released()
        chroma = harmonic_template(Chord.parse("G").pitch_classes)
        for _ in range(5):
            self.snap(panel, chroma)
        assert panel.note_label.text() == "G"

    def test_chord_emits_fretboard_highlight(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        seen = []
        panel.chord_highlight_requested.connect(seen.append)
        chroma = harmonic_template(Chord.parse("D").pitch_classes)
        for _ in range(5):
            self.snap(panel, chroma)
        assert seen and sorted(seen[-1]) == sorted(Chord.parse("D").pitch_classes)

    def test_noise_is_not_shown_as_a_chord(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        for _ in range(6):
            self.snap(panel, np.full(12, 1 / np.sqrt(12)))
        assert panel.note_label.text() == "—"

    def test_deactivation_clears_chord_state(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        chroma = harmonic_template(Chord.parse("Am").pitch_classes)
        for _ in range(4):
            self.snap(panel, chroma)
        panel.on_deactivated()
        panel.on_activated()
        # The in-progress streak must not carry over across a deactivation: one more
        # frame is not enough to settle a chord that hasn't just resumed from scratch.
        self.snap(panel, chroma)
        assert panel.note_label.text() == "—"

    def test_inactive_panel_ignores_chroma(self, qapp):
        panel = FreeDetectPanel()
        chroma = harmonic_template(Chord.parse("Am").pitch_classes)
        for _ in range(5):
            self.snap(panel, chroma)
        assert panel.note_label.text() == "—"

    def test_live_meter_readout_is_smoothed(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        base = name_to_midi("A2")
        for _ in range(6):
            panel.on_pitch(PitchResult.from_freq(midi_to_freq(base), 0.9, 0.1))
        outlier = PitchResult.from_freq(midi_to_freq(base + 1.0), 0.9, 0.1)  # 100c off
        panel.on_pitch(outlier)
        assert abs(panel.meter._cents) < 20.0

    def test_deactivation_resets_smoothing(self, qapp):
        panel = FreeDetectPanel()
        panel.on_activated()
        for _ in range(3):
            panel.on_pitch(pitch(name_to_midi("A2")))
        panel.on_deactivated()
        panel.on_activated()
        panel.on_pitch(pitch(name_to_midi("E4")))
        assert panel.meter._cents == pytest.approx(0.0, abs=1.0)


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

    def test_accidental_style_defaults_to_sharps(self, qapp, config):
        panel = NotePracticePanel(config)
        challenges = {c.pitch_class: c for c in panel.build_challenges()}
        assert challenges[1].prompt == "C#"  # C#/Db

    def test_accidental_style_flats(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.accidental_combo.setCurrentIndex(panel.accidental_combo.findData("flats"))
        challenges = {c.pitch_class: c for c in panel.build_challenges()}
        assert challenges[1].prompt == "Db"

    def test_accidental_style_naturals_unaffected_by_flats(self, qapp, config):
        """Naturals have no accidental to flip either way."""
        panel = NotePracticePanel(config)
        panel.accidental_combo.setCurrentIndex(panel.accidental_combo.findData("flats"))
        challenges = {c.pitch_class: c for c in panel.build_challenges()}
        assert challenges[0].prompt == "C"
        assert challenges[4].prompt == "E"

    def test_accidental_style_mix_only_varies_accidentals(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.accidental_combo.setCurrentIndex(panel.accidental_combo.findData("mix"))
        challenges = {c.pitch_class: c for c in panel.build_challenges()}
        # Naturals are never spelled with an accidental regardless of the coin flip.
        for pc in (0, 2, 4, 5, 7, 9, 11):
            assert challenges[pc].prompt == pitch_class_name(pc)

    def test_accidental_style_mix_produces_both_spellings_over_many_sessions(self, qapp, config):
        """Not a fixed 50/50-per-note guarantee (it's one coin flip per note per
        session), but across many sessions both sharp and flat spellings must appear
        somewhere for at least one accidental — otherwise "mix" would be a no-op."""
        panel = NotePracticePanel(config)
        panel.accidental_combo.setCurrentIndex(panel.accidental_combo.findData("mix"))
        accidentals = (1, 3, 6, 8, 10)
        seen_sharp = seen_flat = False
        for _ in range(50):
            challenges = {c.pitch_class: c for c in panel.build_challenges()}
            for pc in accidentals:
                if challenges[pc].prompt == pitch_class_name(pc, use_flats=True):
                    seen_flat = True
                else:
                    seen_sharp = True
        assert seen_sharp and seen_flat

    def test_accidental_style_persists_to_config(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.accidental_combo.setCurrentIndex(panel.accidental_combo.findData("flats"))
        assert config.note_accidental_style == "flats"

    def test_accidental_style_change_emits_config_changed(self, qapp, config):
        panel = NotePracticePanel(config)
        seen = []
        panel.config_changed.connect(lambda: seen.append(True))
        panel.accidental_combo.setCurrentIndex(panel.accidental_combo.findData("mix"))
        assert seen

    def test_accidental_style_loaded_from_config(self, qapp):
        config = Config(note_accidental_style="flats")
        panel = NotePracticePanel(config)
        assert panel.accidental_combo.currentData() == "flats"

    def test_custom_checkbox_labels_follow_the_selected_style(self, qapp, config):
        panel = NotePracticePanel(config)
        assert panel.note_checks[1].text() == "C#"
        panel.accidental_combo.setCurrentIndex(panel.accidental_combo.findData("flats"))
        assert panel.note_checks[1].text() == "Db"

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

    def test_rhythm_mode_does_not_advance_on_a_correct_answer(self, qapp, config):
        """The fix: answering correctly must not reset the countdown off-beat —
        the next prompt only appears once the beat window elapses on its own."""
        panel = NotePracticePanel(config)
        panel.on_activated()
        panel.rhythm_radio.setChecked(True)
        panel.bpm_spin.setValue(120)
        panel.beats_spin.setValue(4)
        panel.start_session()

        challenge = panel.engine.current
        panel.on_note_stable(pitch(60 + challenge.pitch_class))

        assert panel.engine.correct_count == 1
        assert panel.engine.current is challenge  # still the same prompt
        panel.stop_session()

    def test_free_mode_still_advances_immediately(self, qapp, config):
        panel = NotePracticePanel(config)
        panel.on_activated()
        panel.free_radio.setChecked(True)
        panel.start_session()

        challenge = panel.engine.current
        panel.on_note_stable(pitch(60 + challenge.pitch_class))

        assert panel.engine.current is not challenge
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

    def test_purity_control_updates_config(self, qapp, tmp_path, monkeypatch):
        import guitar_trainer.ui.main_window as mw

        monkeypatch.setattr(mw, "list_input_devices", lambda: [])
        with StatsStore(tmp_path / "s.db") as store:
            window = mw.MainWindow(Config(), store)
            window.purity_spin.setValue(0.75)
            assert window.config.min_harmonic_ratio == pytest.approx(0.75)
            window.analysis = None
            window.close()

    def test_calibrate_is_a_noop_without_a_running_stream(self, qapp, tmp_path, monkeypatch):
        import guitar_trainer.ui.main_window as mw

        monkeypatch.setattr(mw, "list_input_devices", lambda: [])
        with StatsStore(tmp_path / "s.db") as store:
            window = mw.MainWindow(Config(), store)
            window._start_gate_calibration()
            assert not window._calibrating
            window.analysis = None
            window.close()

    def test_calibrate_sets_gate_above_the_observed_floor(self, qapp, tmp_path, monkeypatch):
        import guitar_trainer.ui.main_window as mw

        monkeypatch.setattr(mw, "list_input_devices", lambda: [])
        with StatsStore(tmp_path / "s.db") as store:
            window = mw.MainWindow(Config(), store)

            class _FakeWorker:
                def set_noise_gate(self, value):
                    pass

            class _FakeAnalysis:
                worker = _FakeWorker()

            window.analysis = _FakeAnalysis()  # truthy, and tolerates the gate setter
            window._calibrating = True
            window._calibration_samples = [0.01, 0.03, 0.02]
            window._finish_gate_calibration()
            assert window.gate_spin.value() > 0.03
            assert not window._calibrating
            assert window.calibrate_button.isEnabled()
            window.analysis = None
            window.close()

    def test_calibrate_with_no_samples_leaves_gate_unchanged(self, qapp, tmp_path, monkeypatch):
        import guitar_trainer.ui.main_window as mw

        monkeypatch.setattr(mw, "list_input_devices", lambda: [])
        with StatsStore(tmp_path / "s.db") as store:
            window = mw.MainWindow(Config(), store)
            before = window.gate_spin.value()
            window._calibrating = True
            window._calibration_samples = []
            window._finish_gate_calibration()
            assert window.gate_spin.value() == before
            window.analysis = None
            window.close()

    def test_level_updates_feed_calibration_only_while_calibrating(self, qapp, tmp_path, monkeypatch):
        import guitar_trainer.ui.main_window as mw

        monkeypatch.setattr(mw, "list_input_devices", lambda: [])
        with StatsStore(tmp_path / "s.db") as store:
            window = mw.MainWindow(Config(), store)
            window._on_level(0.02)
            assert window._calibration_samples == []
            window._calibrating = True
            window._on_level(0.03)
            assert window._calibration_samples == [0.03]
            window.analysis = None
            window.close()


class TestSettingsPersistence:
    """Regression coverage for selections not surviving a quick close.

    A change used to only be written to disk by a graceful window close or the 30s
    periodic timer; stopping the app via Ctrl-C in the launching terminal skipped
    closeEvent entirely (Qt doesn't route SIGINT there by default), and even a
    graceful close moments after a change relied on that single save happening to
    catch it. Settings changes now save promptly on their own via a debounced timer.
    """

    def _window(self, tmp_path, monkeypatch, config=None):
        import guitar_trainer.ui.main_window as mw

        monkeypatch.setattr(mw, "list_input_devices", lambda: [])
        store = StatsStore(tmp_path / "s.db")
        window = mw.MainWindow(config or Config(), store)
        return window, store

    def test_toolbar_change_schedules_a_prompt_save(self, qapp, tmp_path, monkeypatch):
        window, store = self._window(tmp_path, monkeypatch)
        assert not window._dirty_timer.isActive()
        window.fret_spin.setValue(24)
        assert window._dirty_timer.isActive()
        window.analysis = None
        window.close()
        store.close()

    def test_note_selection_change_schedules_a_prompt_save(self, qapp, tmp_path, monkeypatch):
        window, store = self._window(tmp_path, monkeypatch)
        window.note_practice.note_set_combo.setCurrentText("Naturals only")
        assert window._dirty_timer.isActive()
        window.analysis = None
        window.close()
        store.close()

    def test_chord_selection_change_schedules_a_prompt_save(self, qapp, tmp_path, monkeypatch):
        window, store = self._window(tmp_path, monkeypatch)
        window.chord_practice._clear_selection()
        assert window._dirty_timer.isActive()
        window.analysis = None
        window.close()
        store.close()

    def test_dirty_timer_actually_saves(self, qapp, tmp_path, monkeypatch):
        import guitar_trainer.core.config as config_module

        config_path = tmp_path / "config.toml"
        monkeypatch.setattr(config_module, "config_path", lambda: config_path)
        window, store = self._window(tmp_path, monkeypatch)

        window.fret_spin.setValue(24)
        window._dirty_timer.stop()
        window._save_config()  # simulate the debounce timer firing, without the wait

        assert Config.load(config_path).fret_count == 24
        window.analysis = None
        window.close()
        store.close()

    def test_a_quick_change_then_close_still_persists(self, qapp, tmp_path, monkeypatch):
        """The scenario this whole fix is for: change something, close moments later."""
        import guitar_trainer.core.config as config_module

        config_path = tmp_path / "config.toml"
        monkeypatch.setattr(config_module, "config_path", lambda: config_path)
        window, store = self._window(tmp_path, monkeypatch)

        window.note_practice.note_set_combo.setCurrentText("Naturals only")
        window.analysis = None
        window.close()  # closeEvent saves immediately; doesn't need the debounce to fire
        store.close()

        assert Config.load(config_path).note_set == "Naturals only"
