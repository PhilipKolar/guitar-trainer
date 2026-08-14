import random

import numpy as np
import pytest

from guitar_trainer.audio.metronome import (
    BeatCounter,
    Metronome,
    render_click,
)
from guitar_trainer.audio.pitch import PitchResult
from guitar_trainer.core.chords import Chord, ChordMatch
from guitar_trainer.core.notes import ALL_NOTES, NATURAL_NOTES, NoteSet, midi_to_freq
from guitar_trainer.core.session import (
    Attempt,
    ChallengePicker,
    ChordChallenge,
    NoteChallenge,
    Outcome,
    SessionEngine,
    SessionState,
    chord_challenges,
    note_challenges,
)


class FakeClock:
    """Controllable monotonic clock, so timeouts are tested without waiting."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def pitch(midi: int) -> PitchResult:
    return PitchResult.from_freq(midi_to_freq(midi), confidence=0.9, rms=0.1)


class TestChallenges:
    def test_note_challenge_matches_any_octave(self):
        challenge = NoteChallenge(9)  # A
        for midi in [45, 57, 69, 81]:  # A2 through A5
            assert challenge.matches(pitch(midi))

    def test_note_challenge_rejects_other_notes(self):
        challenge = NoteChallenge(9)
        assert not challenge.matches(pitch(70))  # A#
        assert not challenge.matches(pitch(68))  # G#
        assert not challenge.matches(None)

    def test_note_challenge_prompt(self):
        assert NoteChallenge(1).prompt == "C#"
        assert NoteChallenge(1, use_flats=True).prompt == "Db"

    def test_chord_challenge_matches_a_chord_match(self):
        challenge = ChordChallenge(Chord.parse("Am"))
        match = ChordMatch(Chord.parse("Am"), score=0.9, margin=0.1)
        assert challenge.matches(match)
        assert challenge.matches(Chord.parse("Am"))

    def test_chord_challenge_rejects_others(self):
        challenge = ChordChallenge(Chord.parse("Am"))
        assert not challenge.matches(ChordMatch(Chord.parse("A"), 0.9, 0.1))
        assert not challenge.matches(None)

    def test_keys_are_distinct_and_stable(self):
        keys = {c.key for c in note_challenges(ALL_NOTES)}
        assert len(keys) == 12
        assert NoteChallenge(9).key == NoteChallenge(9).key
        assert NoteChallenge(9).key != ChordChallenge(Chord(9, "min")).key

    def test_builders(self):
        assert len(note_challenges(NATURAL_NOTES)) == 7
        assert len(chord_challenges([Chord.parse("C"), Chord.parse("G")])) == 2


class TestChallengePicker:
    def test_never_repeats_immediately(self):
        picker = ChallengePicker(note_challenges(ALL_NOTES), rng=random.Random(0))
        picks = [picker.next() for _ in range(200)]
        assert all(a != b for a, b in zip(picks, picks[1:]))

    def test_covers_the_whole_set(self):
        picker = ChallengePicker(note_challenges(ALL_NOTES), rng=random.Random(0))
        assert len({picker.next().prompt for _ in range(400)}) == 12

    def test_single_challenge_set_repeats(self):
        picker = ChallengePicker([NoteChallenge(0)], rng=random.Random(0))
        assert picker.next().pitch_class == 0
        assert picker.next().pitch_class == 0

    def test_empty_set_rejected(self):
        with pytest.raises(ValueError):
            ChallengePicker([])

    def test_weighting_biases_towards_weak_challenges(self):
        challenges = note_challenges(NoteSet("test", (0, 2, 4)))
        weak = challenges[0].key

        picker = ChallengePicker(
            challenges,
            rng=random.Random(0),
            weight_fn=lambda key: 20.0 if key == weak else 1.0,
        )
        counts = {}
        for _ in range(600):
            key = picker.next().key
            counts[key] = counts.get(key, 0) + 1

        # The no-repeat rule caps any single challenge at every other pick, so a
        # heavily weighted one should approach — but never exceed — half the picks.
        assert 0.40 < counts[weak] / 600 <= 0.5
        assert all(counts[weak] > v for k, v in counts.items() if k != weak)

    def test_reset_clears_history(self):
        picker = ChallengePicker(note_challenges(ALL_NOTES), rng=random.Random(0))
        picker.next()
        picker.reset()
        assert picker._previous is None


class TestSessionEngineFreeMode:
    def make(self, clock=None, **kwargs):
        return SessionEngine(
            picker=ChallengePicker(note_challenges(ALL_NOTES), rng=random.Random(0)),
            clock=clock or FakeClock(),
            **kwargs,
        )

    def test_starts_listening(self):
        engine = self.make()
        challenge = engine.start()
        assert engine.state is SessionState.LISTENING
        assert engine.current is challenge

    def test_correct_answer_advances(self):
        engine = self.make()
        challenge = engine.start()
        attempt = engine.submit(pitch(60 + challenge.pitch_class))
        assert attempt is not None
        assert attempt.outcome is Outcome.CORRECT
        assert engine.current is not challenge
        assert engine.state is SessionState.LISTENING

    def test_wrong_note_is_ignored_not_failed(self):
        """Hunting for a note sounds others on the way; those must not fail the round."""
        engine = self.make()
        challenge = engine.start()
        wrong = 60 + (challenge.pitch_class + 1) % 12
        assert engine.submit(pitch(wrong)) is None
        assert engine.current is challenge
        assert engine.attempts == []

    def test_no_timeout_in_free_mode(self):
        clock = FakeClock()
        engine = self.make(clock=clock, timeout_seconds=None)
        engine.start()
        clock.advance(3600)
        assert engine.tick() is None
        assert engine.time_remaining is None

    def test_response_time_recorded(self):
        clock = FakeClock()
        engine = self.make(clock=clock)
        challenge = engine.start()
        clock.advance(1.25)
        attempt = engine.submit(pitch(60 + challenge.pitch_class))
        assert attempt.response_ms == 1250

    def test_skip_records_an_attempt(self):
        engine = self.make()
        engine.start()
        attempt = engine.skip()
        assert attempt.outcome is Outcome.SKIPPED
        assert not attempt.correct

    def test_submit_before_start_does_nothing(self):
        assert self.make().submit(pitch(60)) is None

    def test_stop_ends_the_session(self):
        engine = self.make()
        engine.start()
        engine.stop()
        assert engine.state is SessionState.FINISHED
        assert engine.submit(pitch(60)) is None

    def test_callbacks_fire(self):
        presented, results = [], []
        engine = SessionEngine(
            picker=ChallengePicker(note_challenges(ALL_NOTES), rng=random.Random(0)),
            clock=FakeClock(),
            on_challenge=presented.append,
            on_result=results.append,
        )
        challenge = engine.start()
        engine.submit(pitch(60 + challenge.pitch_class))
        assert len(presented) == 2  # first challenge, plus the next one
        assert len(results) == 1


class TestSessionEngineRhythmMode:
    def make(self, clock, timeout=2.0):
        return SessionEngine(
            picker=ChallengePicker(note_challenges(ALL_NOTES), rng=random.Random(0)),
            timeout_seconds=timeout,
            clock=clock,
        )

    def test_times_out_and_advances(self):
        clock = FakeClock()
        engine = self.make(clock)
        challenge = engine.start()
        clock.advance(2.1)
        attempt = engine.tick()
        assert attempt is not None
        assert attempt.outcome is Outcome.TIMEOUT
        assert engine.current is not challenge

    def test_no_timeout_before_the_deadline(self):
        clock = FakeClock()
        engine = self.make(clock)
        engine.start()
        clock.advance(1.9)
        assert engine.tick() is None

    def test_time_remaining_counts_down(self):
        clock = FakeClock()
        engine = self.make(clock)
        engine.start()
        assert engine.time_remaining == pytest.approx(2.0)
        clock.advance(0.5)
        assert engine.time_remaining == pytest.approx(1.5)
        clock.advance(5.0)
        assert engine.time_remaining == 0.0

    def test_answering_in_time_beats_the_timeout(self):
        clock = FakeClock()
        engine = self.make(clock)
        challenge = engine.start()
        clock.advance(1.0)
        attempt = engine.submit(pitch(60 + challenge.pitch_class))
        assert attempt.outcome is Outcome.CORRECT
        assert engine.tick() is None  # clock was reset for the new challenge


class TestSessionSummary:
    def test_accuracy_and_median(self):
        clock = FakeClock()
        engine = SessionEngine(
            picker=ChallengePicker(note_challenges(ALL_NOTES), rng=random.Random(0)),
            timeout_seconds=5.0,
            clock=clock,
        )
        engine.start()
        for i in range(4):
            challenge = engine.current
            if i < 3:
                clock.advance(1.0 + i)
                engine.submit(pitch(60 + challenge.pitch_class))
            else:
                clock.advance(6.0)
                engine.tick()

        assert engine.total == 4
        assert engine.correct_count == 3
        assert engine.accuracy == pytest.approx(0.75)
        assert engine.median_response_ms == 2000

    def test_empty_session_summary(self):
        engine = SessionEngine(
            picker=ChallengePicker(note_challenges(ALL_NOTES), rng=random.Random(0)),
            clock=FakeClock(),
        )
        assert engine.accuracy == 0.0
        assert engine.median_response_ms is None

    def test_median_over_even_count(self):
        attempts = [
            Attempt("k", "A", Outcome.CORRECT, ms) for ms in [1000, 2000, 3000, 4000]
        ]
        engine = SessionEngine(
            picker=ChallengePicker(note_challenges(ALL_NOTES), rng=random.Random(0)),
            clock=FakeClock(),
        )
        engine.attempts = attempts
        assert engine.median_response_ms == 2500


class TestMetronome:
    def test_click_is_short_and_windowed(self):
        click = render_click(1600.0, 48000)
        assert len(click) == int(48000 * 25 / 1000)
        assert abs(click[0]) < 1e-6           # starts silent
        assert abs(click[-1]) < abs(click).max() / 10   # decayed by the end
        assert abs(click).max() <= 0.5

    def test_bpm_clamped_to_a_sane_range(self):
        m = Metronome(bpm=1000)
        assert m.bpm == 240
        m.bpm = 1
        assert m.bpm == 30

    def test_seconds_per_beat(self):
        assert Metronome(bpm=60).seconds_per_beat == pytest.approx(1.0)
        assert Metronome(bpm=120).seconds_per_beat == pytest.approx(0.5)

    def test_callback_places_beats_at_the_right_samples(self):
        """The core guarantee: beats land on exact sample positions, without drift."""
        sr, bpm, blocksize = 48000, 120, 512
        m = Metronome(sample_rate=sr, bpm=bpm, blocksize=blocksize)
        samples_per_beat = sr * 60 // bpm  # 24000

        # Run 10 seconds of audio through the callback and find every click onset.
        total_blocks = int(10 * sr / blocksize)
        rendered = np.zeros((total_blocks * blocksize, 1), dtype=np.float32)
        for i in range(total_blocks):
            block = np.zeros((blocksize, 1), dtype=np.float32)
            m._callback(block, blocksize, None, None)
            rendered[i * blocksize : (i + 1) * blocksize] = block

        # Detect the first non-silent sample of each click rather than an amplitude
        # crossing: downbeats and offbeats are different pitches, so a fixed threshold
        # would be reached a sample or two apart and look like jitter that isn't there.
        loud = np.flatnonzero(np.abs(rendered[:, 0]) > 1e-9)
        # Group contiguous samples into onsets.
        onsets = [loud[0]] + [b for a, b in zip(loud, loud[1:]) if b - a > 100]

        assert len(onsets) == 20, f"expected 20 beats in 10s at 120bpm, got {len(onsets)}"

        # Each click ramps in over a millisecond, so the detected onset sits a few
        # samples late — uniformly. Drift is what matters, so check the intervals:
        # after 10 seconds a timer-driven metronome would be milliseconds out, while
        # sample-accurate scheduling stays exact.
        intervals = np.diff(onsets)
        assert np.all(intervals == samples_per_beat), f"uneven beats: {set(intervals)}"
        assert onsets[-1] - onsets[0] == 19 * samples_per_beat

    def test_beat_index_advances(self):
        m = Metronome(sample_rate=48000, bpm=120, blocksize=512)
        for _ in range(int(48000 / 512)):  # one second
            m._callback(np.zeros((512, 1), dtype=np.float32), 512, None, None)
        assert m.beat_index == 2

    def test_downbeat_differs_from_offbeat(self):
        m = Metronome(sample_rate=48000, bpm=120, beats_per_bar=4)
        assert not np.array_equal(m._downbeat, m._offbeat)

    def test_beat_in_bar_wraps(self):
        # 240bpm is a beat every 12000 samples, so one second covers beats 0..4.
        m = Metronome(sample_rate=48000, bpm=240, beats_per_bar=4, blocksize=512)
        for _ in range(48000 // 512):
            m._callback(np.zeros((512, 1), dtype=np.float32), 512, None, None)
        assert m.beat_index == 4
        assert m.beat_in_bar == 0

    def test_muted_produces_silence_but_still_counts(self):
        m = Metronome(sample_rate=48000, bpm=120, blocksize=512)
        m.muted = True
        peak = 0.0
        for _ in range(int(48000 / 512)):
            block = np.zeros((512, 1), dtype=np.float32)
            m._callback(block, 512, None, None)
            peak = max(peak, float(np.abs(block).max()))
        assert peak == 0.0
        assert m.beat_index == 2

    def test_stereo_output_is_duplicated(self):
        m = Metronome(sample_rate=48000, bpm=240, blocksize=512)
        block = np.zeros((512, 2), dtype=np.float32)
        m._callback(block, 512, None, None)
        assert np.array_equal(block[:, 0], block[:, 1])

    def test_tempo_change_takes_effect_promptly(self):
        m = Metronome(sample_rate=48000, bpm=30, blocksize=512)
        m._callback(np.zeros((512, 1), dtype=np.float32), 512, None, None)
        m.bpm = 240
        # At 240bpm a beat is 12000 samples; within a second we must see several.
        for _ in range(int(48000 / 512)):
            m._callback(np.zeros((512, 1), dtype=np.float32), 512, None, None)
        assert m.beat_index >= 3

    def test_not_running_before_start(self):
        assert not Metronome().is_running


class TestBeatCounter:
    def test_fires_every_n_beats(self):
        counter = BeatCounter(beats_per_challenge=4)
        due = [counter.update(i)[1] for i in range(1, 13)]
        assert due == [False, False, False, True] * 3

    def test_handles_a_skipped_beat(self):
        """If the GUI thread stalls the index jumps; challenges must stay aligned."""
        counter = BeatCounter(beats_per_challenge=4)
        assert counter.update(3)[1] is False
        assert counter.update(5)[1] is True   # crossed beat 4
        assert counter.update(8)[1] is True   # crossed beat 8

    def test_reports_beats_elapsed(self):
        counter = BeatCounter(beats_per_challenge=4)
        assert counter.update(3)[0] == 3

    def test_no_change_is_not_due(self):
        counter = BeatCounter(beats_per_challenge=1)
        counter.update(1)
        assert counter.update(1) == (0, False)

    def test_reset(self):
        counter = BeatCounter(beats_per_challenge=2)
        counter.update(1)
        counter.reset(10)
        assert counter.update(11)[1] is False
        assert counter.update(12)[1] is True

    def test_minimum_of_one_beat(self):
        counter = BeatCounter(beats_per_challenge=0)
        assert counter.beats_per_challenge == 1
        assert counter.update(1)[1] is True
