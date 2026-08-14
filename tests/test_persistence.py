import pytest

from guitar_trainer.core.config import Config, _to_toml, default_chord_symbols
from guitar_trainer.core.notes import STANDARD, name_to_midi
from guitar_trainer.core.session import Attempt, Outcome
from guitar_trainer.core.stats import StatsStore


@pytest.fixture
def store(tmp_path):
    with StatsStore(tmp_path / "stats.db") as s:
        yield s


def attempt(key="note:9", prompt="A", outcome=Outcome.CORRECT, ms=1000):
    return Attempt(key, prompt, outcome, ms)


class TestConfig:
    def test_defaults_are_usable(self):
        config = Config()
        assert config.tuning().open_midi == STANDARD.open_midi
        assert len(config.chords()) == len(default_chord_symbols())

    def test_round_trip(self, tmp_path):
        path = tmp_path / "config.toml"
        original = Config(
            device_name="Blue Microphones",
            noise_gate=0.025,
            tuning_name="Drop D",
            fret_count=24,
            label_mode="All notes",
            use_flats=True,
            chord_symbols=["C", "Am", "F#m7"],
            bpm=132,
            rhythm_mode=True,
            note_accidental_style="mix",
        )
        original.save(path)
        loaded = Config.load(path)
        assert loaded == original

    def test_missing_file_yields_defaults(self, tmp_path):
        assert Config.load(tmp_path / "nope.toml") == Config()

    def test_corrupt_file_yields_defaults(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text("this is not [valid toml")
        assert Config.load(path) == Config()

    def test_unknown_keys_ignored(self, tmp_path):
        """A config written by a newer version must not stop an older one starting."""
        path = tmp_path / "config.toml"
        path.write_text('bpm = 100\nfuture_setting = "x"\n')
        assert Config.load(path).bpm == 100

    def test_custom_tuning_overrides_preset(self):
        config = Config(custom_tuning=[name_to_midi(n) for n in ["D2", "A2", "D3", "F#3", "A3", "D4"]])
        assert config.tuning().name == "Custom"
        assert config.tuning().open_midi[0] == name_to_midi("D2")

    def test_fret_count_applied_to_preset(self):
        assert Config(tuning_name="Drop D", fret_count=24).tuning().fret_count == 24

    def test_unknown_tuning_falls_back_to_standard(self):
        assert Config(tuning_name="Nonsense").tuning().open_midi == STANDARD.open_midi

    def test_unparseable_chords_skipped(self):
        config = Config(chord_symbols=["C", "not-a-chord", "Am"])
        assert [str(c) for c in config.chords()] == ["C", "Am"]

    def test_all_bad_chords_falls_back_to_defaults(self):
        assert len(Config(chord_symbols=["xxx"]).chords()) == len(default_chord_symbols())

    def test_save_creates_parent_directory(self, tmp_path):
        path = tmp_path / "nested" / "deeper" / "config.toml"
        Config().save(path)
        assert path.exists()

    def test_save_is_atomic(self, tmp_path):
        """No temp file should survive a successful write."""
        path = tmp_path / "config.toml"
        Config().save(path)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_toml_escapes_quotes(self):
        assert '\\"' in _to_toml({"name": 'a "quoted" device'})

    def test_toml_booleans_lowercase(self):
        assert _to_toml({"flag": True}).strip() == "flag = true"

    def test_toml_rejects_unsupported_types(self):
        with pytest.raises(TypeError):
            _to_toml({"bad": {"nested": 1}})


class TestStatsStore:
    def test_schema_created(self, store):
        assert store.totals() == (0, 0)
        assert store.session_count() == 0

    def test_record_and_total(self, store):
        session = store.start_session("note", bpm=80)
        store.record(session, attempt())
        store.record(session, attempt(outcome=Outcome.TIMEOUT))
        store.end_session(session)

        assert store.totals() == (2, 1)
        assert store.session_count() == 1

    def test_challenge_stats_aggregate(self, store):
        session = store.start_session("note")
        for ms in [1000, 2000, 3000]:
            store.record(session, attempt(ms=ms))
        store.record(session, attempt(outcome=Outcome.TIMEOUT, ms=5000))

        stats = store.challenge_stats()
        assert len(stats) == 1
        assert stats[0].attempts == 4
        assert stats[0].correct == 3
        assert stats[0].accuracy == pytest.approx(0.75)
        assert stats[0].median_response_ms == 2000

    def test_median_ignores_wrong_answers(self, store):
        session = store.start_session("note")
        store.record(session, attempt(ms=1000))
        store.record(session, attempt(outcome=Outcome.TIMEOUT, ms=99000))
        assert store.challenge_stats()[0].median_response_ms == 1000

    def test_prefix_filter(self, store):
        session = store.start_session("mixed")
        store.record(session, attempt(key="note:9", prompt="A"))
        store.record(session, attempt(key="chord:0:maj", prompt="C"))

        assert len(store.challenge_stats(prefix="note:")) == 1
        assert store.challenge_stats(prefix="chord:")[0].prompt == "C"

    def test_persists_across_reopen(self, tmp_path):
        path = tmp_path / "stats.db"
        with StatsStore(path) as store:
            session = store.start_session("note")
            store.record(session, attempt())
        with StatsStore(path) as store:
            assert store.totals() == (1, 0 + 1)

    def test_reset_clears_everything(self, store):
        session = store.start_session("note")
        store.record(session, attempt())
        store.reset()
        assert store.totals() == (0, 0)
        assert store.session_count() == 0

    def test_no_stats_for_untouched_challenges(self, store):
        assert store.challenge_stats() == []


class TestWeaknessWeights:
    def test_empty_history_gives_no_weights(self, store):
        assert store.weakness_weights() == {}

    def test_inaccurate_challenges_weighted_higher(self, store):
        session = store.start_session("note")
        for _ in range(5):
            store.record(session, attempt(key="note:0", prompt="C"))
        for _ in range(5):
            store.record(session, attempt(key="note:1", prompt="C#", outcome=Outcome.TIMEOUT))

        weights = store.weakness_weights()
        assert weights["note:1"] > weights["note:0"]

    def test_slow_challenges_weighted_higher(self, store):
        session = store.start_session("note")
        for _ in range(4):
            store.record(session, attempt(key="note:0", prompt="C", ms=1000))
            store.record(session, attempt(key="note:1", prompt="C#", ms=6000))

        weights = store.weakness_weights()
        assert weights["note:1"] > weights["note:0"]

    def test_weights_stay_positive(self, store):
        session = store.start_session("note")
        for _ in range(10):
            store.record(session, attempt(ms=500))
        assert all(w > 0 for w in store.weakness_weights().values())

    def test_perfect_accuracy_gets_the_floor(self, store):
        session = store.start_session("note")
        for _ in range(5):
            store.record(session, attempt(ms=1000))
        assert store.weakness_weights()["note:9"] == pytest.approx(1.0)
