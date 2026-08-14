"""Practice history, stored in SQLite.

Every attempt is recorded so the app can show which notes and chords are slow or
error-prone, and optionally drill those more often.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import data_dir
from .session import Attempt

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mode        TEXT    NOT NULL,
    started_at  TEXT    NOT NULL,
    ended_at    TEXT,
    bpm         INTEGER,
    config      TEXT
);

CREATE TABLE IF NOT EXISTS attempts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    challenge_key TEXT    NOT NULL,
    prompt        TEXT    NOT NULL,
    outcome       TEXT    NOT NULL,
    correct       INTEGER NOT NULL,
    response_ms   INTEGER,
    detected      TEXT,
    created_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attempts_session ON attempts(session_id);
CREATE INDEX IF NOT EXISTS idx_attempts_key ON attempts(challenge_key);
"""


@dataclass(frozen=True)
class ChallengeStats:
    """Aggregated history for one note or chord."""

    challenge_key: str
    prompt: str
    attempts: int
    correct: int
    median_response_ms: int | None

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempts if self.attempts else 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StatsStore:
    """Thin wrapper over the SQLite database."""

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            data_dir().mkdir(parents=True, exist_ok=True)
            path = data_dir() / "stats.db"
        self.path = str(path)
        # The GUI thread opens sessions while the analysis thread may record attempts.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> StatsStore:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -------------------------------------------------------------- writing

    def start_session(self, mode: str, *, bpm: int | None = None, config: dict | None = None) -> int:
        cursor = self._conn.execute(
            "INSERT INTO sessions (mode, started_at, bpm, config) VALUES (?, ?, ?, ?)",
            (mode, _now(), bpm, json.dumps(config) if config else None),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def end_session(self, session_id: int) -> None:
        self._conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?", (_now(), session_id)
        )
        self._conn.commit()

    def record(self, session_id: int, attempt: Attempt) -> None:
        self._conn.execute(
            """INSERT INTO attempts
               (session_id, challenge_key, prompt, outcome, correct, response_ms,
                detected, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                attempt.challenge_key,
                attempt.prompt,
                attempt.outcome.value,
                int(attempt.correct),
                attempt.response_ms,
                attempt.detected,
                _now(),
            ),
        )
        self._conn.commit()

    # -------------------------------------------------------------- reading

    def challenge_stats(self, *, prefix: str | None = None) -> list[ChallengeStats]:
        """Per-challenge totals, best accuracy first.

        ``prefix`` filters by challenge key, so ``"note:"`` gets just the note drills.
        """
        where = "WHERE challenge_key LIKE ?" if prefix else ""
        params = (f"{prefix}%",) if prefix else ()

        rows = self._conn.execute(
            f"""SELECT challenge_key, prompt, COUNT(*) AS attempts,
                       SUM(correct) AS correct
                FROM attempts {where}
                GROUP BY challenge_key
                ORDER BY prompt""",
            params,
        ).fetchall()

        out = []
        for row in rows:
            out.append(
                ChallengeStats(
                    challenge_key=row["challenge_key"],
                    prompt=row["prompt"],
                    attempts=row["attempts"],
                    correct=row["correct"] or 0,
                    median_response_ms=self._median_response(row["challenge_key"]),
                )
            )
        return out

    def _median_response(self, challenge_key: str) -> int | None:
        """Median over correct answers.

        SQLite has no median function, and the per-challenge row counts here are small
        enough that fetching and indexing is simpler than a window-function query.
        """
        times = [
            row[0]
            for row in self._conn.execute(
                """SELECT response_ms FROM attempts
                   WHERE challenge_key = ? AND correct = 1 AND response_ms IS NOT NULL
                   ORDER BY response_ms""",
                (challenge_key,),
            )
        ]
        if not times:
            return None
        mid = len(times) // 2
        return times[mid] if len(times) % 2 else (times[mid - 1] + times[mid]) // 2

    def totals(self) -> tuple[int, int]:
        """Overall ``(attempts, correct)`` across all history."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(correct), 0) AS c FROM attempts"
        ).fetchone()
        return int(row["n"]), int(row["c"])

    def session_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"])

    def weakness_weights(self, *, prefix: str | None = None, floor: float = 1.0) -> dict[str, float]:
        """Selection weights that favour challenges the user gets wrong or answers slowly.

        Unseen challenges keep the default weight rather than being starved: they carry
        no evidence either way, and biasing against them would mean never drilling
        anything new.
        """
        stats = self.challenge_stats(prefix=prefix)
        if not stats:
            return {}

        times = [s.median_response_ms for s in stats if s.median_response_ms]
        typical = sorted(times)[len(times) // 2] if times else None

        weights = {}
        for entry in stats:
            # Accuracy contributes most: 100% correct scores 1.0, 0% scores 4.0.
            weight = floor + 3.0 * (1.0 - entry.accuracy)
            if typical and entry.median_response_ms:
                # Being twice as slow as usual is worth roughly one extra point.
                weight += min(2.0, entry.median_response_ms / typical - 1.0)
            weights[entry.challenge_key] = max(floor * 0.25, weight)
        return weights

    def reset(self) -> None:
        """Delete all history."""
        self._conn.executescript("DELETE FROM attempts; DELETE FROM sessions;")
        self._conn.commit()
