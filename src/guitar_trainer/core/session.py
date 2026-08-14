"""The practice loop.

Both practice modes are the same state machine over different challenge types, so the
engine knows nothing about notes or chords beyond asking a challenge whether what was
detected satisfies it. Timing is injected as a clock so the whole thing is testable
without real time passing.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Protocol, Sequence, runtime_checkable

from .chords import Chord
from .notes import NoteSet, pitch_class_name


class SessionState(Enum):
    IDLE = auto()
    LISTENING = auto()
    FINISHED = auto()


class Outcome(Enum):
    CORRECT = "correct"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@runtime_checkable
class Challenge(Protocol):
    """Something the user is asked to play."""

    @property
    def prompt(self) -> str:
        """What to show the user, e.g. ``"A"`` or ``"Am7"``."""

    @property
    def key(self) -> str:
        """Stable identifier for stats storage."""

    def matches(self, detection) -> bool:
        """Whether a detection satisfies this challenge."""


@dataclass(frozen=True)
class NoteChallenge:
    """Play a given note, in any octave.

    Octave is deliberately ignored: the user asked to be able to answer anywhere on the
    neck, and comparing pitch classes is exactly that.
    """

    pitch_class: int
    use_flats: bool = False

    @property
    def prompt(self) -> str:
        return pitch_class_name(self.pitch_class, use_flats=self.use_flats)

    @property
    def key(self) -> str:
        return f"note:{self.pitch_class}"

    def matches(self, detection) -> bool:
        return detection is not None and getattr(detection, "pitch_class", None) == self.pitch_class


@dataclass(frozen=True)
class ChordChallenge:
    """Play a given chord, in any voicing."""

    chord: Chord

    @property
    def prompt(self) -> str:
        return self.chord.name()

    @property
    def key(self) -> str:
        return f"chord:{self.chord.root}:{self.chord.quality}"

    def matches(self, detection) -> bool:
        if detection is None:
            return False
        # Accepts either a ChordMatch or a bare Chord.
        chord = getattr(detection, "chord", detection)
        return chord == self.chord


@dataclass
class Attempt:
    """One completed challenge, as recorded in the stats database."""

    challenge_key: str
    prompt: str
    outcome: Outcome
    response_ms: int | None
    detected: str | None = None

    @property
    def correct(self) -> bool:
        return self.outcome is Outcome.CORRECT


class ChallengePicker:
    """Random selection that never repeats the previous challenge.

    Optionally biases towards challenges the user is worse at, using per-key weights
    supplied by the stats layer — practising what you already know is a poor use of
    practice time.
    """

    def __init__(
        self,
        challenges: Sequence[Challenge],
        *,
        rng: random.Random | None = None,
        weight_fn: Callable[[str], float] | None = None,
    ) -> None:
        if not challenges:
            raise ValueError("a practice session needs at least one challenge")
        self.challenges = list(challenges)
        self.rng = rng or random.Random()
        self.weight_fn = weight_fn
        self._previous: Challenge | None = None
        # Generated ahead of being consumed, so peek() can show what's coming up
        # without that answer changing between the peek and the actual presentation.
        self._queue: deque[Challenge] = deque()

    def _generate(self) -> Challenge:
        candidates = [c for c in self.challenges if c != self._previous]
        if not candidates:
            # Only one challenge in the set, so repetition is unavoidable.
            candidates = self.challenges

        if self.weight_fn is not None:
            weights = [max(1e-6, self.weight_fn(c.key)) for c in candidates]
            choice = self.rng.choices(candidates, weights=weights, k=1)[0]
        else:
            choice = self.rng.choice(candidates)

        self._previous = choice
        return choice

    def _ensure_queued(self, n: int) -> None:
        while len(self._queue) < n:
            self._queue.append(self._generate())

    def next(self) -> Challenge:
        self._ensure_queued(1)
        return self._queue.popleft()

    def peek(self, ahead: int = 1) -> Challenge | None:
        """The challenge ``ahead`` positions past the one about to be returned by
        :meth:`next`, without consuming it — what a "coming up" preview shows.

        ``peek(1)`` is the challenge that will follow whichever one is presented next;
        it does not change on repeated calls, since it's already been decided (with
        the same no-immediate-repeat rule ``next()`` uses), just not handed out yet.
        """
        if ahead < 1:
            return None
        self._ensure_queued(ahead)
        return self._queue[ahead - 1]

    def reset(self) -> None:
        self._previous = None
        self._queue.clear()


@dataclass
class SessionEngine:
    """Drives prompt → listen → resolve, over any challenge type.

    The engine is pure logic and never touches Qt or audio; the UI polls
    :meth:`tick` and pushes detections in. That keeps the whole practice loop testable
    with a fake clock.
    """

    picker: ChallengePicker
    timeout_seconds: float | None = None
    """``None`` is free mode — wait as long as it takes."""

    clock: Callable[[], float] = time.monotonic
    on_challenge: Callable[[Challenge], None] | None = None
    on_result: Callable[[Attempt], None] | None = None

    state: SessionState = SessionState.IDLE
    current: Challenge | None = None
    previous: Challenge | None = None
    """The challenge presented immediately before ``current`` — what was just
    resolved, for a "last played" reference alongside the current prompt."""
    attempts: list[Attempt] = field(default_factory=list)
    _started_at: float = 0.0

    # ---------------------------------------------------------------- control

    def start(self) -> Challenge:
        """Begin a session and present the first challenge."""
        self.attempts.clear()
        self.current = None  # so _present()'s `previous = current` starts at None
        self.picker.reset()
        return self._present()

    def stop(self) -> None:
        self.state = SessionState.FINISHED
        self.current = None

    def upcoming(self, n: int = 1) -> list[Challenge]:
        """The next ``n`` challenges after ``current``, without consuming them."""
        result = []
        for i in range(1, n + 1):
            challenge = self.picker.peek(i)
            if challenge is None:
                break
            result.append(challenge)
        return result

    def _present(self) -> Challenge:
        self.previous = self.current
        self.current = self.picker.next()
        self.state = SessionState.LISTENING
        self._started_at = self.clock()
        if self.on_challenge:
            self.on_challenge(self.current)
        return self.current

    @property
    def elapsed(self) -> float:
        """Seconds since the current challenge was presented."""
        return self.clock() - self._started_at if self.state is SessionState.LISTENING else 0.0

    @property
    def time_remaining(self) -> float | None:
        """Seconds left, or ``None`` in free mode."""
        if self.timeout_seconds is None or self.state is not SessionState.LISTENING:
            return None
        return max(0.0, self.timeout_seconds - self.elapsed)

    # ----------------------------------------------------------------- input

    def submit(self, detection, *, label: str | None = None) -> Attempt | None:
        """Offer a detection. Returns an Attempt if it resolved the challenge.

        Wrong notes are ignored rather than failing the challenge: while hunting for a
        note on the neck you will sound several others on the way, and failing on those
        would make the mode unusable.
        """
        if self.state is not SessionState.LISTENING or self.current is None:
            return None
        if not self.current.matches(detection):
            return None
        return self._resolve(Outcome.CORRECT, label)

    def skip(self) -> Attempt | None:
        """Give up on the current challenge and move on."""
        if self.state is not SessionState.LISTENING:
            return None
        return self._resolve(Outcome.SKIPPED, None)

    def tick(self) -> Attempt | None:
        """Advance time. Returns an Attempt if the challenge timed out."""
        if self.state is not SessionState.LISTENING or self.timeout_seconds is None:
            return None
        if self.elapsed < self.timeout_seconds:
            return None
        return self._resolve(Outcome.TIMEOUT, None)

    def _resolve(self, outcome: Outcome, label: str | None) -> Attempt:
        assert self.current is not None
        attempt = Attempt(
            challenge_key=self.current.key,
            prompt=self.current.prompt,
            outcome=outcome,
            response_ms=int((self.clock() - self._started_at) * 1000),
            detected=label,
        )
        self.attempts.append(attempt)
        if self.on_result:
            self.on_result(attempt)
        self._present()
        return attempt

    # --------------------------------------------------------------- summary

    @property
    def total(self) -> int:
        return len(self.attempts)

    @property
    def correct_count(self) -> int:
        return sum(1 for a in self.attempts if a.correct)

    @property
    def accuracy(self) -> float:
        """Fraction correct, 0.0 when nothing has been attempted."""
        return self.correct_count / self.total if self.total else 0.0

    @property
    def median_response_ms(self) -> int | None:
        """Median response time over correct answers only."""
        times = sorted(a.response_ms for a in self.attempts if a.correct and a.response_ms)
        if not times:
            return None
        mid = len(times) // 2
        if len(times) % 2:
            return times[mid]
        return (times[mid - 1] + times[mid]) // 2


def note_challenges(note_set: NoteSet, *, use_flats: bool = False) -> list[NoteChallenge]:
    return [NoteChallenge(pc, use_flats=use_flats) for pc in note_set.pitch_classes]


def chord_challenges(chords: Sequence[Chord]) -> list[ChordChallenge]:
    return [ChordChallenge(c) for c in chords]
