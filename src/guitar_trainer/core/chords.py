"""Chord definitions, templates and matching.

Templates are synthesised from the harmonic series of each chord's notes rather than
being binary "these three pitch classes are on" vectors. A guitar string radiates
strong partials — the 2nd harmonic is an octave up (same pitch class), the 3rd is a
twelfth (a fifth above), the 5th is a major third — so a real chord's chroma always has
energy on pitch classes outside the chord. Modelling that is what lets one template set
recognise a chord across open and barre voicings alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .notes import name_to_pitch_class, pitch_class_name

#: Interval patterns in semitones from the root.
CHORD_QUALITIES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "5": (0, 7),
}

#: How each quality is written after the root, e.g. "A" + "m" -> "Am".
QUALITY_SUFFIX: dict[str, str] = {
    "maj": "",
    "min": "m",
    "7": "7",
    "maj7": "maj7",
    "min7": "m7",
    "sus2": "sus2",
    "sus4": "sus4",
    "dim": "dim",
    "aug": "aug",
    "5": "5",
}

#: A sensible starting set: the first chords most players learn.
COMMON_OPEN_CHORDS = (
    ("E", "maj"), ("A", "maj"), ("D", "maj"), ("G", "maj"), ("C", "maj"),
    ("E", "min"), ("A", "min"), ("D", "min"),
)

#: How quickly harmonics fall away. Roughly matches a plucked steel string, and is the
#: single knob that most affects how much templates differ from the binary ideal.
HARMONIC_DECAY = 0.6
N_HARMONICS = 6

#: Guitar voicings double and triple the root far more often than the other chord
#: tones — an open Em sounds three E strings against one G — so the root is expected
#: to carry more energy than an equal-weight template would predict.
ROOT_WEIGHT = 1.35

#: Weight of the unexplained-energy penalty. See :meth:`ChordMatcher.score_all`.
UNEXPLAINED_PENALTY = 0.55
#: A pitch class counts as "predicted" above this fraction of the template's peak.
SUPPORT_THRESHOLD = 0.1


@dataclass(frozen=True)
class Chord:
    """A chord as a root pitch class plus a quality."""

    root: int
    quality: str

    def __post_init__(self) -> None:
        if self.quality not in CHORD_QUALITIES:
            raise ValueError(f"unknown chord quality: {self.quality!r}")
        object.__setattr__(self, "root", self.root % 12)

    @classmethod
    def parse(cls, text: str) -> Chord:
        """Parse a chord symbol such as ``"Am"``, ``"C"``, ``"F#m7"``, ``"Gsus4"``."""
        cleaned = text.strip().replace("♯", "#").replace("♭", "b")
        if not cleaned:
            raise ValueError("empty chord symbol")

        split = 2 if len(cleaned) > 1 and cleaned[1] in "#b" else 1
        root = name_to_pitch_class(cleaned[:split])
        suffix = cleaned[split:]

        # Longest suffix first, so "m7" is not read as "m" with a stray 7.
        for quality, quality_suffix in sorted(
            QUALITY_SUFFIX.items(), key=lambda kv: -len(kv[1])
        ):
            if suffix == quality_suffix:
                return cls(root, quality)
        raise ValueError(f"unrecognised chord symbol: {text!r}")

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return tuple((self.root + i) % 12 for i in CHORD_QUALITIES[self.quality])

    def name(self, *, use_flats: bool = False) -> str:
        return pitch_class_name(self.root, use_flats=use_flats) + QUALITY_SUFFIX[self.quality]

    def __str__(self) -> str:
        return self.name()


def harmonic_template(
    pitch_classes,
    *,
    decay: float = HARMONIC_DECAY,
    n_harmonics: int = N_HARMONICS,
    root_weight: float = ROOT_WEIGHT,
):
    """Expected chroma for a set of pitch classes, including harmonic spill.

    Harmonic *k* of a note sits ``12*log2(k)`` semitones above it, which folds onto a
    pitch class; its weight falls off geometrically. The first entry of
    ``pitch_classes`` is taken as the root and weighted up, since guitar voicings
    double it. The result is L2-normalised for cosine comparison.
    """
    template = np.zeros(12)
    for index, pitch_class in enumerate(pitch_classes):
        note_weight = root_weight if index == 0 else 1.0
        for k in range(1, n_harmonics + 1):
            offset = int(round(12 * np.log2(k))) % 12
            template[(pitch_class + offset) % 12] += note_weight * decay ** (k - 1)
    norm = np.linalg.norm(template)
    return template / norm if norm > 0 else template


def all_chords(qualities=None) -> list[Chord]:
    """Every chord across all twelve roots, for the given qualities."""
    qualities = qualities or list(CHORD_QUALITIES)
    return [Chord(root, quality) for quality in qualities for root in range(12)]


@dataclass(frozen=True)
class ChordMatch:
    """A scored chord hypothesis."""

    chord: Chord
    score: float
    margin: float
    """How far ahead of the runner-up this was — low means the answer was a coin toss."""


class ChordMatcher:
    """Scores chroma vectors against a fixed set of chord templates."""

    def __init__(
        self,
        chords=None,
        *,
        root_bonus: float = 0.06,
        unexplained_penalty: float = UNEXPLAINED_PENALTY,
        support_threshold: float = SUPPORT_THRESHOLD,
    ) -> None:
        self.chords: list[Chord] = list(chords) if chords is not None else all_chords()
        self.root_bonus = root_bonus
        self.unexplained_penalty = unexplained_penalty
        self._templates = np.array(
            [harmonic_template(c.pitch_classes) for c in self.chords]
        ).reshape(len(self.chords), 12)
        if len(self.chords):
            peaks = self._templates.max(axis=1, keepdims=True)
            self._unsupported = self._templates < support_threshold * peaks
        else:
            self._unsupported = np.zeros((0, 12), dtype=bool)

    def score_all(self, chroma: np.ndarray, bass_pitch_class: int | None = None) -> np.ndarray:
        """Score every candidate chord against a chroma vector.

        Cosine similarity alone is biased towards sparse templates: a power chord
        predicts energy on only two pitch classes, so it scores well on any chord
        containing that root and fifth, and would beat the minor triad it is a subset
        of. The fix is to also charge each template for the chroma energy it fails to
        predict — the third of a minor chord is unexplained by the power chord, but
        explained by the minor triad, and that difference decides it.
        """
        norm = np.linalg.norm(chroma)
        if norm <= 0 or not len(self.chords):
            return np.zeros(len(self.chords))
        unit = chroma / norm

        scores = self._templates @ unit
        if self.unexplained_penalty:
            unexplained = np.linalg.norm(unit * self._unsupported, axis=1)
            scores = scores - self.unexplained_penalty * unexplained

        if bass_pitch_class is not None:
            roots = np.array([c.root for c in self.chords])
            scores = scores + self.root_bonus * (roots == bass_pitch_class % 12)
        return scores

    def match(self, chroma: np.ndarray, bass_pitch_class: int | None = None) -> ChordMatch | None:
        """Best-scoring chord, or ``None`` if there are no candidates."""
        if not self.chords:
            return None
        scores = self.score_all(chroma, bass_pitch_class)
        best = int(np.argmax(scores))

        if len(scores) > 1:
            runner_up = float(np.partition(scores, -2)[-2])
        else:
            runner_up = 0.0
        return ChordMatch(self.chords[best], float(scores[best]), float(scores[best] - runner_up))

    def top_n(self, chroma: np.ndarray, n: int = 3, bass_pitch_class: int | None = None):
        """The ``n`` best candidates, best first — useful for debugging near-misses."""
        scores = self.score_all(chroma, bass_pitch_class)
        order = np.argsort(scores)[::-1][:n]
        return [(self.chords[i], float(scores[i])) for i in order]


def note_template(pitch_class: int, **kwargs) -> np.ndarray:
    """Expected chroma for a single ringing note — a harmonic_template of one note.

    Used to tell a genuine chord apart from a single note whose own harmonics happen
    to land on other chord-tone positions: a major triad's third and fifth *are* the
    low harmonics of its root, so this needs the same scoring machinery as chord-vs-
    chord matching, not a cruder single-note-vs-multi-note heuristic.
    """
    return harmonic_template((pitch_class,), **kwargs)


@dataclass(frozen=True)
class NoteOrChordMatch:
    """The result of :meth:`NoteOrChordClassifier.classify`."""

    kind: str  # "note" or "chord"
    pitch_class: int | None
    """Set when ``kind == "note"``."""
    chord: Chord | None
    """Set when ``kind == "chord"``."""
    score: float


class NoteOrChordClassifier:
    """Decides whether a chroma vector is better explained as a single ringing note
    or as a chord.

    This exists because a monophonic pitch tracker (YIN) can look perfectly "stable"
    even while a chord is ringing — it just locks onto whichever string happens to
    dominate a given frame, drifting between strings as their relative loudness
    shifts as they decay. So "YIN currently reports a stable note" is not a reliable
    signal for "this is actually a single note, not a chord": it will just as
    confidently claim a strummed E major is "B2" for a while, then jump to another
    string. Chroma-based classification is what actually distinguishes the two.

    Both candidate types (12 single notes, every chord) are scored with the identical
    cosine-minus-unexplained-energy formula :class:`ChordMatcher` uses, and whichever
    scores higher wins. Plain cosine similarity against a single-note template alone
    would not be enough — see the harmonic-overlap note above — but the unexplained-
    energy penalty is what actually separates them: a real chord's chroma carries
    energy on its other tones from their *own* fundamentals, well beyond what one
    root's harmonic decay alone would predict, so a single-note template leaves much
    more of a real chord's chroma "unexplained" than the true chord template does.
    Validated against real single notes and real chord voicings — E major
    specifically, the case most likely to be confused — in
    tests/test_chords.py::TestNoteOrChordClassifier.
    """

    def __init__(
        self,
        chords=None,
        *,
        unexplained_penalty: float = UNEXPLAINED_PENALTY,
        support_threshold: float = SUPPORT_THRESHOLD,
        root_bonus: float = 0.06,
    ) -> None:
        self._chord_matcher = ChordMatcher(
            chords,
            root_bonus=root_bonus,
            unexplained_penalty=unexplained_penalty,
            support_threshold=support_threshold,
        )
        self._note_templates = np.array([note_template(pc) for pc in range(12)])
        peaks = self._note_templates.max(axis=1, keepdims=True)
        self._note_unsupported = self._note_templates < support_threshold * peaks
        self._unexplained_penalty = unexplained_penalty

    def classify(
        self, chroma: np.ndarray, bass_pitch_class: int | None = None
    ) -> NoteOrChordMatch | None:
        norm = np.linalg.norm(chroma)
        if norm <= 0:
            return None
        unit = chroma / norm

        note_scores = self._note_templates @ unit
        if self._unexplained_penalty:
            unexplained = np.linalg.norm(unit * self._note_unsupported, axis=1)
            note_scores = note_scores - self._unexplained_penalty * unexplained
        best_note_pc = int(np.argmax(note_scores))
        best_note_score = float(note_scores[best_note_pc])

        chord_match = self._chord_matcher.match(chroma, bass_pitch_class)
        chord_score = chord_match.score if chord_match is not None else -np.inf

        if chord_match is not None and chord_score > best_note_score:
            return NoteOrChordMatch("chord", None, chord_match.chord, chord_score)
        return NoteOrChordMatch("note", best_note_pc, None, best_note_score)


class NoteOrChordGate:
    """Accepts a note-or-chord classification only once it has held steadily.

    Mirrors :class:`ChordGate`, generalised over the "note" vs "chord" outcomes a
    :class:`NoteOrChordClassifier` produces.
    """

    def __init__(
        self,
        classifier: NoteOrChordClassifier,
        *,
        min_score: float = 0.35,
        stable_frames: int = 3,
    ) -> None:
        self.classifier = classifier
        self.min_score = min_score
        self.stable_frames = stable_frames
        self._streak = 0
        self._candidate_key: tuple | None = None
        self._current: NoteOrChordMatch | None = None

    @property
    def current(self) -> NoteOrChordMatch | None:
        return self._current

    def reset(self) -> None:
        self._streak = 0
        self._candidate_key = None
        self._current = None

    def push(
        self, chroma: np.ndarray | None, bass_pitch_class: int | None = None
    ) -> NoteOrChordMatch | None:
        if chroma is None:
            self.reset()
            return None

        match = self.classifier.classify(chroma, bass_pitch_class)
        if match is None or match.score < self.min_score:
            self.reset()
            return None

        key = (match.kind, match.pitch_class if match.kind == "note" else match.chord)
        if key == self._candidate_key:
            self._streak += 1
        else:
            self._candidate_key = key
            self._streak = 1

        if self._streak >= self.stable_frames:
            self._current = match
            return match

        self._current = None
        return None


class ChordGate:
    """Accepts a chord only once it has been held steadily.

    A strum passes through a moment of noise and, on a slow strum, through partial
    chords as strings are still being struck. Requiring the same answer across several
    consecutive frames with a clear margin keeps those transients from being scored.
    """

    def __init__(
        self,
        matcher: ChordMatcher,
        *,
        min_score: float = 0.82,
        min_margin: float = 0.01,
        stable_frames: int = 3,
    ) -> None:
        self.matcher = matcher
        self.min_score = min_score
        self.min_margin = min_margin
        self.stable_frames = stable_frames
        self._streak = 0
        self._candidate: Chord | None = None
        self._current: ChordMatch | None = None

    @property
    def current(self) -> ChordMatch | None:
        return self._current

    def reset(self) -> None:
        self._streak = 0
        self._candidate = None
        self._current = None

    def push(self, chroma: np.ndarray | None, bass_pitch_class: int | None = None):
        """Feed one chroma frame. Returns the accepted chord, or ``None``."""
        if chroma is None:
            self.reset()
            return None

        match = self.matcher.match(chroma, bass_pitch_class)
        if match is None or match.score < self.min_score or match.margin < self.min_margin:
            self._streak = 0
            self._candidate = None
            self._current = None
            return None

        if match.chord == self._candidate:
            self._streak += 1
        else:
            self._candidate = match.chord
            self._streak = 1

        if self._streak >= self.stable_frames:
            self._current = match
            return match

        self._current = None
        return None
