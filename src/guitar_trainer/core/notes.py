"""Pitch, note-name and fretboard geometry.

This is the single source of truth for musical maths in the app: frequency to MIDI
to note name, cents deviation, and where a given pitch lives on the fretboard.
Everything else (tuner, fretboard widget, practice modes) is data-driven off it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

A4_MIDI = 69
A4_FREQ = 440.0

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

#: Pitch classes with no accidental, in order.
NATURAL_PITCH_CLASSES = (0, 2, 4, 5, 7, 9, 11)

#: Frets that carry a position marker on a standard guitar neck.
INLAY_FRETS = (3, 5, 7, 9, 15, 17, 19, 21)
#: Frets marked with a double dot (an octave above the nut, and above that again).
DOUBLE_INLAY_FRETS = (12, 24)


def freq_to_midi(freq: float) -> float:
    """Frequency in Hz to a fractional MIDI note number."""
    if freq <= 0:
        raise ValueError(f"frequency must be positive, got {freq}")
    return A4_MIDI + 12.0 * math.log2(freq / A4_FREQ)


def midi_to_freq(midi: float) -> float:
    """MIDI note number (fractional allowed) to frequency in Hz."""
    return A4_FREQ * (2.0 ** ((midi - A4_MIDI) / 12.0))


def note_name(midi: int, *, use_flats: bool = False, with_octave: bool = True) -> str:
    """Name a MIDI note, e.g. ``69`` -> ``"A4"``."""
    names = FLAT_NAMES if use_flats else SHARP_NAMES
    name = names[midi % 12]
    if with_octave:
        # Scientific pitch notation: MIDI 60 is C4, so octave -1 is the MIDI 0..11 block.
        return f"{name}{midi // 12 - 1}"
    return name


def pitch_class_name(pitch_class: int, *, use_flats: bool = False) -> str:
    """Name a pitch class 0..11, e.g. ``9`` -> ``"A"``."""
    names = FLAT_NAMES if use_flats else SHARP_NAMES
    return names[pitch_class % 12]


def name_to_pitch_class(name: str) -> int:
    """Parse a note name (``"A"``, ``"C#"``, ``"Db"``) to a pitch class 0..11."""
    cleaned = name.strip().replace("♯", "#").replace("♭", "b")
    if not cleaned:
        raise ValueError("empty note name")
    head = cleaned[0].upper() + cleaned[1:]
    for table in (SHARP_NAMES, FLAT_NAMES):
        if head in table:
            return table.index(head)
    raise ValueError(f"unrecognised note name: {name!r}")


def name_to_midi(name: str) -> int:
    """Parse a note name with an octave (``"A4"``, ``"E2"``, ``"C#3"``) to MIDI."""
    cleaned = name.strip().replace("♯", "#").replace("♭", "b")
    idx = len(cleaned)
    while idx > 0 and (cleaned[idx - 1].isdigit() or cleaned[idx - 1] == "-"):
        idx -= 1
    if idx == len(cleaned):
        raise ValueError(f"note name is missing an octave: {name!r}")
    pitch_class = name_to_pitch_class(cleaned[:idx])
    octave = int(cleaned[idx:])
    return (octave + 1) * 12 + pitch_class


def cents_off(freq: float, target_midi: float) -> float:
    """How far ``freq`` is from ``target_midi``, in cents (positive = sharp)."""
    return (freq_to_midi(freq) - target_midi) * 100.0


def nearest_note(freq: float) -> tuple[int, float]:
    """Nearest MIDI note to ``freq``, and the deviation from it in cents."""
    midi_float = freq_to_midi(freq)
    nearest = int(round(midi_float))
    return nearest, (midi_float - nearest) * 100.0


@dataclass(frozen=True)
class Tuning:
    """An instrument tuning, as the open-string pitches from lowest string to highest.

    ``open_midi[0]`` is the thickest/lowest string (the 6th on a guitar). The UI draws
    strings in this order too, flipped so the low string is at the bottom.
    """

    name: str
    open_midi: tuple[int, ...]
    fret_count: int = 22

    @classmethod
    def from_names(cls, name: str, names: list[str], fret_count: int = 22) -> Tuning:
        """Build a tuning from note names, e.g. ``["E2", "A2", "D3", "G3", "B3", "E4"]``."""
        return cls(name, tuple(name_to_midi(n) for n in names), fret_count)

    @property
    def string_count(self) -> int:
        return len(self.open_midi)

    def midi_at(self, string: int, fret: int) -> int:
        """MIDI note at a fret. ``string`` is indexed from the lowest string (0)."""
        if not 0 <= string < self.string_count:
            raise IndexError(f"string {string} out of range for {self.string_count}-string tuning")
        if not 0 <= fret <= self.fret_count:
            raise IndexError(f"fret {fret} out of range 0..{self.fret_count}")
        return self.open_midi[string] + fret

    def string_label(self, string: int) -> str:
        """Open-note name for a string, e.g. ``"E2"``."""
        return note_name(self.open_midi[string])

    def positions_for_pitch_class(self, pitch_class: int) -> list[tuple[int, int]]:
        """Every ``(string, fret)`` playing this pitch class, in any octave."""
        pitch_class %= 12
        return [
            (string, fret)
            for string in range(self.string_count)
            for fret in range(self.fret_count + 1)
            if self.midi_at(string, fret) % 12 == pitch_class
        ]

    def positions_for_midi(self, midi: int) -> list[tuple[int, int]]:
        """Every ``(string, fret)`` playing this exact pitch."""
        return [
            (string, fret)
            for string in range(self.string_count)
            for fret in range(self.fret_count + 1)
            if self.midi_at(string, fret) == midi
        ]

    def nearest_string(self, freq: float) -> tuple[int, float]:
        """The open string closest to ``freq``, and the deviation in cents.

        Used by the tuner. Comparison is on the log-frequency axis so that "closest"
        means closest in pitch rather than in Hz.
        """
        midi_float = freq_to_midi(freq)
        string = min(
            range(self.string_count),
            key=lambda s: abs(midi_float - self.open_midi[s]),
        )
        return string, (midi_float - self.open_midi[string]) * 100.0

    def midi_range(self) -> tuple[int, int]:
        """Lowest and highest MIDI note reachable in this tuning."""
        return min(self.open_midi), max(m + self.fret_count for m in self.open_midi)


STANDARD = Tuning.from_names("Standard E", ["E2", "A2", "D3", "G3", "B3", "E4"])
DROP_D = Tuning.from_names("Drop D", ["D2", "A2", "D3", "G3", "B3", "E4"])
HALF_STEP_DOWN = Tuning.from_names("Half-step down", ["D#2", "G#2", "C#3", "F#3", "A#3", "D#4"])
OPEN_G = Tuning.from_names("Open G", ["D2", "G2", "D3", "G3", "B3", "D4"])
DADGAD = Tuning.from_names("DADGAD", ["D2", "A2", "D3", "G3", "A3", "D4"])

TUNING_PRESETS: tuple[Tuning, ...] = (STANDARD, DROP_D, HALF_STEP_DOWN, OPEN_G, DADGAD)


def preset_by_name(name: str) -> Tuning | None:
    """Look up a preset tuning by its display name."""
    return next((t for t in TUNING_PRESETS if t.name == name), None)


def fret_x_positions(fret_count: int, scale_length: float = 1.0) -> list[float]:
    """Distance from the nut to each fret, as a fraction of ``scale_length``.

    Uses the standard rule ``d_n = L - L / 2^(n/12)``, so index 0 is the nut (0.0) and
    index 12 sits at exactly half the scale length. Returns ``fret_count + 1`` values.
    """
    return [scale_length * (1.0 - 2.0 ** (-n / 12.0)) for n in range(fret_count + 1)]


@dataclass
class NoteSet:
    """A user-selected group of pitch classes to be drilled on."""

    name: str
    pitch_classes: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.pitch_classes = tuple(sorted({pc % 12 for pc in self.pitch_classes}))
        if not self.pitch_classes:
            raise ValueError("a note set needs at least one pitch class")

    def labels(self, *, use_flats: bool = False) -> list[str]:
        return [pitch_class_name(pc, use_flats=use_flats) for pc in self.pitch_classes]


ALL_NOTES = NoteSet("All 12 notes", tuple(range(12)))
NATURAL_NOTES = NoteSet("Naturals only", NATURAL_PITCH_CLASSES)

NOTE_SET_PRESETS: tuple[NoteSet, ...] = (ALL_NOTES, NATURAL_NOTES)
