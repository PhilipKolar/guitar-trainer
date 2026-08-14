"""Settings, persisted to the XDG config directory as TOML."""

from __future__ import annotations

import os
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .chords import COMMON_OPEN_CHORDS, Chord
from .notes import STANDARD, Tuning, name_to_pitch_class, preset_by_name

APP_NAME = "guitar-trainer"


def config_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / APP_NAME


def data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(base) / APP_NAME


def config_path() -> Path:
    return config_dir() / "config.toml"


def default_chord_symbols() -> list[str]:
    return [Chord(name_to_pitch_class(r), q).name() for r, q in COMMON_OPEN_CHORDS]


@dataclass
class Config:
    """Everything the app remembers between runs."""

    # Audio
    device_name: str | None = None
    """Matched by name rather than index, since indices shift as devices come and go."""
    noise_gate: float = 0.01
    min_harmonic_ratio: float = 0.6
    """Mirrors audio.pitch.DEFAULT_MIN_HARMONIC_RATIO; kept independent so core/ has no
    audio import (see AGENTS.md)."""

    # Instrument
    tuning_name: str = STANDARD.name
    custom_tuning: list[int] = field(default_factory=list)
    fret_count: int = 22

    # Display
    label_mode: str = "None"
    use_flats: bool = False

    # Practice
    note_set: str = "All 12 notes"
    custom_note_set: list[int] = field(default_factory=list)
    chord_symbols: list[str] = field(default_factory=default_chord_symbols)
    bpm: int = 80
    beats_per_challenge: int = 4
    beats_per_bar: int = 4
    rhythm_mode: bool = False
    metronome_muted: bool = False
    weight_by_weakness: bool = False

    # ------------------------------------------------------------ derived

    def tuning(self) -> Tuning:
        """The configured tuning, falling back to standard if it can't be resolved."""
        if self.custom_tuning:
            return Tuning("Custom", tuple(self.custom_tuning), self.fret_count)
        preset = preset_by_name(self.tuning_name) or STANDARD
        # Apply the configured fret count on top of the preset's default.
        return Tuning(preset.name, preset.open_midi, self.fret_count)

    def chords(self) -> list[Chord]:
        """Parsed chord set, skipping anything unparseable rather than failing to start."""
        out = []
        for symbol in self.chord_symbols:
            try:
                out.append(Chord.parse(symbol))
            except ValueError:
                continue
        return out or [Chord.parse(s) for s in default_chord_symbols()]

    # ------------------------------------------------------- serialisation

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        """Read config from disk. Missing or corrupt files yield defaults."""
        path = path or config_path()
        try:
            with open(path, "rb") as handle:
                raw = tomllib.load(handle)
        except (FileNotFoundError, tomllib.TOMLDecodeError, OSError):
            return cls()

        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path | None = None) -> None:
        """Write config to disk, creating the directory if needed."""
        path = path or config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".toml.tmp")
        tmp.write_text(_to_toml(asdict(self)))
        # Atomic replace, so an interrupted write can't leave a truncated config.
        tmp.replace(path)


def _to_toml(data: dict) -> str:
    """Minimal TOML writer.

    Python ships a TOML reader but no writer, and the config is a flat table of
    scalars and string/int lists — not worth a dependency.
    """
    lines = []
    for key, value in data.items():
        if value is None:
            continue
        lines.append(f"{key} = {_format(value)}")
    return "\n".join(lines) + "\n"


def _format(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format(v) for v in value) + "]"
    raise TypeError(f"cannot serialise {type(value).__name__} to TOML")
