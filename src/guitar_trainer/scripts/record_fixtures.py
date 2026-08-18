"""Records real guitar audio (single notes or chords) into tests/fixtures/audio/.

Synthesised test signals validate the algorithm; they can't catch anything specific
to a particular guitar, pickup, mic, or player — the exact things behind "detection
feels erratic" reports. This walks through recording one short clip per note or chord
so those reports become a concrete, re-runnable pytest assertion instead
(tests/test_real_recordings.py, tests/test_real_chords.py) — run once, see exactly
which take is wrong.

    python -m guitar_trainer.scripts.record_fixtures [--device N] [--duration 4]
    python -m guitar_trainer.scripts.record_fixtures --kind chords

Tune the guitar with a hardware tuner first — this script only records and checks
what it hears against what the app's own detector reports, it doesn't verify
intonation.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio.capture import DEFAULT_SAMPLE_RATE, AudioCapture, list_input_devices
from ..audio.chroma import ChromaAnalyser
from ..audio.pitch import DEFAULT_HOP, DEFAULT_WINDOW, NoteGate, YinDetector
from ..core.chords import Chord, ChordGate, ChordMatcher
from ..core.notes import STANDARD, note_name

#: One full chromatic octave, played on a single string (frets 0-11 on the low E) so
#: there's no string-to-string jump to get wrong while recording.
NOTE_PLAN = [(fret, STANDARD.midi_at(0, fret)) for fret in range(12)]

#: The first chords most players learn: E, A and D, each as a major triad, minor
#: triad and dominant seventh. More roots/qualities can be added to this list the
#: same way later — nothing else here is specific to these nine.
CHORD_PLAN = [Chord.parse(s) for s in ("E", "Em", "E7", "A", "Am", "A7", "D", "Dm", "D7")]

_QUALITY_LABEL = {"maj": "major", "min": "minor", "7": "dominant 7th"}

FIXTURES_ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "audio"
NOTES_FIXTURES_DIR = FIXTURES_ROOT / "notes"
CHORDS_FIXTURES_DIR = FIXTURES_ROOT / "chords"


def _load_wavfile():
    from scipy.io import wavfile

    return wavfile


def save_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    wavfile = _load_wavfile()
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    wavfile.write(str(path), sample_rate, (clipped * 32767).astype(np.int16))


def _level_bar(rms: float, width: int = 30) -> str:
    filled = int(min(width, rms * width / 0.2))
    return "#" * filled + "." * (width - filled)


#: How long to wait after Enter is pressed before the timed capture window starts.
#: The keyboard's own acoustic tap — picked up by a sensitive mic sitting right next
#: to it — otherwise ends up as the first ~0.3-0.4s of every clip (confirmed on real
#: recordings: a clear amplitude spike right at the start, then a quiet gap before the
#: actual note begins). Long enough for the tap to fully decay, short enough not to
#: feel like a real pause.
PRE_ROLL_SECONDS = 0.6


def record_clip(capture: AudioCapture, duration: float, sample_rate: int) -> np.ndarray:
    print(f"  (settling for {PRE_ROLL_SECONDS:.1f}s so the Enter key-tap isn't in the clip)")
    time.sleep(PRE_ROLL_SECONDS)
    capture.buffer.clear()
    samples_needed = int(duration * sample_rate)
    start = time.monotonic()
    while capture.buffer.total_written < samples_needed:
        recent = capture.buffer.read_latest(min(4096, capture.buffer.capacity))
        rms = float(np.sqrt(np.mean(recent.astype(np.float64) ** 2))) if len(recent) else 0.0
        elapsed = time.monotonic() - start
        print(
            f"\r  recording {elapsed:4.1f}s/{duration:.1f}s  [{_level_bar(rms)}]",
            end="",
            flush=True,
        )
        time.sleep(0.05)
    print()
    return capture.buffer.read_latest(samples_needed)


def sanity_check(samples: np.ndarray, expected_midi: int, sample_rate: int) -> str:
    """Run the take through the real detector so a bad recording is caught right
    away, not after the whole batch is done."""
    detector = YinDetector(sample_rate)
    gate = NoteGate()
    settled = None
    for start in range(0, len(samples) - DEFAULT_WINDOW, DEFAULT_WINDOW // 4):
        stable = gate.push(detector.detect(samples[start : start + DEFAULT_WINDOW]))
        if stable is not None:
            settled = stable

    if settled is None:
        return "no stable note detected — too quiet, or didn't settle in time"
    if settled.midi != expected_midi:
        return f"detected {settled.name} ({settled.cents:+.0f}¢) — expected {note_name(expected_midi)}"
    return f"detected {settled.name} ({settled.cents:+.0f}¢) — looks right"


def sanity_check_chord(samples: np.ndarray, expected: Chord, sample_rate: int) -> str:
    """Run the take through the real chord-detection path — chroma extraction plus
    matching against every chord, not just the one expected, so a take that actually
    reads as some *other* chord is caught right away rather than reported as a vague
    "not this one". Scored against the full catalogue (``ChordMatcher()`` with no
    restriction) to mirror Free Detect, the most demanding real path — a practice
    session narrows the candidates, which is easier, not harder.
    """
    analyser = ChromaAnalyser(sample_rate)
    gate = ChordGate(ChordMatcher())
    window = max(DEFAULT_WINDOW, analyser.window)
    settled = None
    for start in range(0, len(samples) - window, DEFAULT_HOP):
        frame = samples[start : start + window]
        chroma = analyser.analyse(frame)
        bass = analyser.bass_pitch_class(frame) if chroma is not None else None
        stable = gate.push(chroma, bass)
        if stable is not None:
            settled = stable

    if settled is None:
        return "no stable chord detected — too quiet, ambiguous, or didn't settle in time"
    if settled.chord != expected:
        return f"detected {settled.chord.name()} (score {settled.score:.2f}) — expected {expected.name()}"
    return f"detected {settled.chord.name()} (score {settled.score:.2f}, margin {settled.margin:.2f}) — looks right"


@dataclass(frozen=True)
class _Item:
    """One thing to record: a prompt to show, the filename to save under, and how
    to sanity-check the take. Notes and chords differ only in how these are built —
    the recording loop in main() doesn't need to know which it's holding."""

    heading: str
    filename: str
    check: Callable[[np.ndarray, int], str]


def _note_items(plan) -> list[_Item]:
    items = []
    for fret, midi in plan:
        name = note_name(midi)
        where = "open string" if fret == 0 else f"fret {fret}"
        items.append(
            _Item(
                heading=f"{name} — low E string, {where}",
                filename=f"{name}.wav",
                check=lambda samples, sr, midi=midi: sanity_check(samples, midi, sr),
            )
        )
    return items


def _chord_items(plan) -> list[_Item]:
    items = []
    for chord in plan:
        symbol = chord.name()
        items.append(
            _Item(
                heading=f"{symbol} ({_QUALITY_LABEL[chord.quality]}) — strum it, any voicing, let it ring",
                filename=f"{symbol}.wav",
                check=lambda samples, sr, chord=chord: sanity_check_chord(samples, chord, sr),
            )
        )
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=None, help="input device index")
    parser.add_argument("--list", action="store_true", help="list input devices and exit")
    parser.add_argument("--duration", type=float, default=4.0, help="seconds per clip (default 4)")
    parser.add_argument(
        "--kind", choices=("notes", "chords"), default="notes", help="what to record (default notes)"
    )
    parser.add_argument(
        "--force", action="store_true", help="re-record items that already have a saved clip"
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output directory (default tests/fixtures/audio/<kind>)",
    )
    args = parser.parse_args()

    devices = list_input_devices()
    if args.list:
        if not devices:
            print("No input devices found.")
        for dev in devices:
            marker = " (default)" if dev.is_default else ""
            print(f"  [{dev.index}] {dev.name} — {dev.channels}ch{marker}")
        return 0
    if not devices:
        print("No input devices found. Is PortAudio installed (libportaudio2)?")
        return 1

    if args.kind == "chords":
        all_items = _chord_items(CHORD_PLAN)
        default_out = CHORDS_FIXTURES_DIR
        noun = "chord"
    else:
        all_items = _note_items(NOTE_PLAN)
        default_out = NOTES_FIXTURES_DIR
        noun = "note"
    out = args.out if args.out is not None else default_out

    plan = [item for item in all_items if args.force or not (out / item.filename).exists()]
    if not plan:
        print(f"All {len(all_items)} {noun}s already recorded in {out}. Use --force to redo them.")
        return 0

    # The ring buffer must hold at least one full clip — its default (~1.4s, sized for
    # live detection, not recording) is far shorter than a multi-second take, and
    # read_latest() raises rather than silently truncating.
    capture = AudioCapture(
        device=args.device,
        sample_rate=DEFAULT_SAMPLE_RATE,
        buffer_seconds=args.duration + 1.0,
    )
    try:
        capture.start()
    except Exception as exc:
        print(f"Could not open the input stream: {exc}")
        return 1

    print(f"Recording {len(plan)} {noun}(s) to {out}")
    print("Tune with a hardware tuner first — this only checks pitch, not intonation.\n")

    try:
        with capture:
            for i, item in enumerate(plan, 1):
                print(f"[{i}/{len(plan)}] {item.heading}")

                while True:
                    input("  Press Enter to start recording, or Ctrl-C to stop early...")
                    samples = record_clip(capture, args.duration, DEFAULT_SAMPLE_RATE)
                    print(f"  {item.check(samples, DEFAULT_SAMPLE_RATE)}")
                    choice = input("  Keep this take? [Y/n] ").strip().lower()
                    if choice in ("", "y", "yes"):
                        break
                    print("  Retrying...")

                save_wav(out / item.filename, samples, DEFAULT_SAMPLE_RATE)
                print(f"  Saved {item.filename}\n")
    except KeyboardInterrupt:
        print("\nStopped early — clips saved so far are kept.")

    if capture.overflow_count:
        print(f"({capture.overflow_count} input overflows during recording)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
