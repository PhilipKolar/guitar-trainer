"""Records real single-note guitar audio into tests/fixtures/audio/notes/.

Synthesised test signals validate the algorithm; they can't catch anything specific
to a particular guitar, pickup, mic, or player — the exact things behind "detection
feels erratic" reports. This walks through recording one short clip per note so those
reports become a concrete, re-runnable pytest assertion instead
(tests/test_real_recordings.py) — run once, see exactly which note/take is wrong.

    python -m guitar_trainer.scripts.record_fixtures [--device N] [--duration 4]

Tune the guitar with a hardware tuner first — this script only records and checks
what it hears against what YIN detects, it doesn't verify intonation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ..audio.capture import DEFAULT_SAMPLE_RATE, AudioCapture, list_input_devices
from ..audio.pitch import DEFAULT_WINDOW, NoteGate, YinDetector
from ..core.notes import STANDARD, note_name

#: One full chromatic octave, played on a single string (frets 0-11 on the low E) so
#: there's no string-to-string jump to get wrong while recording.
NOTE_PLAN = [(fret, STANDARD.midi_at(0, fret)) for fret in range(12)]

FIXTURES_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "audio" / "notes"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=None, help="input device index")
    parser.add_argument("--list", action="store_true", help="list input devices and exit")
    parser.add_argument("--duration", type=float, default=4.0, help="seconds per clip (default 4)")
    parser.add_argument(
        "--force", action="store_true", help="re-record notes that already have a saved clip"
    )
    parser.add_argument(
        "--out", type=Path, default=FIXTURES_DIR, help=f"output directory (default {FIXTURES_DIR})"
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

    plan = [
        (fret, midi) for fret, midi in NOTE_PLAN
        if args.force or not (args.out / f"{note_name(midi)}.wav").exists()
    ]
    if not plan:
        print(f"All {len(NOTE_PLAN)} notes already recorded in {args.out}. Use --force to redo them.")
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

    print(f"Recording {len(plan)} note(s) to {args.out}")
    print("Tune with a hardware tuner first — this only checks pitch, not intonation.\n")

    try:
        with capture:
            for i, (fret, midi) in enumerate(plan, 1):
                name = note_name(midi)
                where = "open string" if fret == 0 else f"fret {fret}"
                print(f"[{i}/{len(plan)}] {name} — low E string, {where}")

                while True:
                    input("  Press Enter to start recording, or Ctrl-C to stop early...")
                    samples = record_clip(capture, args.duration, DEFAULT_SAMPLE_RATE)
                    print(f"  {sanity_check(samples, midi, DEFAULT_SAMPLE_RATE)}")
                    choice = input("  Keep this take? [Y/n] ").strip().lower()
                    if choice in ("", "y", "yes"):
                        break
                    print("  Retrying...")

                save_wav(args.out / f"{name}.wav", samples, DEFAULT_SAMPLE_RATE)
                print(f"  Saved {name}.wav\n")
    except KeyboardInterrupt:
        print("\nStopped early — clips saved so far are kept.")

    if capture.overflow_count:
        print(f"({capture.overflow_count} input overflows during recording)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
