"""Terminal pitch-detection smoke test.

Run this before touching the GUI when something seems wrong with detection — it
exercises the exact capture and analysis path the app uses, with nothing else in the
way.

    python -m guitar_trainer.scripts.listen [--device N] [--list]
"""

from __future__ import annotations

import argparse
import time

from ..audio.capture import DEFAULT_SAMPLE_RATE, AudioCapture, list_input_devices
from ..audio.pitch import DEFAULT_WINDOW, NoteGate, YinDetector
from ..core.notes import STANDARD


def main() -> int:
    parser = argparse.ArgumentParser(description="Print detected guitar notes from the mic.")
    parser.add_argument("--device", type=int, default=None, help="input device index")
    parser.add_argument("--list", action="store_true", help="list input devices and exit")
    parser.add_argument("--gate", type=float, default=0.01, help="RMS noise gate (default 0.01)")
    parser.add_argument("--raw", action="store_true", help="show every frame, ungated")
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

    detector = YinDetector(DEFAULT_SAMPLE_RATE)
    gate = NoteGate(rms_threshold=args.gate)
    capture = AudioCapture(device=args.device, sample_rate=DEFAULT_SAMPLE_RATE)

    try:
        capture.start()
    except Exception as exc:
        print(f"Could not open the input stream: {exc}")
        return 1

    print("Listening — play something. Ctrl-C to stop.\n")
    last_reported = None
    try:
        with capture:
            while True:
                frame = capture.buffer.read_latest(DEFAULT_WINDOW)
                result = detector.detect(frame)

                if args.raw:
                    if result is not None:
                        print(
                            f"  {result.name:<4} {result.freq:7.2f} Hz  "
                            f"{result.cents:+6.1f}¢  conf {result.confidence:.2f}  "
                            f"rms {result.rms:.4f}"
                        )
                else:
                    stable = gate.push(result)
                    if stable is not None and stable.midi != last_reported:
                        string, _ = STANDARD.nearest_string(stable.freq)
                        positions = STANDARD.positions_for_midi(stable.midi)
                        where = ", ".join(f"s{s + 1}f{f}" for s, f in positions[:4]) or "—"
                        print(
                            f"  {stable.name:<4} {stable.freq:7.2f} Hz  "
                            f"{stable.cents:+6.1f}¢  [{where}]"
                        )
                        last_reported = stable.midi
                    elif stable is None:
                        last_reported = None

                time.sleep(DEFAULT_WINDOW / 4 / DEFAULT_SAMPLE_RATE)
    except KeyboardInterrupt:
        print("\nStopped.")

    if capture.overflow_count:
        print(f"({capture.overflow_count} input overflows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
