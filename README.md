# Guitar Trainer

A desktop guitar practice app for Linux. It listens through your microphone, works out what you're
playing in real time, and drills you on notes and chords.

## Features

- **Tuner** — per-string cents readout with an in-tune indicator.
- **Free detect** — plays back what note it hears, live, and lights up every matching position on
  the fretboard.
- **Fretboard map** — realistic fret spacing, with note names optionally shown (none / naturals
  only / all notes).
- **Note practice** — prompts a random note; you play it in any octave.
- **Chord practice** — prompts a random chord from a set you configure.
- **Timing** — free mode, or rhythm mode driven by an audible metronome at a set BPM.
- **Stats** — accuracy and response times per note and chord, stored locally in SQLite.

## Requirements

- Python 3.10+
- PortAudio (`sudo apt install libportaudio2`)
- A microphone

## Setup

```bash
git clone https://github.com/PhilipKolar/guitar-trainer.git
cd guitar-trainer
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Running

```bash
./run.sh
```

Pick your input device from the dropdown in the toolbar and check the level meter in the status bar
responds when you play. The meter's faint vertical marker is the noise gate — set it with the
**Gate** control in the toolbar so it sits above where the room idles and below where your playing
peaks. If quiet notes are missed, lower it; if room noise triggers detection, raise it.

## Development

```bash
.venv/bin/pytest                                     # full suite, no audio hardware needed
.venv/bin/python -m guitar_trainer.scripts.listen    # CLI pitch-detection smoke test
.venv/bin/python -m guitar_trainer.scripts.listen --list   # available input devices
```

All DSP tests run against synthesised signals, so the suite is deterministic and runs headless in
CI.

## How it works

Audio is captured on PortAudio's realtime thread, which does nothing but copy samples into a ring
buffer. A worker thread runs the analysis and emits results to the UI as queued Qt signals.

- **Pitch**: YIN, computed over a 4096-sample window at 48 kHz with an FFT-accelerated difference
  function and parabolic interpolation for sub-cent accuracy. A note must hold steady across several
  frames before it counts, which keeps pick attacks and fret noise from registering as notes.
- **Chords**: a chromagram is matched against templates synthesised from the harmonic series of each
  chord's tones, so recognition works across voicings rather than only the shapes it was tuned on.

## Desktop entry

To add it to the Mint application menu:

```bash
cp guitar-trainer.desktop ~/.local/share/applications/
```

Edit the `Exec` and `Path` lines first if the repository lives somewhere other than
`~/repos/guitar-trainer`.

## Licence

MIT
