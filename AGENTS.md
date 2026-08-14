# AGENTS.md

Context for AI coding assistants working on this repository. Tool-neutral by
convention — `CLAUDE.md` is a pointer to this file, and any other assistant's config
should point here too rather than duplicating content.

**Keep this file current.** See [Maintaining this file](#maintaining-this-file) at the
bottom; it is part of the definition of done for any non-trivial change.

## What this is

A desktop guitar practice app for Linux (Linux Mint, PipeWire). It listens through the
microphone, works out what is being played in real time, and drills the user on notes
and chords.

Single-user personal project. Author: Philip Kolar (@PhilipKolar).
Repo: https://github.com/PhilipKolar/guitar-trainer (public, MIT).

The five modes:

| Mode | What it does |
|---|---|
| Tuner | Nearest open string plus a cents needle. |
| Free detect | Live note *or* chord readout — a clean single note wins if one is ringing, otherwise a sustained chord match is shown. Lights up matching fretboard positions either way. |
| Note practice | Prompts a random note; **any octave counts**. |
| Chord practice | Prompts a random chord from a user-selected set; any voicing counts. |
| Stats | Accuracy and median response time per note/chord, weakest first. |

Both practice modes offer **free** timing (take as long as you like) or **rhythm**
timing (a metronome at a set BPM, N beats per challenge).

## Stack

Python 3.12 · PySide6 (Qt6) · numpy/scipy · sounddevice (PortAudio) · SQLite · pytest.

No build step. The venv lives at `.venv/`; `./run.sh` creates it on first use.

```bash
./run.sh                                                  # launch the app
.venv/bin/pytest                                          # full suite (~7s, no hardware needed)
.venv/bin/python -m guitar_trainer.scripts.listen         # CLI pitch-detection smoke test
.venv/bin/python -m guitar_trainer.scripts.listen --list  # list input devices
```

CI (`.github/workflows/ci.yml`) runs `pytest` headless on push and PR.

## Architecture

Audio never touches the GUI thread:

```
PortAudio RT callback ──write──> ring buffer ──read──> AnalysisWorker (QThread)
   (copies samples only)                                      │ queued Qt signals
                                                              ▼
                                        UI panels ──> SessionEngine ──> SQLite
```

The realtime callback does nothing but copy samples into a preallocated ring buffer.
All DSP runs on a worker thread and reaches widgets as queued signals. **Do not do
work in the audio callback and do not touch Qt widgets off the GUI thread.**

### Layout

```
src/guitar_trainer/
  __main__.py           entry point; sets the app font BEFORE the stylesheet (see gotchas)
  audio/
    capture.py          InputStream + RingBuffer + device enumeration
    pitch.py            YIN detector + NoteGate (frame-to-note-event stability)
    chroma.py           chromagram + bass-note detection
    metronome.py        sample-accurate click track + BeatCounter
  core/
    notes.py            freq/MIDI/name, cents, Tuning, fretboard geometry
    chords.py           Chord model, harmonic templates, matcher, ChordGate
    session.py          Challenge protocol, ChallengePicker, SessionEngine
    config.py           TOML settings (XDG config dir)
    stats.py            SQLite attempt log + weakness weights
  ui/
    main_window.py      toolbar, shared fretboard, tabs, signal routing
    fretboard.py        QPainter fretboard widget
    modes.py            the four mode panels (PracticePanel is shared by both drills)
    widgets.py          LevelMeter, CentsMeter, BigNoteLabel
    stats_panel.py      history table
    worker.py           AnalysisWorker/AnalysisThread
    theme.py            colours + stylesheet
  scripts/listen.py     terminal pitch-detection harness
tests/
  synth.py              signal synthesis helpers (NOT a test file)
  test_*.py
```

`core/` is pure logic with no Qt and no audio imports — keep it that way, it is what
makes the suite fast and headless.

`core/notes.py` is the single source of truth for musical maths. Tuning changes and
fret counts propagate from there; nothing else should hardcode string pitches or fret
spacing.

## Design decisions worth knowing

These were reached deliberately. Changing them is fine, but know what you are undoing.

**Notes are matched by pitch class, not exact pitch.** The user asked to answer a
prompt anywhere on the neck. `NoteChallenge.matches` compares `pitch_class`.

**Wrong answers are ignored, not failed.** Hunting for a note on the neck necessarily
sounds others on the way; failing on those makes the mode unusable. Only a timeout or
an explicit skip ends a round unsuccessfully.

**YIN, not autocorrelation peak-picking.** A plucked low E often has a stronger 2nd
harmonic than fundamental, which fools a naive peak-picker into reporting the octave
above. YIN's cumulative mean normalised difference handles this; picking the *first*
dip below threshold rather than the global minimum is what avoids octave-down errors.
A spectral check (`_has_energy_at`) additionally rejects subharmonics — a signal is
periodic at every subharmonic of its true pitch, so YIN can settle on a period 2× or 3×
too long, and the tell is that there is no spectral peak there.

**Detections are gated for stability.** `NoteGate` requires several consecutive frames
within ±40 cents before reporting a note. Without it, pick attacks and fret noise score
as answers and the practice modes feel twitchy and wrong.

**Chord templates model the harmonic series, and scoring penalises unexplained
energy.** This is the subtlest part of the codebase. Binary "three pitch classes are
on" templates do not work, because a real strummed chord puts energy on pitch classes
outside the chord (a string's 3rd harmonic is a fifth up, its 5th a major third).
Worse, plain cosine similarity is biased towards *sparse* templates: a power chord
(root + fifth) scored higher than the minor triad it is a subset of, because nothing
charged it for the third it failed to predict. So `ChordMatcher.score_all` is:

```
score = cosine(template, chroma) - UNEXPLAINED_PENALTY * ‖chroma restricted to
                                    pitch classes the template does not predict‖
```

plus `ROOT_WEIGHT` on the root (guitar voicings double and triple it — an open Em
sounds three E strings against one G) and an optional bass-note bonus.

The constants in `core/chords.py` were tuned by sweep against 17 synthesised real
voicings; **`ROOT_WEIGHT` has a narrow working window of roughly 1.3–1.4** and
`UNEXPLAINED_PENALTY` works over roughly 0.4–0.7. If you change either, re-run
`tests/test_chords.py::TestChordRecognition` — it is the regression net for exactly
this. Absolute scores are on a shifted scale because of the penalty term; do not
compare them against pre-penalty thresholds.

**Chord practice restricts the matcher to the selected chords.** Scoring against only
what is being drilled removes most near-misses — nothing can be mistaken for a chord
that is not in the exercise. Full 120-chord matching is inherently harder (Cmaj7
contains Em; Am7 contains C) and is only used where the candidate set is unknown.

**The metronome schedules clicks by sample position, not by timer.** A QTimer at 120
BPM drifts audibly within a minute or two, which is precisely wrong for a mode whose
purpose is keeping time. The output callback writes clicks at exact sample offsets and
the UI polls a beat counter.

**Detection is suppressed around each click** (`modes.py`, `_on_tick`, currently 50 ms)
so metronome bleed through speakers is never scored as a played note. Click pitches are
high and narrowband to sit far from guitar fundamentals.

**Live meters (Tuner, Free Detect) are smoothed; NoteGate's stability check is a
different thing and doesn't substitute for it.** `PitchSmoother` (`audio/pitch.py`)
median-filters the last 3 raw frames then applies an EMA, both on the MIDI
(log-frequency) axis since cents are linear there, not in Hz. Without this the needle
visibly jitters on a perfectly clean, steady note — that's real frame-to-frame YIN
noise plus the string's own micro-vibrato and decay, not a bug in the detector. NoteGate
doesn't help here: it decides whether a note *qualifies* as a scored answer and only
reports once, whereas the meters need something that steadies a continuous stream. A
jump past `reset_threshold_cents` (50c) snaps immediately instead of easing through —
needed so switching strings feels instant rather than gliding. `PracticePanel`'s
scoring path is deliberately **not** run through this smoother; scoring wants the raw,
fast NoteGate-filtered result, not something with even a couple of frames of EMA lag.

**A harmonic-energy-ratio filter rejects noise that is loud and periodic enough to
fool YIN.** `YinDetector._harmonic_energy_ratio` measures what fraction of a frame's
spectral energy sits at multiples of the detected fundamental. A plucked string
concentrates nearly all of it there (ratio > 0.98 across the neck, even at 10dB SNR); a
keyboard click spreads it across the spectrum instead (ratio < 0.52 for every
synthesised click that fools YIN's periodicity check at all — see
`tests/test_pitch.py::TestPurityFilter`). `DEFAULT_MIN_HARMONIC_RATIO = 0.6` sits well
clear of both populations. **This does not, and cannot, reliably reject speech or
singing** — a sustained vowel is a genuine harmonic source, acoustically much closer to
a plucked string than to a keyboard click, so it scores similarly high. Rejecting voice
would need timbral classification (formants, MFCCs), not a purity ratio; the honest
mitigation is the RMS gate plus mic placement, not this filter. Don't oversell it in UI
copy or docs — say what it actually does.

## Tuned constants

Changing these has non-obvious consequences; the referenced tests are the safety net.

| Constant | Value | Where | Guarded by |
|---|---|---|---|
| `DEFAULT_WINDOW` / `DEFAULT_HOP` | 4096 / 1024 @ 48 kHz | `audio/pitch.py` | `test_pitch.py` |
| `MIN_FREQ` / `MAX_FREQ` | 70 / 1400 Hz | `audio/pitch.py` | `test_pitch.py::TestRange` |
| `DEFAULT_THRESHOLD` | 0.12 | `audio/pitch.py` | `test_pitch.py` |
| `CHROMA_WINDOW` / `FFT_SIZE` | 8192 / 16384 | `audio/chroma.py` | `test_chords.py` |
| `ROOT_WEIGHT` | 1.35 (works 1.3–1.4) | `core/chords.py` | `test_chords.py` |
| `UNEXPLAINED_PENALTY` | 0.55 (works 0.4–0.7) | `core/chords.py` | `test_chords.py` |
| `HARMONIC_DECAY` / `N_HARMONICS` | 0.6 / 6 | `core/chords.py` | `test_chords.py` |
| `DEFAULT_MIN_HARMONIC_RATIO` | 0.6 (guitar floor ~0.98, click ceiling ~0.51) | `audio/pitch.py` | `test_pitch.py::TestPurityFilter` |
| `SMOOTHER_ALPHA` / `SMOOTHER_RESET_CENTS` / `SMOOTHER_MEDIAN_WINDOW` | 0.2 / 50 / 3 | `audio/pitch.py` | `test_pitch.py::TestPitchSmoother` |

## Testing

644 tests, ~7 seconds, no audio hardware and no display required.

**All DSP tests run against synthesised signals** (`tests/synth.py`), which is what
keeps them deterministic and CI-able. The plucked-string model is the important one: it
reproduces what actually breaks naive detectors — decaying harmonics, weak
fundamentals, inharmonic partials, attack noise.

- Prefer adding cases to `tests/synth.py`-driven tests over recording audio fixtures.
- UI tests run under `QT_QPA_PLATFORM=offscreen` (set automatically in
  `tests/conftest.py`). `widget.grab()` forces a real `paintEvent` and is the cheapest
  way to catch painting crashes.
- When a test fails, check whether the *test's expectation* is wrong before changing
  code. Several early failures here were bad test arithmetic (guitar fret positions,
  beat counts) rather than bugs, and one was a threshold comparing two different click
  pitches. Verify the musical or DSP claim first.

## Gotchas

**Qt stylesheets override `setFont`.** A `font-size` rule in the app stylesheet
silently flattens every widget that sets its own font, including the large note and
prompt readouts. The base size is therefore applied as the *application font* in
`__main__.py` before the stylesheet, and `theme.STYLESHEET` must not contain
`font-size`.

**Fret positions are a fraction of scale length, not of the widget.** A 22-fret neck
spans only ~72% of the scale length, so `fretboard.py` rescales so the last fret lands
near the right edge (`TAIL_FRACTION`). Relative spacing is preserved — that is what
makes the neck recognisable.

**String index 0 is the lowest string** (thickest, the 6th on a guitar) and is drawn at
the *bottom*. Tuning tuples are ordered low to high.

**Devices are remembered by name, not index.** PortAudio indices shift as devices come
and go; `config.device_name` is matched against the combo text.

**Ambient noise on this machine is significant** — measured idle RMS ~0.05 against a
default gate of 0.01. There's a **Calibrate** button next to the Gate control
(`main_window.py`, `_start_gate_calibration`) that samples ambient level for ~1.2s and
sets the gate above it; point users at it rather than having them guess a number.

**The Gate control used to be wired to only half the pipeline — this was a real bug,
not just an unconfigured default.** `frame_analysed` (which drives `Tuner.on_pitch` and
`FreeDetectPanel`'s live cents/detail readout) was emitted straight from
`YinDetector.detect()`, bypassing `NoteGate.rms_threshold` entirely; only the
practice-modes' "stable note" path checked it. Raising the Gate control did nothing for
what the Tuner tab actually shows. Fixed in `AnalysisWorker._process` by gating
`result` itself on `rms >= self.gate.rms_threshold` before running detection at all.
`ChromaAnalyser` also had its own fixed, lower threshold (0.005) independent of the
user's setting; `set_noise_gate` now updates both. If you add a new consumer of
per-frame analysis, route it through this same gate rather than reading raw detector
output — that's exactly how this bug happened. Guarded by `tests/test_worker.py`.

**A defined-but-never-connected signal handler is a silent bug, not a no-op.**
`ChordPracticePanel` had a `set_bass()` method that was never wired to anything in
`main_window.py` — bass-note disambiguation (C vs Am, Em vs G) only ever worked in
tests that called it directly, never in the running app. Renamed to the `on_bass()`
hook every panel now gets (see `ModePanel.on_bass`, fed by `AnalysisWorker.bass_changed`,
emitted immediately before `chroma_updated` for the same frame so a panel's stored bass
value is always fresh when it matches). When adding a new panel hook like this, grep for
where it's actually *connected*, not just defined — a plausible-looking method with no
caller passes review easily.

## Data locations

- Settings: `~/.config/guitar-trainer/config.toml` (written atomically; unknown keys
  ignored so a newer version's config cannot stop an older one starting)
- History: `~/.local/share/guitar-trainer/stats.db`

## Conventions

- Comments explain *why*, not what. The DSP here has non-obvious reasoning behind it;
  that reasoning belongs next to the code.
- Match the surrounding style — dataclasses for value types, `Protocol` for
  duck-typed interfaces, keyword-only args for tuning parameters.
- Commit messages: summary line, then prose explaining the reasoning behind
  non-obvious choices. See `git log` for the established tone.
- Do not commit or push unless asked.

## Status and possible next steps

Everything in the original plan is built, tested and pushed; CI is green. Since then,
in response to real usage on the author's machine (very sensitive to speech/typing):

- Added a harmonic-purity filter that rejects percussive/noise false positives
  (keyboard clicks) — see the design decision above. **Does not filter speech.**
- Fixed the Gate control not actually reaching the Tuner/Free Detect live readout
  (`frame_analysed` bypassed it) — see the Gotchas entry. This was likely the bigger
  contributor to "detection reacts to talking/typing" than the missing purity filter.
- Added a **Calibrate** button that samples ambient level and sets the gate above it.
- Fixed `ChordPracticePanel`'s bass-note disambiguation, which was dead code — never
  wired to the live signal path, only exercised by tests calling it directly.
- Free detect now also recognises chords (full 120-chord candidate set, since there's
  no drilled subset to narrow it to — an honest best-effort display, not scored).
- Added `PitchSmoother` — the Tuner and Free Detect live meters were showing raw,
  unsmoothed per-frame YIN output, which jitters noticeably even on a clean, steady
  note. Median-of-3 plus EMA now smooths the display; practice-mode scoring is
  untouched (still reads the raw NoteGate-filtered result, not this).

Still not verified by ear against a real guitar beyond the fixes above. Acceptance
checks that still want a human with an instrument:

1. **Talking/typing near the mic** — with the gate fix and Calibrate button, does
   raising the gate above ambient actually stop the Tuner/Free Detect from reacting?
   Loud sustained speech that clears the gate will still register — that's expected,
   not a bug (see the purity-filter design decision) — but it should take real volume
   to do it, not just a normal-volume comment.
2. Tuner: each open string reads correctly, needle centres when in tune.
3. Free detect: chromatic scale up the low E, then above the 12th fret (octave errors);
   strum a chord and confirm it's recognised once the note readout goes quiet.
4. Note practice: ~20 prompts register promptly; wrong notes are not accepted.
5. **Rhythm mode at 60 BPM through speakers** — confirm the click is never scored as a
   played note. This is the least-trusted part; if it leaks, widen the suppression
   window in `modes.py:_on_tick`.
6. Chord practice: open vs barre voicings; Am vs C and Em vs G not confused (this now
   has a real bass-note signal behind it, worth specifically re-checking).

Ideas, none committed to: per-string practice restriction, scale/arpeggio drills,
detecting *which* string was played (needs more than a mono signal), PyInstaller
packaging, a "pause detection" toggle/hotkey for mid-session conversation (the purity
filter can't help with that — it's a genuine gap, not a bug).

## Maintaining this file

Update this file as part of the same change, not as a follow-up, whenever you:

- add, remove or rename a module, or move responsibility between them → update
  **Layout**
- make a decision a future reader would otherwise undo by accident → add it to
  **Design decisions worth knowing**, with the reasoning
- retune a constant that has a narrow working range → update **Tuned constants**
- hit a non-obvious trap that cost you time → add it to **Gotchas** so nobody pays
  twice
- finish or start significant work → update **Status and possible next steps**

Keep it honest: it should describe what the code does now, not what was intended. If
something here contradicts the code, the code wins — fix this file. Prune anything that
has stopped being true rather than letting it accumulate; a stale entry is worse than a
missing one. Do not turn it into API documentation — that belongs in docstrings. This
file is for context that is not recoverable by reading the source.
