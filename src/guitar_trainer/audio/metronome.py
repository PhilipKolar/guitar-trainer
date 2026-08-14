"""Metronome.

Clicks are placed at sample-accurate positions inside the output callback rather than
fired from a timer. A QTimer at 120 BPM drifts audibly within a minute or two, which is
exactly the wrong thing when the point of the mode is to practise keeping time.
"""

from __future__ import annotations

import threading

import numpy as np

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BPM = 80
MIN_BPM = 30
MAX_BPM = 240

CLICK_MS = 25
#: Downbeat and off-beat click pitches, chosen high and narrowband so that whatever
#: leaks into the microphone is easy to gate out and lands far from guitar fundamentals.
DOWNBEAT_HZ = 1600.0
OFFBEAT_HZ = 1100.0


def render_click(
    freq: float, sample_rate: int = DEFAULT_SAMPLE_RATE, duration_ms: int = CLICK_MS
) -> np.ndarray:
    """A short decaying sine, windowed so it starts and ends without a pop."""
    n = int(sample_rate * duration_ms / 1000)
    t = np.arange(n) / sample_rate
    envelope = np.exp(-t * 90.0)
    # Ramp the first millisecond in to avoid a click-on-the-click.
    ramp_len = max(1, int(sample_rate * 0.001))
    envelope[:ramp_len] *= np.linspace(0.0, 1.0, ramp_len)
    return (np.sin(2 * np.pi * freq * t) * envelope * 0.4).astype(np.float32)


class Metronome:
    """Sample-accurate click track.

    The audio callback advances a sample counter and writes clicks wherever a beat
    falls inside the current block, so beats stay locked to the audio clock no matter
    what the GUI thread is doing.
    """

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        bpm: int = DEFAULT_BPM,
        beats_per_bar: int = 4,
        blocksize: int = 512,
        device: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.beats_per_bar = beats_per_bar
        self.blocksize = blocksize
        self.device = device
        self.muted = False

        self._downbeat = render_click(DOWNBEAT_HZ, sample_rate)
        self._offbeat = render_click(OFFBEAT_HZ, sample_rate)

        self._lock = threading.Lock()
        self._bpm = int(np.clip(bpm, MIN_BPM, MAX_BPM))
        self._samples_per_beat = self._compute_period()
        self._sample_pos = 0
        self._beat_index = 0
        self._next_beat_at = 0
        self._stream = None
        self._error: str | None = None

        # A click (25ms, 1200 samples) is longer than a typical output block (512
        # samples, ~11ms) — most clicks span two or three callbacks. This buffer
        # carries whatever didn't fit in the current block forward to the next one(s),
        # indexed relative to the sample right after the current block. Sized to the
        # longest click, which bounds how far a click can ever need to carry over.
        self._tail_len = max(len(self._downbeat), len(self._offbeat))
        self._tail = np.zeros(self._tail_len, dtype=np.float32)

    # ---------------------------------------------------------------- config

    def _compute_period(self) -> int:
        return max(1, int(round(self.sample_rate * 60.0 / self._bpm)))

    @property
    def bpm(self) -> int:
        return self._bpm

    @bpm.setter
    def bpm(self, value: int) -> None:
        with self._lock:
            self._bpm = int(np.clip(value, MIN_BPM, MAX_BPM))
            self._samples_per_beat = self._compute_period()
            # Re-anchor the next beat so a tempo change takes effect immediately
            # instead of after the old, longer interval has elapsed.
            self._next_beat_at = self._sample_pos + self._samples_per_beat

    @property
    def seconds_per_beat(self) -> float:
        return 60.0 / self._bpm

    @property
    def beat_index(self) -> int:
        """Beats elapsed since start. Polled by the UI for the visual pulse."""
        with self._lock:
            return self._beat_index

    @property
    def beat_in_bar(self) -> int:
        return self.beat_index % self.beats_per_bar

    @property
    def is_running(self) -> bool:
        return self._stream is not None

    @property
    def error(self) -> str | None:
        return self._error

    # ------------------------------------------------------------------ loop

    def _callback(self, outdata, frames, time_info, status) -> None:
        outdata.fill(0.0)
        with self._lock:
            block_start = self._sample_pos
            block_end = block_start + frames
            samples_per_beat = self._samples_per_beat

            # Start this block with whatever carried over from a click that ran past
            # the end of the previous one.
            carry = min(frames, self._tail_len)
            if carry:
                outdata[:carry, 0] += self._tail[:carry]
            # Shift the tail buffer left by a block's worth of samples: what was at
            # position `frames` is now relative to the start of the *next* block.
            if frames < self._tail_len:
                self._tail[: self._tail_len - frames] = self._tail[frames:]
                self._tail[self._tail_len - frames :] = 0.0
            else:
                self._tail[:] = 0.0

            while self._next_beat_at < block_end:
                offset = self._next_beat_at - block_start
                if offset >= 0 and not self.muted:
                    is_downbeat = self._beat_index % self.beats_per_bar == 0
                    click = self._downbeat if is_downbeat else self._offbeat

                    in_block = min(len(click), frames - offset)
                    if in_block > 0:
                        outdata[offset : offset + in_block, 0] += click[:in_block]
                    # Whatever didn't fit carries into the tail buffer instead of
                    # being dropped, so the click finishes its decay in a later block
                    # rather than cutting off mid-waveform.
                    remaining = len(click) - in_block
                    if remaining > 0:
                        self._tail[:remaining] += click[in_block:]

                self._beat_index += 1
                self._next_beat_at += samples_per_beat

            self._sample_pos = block_end

        if outdata.shape[1] > 1:
            outdata[:, 1:] = outdata[:, :1]

    def start(self) -> None:
        """Open the output stream and begin clicking. Raises on failure."""
        if self._stream is not None:
            return
        import sounddevice as sd

        with self._lock:
            self._sample_pos = 0
            self._beat_index = 0
            self._next_beat_at = 0
            self._tail[:] = 0.0

        try:
            self._stream = sd.OutputStream(
                device=self.device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.blocksize,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            self._error = None
        except Exception as exc:
            self._stream = None
            self._error = str(exc)
            raise

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def __enter__(self) -> Metronome:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()


class BeatCounter:
    """Turns a rising beat index into "a new challenge is due" events.

    Used by rhythm mode: the metronome owns the tempo, and this decides how many beats
    each challenge gets.
    """

    def __init__(self, beats_per_challenge: int = 4) -> None:
        self.beats_per_challenge = max(1, beats_per_challenge)
        self._last_seen = 0
        self._countdown = self.beats_per_challenge

    def reset(self, beat_index: int = 0) -> None:
        self._last_seen = beat_index
        self._countdown = self.beats_per_challenge

    def update(self, beat_index: int) -> tuple[int, bool]:
        """Feed the current beat index.

        Returns ``(beats_elapsed, challenge_due)``. Handles the index jumping by more
        than one, which happens if the GUI thread stalls.
        """
        elapsed = max(0, beat_index - self._last_seen)
        self._last_seen = beat_index
        if elapsed == 0:
            return 0, False

        self._countdown -= elapsed
        if self._countdown <= 0:
            # Carry the remainder so challenges stay aligned to the bar.
            self._countdown += self.beats_per_challenge * (
                1 + (-self._countdown) // self.beats_per_challenge
            )
            return elapsed, True
        return elapsed, False
