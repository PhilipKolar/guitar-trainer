"""Microphone capture.

PortAudio's callback runs on a realtime thread: anything slow or allocating in there
causes dropouts. So the callback does one thing only — copy samples into a preallocated
ring buffer — and all analysis happens on a worker thread that reads from it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_BLOCKSIZE = 1024


@dataclass(frozen=True)
class AudioDevice:
    """An input device as offered in the UI."""

    index: int
    name: str
    channels: int
    default_sample_rate: float
    is_default: bool = False

    def __str__(self) -> str:
        return f"{self.name} ({self.channels}ch)"


def list_input_devices() -> list[AudioDevice]:
    """Enumerate available input devices. Returns ``[]`` if PortAudio is unavailable."""
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        try:
            default_index = sd.default.device[0]
        except (TypeError, IndexError):
            default_index = None
    except Exception:
        return []

    return [
        AudioDevice(
            index=i,
            name=dev["name"],
            channels=dev["max_input_channels"],
            default_sample_rate=dev["default_samplerate"],
            is_default=(i == default_index),
        )
        for i, dev in enumerate(devices)
        if dev["max_input_channels"] > 0
    ]


class RingBuffer:
    """Single-producer/single-consumer float ring buffer.

    The lock is held only for the duration of a memcpy, never across analysis, so the
    realtime thread is never blocked behind DSP work.
    """

    def __init__(self, capacity: int) -> None:
        self._buffer = np.zeros(capacity, dtype=np.float32)
        self._capacity = capacity
        self._write_pos = 0
        self._total_written = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def total_written(self) -> int:
        """Running count of samples ever written; used to detect fresh audio."""
        with self._lock:
            return self._total_written

    def write(self, data: np.ndarray) -> None:
        """Append samples, overwriting the oldest if the buffer wraps."""
        n = len(data)
        if n == 0:
            return
        if n >= self._capacity:
            data = data[-self._capacity :]
            n = self._capacity

        with self._lock:
            end = self._write_pos + n
            if end <= self._capacity:
                self._buffer[self._write_pos : end] = data
            else:
                split = self._capacity - self._write_pos
                self._buffer[self._write_pos :] = data[:split]
                self._buffer[: n - split] = data[split:]
            self._write_pos = end % self._capacity
            self._total_written += n

    def read_latest(self, n: int) -> np.ndarray:
        """The most recent ``n`` samples, oldest first. Zero-padded early on."""
        if n > self._capacity:
            raise ValueError(f"requested {n} samples from a {self._capacity}-sample buffer")

        with self._lock:
            if self._total_written < n:
                out = np.zeros(n, dtype=np.float32)
                available = self._total_written
                if available:
                    out[-available:] = np.concatenate(
                        (self._buffer[self._write_pos :], self._buffer[: self._write_pos])
                    )[-available:]
                return out

            start = (self._write_pos - n) % self._capacity
            if start + n <= self._capacity:
                return self._buffer[start : start + n].copy()
            split = self._capacity - start
            return np.concatenate((self._buffer[start:], self._buffer[: n - split]))

    def clear(self) -> None:
        with self._lock:
            self._buffer.fill(0.0)
            self._write_pos = 0
            self._total_written = 0


class AudioCapture:
    """Owns the input stream and the ring buffer the analysis thread reads from."""

    def __init__(
        self,
        *,
        device: int | None = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        blocksize: int = DEFAULT_BLOCKSIZE,
        buffer_seconds: float = 1.0,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        capacity = 1 << int(np.ceil(np.log2(sample_rate * buffer_seconds)))
        self.buffer = RingBuffer(capacity)
        self._stream = None
        self._error: str | None = None
        self._overflow_count = 0

    @property
    def is_running(self) -> bool:
        return self._stream is not None

    @property
    def error(self) -> str | None:
        """The last stream error, for surfacing in the UI."""
        return self._error

    @property
    def overflow_count(self) -> int:
        """Input overflows seen; a rising count means the analysis loop is too slow."""
        return self._overflow_count

    def _callback(self, indata, frames, time_info, status) -> None:
        if status and status.input_overflow:
            self._overflow_count += 1
        # Mix to mono. Most USB mics (the Blue Yeti included) present as stereo even
        # for a single capsule, so averaging is both correct and cheap.
        mono = indata[:, 0] if indata.shape[1] == 1 else indata.mean(axis=1)
        self.buffer.write(mono)

    def start(self) -> None:
        """Open and start the stream. Raises on failure."""
        if self._stream is not None:
            return
        import sounddevice as sd

        self.buffer.clear()
        self._overflow_count = 0
        try:
            self._stream = sd.InputStream(
                device=self.device,
                channels=self._input_channels(),
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

    def _input_channels(self) -> int:
        """Open the fewest channels the device allows, capped at stereo."""
        import sounddevice as sd

        try:
            info = sd.query_devices(self.device, "input")
            return min(2, max(1, int(info["max_input_channels"])))
        except Exception:
            return 1

    def stop(self) -> None:
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None

    def __enter__(self) -> AudioCapture:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()
