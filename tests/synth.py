"""Synthetic test signals.

The DSP suite runs against generated audio rather than recordings so it stays
deterministic and needs no audio hardware in CI. The plucked-string model is the
important one: it reproduces the things that actually break naive pitch detectors on a
guitar — a fundamental weaker than its harmonics, and a noisy attack transient.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 48000


def sine(freq: float, duration: float = 0.2, sr: int = SAMPLE_RATE, amplitude: float = 0.3):
    t = np.arange(int(duration * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def sawtooth(freq: float, duration: float = 0.2, sr: int = SAMPLE_RATE, amplitude: float = 0.3):
    """Band-limited sawtooth — every harmonic present, fundamental strongest."""
    t = np.arange(int(duration * sr)) / sr
    out = np.zeros_like(t)
    for k in range(1, int(sr / 2 / freq) + 1):
        out += np.sin(2 * np.pi * freq * k * t) / k
    return (amplitude * out / np.max(np.abs(out))).astype(np.float32)


def plucked_string(
    freq: float,
    duration: float = 0.5,
    sr: int = SAMPLE_RATE,
    amplitude: float = 0.3,
    *,
    weak_fundamental: bool = False,
    attack_noise: float = 0.0,
    inharmonicity: float = 0.0,
    seed: int = 0,
):
    """Additive plucked-string model.

    Harmonics decay in amplitude and higher partials die away faster, as on a real
    string. ``weak_fundamental`` models the common case where the body and the pickup
    position leave the second harmonic louder than the first — the classic cause of
    octave errors. ``inharmonicity`` stretches the partials the way real stiff strings
    do, most noticeably on a wound low E.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    out = np.zeros_like(t)

    n_harmonics = max(1, min(20, int(sr / 2 / freq)))
    for k in range(1, n_harmonics + 1):
        partial_freq = freq * k * np.sqrt(1.0 + inharmonicity * k * k)
        if partial_freq >= sr / 2:
            break
        weight = 1.0 / k
        if weak_fundamental and k == 1:
            weight *= 0.25
        decay = np.exp(-t * (2.0 + 0.8 * k))
        phase = rng.uniform(0, 2 * np.pi)
        out += weight * decay * np.sin(2 * np.pi * partial_freq * t + phase)

    out /= np.max(np.abs(out))

    if attack_noise > 0:
        attack_len = int(0.01 * sr)
        noise = rng.normal(0, attack_noise, len(out))
        envelope = np.zeros_like(out)
        envelope[:attack_len] = np.linspace(1.0, 0.0, attack_len)
        out += noise * envelope

    return (amplitude * out).astype(np.float32)


def add_noise(signal: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """Mix in white noise at a given signal-to-noise ratio."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(signal.astype(np.float64) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(signal))
    return (signal + noise).astype(np.float32)


def strum(midi_notes, duration: float = 0.6, sr: int = SAMPLE_RATE, spread: float = 0.02, **kwargs):
    """A chord voicing, strummed with a small delay between strings."""
    from guitar_trainer.core.notes import midi_to_freq

    total = int(duration * sr)
    out = np.zeros(total, dtype=np.float64)
    for i, midi in enumerate(midi_notes):
        offset = int(i * spread * sr)
        note = plucked_string(
            midi_to_freq(midi), duration=duration, sr=sr, seed=midi, **kwargs
        )
        usable = min(len(note), total - offset)
        if usable > 0:
            out[offset : offset + usable] += note[:usable]
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.3
    return out.astype(np.float32)
