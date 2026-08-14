import numpy as np
import pytest

from guitar_trainer.audio.capture import AudioCapture, RingBuffer, list_input_devices


class TestRingBuffer:
    def test_write_then_read(self):
        buf = RingBuffer(16)
        buf.write(np.arange(8, dtype=np.float32))
        assert np.array_equal(buf.read_latest(8), np.arange(8))

    def test_reads_the_most_recent_samples(self):
        buf = RingBuffer(16)
        buf.write(np.arange(16, dtype=np.float32))
        assert np.array_equal(buf.read_latest(4), np.array([12, 13, 14, 15]))

    def test_wraps_around(self):
        buf = RingBuffer(8)
        buf.write(np.arange(6, dtype=np.float32))
        buf.write(np.arange(6, 12, dtype=np.float32))
        # Capacity 8, 12 written, so the last 8 are 4..11.
        assert np.array_equal(buf.read_latest(8), np.arange(4, 12))

    def test_read_spanning_the_wrap_point(self):
        buf = RingBuffer(8)
        buf.write(np.arange(10, dtype=np.float32))
        assert np.array_equal(buf.read_latest(5), np.arange(5, 10))

    def test_zero_pads_before_enough_audio_arrives(self):
        buf = RingBuffer(16)
        buf.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        out = buf.read_latest(8)
        assert len(out) == 8
        assert np.array_equal(out[:5], np.zeros(5))
        assert np.array_equal(out[5:], [1.0, 2.0, 3.0])

    def test_read_from_empty(self):
        buf = RingBuffer(16)
        assert np.array_equal(buf.read_latest(4), np.zeros(4))

    def test_oversized_write_keeps_the_newest(self):
        buf = RingBuffer(4)
        buf.write(np.arange(10, dtype=np.float32))
        assert np.array_equal(buf.read_latest(4), np.arange(6, 10))

    def test_empty_write_is_a_noop(self):
        buf = RingBuffer(8)
        buf.write(np.array([], dtype=np.float32))
        assert buf.total_written == 0

    def test_read_beyond_capacity_raises(self):
        buf = RingBuffer(8)
        with pytest.raises(ValueError):
            buf.read_latest(9)

    def test_total_written_tracks_every_sample(self):
        buf = RingBuffer(8)
        buf.write(np.zeros(5, dtype=np.float32))
        buf.write(np.zeros(7, dtype=np.float32))
        assert buf.total_written == 12

    def test_clear_resets(self):
        buf = RingBuffer(8)
        buf.write(np.arange(8, dtype=np.float32))
        buf.clear()
        assert buf.total_written == 0
        assert np.array_equal(buf.read_latest(8), np.zeros(8))

    def test_survives_concurrent_writer_and_reader(self):
        """The realtime thread writes while the analysis thread reads; neither may fail."""
        import threading

        buf = RingBuffer(4096)
        stop = threading.Event()
        errors = []

        def writer():
            block = np.ones(256, dtype=np.float32)
            try:
                while not stop.is_set():
                    buf.write(block)
            except Exception as exc:  # pragma: no cover - only on a real race
                errors.append(exc)

        def reader():
            try:
                while not stop.is_set():
                    out = buf.read_latest(1024)
                    assert len(out) == 1024
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        stop.wait(0.25)
        stop.set()
        for t in threads:
            t.join(timeout=2)

        assert not errors


class TestAudioCapture:
    def test_buffer_capacity_is_a_power_of_two_covering_the_window(self):
        cap = AudioCapture(sample_rate=48000, buffer_seconds=1.0)
        assert cap.buffer.capacity == 65536
        assert cap.buffer.capacity >= 48000

    def test_starts_not_running(self):
        assert not AudioCapture().is_running

    def test_callback_mixes_stereo_to_mono(self):
        cap = AudioCapture(sample_rate=48000)
        stereo = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        cap._callback(stereo, 2, None, None)
        assert np.array_equal(cap.buffer.read_latest(2), [2.0, 3.0])

    def test_callback_passes_mono_through(self):
        cap = AudioCapture(sample_rate=48000)
        mono = np.array([[0.5], [0.25]], dtype=np.float32)
        cap._callback(mono, 2, None, None)
        assert np.array_equal(cap.buffer.read_latest(2), [0.5, 0.25])

    def test_stop_is_safe_when_never_started(self):
        AudioCapture().stop()

    def test_overflow_counter_starts_at_zero(self):
        assert AudioCapture().overflow_count == 0


class TestDeviceEnumeration:
    def test_returns_a_list_without_raising(self):
        """Must degrade gracefully: CI has no audio devices at all."""
        devices = list_input_devices()
        assert isinstance(devices, list)
        for dev in devices:
            assert dev.channels > 0
            assert isinstance(dev.name, str)
            assert str(dev)
