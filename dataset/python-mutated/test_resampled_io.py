import time
import pytest
from pedalboard import Resample
from pedalboard.io import AudioFile, StreamResampler, ResampledReadableAudioFile
from io import BytesIO
import numpy as np
from .utils import generate_sine_at

def expected_output(input_signal, sample_rate: float, target_sample_rate: float, num_channels: int, quality) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    if sample_rate == target_sample_rate:
        if len(input_signal.shape) == 1:
            return np.expand_dims(input_signal, 0)
        else:
            return input_signal
    resampler = StreamResampler(sample_rate, target_sample_rate, num_channels, quality)
    output = np.concatenate([resampler.process(input_signal), resampler.process(None)], axis=1)
    return output
QUALITIES = [v[0] for v in Resample.Quality.__entries.values()]

def test_read_resampled_constructor():
    if False:
        print('Hello World!')
    sine_wave = generate_sine_at(44100, 440, num_seconds=1, num_channels=1).astype(np.float32)
    read_buffer = BytesIO()
    read_buffer.name = 'test.wav'
    with AudioFile(read_buffer, 'w', 44100, 1, bit_depth=32) as f:
        f.write(sine_wave)
    with AudioFile(BytesIO(read_buffer.getvalue())) as f:
        with f.resampled_to(22050) as r:
            assert isinstance(r, ResampledReadableAudioFile)
        assert r.closed
        assert not f.closed
    assert f.closed

def test_read_resampled_constructor_does_nothing():
    if False:
        for i in range(10):
            print('nop')
    sine_wave = generate_sine_at(44100, 440, num_seconds=1, num_channels=1).astype(np.float32)
    read_buffer = BytesIO()
    read_buffer.name = 'test.wav'
    with AudioFile(read_buffer, 'w', 44100, 1, bit_depth=32) as f:
        f.write(sine_wave)
    with AudioFile(BytesIO(read_buffer.getvalue())) as f:
        with f.resampled_to(44100) as r:
            assert r is f

def test_read_zero():
    if False:
        print('Hello World!')
    sine_wave = generate_sine_at(44100, 440, num_seconds=1, num_channels=1).astype(np.float32)
    read_buffer = BytesIO()
    read_buffer.name = 'test.wav'
    with AudioFile(read_buffer, 'w', 44100, 1, bit_depth=32) as f:
        f.write(sine_wave)
    with AudioFile(BytesIO(read_buffer.getvalue())).resampled_to(22050) as f:
        with pytest.raises(ValueError):
            f.read()

@pytest.mark.parametrize('fundamental_hz', [440])
@pytest.mark.parametrize('sample_rate', [8000, 11025, 22050, 44100, 48000])
@pytest.mark.parametrize('target_sample_rate', [123.45, 8000, 11025, 12345.67, 22050, 44100, 48000])
@pytest.mark.parametrize('num_channels', [1, 2])
@pytest.mark.parametrize('quality', QUALITIES)
def test_read_resampled(fundamental_hz: float, sample_rate: float, target_sample_rate: float, num_channels: int, quality):
    if False:
        return 10
    sine_wave = generate_sine_at(sample_rate, fundamental_hz, num_seconds=1, num_channels=num_channels).astype(np.float32)
    expected_sine_wave = expected_output(sine_wave, sample_rate, target_sample_rate, num_channels, quality)
    read_buffer = BytesIO()
    read_buffer.name = 'test.wav'
    with AudioFile(read_buffer, 'w', sample_rate, num_channels, bit_depth=32) as f:
        f.write(sine_wave)
    with AudioFile(BytesIO(read_buffer.getvalue())).resampled_to(target_sample_rate, quality) as f:
        actual = f.read(float(f.frames))
        np.testing.assert_allclose(expected_sine_wave, actual)

@pytest.mark.parametrize('sample_rate', [8000, 11025, 22050, 44100, 48000])
@pytest.mark.parametrize('target_sample_rate', [8000, 11025, 12345.67, 22050, 44100, 48000])
@pytest.mark.parametrize('chunk_size', [10, 100])
@pytest.mark.parametrize('quality', QUALITIES)
def test_tell_resampled(sample_rate: float, target_sample_rate: float, chunk_size: int, quality):
    if False:
        i = 10
        return i + 15
    signal = np.linspace(1, sample_rate, sample_rate).astype(np.float32)
    read_buffer = BytesIO()
    read_buffer.name = 'test.wav'
    with AudioFile(read_buffer, 'w', sample_rate, 1, bit_depth=32) as f:
        f.write(signal)
    with AudioFile(BytesIO(read_buffer.getvalue())).resampled_to(target_sample_rate, quality) as f:
        for i in range(0, f.frames, chunk_size):
            assert f.tell() == i
            if f.read(chunk_size).shape[-1] < chunk_size:
                break

@pytest.mark.parametrize('sample_rate', [8000, 11025, 22050, 44100, 48000])
@pytest.mark.parametrize('target_sample_rate', [8000, 11025, 12345.67, 22050, 44100, 48000])
@pytest.mark.parametrize('offset', [2, 10, 100, -10, -1000])
@pytest.mark.parametrize('chunk_size', [2, 10, 50, 100, 1000000])
@pytest.mark.parametrize('quality', QUALITIES)
def test_seek_resampled(sample_rate: float, target_sample_rate: float, offset: int, chunk_size: int, quality):
    if False:
        for i in range(10):
            print('nop')
    signal = np.linspace(1, sample_rate, sample_rate).astype(np.float32)
    read_buffer = BytesIO()
    read_buffer.name = 'test.wav'
    with AudioFile(read_buffer, 'w', sample_rate, 1, bit_depth=32) as f:
        f.write(signal)
    with AudioFile(BytesIO(read_buffer.getvalue())).resampled_to(target_sample_rate, quality) as f:
        effective_offset = offset if offset >= 0 else f.frames + offset
        f.read(effective_offset)
        expected = f.read(chunk_size)
        f.seek(effective_offset)
        actual = f.read(chunk_size)
        np.testing.assert_allclose(expected, actual)

@pytest.mark.parametrize('sample_rate', [8000, 11025])
@pytest.mark.parametrize('target_sample_rate', [8000, 11025, 12345.67])
def test_seek_resampled_is_constant_time(sample_rate: float, target_sample_rate: float):
    if False:
        for i in range(10):
            print('nop')
    signal = np.random.rand(sample_rate * 60).astype(np.float32)
    read_buffer = BytesIO()
    read_buffer.name = 'test.wav'
    with AudioFile(read_buffer, 'w', sample_rate, 1) as f:
        f.write(signal)
    with AudioFile(BytesIO(read_buffer.getvalue())).resampled_to(target_sample_rate) as f:
        timings = []
        for i in range(0, f.frames, sample_rate):
            a = time.time()
            f.seek(i)
            b = time.time()
            timings.append(b - a)
        assert np.std(timings) < 0.02

@pytest.mark.parametrize('sample_rate', [8000, 11025, 22050, 44100, 48000])
@pytest.mark.parametrize('target_sample_rate', [8000, 11025, 12345.67, 22050, 44100, 48000])
@pytest.mark.parametrize('chunk_size', [1000])
@pytest.mark.parametrize('duration', [1.0])
@pytest.mark.parametrize('quality', QUALITIES)
def test_read_resampled_in_chunks(sample_rate: float, target_sample_rate: float, chunk_size: int, duration: float, quality):
    if False:
        i = 10
        return i + 15
    signal = np.linspace(1, sample_rate, int(sample_rate * duration)).astype(np.float32)
    expected_signal = expected_output(signal, sample_rate, target_sample_rate, 1, quality)
    read_buffer = BytesIO()
    read_buffer.name = 'test.wav'
    with AudioFile(read_buffer, 'w', sample_rate, 1, bit_depth=32) as f:
        f.write(signal)
    with AudioFile(BytesIO(read_buffer.getvalue())).resampled_to(target_sample_rate, quality) as f:
        samples_received = 0
        while f.tell() < expected_signal.shape[-1]:
            expected_num_frames = min(chunk_size, expected_signal.shape[-1] - f.tell())
            pos = f.tell()
            output = f.read(chunk_size)
            output_size = output.shape[-1]
            assert output_size == expected_num_frames
            np.testing.assert_allclose(expected_signal[:, samples_received:samples_received + output_size], output, err_msg=f'Output mismatch from {pos:,} to {f.tell():,} of {f.frames:,} samples.')
            samples_received += output_size
        assert samples_received == f.tell()

@pytest.mark.parametrize('sample_rate', [8000, 48000])
@pytest.mark.parametrize('target_sample_rate', [8000, 12345.67, 22050, 44100, 48000])
@pytest.mark.parametrize('chunk_size', [1, 4, 5])
@pytest.mark.parametrize('duration', [0.1])
@pytest.mark.parametrize('quality', QUALITIES)
def test_read_resampled_with_tiny_chunks(sample_rate: float, target_sample_rate: float, chunk_size: int, duration, quality):
    if False:
        for i in range(10):
            print('nop')
    if sample_rate == target_sample_rate:
        return
    test_read_resampled_in_chunks(sample_rate, target_sample_rate, chunk_size, duration, quality)

@pytest.mark.parametrize('sample_rate', [8000, 11025, 22050, 44100, 48000])
@pytest.mark.parametrize('target_sample_rate', [8000, 11025, 12345.67, 22050, 44100, 48000])
@pytest.mark.parametrize('quality', QUALITIES)
def test_frame_count(sample_rate: float, target_sample_rate: float, quality):
    if False:
        return 10
    signal = np.linspace(1, sample_rate, sample_rate).astype(np.float32)
    expected_signal = expected_output(signal, sample_rate, target_sample_rate, 1, quality)
    read_buffer = BytesIO()
    read_buffer.name = 'test.wav'
    with AudioFile(read_buffer, 'w', sample_rate, 1, bit_depth=32) as f:
        f.write(signal)
    with AudioFile(BytesIO(read_buffer.getvalue())).resampled_to(target_sample_rate, quality) as f:
        assert f.frames == expected_signal.shape[-1]