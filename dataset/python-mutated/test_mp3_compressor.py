import pytest
import numpy as np
from pedalboard import MP3Compressor
MP3_ABSOLUTE_TOLERANCE = 0.25

def generate_sine_at(sample_rate: float, fundamental_hz: float=440.0, num_seconds: float=3.0, num_channels: int=1) -> np.ndarray:
    if False:
        print('Hello World!')
    samples = np.arange(num_seconds * sample_rate)
    sine_wave = np.sin(2 * np.pi * fundamental_hz * samples / sample_rate)
    if num_channels == 2:
        sine_wave = np.stack([sine_wave, sine_wave])
    return sine_wave

@pytest.mark.parametrize('vbr_quality', [0.0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5])
@pytest.mark.parametrize('sample_rate', [44100, 48000])
@pytest.mark.parametrize('num_channels', [1, 2])
def test_mp3_compressor(vbr_quality: float, sample_rate: int, num_channels: int):
    if False:
        print('Hello World!')
    sine_wave = generate_sine_at(sample_rate, num_channels=num_channels)
    compressed = MP3Compressor(vbr_quality)(sine_wave, sample_rate)
    np.testing.assert_allclose(sine_wave, compressed, atol=MP3_ABSOLUTE_TOLERANCE)

@pytest.mark.parametrize('vbr_quality', [0.0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5])
@pytest.mark.parametrize('sample_rate', [44100, 48000])
@pytest.mark.parametrize('num_channels', [1, 2])
def test_mp3_compressor_invariant_to_buffer_size(vbr_quality: float, sample_rate: int, num_channels: int):
    if False:
        i = 10
        return i + 15
    sine_wave = generate_sine_at(sample_rate, num_channels=num_channels)
    compressed_different_buffer_sizes = [MP3Compressor(vbr_quality)(sine_wave, sample_rate, buffer_size=buffer_size) for buffer_size in (1, 32, 128, 1024, 1152, 8192, 65536)]
    for (a, b) in zip(compressed_different_buffer_sizes, compressed_different_buffer_sizes[:1]):
        np.testing.assert_allclose(a, b, atol=MP3_ABSOLUTE_TOLERANCE)

@pytest.mark.parametrize('vbr_quality', [2])
@pytest.mark.parametrize('sample_rate', [48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000])
@pytest.mark.parametrize('num_channels', [1, 2])
@pytest.mark.parametrize('buffer_size', [32, 8192])
def test_mp3_compressor_arbitrary_sample_rate(vbr_quality: float, sample_rate: int, num_channels: int, buffer_size: int):
    if False:
        return 10
    sine_wave = generate_sine_at(sample_rate, num_channels=num_channels)
    compressed = MP3Compressor(vbr_quality)(sine_wave, sample_rate)
    np.testing.assert_allclose(sine_wave, compressed, atol=MP3_ABSOLUTE_TOLERANCE)

@pytest.mark.parametrize('sample_rate', [96000, 6000, 44101])
@pytest.mark.parametrize('num_channels', [1, 2])
def test_mp3_compressor_fails_on_invalid_sample_rate(sample_rate: int, num_channels: int):
    if False:
        for i in range(10):
            print('nop')
    sine_wave = generate_sine_at(sample_rate, num_channels=num_channels)
    with pytest.raises(ValueError):
        MP3Compressor(1)(sine_wave, sample_rate)