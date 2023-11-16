import pytest
import numpy as np
from pedalboard import GSMFullRateCompressor, Resample
from .utils import generate_sine_at
GSM_ABSOLUTE_TOLERANCE = 0.75
SINE_WAVE_VOLUME = 0.9

@pytest.mark.parametrize('fundamental_hz', [440.0])
@pytest.mark.parametrize('sample_rate', [8000, 11025, 32001.2345, 44100, 48000])
@pytest.mark.parametrize('buffer_size', [1, 32, 160, 1000000])
@pytest.mark.parametrize('duration', [1.0])
@pytest.mark.parametrize('quality', [Resample.Quality.ZeroOrderHold, Resample.Quality.Linear, Resample.Quality.Lagrange, Resample.Quality.CatmullRom, Resample.Quality.WindowedSinc])
@pytest.mark.parametrize('num_channels', [1, 2])
def test_gsm_compressor(fundamental_hz: float, sample_rate: float, buffer_size: int, duration: float, quality: Resample.Quality, num_channels: int):
    if False:
        for i in range(10):
            print('nop')
    signal = generate_sine_at(sample_rate, fundamental_hz, duration, num_channels) * SINE_WAVE_VOLUME
    output = GSMFullRateCompressor(quality=quality)(signal, sample_rate, buffer_size=buffer_size)
    np.testing.assert_allclose(signal, output, atol=GSM_ABSOLUTE_TOLERANCE)

@pytest.mark.parametrize('sample_rate', [8000, 44100])
@pytest.mark.parametrize('quality', [Resample.Quality.ZeroOrderHold, Resample.Quality.Linear, Resample.Quality.Lagrange, Resample.Quality.CatmullRom, Resample.Quality.WindowedSinc])
@pytest.mark.parametrize('num_channels', [1, 2])
def test_gsm_compressor_invariant_to_buffer_size(sample_rate: float, quality: Resample.Quality, num_channels: int):
    if False:
        while True:
            i = 10
    fundamental_hz = 400.0
    duration = 3.0
    signal = generate_sine_at(sample_rate, fundamental_hz, duration, num_channels)
    compressed = [GSMFullRateCompressor(quality=quality)(signal, sample_rate, buffer_size=buffer_size) for buffer_size in (1, 32, 7000, 8192)]
    for (a, b) in zip(compressed, compressed[1:]):
        np.testing.assert_allclose(a, b)