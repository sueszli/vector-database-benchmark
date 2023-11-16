import pytest
import numpy as np
from pedalboard import Pedalboard, PitchShift

@pytest.mark.parametrize('semitones', [-12, 0, 12])
@pytest.mark.parametrize('fundamental_hz', [440])
@pytest.mark.parametrize('sample_rate', [22050, 44100, 48000])
def test_pitch_shift(semitones, fundamental_hz, sample_rate):
    if False:
        for i in range(10):
            print('nop')
    num_seconds = 1.0
    samples = np.arange(num_seconds * sample_rate)
    sine_wave = np.sin(2 * np.pi * fundamental_hz * samples / sample_rate)
    plugin = PitchShift(semitones)
    output = plugin.process(sine_wave, sample_rate)
    assert np.all(np.isfinite(output))

@pytest.mark.parametrize('semitones', [-73, 73])
def test_pitch_shift_extremes_throws_errors(semitones):
    if False:
        return 10
    with pytest.raises(ValueError):
        PitchShift(semitones)

@pytest.mark.parametrize('semitones', [-72, 72])
@pytest.mark.parametrize('sample_rate', [48000])
@pytest.mark.parametrize('duration', [0.25])
def test_pitch_shift_extremes(semitones, sample_rate, duration):
    if False:
        i = 10
        return i + 15
    noise = np.random.rand(int(duration * sample_rate))
    output = PitchShift(semitones).process(noise, sample_rate)
    assert np.all(np.isfinite(output))

@pytest.mark.parametrize('fundamental_hz', [440.0, 880.0])
@pytest.mark.parametrize('sample_rate', [22050, 44100, 48000])
@pytest.mark.parametrize('buffer_size', [32, 512, 513, 1024, 1029, 2048, 8192])
def test_pitch_shift_latency_compensation(fundamental_hz, sample_rate, buffer_size):
    if False:
        i = 10
        return i + 15
    num_seconds = 2.0
    samples = np.arange(num_seconds * sample_rate)
    sine_wave = np.sin(2 * np.pi * fundamental_hz * samples / sample_rate)
    plugin = Pedalboard([PitchShift(0)])
    output = plugin.process(sine_wave, sample_rate, buffer_size=buffer_size)
    np.testing.assert_allclose(sine_wave, output, atol=1e-06)