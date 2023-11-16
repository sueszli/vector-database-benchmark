import pytest
import numpy as np
from pedalboard import time_stretch

@pytest.mark.parametrize('semitones', [-1, 0, 1])
@pytest.mark.parametrize('stretch_factor', [0.1, 0.75, 1, 1.25])
@pytest.mark.parametrize('fundamental_hz', [440])
@pytest.mark.parametrize('sample_rate', [22050, 44100, 48000])
@pytest.mark.parametrize('high_quality', [True, False])
def test_time_stretch(semitones, stretch_factor, fundamental_hz, sample_rate, high_quality):
    if False:
        return 10
    num_seconds = 1.0
    samples = np.arange(num_seconds * sample_rate)
    sine_wave = np.sin(2 * np.pi * fundamental_hz * samples / sample_rate).astype(np.float32)
    output = time_stretch(sine_wave, sample_rate, stretch_factor=stretch_factor, pitch_shift_in_semitones=semitones, high_quality=high_quality)
    assert np.all(np.isfinite(output))
    assert output.shape[1] == int(num_seconds * sample_rate / stretch_factor)
    if stretch_factor != 1 or semitones != 0:
        min_samples = min(output.shape[1], sine_wave.shape[0])
        assert not np.allclose(output[:, :min_samples], sine_wave[:min_samples])

@pytest.mark.parametrize('high_quality', [True, False])
@pytest.mark.parametrize('transient_mode', ['crisp', 'mixed', 'smooth'])
@pytest.mark.parametrize('transient_detector', ['compound', 'percussive', 'soft'])
@pytest.mark.parametrize('retain_phase_continuity', [True, False])
@pytest.mark.parametrize('use_long_fft_window', [None, True, False])
@pytest.mark.parametrize('use_time_domain_smoothing', [True, False])
@pytest.mark.parametrize('preserve_formants', [True, False])
def test_time_stretch_extra_options(high_quality, transient_mode, transient_detector, retain_phase_continuity, use_long_fft_window, use_time_domain_smoothing, preserve_formants):
    if False:
        print('Hello World!')
    sample_rate = 22050
    num_seconds = 0.5
    fundamental_hz = 440
    samples = np.arange(num_seconds * sample_rate)
    sine_wave = np.sin(2 * np.pi * fundamental_hz * samples / sample_rate).astype(np.float32)
    output = time_stretch(sine_wave, sample_rate, stretch_factor=1.5, pitch_shift_in_semitones=1, high_quality=high_quality, transient_mode=transient_mode, transient_detector=transient_detector, retain_phase_continuity=retain_phase_continuity, use_long_fft_window=use_long_fft_window, use_time_domain_smoothing=use_time_domain_smoothing, preserve_formants=preserve_formants)
    assert np.all(np.isfinite(output))

@pytest.mark.parametrize('semitones', [0])
@pytest.mark.parametrize('stretch_factor', [1.0])
@pytest.mark.parametrize('fundamental_hz', [440, 220, 110])
@pytest.mark.parametrize('sample_rate', [22050, 44100])
@pytest.mark.parametrize('high_quality', [True, False])
def test_time_stretch_long_passthrough(semitones, stretch_factor, fundamental_hz, sample_rate, high_quality):
    if False:
        for i in range(10):
            print('nop')
    num_seconds = 30.0
    samples = np.arange(num_seconds * sample_rate)
    sine_wave = np.sin(2 * np.pi * fundamental_hz * samples / sample_rate).astype(np.float32)
    output = time_stretch(sine_wave, sample_rate, stretch_factor=stretch_factor, pitch_shift_in_semitones=semitones, high_quality=high_quality)
    np.testing.assert_allclose(output[0], sine_wave, atol=0.25)