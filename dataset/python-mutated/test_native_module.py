import os
import pytest
import numpy as np
from pedalboard import process, Delay, Distortion, Invert, Gain, Compressor, Convolution, Reverb
IMPULSE_RESPONSE_PATH = os.path.join(os.path.dirname(__file__), 'impulse_response.wav')

@pytest.mark.parametrize('shape', [(44100,), (44100, 1), (44100, 2), (1, 44100), (2, 44100)])
def test_no_transforms(shape, sr=44100):
    if False:
        print('Hello World!')
    _input = np.random.rand(*shape).astype(np.float32)
    output = process(_input, sr, [])
    assert _input.shape == output.shape
    assert np.allclose(_input, output, rtol=0.0001)

@pytest.mark.parametrize('shape', [(44100,), (44100, 1), (44100, 2), (1, 44100), (2, 44100)])
def test_noise_gain(shape, sr=44100):
    if False:
        for i in range(10):
            print('nop')
    full_scale_noise = np.random.rand(*shape).astype(np.float32)
    half_noise = process(full_scale_noise, sr, [Gain(-6)])
    assert full_scale_noise.shape == half_noise.shape
    assert np.allclose(full_scale_noise / 2.0, half_noise, rtol=0.01)

def test_throw_on_invalid_compressor_ratio(sr=44100):
    if False:
        while True:
            i = 10
    full_scale_noise = np.random.rand(sr, 1).astype(np.float32)
    process(full_scale_noise, sr, [Compressor(ratio=1.1)])
    with pytest.raises(ValueError):
        Compressor(ratio=0.1)

def test_convolution_file_exists():
    if False:
        i = 10
        return i + 15
    "\n    A meta-test - if this fails, we can't find the file, so the following two tests will fail!\n    "
    assert os.path.isfile(IMPULSE_RESPONSE_PATH)

def test_convolution_works(sr=44100, duration=10):
    if False:
        print('Hello World!')
    full_scale_noise = np.random.rand(sr * duration).astype(np.float32)
    result = process(full_scale_noise, sr, [Convolution(IMPULSE_RESPONSE_PATH, 0.5)])
    assert not np.allclose(full_scale_noise, result, rtol=0.1)

def test_throw_on_inaccessible_convolution_file():
    if False:
        return 10
    Convolution(IMPULSE_RESPONSE_PATH)
    with pytest.raises(RuntimeError):
        Convolution('missing_impulse_response.wav')

@pytest.mark.parametrize('gain_db', [-12, -6, 0, 1.1, 6, 12, 24, 48, 96])
@pytest.mark.parametrize('shape', [(44100,), (44100, 1), (44100, 2), (1, 44100), (2, 44100)])
def test_distortion(gain_db, shape, sr=44100):
    if False:
        i = 10
        return i + 15
    full_scale_noise = np.random.rand(*shape).astype(np.float32)
    result = process(full_scale_noise, sr, [Distortion(gain_db)])
    np.testing.assert_equal(result.shape, full_scale_noise.shape)
    gain_scale = np.power(10.0, 0.05 * gain_db)
    np.testing.assert_allclose(np.tanh(full_scale_noise * gain_scale), result, rtol=4e-07, atol=2e-07)

@pytest.mark.parametrize('shape', [(44100,), (44100, 1), (44100, 2), (1, 44100), (2, 44100)])
def test_invert(shape, sr=44100):
    if False:
        while True:
            i = 10
    full_scale_noise = np.random.rand(*shape).astype(np.float32)
    result = Invert()(full_scale_noise, sr)
    np.testing.assert_allclose(full_scale_noise * -1, result, rtol=4e-07, atol=2e-07)

def test_delay():
    if False:
        print('Hello World!')
    delay_seconds = 2.5
    feedback = 0.0
    mix = 0.5
    duration = 10.0
    sr = 44100
    full_scale_noise = np.random.rand(int(sr * duration)).astype(np.float32)
    result = Delay(delay_seconds, feedback, mix)(full_scale_noise, sr)
    dry_volume = 1.0 - mix
    wet_volume = mix
    delayed_line = np.concatenate([np.zeros(int(delay_seconds * sr)), full_scale_noise])[:len(result)]
    expected = dry_volume * full_scale_noise + wet_volume * delayed_line
    np.testing.assert_equal(result.shape, expected.shape)
    np.testing.assert_allclose(expected, result, rtol=4e-07, atol=2e-07)

@pytest.mark.parametrize('reset', (True, False))
def test_plugin_state_not_cleared_between_invocations(reset: bool):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure that if `reset` is True, we do reset the plugin state\n    (i.e.: we cut off reverb tails). If `reset` is False, plugin\n    state should be maintained between calls to `render`\n    (preserving tails).\n    '
    reverb = Reverb()
    sr = 44100
    noise = np.random.rand(sr)
    silence = np.zeros_like(noise)
    assert np.amax(np.abs(reverb(silence, sr, reset=reset))) == 0.0
    reverb(noise, sr, reset=reset)
    effected_silence = reverb(silence, sr, reset=reset)
    effected_silence_noise_floor = np.amax(np.abs(effected_silence))
    if reset:
        assert effected_silence_noise_floor == 0.0
    else:
        assert effected_silence_noise_floor > 0.25

def test_plugin_state_not_cleared_if_passed_smaller_buffer():
    if False:
        i = 10
        return i + 15
    "\n    Ensure that if `reset` is False, a smaller buffer size can be\n    passed without clearing the plugin's internal state:\n    "
    reverb = Reverb()
    sr = 44100
    noise = np.random.rand(sr)
    silence = np.zeros_like(noise)
    assert np.amax(np.abs(reverb(silence, sr, reset=False))) == 0.0
    reverb(noise, sr, reset=False)
    effected_silence = reverb(silence[:int(len(silence) / 2)], sr, reset=False)
    effected_silence_noise_floor = np.amax(np.abs(effected_silence))
    assert effected_silence_noise_floor > 0.25