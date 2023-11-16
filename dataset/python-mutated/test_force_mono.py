import pytest
import numpy as np
from pedalboard_native._internal import ForceMonoTestPlugin
NUM_SECONDS = 1.0

@pytest.mark.parametrize('sample_rate', [22050, 44100])
@pytest.mark.parametrize('buffer_size', [1, 16, 128, 8192])
def test_force_mono(sample_rate, buffer_size):
    if False:
        i = 10
        return i + 15
    stereo_noise = np.stack([np.random.rand(int(NUM_SECONDS * sample_rate)), np.random.rand(int(NUM_SECONDS * sample_rate))])
    output = ForceMonoTestPlugin().process(stereo_noise, sample_rate, buffer_size=buffer_size)
    np.testing.assert_allclose(output[0], output[1])
    expected_mono = (stereo_noise[0] + stereo_noise[1]) / 2
    np.testing.assert_allclose(output, np.stack([expected_mono, expected_mono]), atol=1e-07)

@pytest.mark.parametrize('sample_rate', [22050, 44100])
@pytest.mark.parametrize('buffer_size', [1, 16, 128, 8192])
def test_force_mono_on_already_mono(sample_rate, buffer_size):
    if False:
        print('Hello World!')
    mono_noise = np.random.rand(int(NUM_SECONDS * sample_rate))
    output = ForceMonoTestPlugin().process(mono_noise, sample_rate, buffer_size=buffer_size)
    np.testing.assert_allclose(output, mono_noise)