import pytest
import numpy as np
from pedalboard_native._internal import PrimeWithSilenceTestPlugin

@pytest.mark.parametrize('sample_rate', [22050, 44100, 48000])
@pytest.mark.parametrize('buffer_size', [16, 40, 128, 160, 8192, 8193])
@pytest.mark.parametrize('silent_samples_to_add', [1, 16, 40, 128, 160, 8192, 8193])
@pytest.mark.parametrize('num_channels', [1, 2])
def test_prime_with_silence(sample_rate, buffer_size, silent_samples_to_add, num_channels):
    if False:
        for i in range(10):
            print('nop')
    num_seconds = 5.0
    noise = np.random.rand(int(num_seconds * sample_rate))
    if num_channels == 2:
        noise = np.stack([noise, noise])
    plugin = PrimeWithSilenceTestPlugin(silent_samples_to_add)
    output = plugin.process(noise, sample_rate, buffer_size=buffer_size)
    np.testing.assert_allclose(output, noise)