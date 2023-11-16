import pytest
import numpy as np
from pedalboard_native._internal import AddLatency

@pytest.mark.parametrize('sample_rate', [22050, 44100, 48000])
@pytest.mark.parametrize('buffer_size', [128, 8192, 65536])
@pytest.mark.parametrize('latency_seconds', [0.25, 1, 2, 10])
def test_latency_compensation(sample_rate, buffer_size, latency_seconds):
    if False:
        print('Hello World!')
    num_seconds = 10.0
    noise = np.random.rand(int(num_seconds * sample_rate))
    plugin = AddLatency(int(latency_seconds * sample_rate))
    output = plugin.process(noise, sample_rate, buffer_size=buffer_size)
    np.testing.assert_allclose(output, noise)