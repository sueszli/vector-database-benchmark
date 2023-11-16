import pytest
import numpy as np
from pedalboard import Bitcrush
from .utils import generate_sine_at

@pytest.mark.parametrize('bit_depth', list(np.arange(1, 32, 0.5)))
@pytest.mark.parametrize('fundamental_hz', [440, 880])
@pytest.mark.parametrize('sample_rate', [22050, 44100, 48000])
@pytest.mark.parametrize('num_channels', [1, 2])
def test_bitcrush(bit_depth: float, fundamental_hz: float, sample_rate: float, num_channels: int):
    if False:
        for i in range(10):
            print('nop')
    sine_wave = generate_sine_at(sample_rate, fundamental_hz, num_seconds=0.1, num_channels=num_channels)
    plugin = Bitcrush(bit_depth)
    output = plugin.process(sine_wave, sample_rate)
    assert np.all(np.isfinite(output))
    expected_output = np.around(sine_wave.astype(np.float64) * 2 ** bit_depth) / 2 ** bit_depth
    np.testing.assert_allclose(output, expected_output, atol=0.01)