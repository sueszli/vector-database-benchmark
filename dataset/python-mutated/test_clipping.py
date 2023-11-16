import pytest
import numpy as np
from pedalboard import Clipping
from .utils import generate_sine_at, db_to_gain

@pytest.mark.parametrize('threshold_db', list(np.arange(0.0, -40, -0.5)))
@pytest.mark.parametrize('fundamental_hz', [440, 880])
@pytest.mark.parametrize('sample_rate', [22050, 44100, 48000])
@pytest.mark.parametrize('num_channels', [1, 2])
def test_bitcrush(threshold_db: float, fundamental_hz: float, sample_rate: float, num_channels: int):
    if False:
        for i in range(10):
            print('nop')
    sine_wave = generate_sine_at(sample_rate, fundamental_hz, num_seconds=0.1, num_channels=num_channels)
    plugin = Clipping(threshold_db)
    output = plugin.process(sine_wave, sample_rate)
    assert np.all(np.isfinite(output))
    (_min, _max) = (-db_to_gain(threshold_db), db_to_gain(threshold_db))
    expected_output = np.clip(sine_wave, _min, _max)
    np.testing.assert_allclose(output, expected_output, atol=0.01)