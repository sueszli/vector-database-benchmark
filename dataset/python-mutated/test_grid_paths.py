import numpy as np
import pytest
from matplotlib.lines import Path
from astropy.visualization.wcsaxes.grid_paths import get_lon_lat_path

@pytest.mark.parametrize('step_in_degrees', [10, 1, 0.01])
def test_round_trip_visibility(step_in_degrees):
    if False:
        while True:
            i = 10
    zero = np.zeros(100)
    pixel = np.stack([zero, zero]).T
    line = np.stack([np.arange(100), zero]).T * step_in_degrees
    line_round = line * 1.05
    path = get_lon_lat_path(line, pixel, line_round)
    codes_check = np.full(100, Path.MOVETO)
    codes_check[line_round[:, 0] - line[:, 0] < step_in_degrees] = Path.LINETO
    assert np.all(path.codes[1:] == codes_check[1:])