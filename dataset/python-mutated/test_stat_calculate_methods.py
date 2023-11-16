import numpy as np
import pandas as pd
import pytest
from plotnine import aes, ggplot, stat_bin, stat_density, xlim
from plotnine.exceptions import PlotnineError, PlotnineWarning

def test_stat_bin():
    if False:
        i = 10
        return i + 15
    x = [1, 2, 3]
    y = [1, 2, 3]
    data = pd.DataFrame({'x': x, 'y': y})
    gg = ggplot(aes(x='x'), data) + stat_bin()
    with pytest.warns(PlotnineWarning) as record:
        gg.draw_test()
    res = ('bins' in str(item.message).lower() for item in record)
    assert any(res)
    gg = ggplot(aes(x='x', y='y'), data) + stat_bin()
    with pytest.raises(PlotnineError):
        gg.draw_test()

def test_changing_xlim_in_stat_density():
    if False:
        for i in range(10):
            print('nop')
    n = 100
    _xlim = (5, 10)
    data = pd.DataFrame({'x': np.linspace(_xlim[0] - 1, _xlim[1] + 1, n)})
    p = ggplot(data, aes('x')) + stat_density() + xlim(*_xlim)
    with pytest.warns(PlotnineWarning):
        p._build()