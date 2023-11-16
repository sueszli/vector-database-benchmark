import numpy as np
import pandas as pd
import pytest
from mizani.transforms import trans_new
from plotnine import aes, coord_fixed, coord_flip, coord_trans, geom_bar, geom_line, ggplot, xlim
n = 10
data = pd.DataFrame({'x': np.repeat(range(n + 1), range(n + 1)), 'z': np.repeat(range(n // 2), range(3, n * 2, 4))})
p = ggplot(data, aes('x')) + geom_bar(aes(fill='factor(z)'), show_legend=False)

def test_coord_flip():
    if False:
        for i in range(10):
            print('nop')
    assert p + coord_flip() == 'coord_flip'

def test_coord_fixed():
    if False:
        print('Hello World!')
    assert p + coord_fixed(0.5) == 'coord_fixed'

def test_coord_trans():
    if False:
        i = 10
        return i + 15
    double_trans = trans_new('double', np.square, np.sqrt)
    with pytest.warns(RuntimeWarning):
        assert p + coord_trans(y=double_trans) == 'coord_trans'

def test_coord_trans_reverse():
    if False:
        i = 10
        return i + 15
    p = ggplot(data, aes('factor(x)')) + geom_bar(aes(fill='factor(z)'), show_legend=False) + coord_trans(x='reverse', y='reverse')
    assert p == 'coord_trans_reverse'

def test_coord_trans_backtransforms():
    if False:
        return 10
    data = pd.DataFrame({'x': [-np.inf, np.inf], 'y': [1, 2]})
    p = ggplot(data, aes('x', 'y')) + geom_line(size=2) + xlim(1, 2) + coord_trans(x='log10')
    assert p == 'coord_trans_backtransform'