import matplotlib.gridspec as gridspec
import pytest

def test_equal():
    if False:
        while True:
            i = 10
    gs = gridspec.GridSpec(2, 1)
    assert gs[0, 0] == gs[0, 0]
    assert gs[:, 0] == gs[:, 0]

def test_width_ratios():
    if False:
        i = 10
        return i + 15
    '\n    Addresses issue #5835.\n    See at https://github.com/matplotlib/matplotlib/issues/5835.\n    '
    with pytest.raises(ValueError):
        gridspec.GridSpec(1, 1, width_ratios=[2, 1, 3])

def test_height_ratios():
    if False:
        i = 10
        return i + 15
    '\n    Addresses issue #5835.\n    See at https://github.com/matplotlib/matplotlib/issues/5835.\n    '
    with pytest.raises(ValueError):
        gridspec.GridSpec(1, 1, height_ratios=[2, 1, 3])

def test_repr():
    if False:
        while True:
            i = 10
    ss = gridspec.GridSpec(3, 3)[2, 1:3]
    assert repr(ss) == 'GridSpec(3, 3)[2:3, 1:3]'
    ss = gridspec.GridSpec(2, 2, height_ratios=(3, 1), width_ratios=(1, 3))
    assert repr(ss) == 'GridSpec(2, 2, height_ratios=(3, 1), width_ratios=(1, 3))'