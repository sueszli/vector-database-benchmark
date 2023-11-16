from common import *
import pytest

def test_count_1d():
    if False:
        while True:
            i = 10
    ds = vaex.example()
    binned_values = ds.count(binby=ds.x, limits=[-50, 50], shape=16)
    assert len(binned_values) == 16
    binned_values = ds.count(binby=ds.x, limits='minmax', shape=16)
    assert len(binned_values) == 16
    binned_values = ds.count(binby=ds.x, limits='95%', shape=16)
    assert len(binned_values) == 16

def test_count_2d():
    if False:
        print('Hello World!')
    ds = vaex.example()
    binned_values = ds.count(binby=[ds.x, ds.y], shape=32, limits=['minmax', '95%'])
    assert list(binned_values.shape) == [32, 32]
    binned_values = ds.count(binby=[ds.x, ds.y], shape=32, limits=None)
    assert list(binned_values.shape) == [32, 32]
    binned_values = ds.count(binby=[ds.x, ds.y], shape=32, limits=[[-50, 50], [-50, 50]])
    assert list(binned_values.shape) == [32, 32]

@pytest.mark.parametrize('limits', ['minmax', '68.2%', '99.7%', '100%'])
def test_count_1d_verify_against_numpy(ds_local, limits):
    if False:
        for i in range(10):
            print('nop')
    df = ds_local
    expression = 'x'
    selection = df.y > 10
    shape = 4
    vaex_counts = df.count(binby=[expression], selection=selection, shape=shape, limits=limits)
    (xmin, xmax) = df.limits(expression=expression, value=limits, selection=selection)
    x_values = df[selection][expression].values
    (numpy_counts, numpy_edges) = np.histogram(x_values, bins=shape, range=(xmin, xmax))
    assert vaex_counts[:-1].tolist() == numpy_counts[:-1].tolist()

def test_count_selection_w_missing_values():
    if False:
        print('Hello World!')
    x = np.arange(10)
    missing_mask = x % 3 == 0
    x_numpy = np.ma.array(x, mask=missing_mask)
    x_arrow = pa.array(x, mask=missing_mask)
    df = vaex.from_arrays(x_numpy=x_numpy, x_arrow=x_arrow)
    assert all(df.count(binby='x_numpy') == df.count(binby='x_arrow'))
    assert all(df.count(binby='x_numpy', shape=2, limits=[0, 10], selection='x_numpy > 0') == df.count(binby='x_arrow', shape=2, limits=[0, 10], selection='x_arrow > 0'))
    assert all(df.count(binby='x_numpy', shape=2, selection='x_numpy > 0') == df.count(binby='x_arrow', shape=2, selection='x_arrow > 0'))