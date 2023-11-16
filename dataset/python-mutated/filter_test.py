import numpy as np
import vaex
import pyarrow as pa

def test_set_active_range_and_trim(df_factory):
    if False:
        i = 10
        return i + 15
    df = df_factory(x=np.arange(8))
    df = df[df.x % 2 == 0]
    assert len(df) == 4
    df.set_active_range(2, 6)
    assert df._cached_filtered_length == 2
    assert df._selection_masks[vaex.dataframe.FILTER_SELECTION_NAME].count() == 4
    dft = df.trim()
    assert dft._cached_filtered_length == 2
    assert dft._selection_masks[vaex.dataframe.FILTER_SELECTION_NAME].count() == 2
    assert dft.x.tolist() == [2, 4]

def test_filter_cache():
    if False:
        i = 10
        return i + 15
    called = 0

    def odd(x):
        if False:
            return 10
        nonlocal called
        called += 1
        return x % 2 == 1
    x = np.arange(10)
    df = vaex.from_arrays(x=x)
    df.add_function('odd', odd)
    dff = df[df.func.odd('x')]
    len(dff)
    assert called == 1
    df_sliced1 = dff[1:2]
    df_sliced2 = dff[2:4]
    assert called == 1
    repr(dff)
    assert called == 1
    len(df_sliced1)
    len(df_sliced2)
    assert called == 1
    df_sliced3 = df_sliced2[1:2]
    assert called == 1
    len(df_sliced3)
    assert called == 1

def test_filter_by_boolean_column():
    if False:
        while True:
            i = 10
    df = vaex.from_scalars(x=1, ok=True)
    dff = df[df.ok]
    assert dff[['x']].x.tolist() == [1]

def test_filter_after_dropna(df_factory):
    if False:
        while True:
            i = 10
    x = pa.array([10, 20, 30, None, None])
    y = pa.array([1, 2, 3, None, None])
    z = pa.array(['1', '2', '3', None, None])
    df = df_factory(x=x, y=y, z=z)
    df = df['x', 'y'].dropna()
    dd = df[df.x > 10]
    assert dd.x.tolist() == [20, 30]
    assert dd.y.tolist() == [2, 3]

def test_filter_arrow_string_scalar():
    if False:
        print('Hello World!')
    df = vaex.from_arrays(x=['red', 'green', 'blue'])
    assert df[df.x == pa.scalar('red')].x.tolist() == ['red']
    assert df[df.x == pa.scalar('green')].shape == (1, 1)
    assert df[df.x != pa.scalar('blue')].shape