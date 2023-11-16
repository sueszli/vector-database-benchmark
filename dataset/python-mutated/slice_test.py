import pytest
from common import *

def test_slice_expression(df):
    if False:
        for i in range(10):
            print('nop')
    assert df.x[:2].tolist() == df[:2].x.tolist()
    assert df.x[2:6].tolist() == df[2:6].x.tolist()
    assert df.x[-3:].tolist() == df[-3:].x.tolist()

def test_slice_against_numpy(df):
    if False:
        while True:
            i = 10
    assert df.x[:2].tolist() == df.x.to_numpy()[:2].tolist()
    assert df.x[2:6].tolist() == df.x.to_numpy()[2:6].tolist()
    assert df.x[-3:].tolist() == df.x.to_numpy()[-3:].tolist()

@pytest.mark.xfail(reason='Not supported yet')
def test_slice_filtered_remte(ds_remote):
    if False:
        return 10
    df = ds_remote
    dff = df[df.x > 0]
    dfs = dff[1:]
    assert dfs.x.values[0] == 2

def test_slice(ds_local):
    if False:
        print('Hello World!')
    ds = ds_local
    ds_sliced = ds[:]
    assert ds_sliced.length_original() == ds_sliced.length_unfiltered() >= 10
    assert ds_sliced.get_active_range() == (0, ds_sliced.length_original())
    assert ds_sliced.x.tolist() == np.arange(10.0).tolist()
    ds_sliced = ds[5:]
    assert ds_sliced.length_original() == ds_sliced.length_unfiltered() == 5
    assert ds_sliced.get_active_range() == (0, ds_sliced.length_original()) == (0, 5)
    assert ds_sliced.x.tolist() == np.arange(5, 10.0).tolist()
    ds_sliced = ds_sliced[1:4]
    assert ds_sliced.length_original() == ds_sliced.length_unfiltered() == 3
    assert ds_sliced.get_active_range() == (0, ds_sliced.length_original()) == (0, 3)
    assert ds_sliced.x.tolist() == np.arange(6, 9.0).tolist()

def test_head(ds_local):
    if False:
        while True:
            i = 10
    ds = ds_local
    df = ds.head(5)
    assert len(df) == 5

def test_tail(ds_local):
    if False:
        for i in range(10):
            print('nop')
    ds = ds_local
    df = ds.tail(5)
    assert len(df) == 5

def test_head_with_selection():
    if False:
        return 10
    df = vaex.example()
    df.select(df.x > 0, name='test')
    df.head()

def test_slice_beyond_end(df):
    if False:
        i = 10
        return i + 15
    df2 = df[:100]
    assert df2.x.tolist() == df.x.tolist()
    assert len(df2) == len(df)

def test_slice_negative(df):
    if False:
        i = 10
        return i + 15
    df2 = df[:-1]
    assert df2.x.tolist() == df.x.to_numpy()[:-1].tolist()
    assert len(df2) == len(df) - 1

def test_getitem():
    if False:
        for i in range(10):
            print('nop')
    x = np.array([[1, 7], [2, 8], [3, 9]])
    df = vaex.from_arrays(x=x)
    assert len(df) == 3
    assert df.x[:, 0].tolist() == [1, 2, 3]
    assert df.x[:, 1].tolist() == [7, 8, 9]
    assert df.x[:, -1].tolist() == [7, 8, 9]
    assert df.x[1:, 0].tolist() == [2, 3]
    assert df.x[:2, 1].tolist() == [7, 8]
    assert df.x[1:-1, -1].tolist() == [8]

def test_slice_empty_df():
    if False:
        for i in range(10):
            print('nop')
    x = np.array([1, 2, 3, 4, 5])
    df = vaex.from_arrays(x=x)
    dff = df[df.x > 100]
    assert len(dff) == 0
    dfs1 = dff[:3]
    assert len(dfs1) == 0
    dfs2 = dff[3:]
    assert len(dfs2) == 0