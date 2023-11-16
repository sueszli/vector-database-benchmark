import pytest
import pickle
import numpy as np
import vaex
N_rows = 1024 * 4

def test_pickle_roundtrip(df_local):
    if False:
        for i in range(10):
            print('nop')
    df = df_local
    data = pickle.dumps(df)
    df2 = pickle.loads(data)
    if 'obj' in df:
        df = df.drop('obj')
        df2 = df2.drop('obj')
    df['x'].tolist() == df2['x'].tolist()
    df.x.tolist() == df2.x.tolist()
    result = list(df.compare(df2))
    result[2] = []
    result = tuple(result)
    assert result == ([], [], [], [])

def test_pick_file(tmpdir, file_extension):
    if False:
        print('Hello World!')
    x = np.arange(N_rows, dtype='i8')
    df = vaex.from_arrays(x=x, x2=-x)
    df['y'] = df.x ** 2
    data = pickle.dumps(df)
    assert len(data) > len(x) * x.itemsize
    xsum = df.x.sum()
    ysum = df.y.sum()
    for ext in 'hdf5 parquet'.split():
        path = tmpdir / f'test.{ext}'
        df.export(path)
        df = vaex.open(path)
        data = pickle.dumps(df)
        assert len(data) < 1000
        assert df.x.sum() == xsum
        assert df.y.sum() == ysum

@pytest.fixture(params=['hdf5', 'parquet', 'arrow'])
def file_extension(request):
    if False:
        while True:
            i = 10
    return request.param

@pytest.fixture()
def df_file(file_extension, tmpdir):
    if False:
        while True:
            i = 10
    x = np.arange(N_rows, dtype='i8')
    df = vaex.from_arrays(x=x, x2=-x)
    df['y'] = df.x ** 2
    path = tmpdir / f'test.{file_extension}'
    df.export(path)
    df = vaex.open(path)
    yield df

def test_slice(df_file):
    if False:
        return 10
    df = df_file[:len(df_file) - 2]
    assert len(pickle.dumps(df)) < 2000
    df2 = pickle.loads(pickle.dumps(df))
    assert df.compare(df2) == ([], [], [], [])

def test_rename(df_file):
    if False:
        while True:
            i = 10
    df = df_file[:len(df_file) - 2]
    df.rename('x', 'a')
    assert len(pickle.dumps(df)) < 2000
    df2 = pickle.loads(pickle.dumps(df))
    assert df.compare(df2) == ([], [], [], [])

def test_drop(df_file):
    if False:
        return 10
    df = df_file.drop('x2')
    assert len(pickle.dumps(df)) < 2000
    df2 = pickle.loads(pickle.dumps(df))
    assert df.compare(df2) == ([], [], [], [])

def test_merge_files(df_file, tmpdir):
    if False:
        i = 10
        return i + 15
    path = tmpdir / 'test2.hdf5'
    df_file[['x']].export(path)
    df_join = vaex.open(path)
    df_join.rename('x', 'z')
    df = df_file.join(df_join)
    assert len(pickle.dumps(df)) < 2300
    df2 = pickle.loads(pickle.dumps(df))
    assert df.compare(df2) == ([], [], [], [])
    assert df2.sum('x-z') == 0

def test_merge_data(df_file, tmpdir):
    if False:
        print('Hello World!')
    df_join = vaex.from_arrays(z=df_file.x.values)
    df = df_file.join(df_join)
    assert len(pickle.dumps(df)) > N_rows * 4, 'transport all'
    df2 = pickle.loads(pickle.dumps(df))
    assert df.compare(df2) == ([], [], [], [])
    assert (df2.x - df2.z).sum() == 0

def test_take(df_file, tmpdir):
    if False:
        return 10
    df = df_file.shuffle()
    assert len(pickle.dumps(df)) > N_rows * 4, 'indices take space'
    df2 = pickle.loads(pickle.dumps(df))
    assert df.compare(df2) == ([], [], [], [])
    assert df2.x.sum() == df_file.x.sum()

def test_concat(df_file, tmpdir):
    if False:
        while True:
            i = 10
    path = tmpdir / 'test2.hdf5'
    df_file[['x']].export(path)
    df_concat = vaex.open(path)
    df = vaex.concat([df_file, df_concat])
    assert len(pickle.dumps(df)) < 2000
    df2 = pickle.loads(pickle.dumps(df))
    assert len(df) == len(df_file) * 2
    assert len(df2) == len(df_file) * 2
    assert df2.x.count() == len(df_file) * 2, 'x is repeated'
    assert df2.x.sum() == df_file.x.sum() * 2, 'x is repeated'
    assert df2.y.sum() == df_file.y.sum(), 'y is not repeated'

def test_state_with_set():
    if False:
        print('Hello World!')
    df = vaex.from_arrays(x=[1, 2, 3])
    df['test'] = df.x.isin([1, 2])
    df2 = pickle.loads(pickle.dumps(df))
    assert df2.x.tolist() == df.x.tolist()
    pickle.dumps(df.state_get())