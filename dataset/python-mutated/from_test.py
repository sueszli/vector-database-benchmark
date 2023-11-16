import pytest
import vaex
from pathlib import Path
HERE = Path(__file__).parent

def test_from_records():
    if False:
        print('Hello World!')
    df = vaex.from_records([{'a': 1, 'b': 2}, {'a': 11, 'c': 13}, {'b': 32}])
    assert df.a.tolist() == [1, 11, None]
    assert df.b.tolist() == [2, None, 32]
    assert df.c.tolist() == [None, 13, None]
    df = vaex.from_records([{'a': 1, 'b': 2}, {'a': 11, 'c': 13}, {'b': 32}], defaults={'a': 111, 'b': 222, 'c': 333})
    assert df.a.tolist() == [1, 11, 111]
    assert df.b.tolist() == [2, 222, 32]
    assert df.c.tolist() == [333, 13, 333]
    df = vaex.from_records([{'a': [1, 1], 'b': 2}, {'a': [11, 12], 'c': 13}, {'a': [13, 14], 'b': 32}], array_type='numpy')
    assert df.a.tolist() == [[1, 1], [11, 12], [13, 14]]
    assert df.a.shape == (3, 2)

def test_from_arrow_dataset():
    if False:
        return 10
    import pyarrow.dataset
    path = HERE / 'data' / 'sample_arrow_dict.parquet'
    ds = pyarrow.dataset.dataset(path)
    df = vaex.from_arrow_dataset(ds)
    assert df.col1.sum() == 45
    assert df.fingerprint() == df.fingerprint()

def test_from_arrow_dataset(tmpdir):
    if False:
        i = 10
        return i + 15
    import numpy as np
    x = np.arange(10)
    df = vaex.from_arrays(x=x)
    import pyarrow.dataset
    path1 = tmpdir / 'sample1.feather'
    path2 = tmpdir / 'sample2.feather'
    df[:5].export(path1)
    df[5:].export(path2)
    ds = pyarrow.dataset.dataset(tmpdir, format='feather')
    df = vaex.from_arrow_dataset(ds)
    assert df.x.sum() == 45