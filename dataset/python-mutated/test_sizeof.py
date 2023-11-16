from __future__ import annotations
import os
import sys
from array import array
import pytest
from dask.multiprocessing import get_context
from dask.sizeof import sizeof
from dask.utils import funcname
try:
    import pandas as pd
except ImportError:
    pd = None
requires_pandas = pytest.mark.skipif(pd is None, reason='requires pandas')

def test_base():
    if False:
        while True:
            i = 10
    assert sizeof(1) == sys.getsizeof(1)

def test_name():
    if False:
        print('Hello World!')
    assert funcname(sizeof) == 'sizeof'

def test_containers():
    if False:
        i = 10
        return i + 15
    assert sizeof([1, 2, [3]]) > sys.getsizeof(3) * 3 + sys.getsizeof([])

def test_bytes_like():
    if False:
        print('Hello World!')
    assert 1000 <= sizeof(bytes(1000)) <= 2000
    assert 1000 <= sizeof(bytearray(1000)) <= 2000
    assert 1000 <= sizeof(memoryview(bytes(1000))) <= 2000
    assert 8000 <= sizeof(array('d', range(1000))) <= 9000

def test_numpy():
    if False:
        return 10
    np = pytest.importorskip('numpy')
    assert 8000 <= sizeof(np.empty(1000, dtype='f8')) <= 9000
    dt = np.dtype('f8')
    assert sizeof(dt) == sys.getsizeof(dt)

def test_numpy_0_strided():
    if False:
        for i in range(10):
            print('nop')
    np = pytest.importorskip('numpy')
    x = np.broadcast_to(1, (100, 100, 100))
    assert sizeof(x) <= 8

@requires_pandas
def test_pandas():
    if False:
        while True:
            i = 10
    df = pd.DataFrame({'x': [1, 2, 3], 'y': ['a' * 100, 'b' * 100, 'c' * 100]}, index=[10, 20, 30])
    assert sizeof(df) >= sizeof(df.x) + sizeof(df.y) - sizeof(df.index)
    assert sizeof(df.x) >= sizeof(df.index)
    assert sizeof(df.y) >= 100 * 3
    assert sizeof(df.index) >= 20
    assert isinstance(sizeof(df), int)
    assert isinstance(sizeof(df.x), int)
    assert isinstance(sizeof(df.index), int)

@requires_pandas
def test_pandas_contiguous_dtypes():
    if False:
        print('Hello World!')
    '2+ contiguous columns of the same dtype in the same DataFrame share the same\n    surface thus have lower overhead\n    '
    df1 = pd.DataFrame([[1, 2.2], [3, 4.4]])
    df2 = pd.DataFrame([[1.1, 2.2], [3.3, 4.4]])
    assert sizeof(df2) < sizeof(df1)

@requires_pandas
def test_pandas_multiindex():
    if False:
        print('Hello World!')
    index = pd.MultiIndex.from_product([range(5), ['a', 'b', 'c', 'd', 'e']])
    actual_size = sys.getsizeof(index)
    assert 0.5 * actual_size < sizeof(index) < 2 * actual_size
    assert isinstance(sizeof(index), int)

@requires_pandas
def test_pandas_repeated_column():
    if False:
        return 10
    df = pd.DataFrame({'x': list(range(10000))})
    df2 = df[['x', 'x', 'x']]
    df3 = pd.DataFrame({'x': list(range(10000)), 'y': list(range(10000))})
    assert 80000 < sizeof(df) < 85000
    assert 80000 < sizeof(df2) < 85000
    assert 160000 < sizeof(df3) < 165000

def test_sparse_matrix():
    if False:
        print('Hello World!')
    sparse = pytest.importorskip('scipy.sparse')
    sp = sparse.eye(10)
    assert sizeof(sp.todia()) >= 152
    assert sizeof(sp.tobsr()) >= 232
    assert sizeof(sp.tocoo()) >= 240
    assert sizeof(sp.tocsc()) >= 232
    assert sizeof(sp.tocsr()) >= 232
    assert sizeof(sp.todok()) >= 188
    assert sizeof(sp.tolil()) >= 204

@requires_pandas
@pytest.mark.parametrize('cls_name', ['Series', 'DataFrame', 'Index'])
@pytest.mark.parametrize('dtype', [object, 'string[python]'])
def test_pandas_object_dtype(dtype, cls_name):
    if False:
        i = 10
        return i + 15
    cls = getattr(pd, cls_name)
    s1 = cls([f'x{i:3d}' for i in range(1000)], dtype=dtype)
    assert sizeof('x000') * 1000 < sizeof(s1) < 2 * sizeof('x000') * 1000
    x = 'x' * 100000
    y = 'y' * 100000
    z = 'z' * 100000
    w = 'w' * 100000
    s2 = cls([x, y, z, w] * 1000, dtype=dtype)
    assert 400000 < sizeof(s2) < 500000
    s3 = cls([x, y, z, w], dtype=dtype)
    s4 = cls([x, y, z, x], dtype=dtype)
    s5 = cls([x, x, x, x], dtype=dtype)
    assert sizeof(s5) < sizeof(s4) < sizeof(s3)

@requires_pandas
@pytest.mark.parametrize('dtype', [object, 'string[python]'])
def test_dataframe_object_dtype(dtype):
    if False:
        return 10
    x = 'x' * 100000
    y = 'y' * 100000
    z = 'z' * 100000
    w = 'w' * 100000
    objs = [x, y, z, w]
    df1 = pd.DataFrame([objs * 3] * 1000, dtype=dtype)
    assert 400000 < sizeof(df1) < 550000
    df2 = pd.DataFrame([[x, y], [z, w]], dtype=dtype)
    df3 = pd.DataFrame([[x, y], [z, x]], dtype=dtype)
    df4 = pd.DataFrame([[x, x], [x, x]], dtype=dtype)
    assert sizeof(df4) < sizeof(df3) < sizeof(df2)

@pytest.mark.parametrize('cls_name', ['Series', 'DataFrame', 'Index'])
def test_pandas_string_arrow_dtype(cls_name):
    if False:
        while True:
            i = 10
    pytest.importorskip('pyarrow')
    cls = getattr(pd, cls_name)
    s = cls(['x' * 100000, 'y' * 50000], dtype='string[pyarrow]')
    assert 150000 < sizeof(s) < 155000

@requires_pandas
def test_pandas_empty():
    if False:
        while True:
            i = 10
    df = pd.DataFrame({'x': [1, 2, 3], 'y': ['a' * 100, 'b' * 100, 'c' * 100]}, index=[10, 20, 30])
    empty = df.head(0)
    assert sizeof(empty) > 0
    assert sizeof(empty.x) > 0
    assert sizeof(empty.y) > 0
    assert sizeof(empty.index) > 0

@requires_pandas
def test_pyarrow_table():
    if False:
        for i in range(10):
            print('nop')
    pa = pytest.importorskip('pyarrow')
    df = pd.DataFrame({'x': [1, 2, 3], 'y': ['a' * 100, 'b' * 100, 'c' * 100]}, index=[10, 20, 30])
    table = pa.Table.from_pandas(df)
    assert sizeof(table) > sizeof(table.schema.metadata)
    assert isinstance(sizeof(table), int)
    assert isinstance(sizeof(table.columns[0]), int)
    assert isinstance(sizeof(table.columns[1]), int)
    assert isinstance(sizeof(table.columns[2]), int)
    empty = pa.Table.from_pandas(df.head(0))
    assert sizeof(empty) > sizeof(empty.schema.metadata)
    assert sizeof(empty.columns[0]) > 0
    assert sizeof(empty.columns[1]) > 0
    assert sizeof(empty.columns[2]) > 0

def test_dict():
    if False:
        for i in range(10):
            print('nop')
    np = pytest.importorskip('numpy')
    x = np.ones(10000)
    assert sizeof({'x': x}) > x.nbytes
    assert sizeof({'x': [x]}) > x.nbytes
    assert sizeof({'x': [{'y': x}]}) > x.nbytes
    d = {i: x for i in range(100)}
    assert sizeof(d) > x.nbytes * 100
    assert isinstance(sizeof(d), int)

def _get_sizeof_on_path(path, size):
    if False:
        for i in range(10):
            print('nop')
    sys.path.append(os.fsdecode(path))
    import dask.sizeof
    dask.sizeof._register_entry_point_plugins()
    import class_impl
    cls = class_impl.Impl(size)
    return sizeof(cls)

def test_register_backend_entrypoint(tmp_path):
    if False:
        print('Hello World!')
    (tmp_path / 'impl_sizeof.py').write_bytes(b'def sizeof_plugin(sizeof):\n    print("REG")\n    @sizeof.register_lazy("class_impl")\n    def register_impl():\n        import class_impl\n        @sizeof.register(class_impl.Impl)\n        def sizeof_impl(obj):\n            return obj.size \n')
    (tmp_path / 'class_impl.py').write_bytes(b'class Impl:\n    def __init__(self, size):\n        self.size = size')
    dist_info = tmp_path / 'impl_sizeof-0.0.0.dist-info'
    dist_info.mkdir()
    (dist_info / 'entry_points.txt').write_bytes(b'[dask.sizeof]\nimpl = impl_sizeof:sizeof_plugin\n')
    with get_context().Pool(1) as pool:
        assert pool.apply(_get_sizeof_on_path, args=(tmp_path, 314159265)) == 314159265
    pool.join()