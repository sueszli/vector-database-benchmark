from __future__ import annotations
import numpy as np
import pytest
pytestmark = pytest.mark.gpu
import dask.array as da
from dask.array.utils import assert_eq
cupy = pytest.importorskip('cupy')

@pytest.mark.parametrize('idx_chunks', [None, 3, 2, 1])
@pytest.mark.parametrize('x_chunks', [(3, 5), (2, 3), (1, 2), (1, 1)])
def test_index_with_int_dask_array(x_chunks, idx_chunks):
    if False:
        print('Hello World!')
    x = cupy.array([[10, 20, 30, 40, 50], [60, 70, 80, 90, 100], [110, 120, 130, 140, 150]])
    idx = cupy.array([3, 0, 1])
    expect = cupy.array([[40, 10, 20], [90, 60, 70], [140, 110, 120]])
    x = da.from_array(x, chunks=x_chunks)
    if idx_chunks is not None:
        idx = da.from_array(idx, chunks=idx_chunks)
    assert_eq(x[:, idx], expect)
    assert_eq(x.T[idx, :], expect.T)

@pytest.mark.parametrize('idx_chunks', [None, 3, 2, 1])
@pytest.mark.parametrize('x_chunks', [(3, 5), (2, 3), (1, 2), (1, 1)])
def test_index_with_int_dask_array_nep35(x_chunks, idx_chunks):
    if False:
        while True:
            i = 10
    x = cupy.array([[10, 20, 30, 40, 50], [60, 70, 80, 90, 100], [110, 120, 130, 140, 150]])
    orig_idx = np.array([3, 0, 1])
    expect = cupy.array([[40, 10, 20], [90, 60, 70], [140, 110, 120]])
    if x_chunks is not None:
        x = da.from_array(x, chunks=x_chunks)
    if idx_chunks is not None:
        idx = da.from_array(orig_idx, chunks=idx_chunks)
    else:
        idx = orig_idx
    assert_eq(x[:, idx], expect)
    assert_eq(x.T[idx, :], expect.T)
    orig_idx = cupy.array(orig_idx)
    if idx_chunks is not None:
        idx = da.from_array(orig_idx, chunks=idx_chunks)
    else:
        idx = orig_idx
    assert_eq(x[:, idx], expect)
    assert_eq(x.T[idx, :], expect.T)

@pytest.mark.parametrize('chunks', [1, 2, 3])
def test_index_with_int_dask_array_0d(chunks):
    if False:
        print('Hello World!')
    x = da.from_array(cupy.array([[10, 20, 30], [40, 50, 60]]), chunks=chunks)
    idx0 = da.from_array(1, chunks=1)
    assert_eq(x[idx0, :], x[1, :])
    assert_eq(x[:, idx0], x[:, 1])
    idx0 = da.from_array(cupy.array(1), chunks=1)
    assert_eq(x[idx0, :], x[1, :])
    assert_eq(x[:, idx0], x[:, 1])

@pytest.mark.skip("dask.Array.nonzero() doesn't support non-NumPy arrays yet")
@pytest.mark.parametrize('chunks', [1, 2, 3, 4, 5])
def test_index_with_int_dask_array_nanchunks(chunks):
    if False:
        while True:
            i = 10
    a = da.from_array(cupy.arange(-2, 3), chunks=chunks)
    assert_eq(a[a.nonzero()], cupy.array([-2, -1, 1, 2]))
    a = da.zeros_like(cupy.array(()), shape=5, chunks=chunks)
    assert_eq(a[a.nonzero()], cupy.array([]))

@pytest.mark.parametrize('chunks', [2, 4])
def test_index_with_int_dask_array_negindex(chunks):
    if False:
        return 10
    a = da.arange(4, chunks=chunks, like=cupy.array(()))
    idx = da.from_array([-1, -4], chunks=1)
    assert_eq(a[idx], cupy.array([3, 0]))
    idx = da.from_array(cupy.array([-1, -4]), chunks=1)
    assert_eq(a[idx], cupy.array([3, 0]))

@pytest.mark.parametrize('chunks', [2, 4])
def test_index_with_int_dask_array_indexerror(chunks):
    if False:
        print('Hello World!')
    a = da.arange(4, chunks=chunks, like=cupy.array(()))
    idx = da.from_array([4], chunks=1)
    with pytest.raises(IndexError):
        a[idx].compute()
    idx = da.from_array([-5], chunks=1)
    with pytest.raises(IndexError):
        a[idx].compute()
    idx = da.from_array(cupy.array([4]), chunks=1)
    with pytest.raises(IndexError):
        a[idx].compute()
    idx = da.from_array(cupy.array([-5]), chunks=1)
    with pytest.raises(IndexError):
        a[idx].compute()

@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'])
def test_index_with_int_dask_array_dtypes(dtype):
    if False:
        i = 10
        return i + 15
    a = da.from_array(cupy.array([10, 20, 30, 40]), chunks=-1)
    idx = da.from_array(np.array([1, 2]).astype(dtype), chunks=1)
    assert_eq(a[idx], cupy.array([20, 30]))
    idx = da.from_array(cupy.array([1, 2]).astype(dtype), chunks=1)
    assert_eq(a[idx], cupy.array([20, 30]))

def test_index_with_int_dask_array_nocompute():
    if False:
        while True:
            i = 10
    'Test that when the indices are a dask array\n    they are not accidentally computed\n    '

    def crash():
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()
    x = da.arange(5, chunks=-1, like=cupy.array(()))
    idx = da.Array({('x', 0): (crash,)}, name='x', chunks=((2,),), dtype=np.int64)
    result = x[idx]
    with pytest.raises(NotImplementedError):
        result.compute()