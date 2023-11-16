import numpy as np
import megengine.functional as F
from megengine import Tensor
from megengine.core._trace_option import use_symbolic_shape

def test_zero_dim():
    if False:
        i = 10
        return i + 15
    a = Tensor(1)
    a_np = np.array(1, dtype=np.int32)
    np.testing.assert_equal(a, a_np)
    if use_symbolic_shape():
        np.testing.assert_equal(a.shape, np.array(a_np.shape))
    else:
        np.testing.assert_equal(a.shape, a_np.shape)

def test_sum():
    if False:
        return 10
    a = Tensor([1, 2])
    a = a.reshape((1, 2))
    assert a.sum().ndim == 0
    assert a.sum(axis=1).ndim == 1

def test_max():
    if False:
        print('Hello World!')
    a = Tensor([1, 2])
    a = a.reshape((1, 2))
    assert a.max().ndim == 0
    assert a.max(axis=1).ndim == 1

def test_reshape():
    if False:
        while True:
            i = 10
    a = Tensor(1)
    a = a.reshape((1, 1))

def test_squeeze():
    if False:
        i = 10
        return i + 15
    a = Tensor(1)
    a = a.reshape((1, 1))
    assert F.squeeze(a).ndim == 0

def test_elemementwise():
    if False:
        i = 10
        return i + 15
    a = Tensor(1.0)
    assert F.exp(a).ndim == 0
    assert (a + a).ndim == 0
    assert (a + 1).ndim == 0

def test_astype():
    if False:
        for i in range(10):
            print('nop')
    a = Tensor(1.0)
    assert a.astype('int32').ndim == 0

def test_tranpose():
    if False:
        i = 10
        return i + 15
    a = Tensor(1.0)
    assert a.transpose().ndim == 0