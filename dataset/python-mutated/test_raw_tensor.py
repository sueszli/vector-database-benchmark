import numpy as np
import megengine.functional as F
from megengine.tensor import Tensor

def test_as_raw_tensor():
    if False:
        i = 10
        return i + 15
    x = np.arange(6, dtype='float32').reshape(2, 3)
    xx = Tensor(x, device='xpux')
    yy = F.add(xx, 1).numpy()
    assert xx.dtype == np.float32
    assert xx.device == 'xpux'
    np.testing.assert_almost_equal(yy, x + 1)

def test_as_raw_tensor_from_int64():
    if False:
        for i in range(10):
            print('nop')
    x = np.arange(6, dtype='int64').reshape(2, 3)
    xx = Tensor(x, dtype='float32', device='xpux')
    yy = F.add(xx, 1).numpy()
    assert xx.dtype == np.float32
    assert xx.device == 'xpux'
    np.testing.assert_almost_equal(yy, x.astype('float32') + 1)

def test_as_raw_tensor_uint16():
    if False:
        while True:
            i = 10
    x = np.arange(6, dtype='uint16').reshape(2, 3)
    xx = Tensor(x, device='xpux')
    assert xx.dtype == np.uint16
    assert xx.device == 'xpux'