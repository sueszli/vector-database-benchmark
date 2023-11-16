import numpy as np
from skimage.transform import frt2, ifrt2

def test_frt():
    if False:
        for i in range(10):
            print('nop')
    SIZE = 59
    L = np.tri(SIZE, dtype=np.int32) + np.tri(SIZE, dtype=np.int32)[::-1]
    f = frt2(L)
    fi = ifrt2(f)
    assert np.array_equal(L, fi)