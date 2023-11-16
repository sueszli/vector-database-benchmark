"""
Tests for tm.makeFoo functions.
"""
import numpy as np
import pandas._testing as tm

def test_make_multiindex_respects_k():
    if False:
        for i in range(10):
            print('nop')
    N = np.random.default_rng(2).integers(0, 100)
    mi = tm.makeMultiIndex(k=N)
    assert len(mi) == N