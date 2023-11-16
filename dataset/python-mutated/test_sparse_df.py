import numpy as np
from cudf import Series

def test_to_dense_array():
    if False:
        for i in range(10):
            print('nop')
    data = np.random.random(8)
    mask = np.asarray([214]).astype(np.byte)
    sr = Series.from_masked_array(data=data, mask=mask, null_count=3)
    assert sr.has_nulls
    assert sr.null_count != len(sr)
    filled = sr.to_numpy(na_value=np.nan)
    dense = sr.dropna().to_numpy()
    assert dense.size < filled.size
    assert filled.size == len(sr)