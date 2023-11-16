import numpy as np
import vaex

def test_to_arrow_mixed_masked():
    if False:
        return 10
    x = np.ma.MaskedArray(data=np.arange(10))
    df = vaex.from_arrays(x=x)
    df['y'] = df.x ** 2
    assert df.to_arrow_table(['x', 'y'])['y'].to_pylist() == df.y.tolist()