import numpy as np
import pandas as pd
from plotnine import aes, geom_bin_2d, ggplot, scale_x_log10
from .conftest import layer_data
n = 20
reps = np.hstack([np.arange(int(np.ceil(n / 2))), np.arange(int(np.ceil(n // 2)))[::-1]])
diagonal = np.repeat(np.arange(n), reps)
data = pd.DataFrame({'x': np.hstack([diagonal, diagonal]), 'y': np.hstack([diagonal, diagonal[::-1]])})

def test_drop_true():
    if False:
        print('Hello World!')
    p = ggplot(data, aes('x', 'y')) + geom_bin_2d(binwidth=2, drop=True)
    assert p == 'drop_true'

def test_drop_false():
    if False:
        for i in range(10):
            print('nop')
    p = ggplot(data, aes('x', 'y')) + geom_bin_2d(binwidth=2, drop=False)
    assert p == 'drop_false'

def test_scale_transformed_breaks():
    if False:
        while True:
            i = 10
    data = pd.DataFrame({'x': [1, 10, 100, 1000], 'y': range(4)})
    p = ggplot(data, aes('x', 'y')) + geom_bin_2d(breaks=([5, 50, 500], [0.5, 1.5, 2.5]))
    out1 = layer_data(p)
    out2 = layer_data(p + scale_x_log10())
    np.testing.assert_allclose(out1.xmax, [50, 500])
    np.testing.assert_allclose(out2.xmax, np.log10([50, 500]))