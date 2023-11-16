import numpy as np
import pandas as pd
import pytest
from plotnine import aes, geom_density, ggplot, lims
from plotnine.exceptions import PlotnineWarning
n = 6
data = pd.DataFrame({'x': np.repeat(range(n + 1), range(n + 1)), 'z': np.repeat(range(n // 2), range(3, n * 2, 4))})
p = ggplot(data, aes('x', fill='factor(z)'))

def test_gaussian():
    if False:
        return 10
    p1 = p + geom_density(kernel='gaussian', alpha=0.3)
    assert p1 == 'gaussian'

def test_gaussian_weighted():
    if False:
        i = 10
        return i + 15
    p1 = p + geom_density(aes(weight='x'), kernel='gaussian', alpha=0.3)
    assert p1 == 'gaussian_weighted'

def test_gaussian_trimmed():
    if False:
        return 10
    p2 = p + geom_density(kernel='gaussian', alpha=0.3, trim=True)
    assert p2 == 'gaussian-trimmed'

def test_triangular():
    if False:
        print('Hello World!')
    p3 = p + geom_density(kernel='triangular', bw='normal_reference', alpha=0.3)
    assert p3 == 'triangular'

def test_few_datapoints():
    if False:
        i = 10
        return i + 15
    data = pd.DataFrame({'x': [1, 2, 2, 3, 3, 3], 'z': list('abbccc')})
    p = ggplot(data, aes('x', color='z')) + geom_density() + lims(x=(-3, 9))
    with pytest.warns(PlotnineWarning) as record:
        p.draw_test()
    record = list(record)
    assert any(('e.g `bw=0.1`' in str(r.message) for r in record))
    assert any(('Groups with fewer than 2' in str(r.message) for r in record))
    p = ggplot(data, aes('x', color='z')) + geom_density(bw=0.1) + lims(x=(0, 4))
    assert p == 'few_datapoints'