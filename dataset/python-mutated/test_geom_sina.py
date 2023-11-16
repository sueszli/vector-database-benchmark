import numpy as np
import pandas as pd
import pytest
from plotnine import aes, coord_flip, geom_sina, geom_violin, ggplot
n = 50
random_state = np.random.RandomState(123)
uni = random_state.normal(5, 0.25, n)
bi = np.hstack([random_state.normal(4, 0.25, n), random_state.normal(6, 0.25, n)])
tri = np.hstack([random_state.normal(4, 0.125, n), random_state.normal(5, 0.125, n), random_state.normal(6, 0.125, n)])
cats = ['uni', 'bi', 'tri']
data = pd.DataFrame({'dist': pd.Categorical(np.repeat(cats, [len(uni), len(bi), len(tri)]), categories=cats), 'value': np.hstack([uni, bi, tri])})

@pytest.mark.skip('Fix the limits')
def test_scale_area():
    if False:
        print('Hello World!')
    p = ggplot(data, aes('dist', 'value')) + geom_violin(scale='area') + geom_sina(scale='area', random_state=123)
    assert p == 'scale_area'

@pytest.mark.skip('Fix the limits')
def test_scale_count():
    if False:
        for i in range(10):
            print('nop')
    p = ggplot(data, aes('dist', 'value')) + geom_violin(scale='count') + geom_sina(scale='count', random_state=123)
    assert p == 'scale_count'

@pytest.mark.skip('Fix the limits')
def test_scale_area_coordatalip():
    if False:
        i = 10
        return i + 15
    p = ggplot(data, aes('dist', 'value')) + geom_violin(scale='area') + geom_sina(scale='area', random_state=123) + coord_flip()
    assert p == 'scale_area+coord_flip'

@pytest.mark.skip('Fix the limits')
def test_method_counts():
    if False:
        return 10
    p = ggplot(data, aes('dist', 'value')) + geom_violin() + geom_sina(method='counts', random_state=123)
    assert p == 'method_counts'