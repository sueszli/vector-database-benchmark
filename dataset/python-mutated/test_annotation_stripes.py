import numpy as np
import pandas as pd
import pytest
from plotnine import aes, annotation_stripes, coord_flip, facet_wrap, geom_point, geom_vline, ggplot
n = 9
data = pd.DataFrame({'x': np.arange(n) % 3 + 1, 'y': range(n)})

def test_annotation_stripes():
    if False:
        return 10
    p = ggplot(data) + annotation_stripes(fill_range='no') + geom_point(aes('factor(x)', 'y')) + geom_vline(xintercept=[0.5, 1.5, 2.5, 3.5])
    assert p == 'annotation_stripes'

def test_annotation_stripes_faceting():
    if False:
        while True:
            i = 10
    n = len(data)
    data2 = pd.DataFrame({'x': np.hstack([data['x'], data['x']]), 'y': np.hstack([data['y'], data['y']]), 'g': list('a' * n + 'b' * n)})
    p = ggplot() + annotation_stripes(fill_range='no') + geom_point(data2, aes('factor(x)', 'y')) + geom_vline(xintercept=[0.5, 1.5, 2.5, 3.5]) + facet_wrap('g')
    assert p == 'annotation_stripes_faceting'

def test_annotation_stripes_fill_range():
    if False:
        for i in range(10):
            print('nop')
    p = ggplot(data) + annotation_stripes() + geom_point(aes('factor(x)', 'y')) + geom_vline(xintercept=[0.5, 1.5, 2.5, 3.5])
    assert p == 'annotation_stripes_fill_range'

def test_annotation_stripes_fill_range_cycle():
    if False:
        while True:
            i = 10
    p = ggplot(data) + annotation_stripes(fill_range='cycle') + geom_point(aes('factor(x)', 'y')) + geom_vline(xintercept=[0.5, 1.5, 2.5, 3.5])
    assert p == 'annotation_stripes_fill_range_cycle'

def test_annotation_stripes_coord_flip():
    if False:
        print('Hello World!')
    p = ggplot(data) + annotation_stripes(fill_range='no') + geom_point(aes('factor(x)', 'y')) + geom_vline(xintercept=[0.5, 1.5, 2.5, 3.5]) + coord_flip()
    assert p == 'annotation_stripes_coord_flip'

def test_annotation_stripes_continuous_scale():
    if False:
        i = 10
        return i + 15
    p = ggplot(data) + annotation_stripes() + geom_point(aes('x', 'y')) + geom_vline(xintercept=[0.5, 1.5, 2.5, 3.5])
    assert p == 'annotation_stripes_continuous_scale'

def test_invalid_orientation():
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        annotation_stripes(direction='diagonal')

def test_annotation_stripes_fill_direction_extend():
    if False:
        while True:
            i = 10
    p = ggplot(data) + annotation_stripes(fill=['red', 'blue', 'green'], fill_range='no', direction='horizontal', extend=(0.15, 0.85), alpha=0.25) + geom_point(aes('factor(x)', 'factor(y)'))
    assert p == 'annotation_stripes_fill_direction_extend'

def test_annotation_stripes_single_stripe():
    if False:
        i = 10
        return i + 15
    p = ggplot(data.assign(x=10)) + annotation_stripes(fill=['#FF0000', '#00FF00']) + geom_point(aes('factor(x)', 'y'))
    assert p == 'annotation_stripes_single_stripe'