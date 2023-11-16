import numpy as np
import pandas as pd
import pytest
from plotnine import aes, geom_point, ggplot, stat_summary
random_state = np.random.RandomState(1234567890)
data = pd.DataFrame({'x': list('aaaaabbbbcccccc'), 'y': [1, 2, 3, 4, 5, 1.5, 1.5, 6, 6, 5, 5, 5, 5, 5, 5]})

def test_mean_cl_boot():
    if False:
        while True:
            i = 10
    p = ggplot(data, aes('x', 'y')) + stat_summary(fun_data='mean_cl_boot', random_state=random_state, size=2)
    assert p == 'mean_cl_boot'

def test_mean_cl_normal():
    if False:
        i = 10
        return i + 15
    p = ggplot(data, aes('x', 'y')) + stat_summary(fun_data='mean_cl_normal', size=2)
    assert p == 'mean_cl_normal'

def test_mean_sdl():
    if False:
        print('Hello World!')
    p = ggplot(data, aes('x', 'y')) + stat_summary(fun_data='mean_sdl', size=2)
    assert p == 'mean_sdl'

def test_median_hilow():
    if False:
        for i in range(10):
            print('nop')
    p = ggplot(data, aes('x', 'y')) + stat_summary(fun_data='median_hilow', size=2)
    assert p == 'median_hilow'

def test_mean_se():
    if False:
        i = 10
        return i + 15
    p = ggplot(data, aes('x', 'y')) + stat_summary(fun_data='mean_se', size=2)
    assert p == 'mean_se'

def test_funargs():
    if False:
        i = 10
        return i + 15
    p = ggplot(data, aes('x', 'y')) + stat_summary(fun_data='mean_cl_normal', size=2, color='blue') + stat_summary(fun_data='mean_cl_normal', fun_args={'confidence_interval': 0.5}, size=2, color='green')
    assert p == 'fun_args'

def test_summary_functions():
    if False:
        i = 10
        return i + 15
    p = ggplot(data, aes('x', 'y')) + stat_summary(fun_y=np.mean, fun_ymin=np.min, fun_ymax=np.max, size=2)
    assert p == 'summary_functions'

def test_stat_summary_raises_on_invalid_paremeters():
    if False:
        while True:
            i = 10
    with pytest.raises(TypeError):
        geom_point(stat_summary(funy=np.mean))
    with pytest.raises(TypeError):
        geom_point(stat_summary(does_not_exist=1))