import numpy as np
import pandas as pd
from plotnine import aes, geom_qq, geom_qq_line, ggplot
random_state = np.random.RandomState(1234567890)
normal_data = pd.DataFrame({'x': random_state.normal(size=100)})

def test_normal():
    if False:
        for i in range(10):
            print('nop')
    p = ggplot(normal_data, aes(sample='x')) + geom_qq()
    assert p == 'normal'

def test_normal_with_line():
    if False:
        i = 10
        return i + 15
    p = ggplot(normal_data, aes(sample='x')) + geom_qq() + geom_qq_line()
    assert p == 'normal_with_line'