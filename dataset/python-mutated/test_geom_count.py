import pandas as pd
from plotnine import aes, geom_count, ggplot
data = pd.DataFrame({'x': list('aaaaaaaaaabbbbbbbbbbcccccccccc'), 'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 1, 1, 1, 1, 6, 6, 8, 10, 10, 1, 1, 2, 4, 4, 4, 4, 9, 9, 9]})

def test_discrete_x():
    if False:
        print('Hello World!')
    p = ggplot(data, aes('x', 'y')) + geom_count()
    assert p == 'discrete_x'

def test_discrete_y():
    if False:
        i = 10
        return i + 15
    p = ggplot(data, aes('y', 'x')) + geom_count()
    assert p == 'discrete_y'

def test_continuous_x_y():
    if False:
        return 10
    p = ggplot(data, aes('y', 'y')) + geom_count()
    assert p == 'continuous_x_y'