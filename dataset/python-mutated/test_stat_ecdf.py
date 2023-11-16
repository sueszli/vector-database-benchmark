import pandas as pd
from plotnine import aes, after_stat, ggplot, stat_ecdf
data = pd.DataFrame({'x': range(10)})
p = ggplot(data, aes('x')) + stat_ecdf(size=2)

def test_ecdf():
    if False:
        print('Hello World!')
    p = ggplot(data, aes('x')) + stat_ecdf(size=2)
    assert p == 'ecdf'

def test_computed_y_column():
    if False:
        i = 10
        return i + 15
    p = ggplot(data, aes('x')) + stat_ecdf(size=2) + stat_ecdf(aes(y=after_stat('ecdf-0.2')), size=2, color='blue')
    assert p == 'computed_y_column'