import numpy as np
import pandas as pd
from plotnine import aes, geom_freqpoly, geom_histogram, geom_point, ggplot
n = 10
data = pd.DataFrame({'x': np.repeat(range(n + 1), range(n + 1)), 'z': np.repeat(range(n // 2), range(3, n * 2, 4))})

def test_midpoint():
    if False:
        while True:
            i = 10
    p = ggplot(data, aes('x')) + geom_histogram(aes(fill='factor(z)'), bins=n, alpha=0.25) + geom_freqpoly(bins=n, size=4) + geom_point(stat='bin', bins=n, size=4, stroke=0, color='red')
    assert p == 'midpoint'