import numpy as np
import pandas as pd
from plotnine import aes, geom_point, geom_quantile, ggplot
n = 200
random_state = np.random.RandomState(1234567890)
data = pd.DataFrame({'x': np.arange(n), 'y': np.arange(n) * (1 + random_state.rand(n))})

def test_lines():
    if False:
        return 10
    p = ggplot(data, aes(x='x', y='y')) + geom_point(alpha=0.5) + geom_quantile(quantiles=[0.001, 0.5, 0.999], formula='y~x', size=2)
    assert p == 'lines'