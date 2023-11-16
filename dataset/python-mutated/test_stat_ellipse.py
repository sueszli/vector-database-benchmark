import numpy as np
import numpy.testing as npt
import pandas as pd
from plotnine import aes, geom_point, ggplot, stat_ellipse
from plotnine.stats.stat_ellipse import cov_trob
data = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [1, 4, 3, 6, 7], 'z': [3, 4, 3, 2, 6]})

def test_ellipse():
    if False:
        print('Hello World!')
    p = ggplot(data, aes('x', 'y')) + geom_point() + stat_ellipse(type='t') + stat_ellipse(type='norm', color='red') + stat_ellipse(type='euclid', color='blue')
    assert p == 'ellipse'

def test_cov_trob_2d():
    if False:
        return 10
    x = np.array(data[['x', 'y']])
    res = cov_trob(x, cor=True)
    n_obs = 5
    center = [3.11013, 4.359847]
    cov = [[1.979174, 2.812684], [2.812684, 4.584488]]
    cor = [[1.0, 0.9337562], [0.9337562, 1.0]]
    assert res['n_obs'] == n_obs
    npt.assert_allclose(res['center'], center)
    npt.assert_allclose(res['cov'], cov)
    npt.assert_allclose(res['cor'], cor)

def test_cov_trob_3d():
    if False:
        return 10
    x = np.array(data[['x', 'y', 'z']])
    res = cov_trob(x, cor=True)
    n_obs = 5
    center = [2.8445, 3.930879, 3.54319]
    cov = [[1.9412275, 2.713547, 0.7242778], [2.7135469, 4.479363, 1.2210262], [0.7242778, 1.221026, 1.6008466]]
    cor = [[1.0, 0.9202185, 0.4108583], [0.9202185, 1.0, 0.455976], [0.4108583, 0.455976, 1.0]]
    assert res['n_obs'] == n_obs
    npt.assert_allclose(res['center'], center, rtol=1e-06)
    npt.assert_allclose(res['cov'], cov, rtol=1e-06)
    npt.assert_allclose(res['cor'], cor, rtol=1e-06)