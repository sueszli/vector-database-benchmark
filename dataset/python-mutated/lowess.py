import numpy as np
import pylab
import seaborn as sns
import statsmodels.api as sm
sns.set_style('darkgrid')
pylab.rc('figure', figsize=(16, 8))
pylab.rc('font', size=14)
np.random.seed(1)
x = np.random.uniform(0, 4 * np.pi, size=200)
y = np.cos(x) + np.random.random(size=len(x))
smoothed = sm.nonparametric.lowess(exog=x, endog=y, frac=0.2)
(fig, ax) = pylab.subplots()
ax.scatter(x, y)
ax.plot(smoothed[:, 0], smoothed[:, 1], c='k')
pylab.autoscale(enable=True, axis='x', tight=True)

def lowess_with_confidence_bounds(x, y, eval_x, N=200, conf_interval=0.95, lowess_kw=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Perform Lowess regression and determine a confidence interval by bootstrap resampling\n    '
    smoothed = sm.nonparametric.lowess(exog=x, endog=y, xvals=eval_x, **lowess_kw)
    smoothed_values = np.empty((N, len(eval_x)))
    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]
        smoothed_values[i] = sm.nonparametric.lowess(exog=sampled_x, endog=sampled_y, xvals=eval_x, **lowess_kw)
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]
    return (smoothed, bottom, top)
eval_x = np.linspace(0, 4 * np.pi, 31)
(smoothed, bottom, top) = lowess_with_confidence_bounds(x, y, eval_x, lowess_kw={'frac': 0.1})
(fig, ax) = pylab.subplots()
ax.scatter(x, y)
ax.plot(eval_x, smoothed, c='k')
ax.fill_between(eval_x, bottom, top, alpha=0.5, color='b')
pylab.autoscale(enable=True, axis='x', tight=True)