"""
Created on Wed May 04 06:09:18 2011

@author: josef
"""
import numpy as np

def mean_residual_life(x, frac=None, alpha=0.05):
    if False:
        for i in range(10):
            print('nop')
    'empirical mean residual life or expected shortfall\n\n    Parameters\n    ----------\n    x : 1-dimensional array_like\n    frac : list[float], optional\n        All entries must be between 0 and 1\n    alpha : float, default 0.05\n        FIXME: not actually used.\n\n    TODO:\n        check formula for std of mean\n        does not include case for all observations\n        last observations std is zero\n        vectorize loop using cumsum\n        frac does not work yet\n    '
    axis = 0
    x = np.asarray(x)
    nobs = x.shape[axis]
    xsorted = np.sort(x, axis=axis)
    if frac is None:
        xthreshold = xsorted
    else:
        xthreshold = xsorted[np.floor(nobs * frac).astype(int)]
    xlargerindex = np.searchsorted(xsorted, xthreshold, side='right')
    result = []
    for i in range(len(xthreshold) - 1):
        k_ind = xlargerindex[i]
        rmean = x[k_ind:].mean()
        rstd = x[k_ind:].std()
        rmstd = rstd / np.sqrt(nobs - k_ind)
        result.append((k_ind, xthreshold[i], rmean, rmstd))
    res = np.array(result)
    crit = 1.96
    confint = res[:, 1:2] + crit * res[:, -1:] * np.array([[-1, 1]])
    return np.column_stack((res, confint))
expected_shortfall = mean_residual_life
if __name__ == '__main__':
    rvs = np.random.standard_t(5, size=10)
    res = mean_residual_life(rvs)
    print(res)
    rmean = [rvs[i:].mean() for i in range(len(rvs))]
    print(res[:, 2] - rmean[1:])
    res_frac = mean_residual_life(rvs, frac=[0.5])
    print(res_frac)