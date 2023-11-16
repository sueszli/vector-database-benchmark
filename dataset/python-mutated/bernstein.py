"""
Created on Wed Feb 17 15:35:23 2021

Author: Josef Perktold
License: BSD-3

"""
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.distributions.tools import _Grid, cdf2prob_grid, prob2cdf_grid, _eval_bernstein_dd, _eval_bernstein_2d, _eval_bernstein_1d

class BernsteinDistribution:
    """Distribution based on Bernstein Polynomials on unit hypercube.

    Parameters
    ----------
    cdf_grid : array_like
        cdf values on a equal spaced grid of the unit hypercube [0, 1]^d.
        The dimension of the arrays define how many random variables are
        included in the multivariate distribution.

    Attributes
    ----------
    cdf_grid : grid of cdf values
    prob_grid : grid of cell or bin probabilities
    k_dim : (int) number of components, dimension of random variable
    k_grid : (tuple) shape of cdf_grid
    k_grid_product : (int) total number of bins in grid
    _grid : Grid instance with helper methods and attributes
    """

    def __init__(self, cdf_grid):
        if False:
            for i in range(10):
                print('nop')
        self.cdf_grid = cdf_grid = np.asarray(cdf_grid)
        self.k_dim = cdf_grid.ndim
        self.k_grid = cdf_grid.shape
        self.k_grid_product = np.prod([i - 1 for i in self.k_grid])
        self._grid = _Grid(self.k_grid)

    @classmethod
    def from_data(cls, data, k_bins):
        if False:
            for i in range(10):
                print('nop')
        'Create distribution instance from data using histogram binning.\n\n        Classmethod to construct a distribution instance.\n\n        Parameters\n        ----------\n        data : array_like\n            Data with observation in rows and random variables in columns.\n            Data can be 1-dimensional in the univariate case.\n        k_bins : int or list\n            Number or edges of bins to be used in numpy histogramdd.\n            If k_bins is a scalar int, then the number of bins of each\n            component will be equal to it.\n\n        Returns\n        -------\n        Instance of a Bernstein distribution\n        '
        data = np.asarray(data)
        if np.any(data < 0) or np.any(data > 1):
            raise ValueError('data needs to be in [0, 1]')
        if data.ndim == 1:
            data = data[:, None]
        k_dim = data.shape[1]
        if np.size(k_bins) == 1:
            k_bins = [k_bins] * k_dim
        bins = [np.linspace(-1 / ni, 1, ni + 2) for ni in k_bins]
        (c, e) = np.histogramdd(data, bins=bins, density=False)
        assert all([ei[1] == 0 for ei in e])
        c /= len(data)
        cdf_grid = prob2cdf_grid(c)
        return cls(cdf_grid)

    @cache_readonly
    def prob_grid(self):
        if False:
            while True:
                i = 10
        return cdf2prob_grid(self.cdf_grid, prepend=None)

    def cdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        'cdf values evaluated at x.\n\n        Parameters\n        ----------\n        x : array_like\n            Points of multivariate random variable at which cdf is evaluated.\n            This can be a single point with length equal to the dimension of\n            the random variable, or two dimensional with points (observations)\n            in rows and random variables in columns.\n            In the univariate case, a 1-dimensional x will be interpreted as\n            different points for evaluation.\n\n        Returns\n        -------\n        pdf values\n\n        Notes\n        -----\n        Warning: 2-dim x with many points can be memory intensive because\n        currently the bernstein polynomials will be evaluated in a fully\n        vectorized computation.\n        '
        x = np.asarray(x)
        if x.ndim == 1 and self.k_dim == 1:
            x = x[:, None]
        cdf_ = _eval_bernstein_dd(x, self.cdf_grid)
        return cdf_

    def pdf(self, x):
        if False:
            i = 10
            return i + 15
        'pdf values evaluated at x.\n\n        Parameters\n        ----------\n        x : array_like\n            Points of multivariate random variable at which pdf is evaluated.\n            This can be a single point with length equal to the dimension of\n            the random variable, or two dimensional with points (observations)\n            in rows and random variables in columns.\n            In the univariate case, a 1-dimensional x will be interpreted as\n            different points for evaluation.\n\n        Returns\n        -------\n        cdf values\n\n        Notes\n        -----\n        Warning: 2-dim x with many points can be memory intensive because\n        currently the bernstein polynomials will be evaluated in a fully\n        vectorized computation.\n        '
        x = np.asarray(x)
        if x.ndim == 1 and self.k_dim == 1:
            x = x[:, None]
        pdf_ = self.k_grid_product * _eval_bernstein_dd(x, self.prob_grid)
        return pdf_

    def get_marginal(self, idx):
        if False:
            for i in range(10):
                print('nop')
        'Get marginal BernsteinDistribution.\n\n        Parameters\n        ----------\n        idx : int or list of int\n            Index or indices of the component for which the marginal\n            distribution is returned.\n\n        Returns\n        -------\n        BernsteinDistribution instance for the marginal distribution.\n        '
        if self.k_dim == 1:
            return self
        sl = [-1] * self.k_dim
        if np.shape(idx) == ():
            idx = [idx]
        for ii in idx:
            sl[ii] = slice(None, None, None)
        cdf_m = self.cdf_grid[tuple(sl)]
        bpd_marginal = BernsteinDistribution(cdf_m)
        return bpd_marginal

    def rvs(self, nobs):
        if False:
            while True:
                i = 10
        'Generate random numbers from distribution.\n\n        Parameters\n        ----------\n        nobs : int\n            Number of random observations to generate.\n        '
        rvs_mnl = np.random.multinomial(nobs, self.prob_grid.flatten())
        k_comp = self.k_dim
        rvs_m = []
        for i in range(len(rvs_mnl)):
            if rvs_mnl[i] != 0:
                idx = np.unravel_index(i, self.prob_grid.shape)
                rvsi = []
                for j in range(k_comp):
                    n = self.k_grid[j]
                    xgi = self._grid.x_marginal[j][idx[j]]
                    rvsi.append(stats.beta.rvs(n * xgi + 1, n * (1 - xgi) + 0, size=rvs_mnl[i]))
                rvs_m.append(np.column_stack(rvsi))
        rvsm = np.concatenate(rvs_m)
        return rvsm

class BernsteinDistributionBV(BernsteinDistribution):

    def cdf(self, x):
        if False:
            while True:
                i = 10
        cdf_ = _eval_bernstein_2d(x, self.cdf_grid)
        return cdf_

    def pdf(self, x):
        if False:
            print('Hello World!')
        pdf_ = self.k_grid_product * _eval_bernstein_2d(x, self.prob_grid)
        return pdf_

class BernsteinDistributionUV(BernsteinDistribution):

    def cdf(self, x, method='binom'):
        if False:
            for i in range(10):
                print('nop')
        cdf_ = _eval_bernstein_1d(x, self.cdf_grid, method=method)
        return cdf_

    def pdf(self, x, method='binom'):
        if False:
            i = 10
            return i + 15
        pdf_ = self.k_grid_product * _eval_bernstein_1d(x, self.prob_grid, method=method)
        return pdf_