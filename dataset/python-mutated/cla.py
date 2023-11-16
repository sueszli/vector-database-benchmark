"""
The ``cla`` module houses the CLA class, which
generates optimal portfolios using the Critical Line Algorithm as implemented
by Marcos Lopez de Prado and David Bailey.
"""
import numpy as np
import pandas as pd
from . import base_optimizer

class CLA(base_optimizer.BaseOptimizer):
    """
    Instance variables:

    - Inputs:

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``mean`` - np.ndarray
        - ``cov_matrix`` - np.ndarray
        - ``expected_returns`` - np.ndarray
        - ``lb`` - np.ndarray
        - ``ub`` - np.ndarray

    - Optimization parameters:

        - ``w`` - np.ndarray list
        - ``ls`` - float list
        - ``g`` - float list
        - ``f`` - float list list

    - Outputs:

        - ``weights`` - np.ndarray
        - ``frontier_values`` - (float list, float list, np.ndarray list)

    Public methods:

    - ``max_sharpe()`` optimizes for maximal Sharpe ratio (a.k.a the tangency portfolio)
    - ``min_volatility()`` optimizes for minimum volatility
    - ``efficient_frontier()`` computes the entire efficient frontier
    - ``portfolio_performance()`` calculates the expected return, volatility and Sharpe ratio for
      the optimized portfolio.
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, expected_returns, cov_matrix, weight_bounds=(0, 1)):
        if False:
            i = 10
            return i + 15
        '\n        :param expected_returns: expected returns for each asset. Set to None if\n                                 optimising for volatility only.\n        :type expected_returns: pd.Series, list, np.ndarray\n        :param cov_matrix: covariance of returns for each asset\n        :type cov_matrix: pd.DataFrame or np.array\n        :param weight_bounds: minimum and maximum weight of an asset, defaults to (0, 1).\n                              Must be changed to (-1, 1) for portfolios with shorting.\n        :type weight_bounds: tuple (float, float) or (list/ndarray, list/ndarray) or list(tuple(float, float))\n        :raises TypeError: if ``expected_returns`` is not a series, list or array\n        :raises TypeError: if ``cov_matrix`` is not a dataframe or array\n        '
        self.mean = np.array(expected_returns).reshape((len(expected_returns), 1))
        self.expected_returns = self.mean.reshape((len(self.mean),))
        self.cov_matrix = np.asarray(cov_matrix)
        if len(weight_bounds) == len(self.mean) and (not isinstance(weight_bounds[0], (float, int))):
            self.lB = np.array([b[0] for b in weight_bounds]).reshape(-1, 1)
            self.uB = np.array([b[1] for b in weight_bounds]).reshape(-1, 1)
        else:
            if isinstance(weight_bounds[0], (float, int)):
                self.lB = np.ones(self.mean.shape) * weight_bounds[0]
            else:
                self.lB = np.array(weight_bounds[0]).reshape(self.mean.shape)
            if isinstance(weight_bounds[0], (float, int)):
                self.uB = np.ones(self.mean.shape) * weight_bounds[1]
            else:
                self.uB = np.array(weight_bounds[1]).reshape(self.mean.shape)
        self.w = []
        self.ls = []
        self.g = []
        self.f = []
        self.frontier_values = None
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        else:
            tickers = list(range(len(self.mean)))
        super().__init__(len(tickers), tickers)

    @staticmethod
    def _infnone(x):
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper method to map None to float infinity.\n\n        :param x: argument\n        :type x: float\n        :return: infinity if the argument was None otherwise x\n        :rtype: float\n        '
        return float('-inf') if x is None else x

    def _init_algo(self):
        if False:
            i = 10
            return i + 15
        a = np.zeros(self.mean.shape[0], dtype=[('id', int), ('mu', float)])
        b = [self.mean[i][0] for i in range(self.mean.shape[0])]
        a[:] = list(zip(list(range(self.mean.shape[0])), b))
        b = np.sort(a, order='mu')
        (i, w) = (b.shape[0], np.copy(self.lB))
        while sum(w) < 1:
            i -= 1
            w[b[i][0]] = self.uB[b[i][0]]
        w[b[i][0]] += 1 - sum(w)
        return ([b[i][0]], w)

    def _compute_bi(self, c, bi):
        if False:
            i = 10
            return i + 15
        if c > 0:
            bi = bi[1][0]
        if c < 0:
            bi = bi[0][0]
        return bi

    def _compute_w(self, covarF_inv, covarFB, meanF, wB):
        if False:
            while True:
                i = 10
        onesF = np.ones(meanF.shape)
        g1 = np.dot(np.dot(onesF.T, covarF_inv), meanF)
        g2 = np.dot(np.dot(onesF.T, covarF_inv), onesF)
        if wB is None:
            (g, w1) = (float(-self.ls[-1] * g1 / g2 + 1 / g2), 0)
        else:
            onesB = np.ones(wB.shape)
            g3 = np.dot(onesB.T, wB)
            g4 = np.dot(covarF_inv, covarFB)
            w1 = np.dot(g4, wB)
            g4 = np.dot(onesF.T, w1)
            g = float(-self.ls[-1] * g1 / g2 + (1 - g3 + g4) / g2)
        w2 = np.dot(covarF_inv, onesF)
        w3 = np.dot(covarF_inv, meanF)
        return (-w1 + g * w2 + self.ls[-1] * w3, g)

    def _compute_lambda(self, covarF_inv, covarFB, meanF, wB, i, bi):
        if False:
            while True:
                i = 10
        onesF = np.ones(meanF.shape)
        c1 = np.dot(np.dot(onesF.T, covarF_inv), onesF)
        c2 = np.dot(covarF_inv, meanF)
        c3 = np.dot(np.dot(onesF.T, covarF_inv), meanF)
        c4 = np.dot(covarF_inv, onesF)
        c = -c1 * c2[i] + c3 * c4[i]
        if c == 0:
            return (None, None)
        if type(bi) == list:
            bi = self._compute_bi(c, bi)
        if wB is None:
            return (float((c4[i] - c1 * bi) / c), bi)
        else:
            onesB = np.ones(wB.shape)
            l1 = np.dot(onesB.T, wB)
            l2 = np.dot(covarF_inv, covarFB)
            l3 = np.dot(l2, wB)
            l2 = np.dot(onesF.T, l3)
            return (float(((1 - l1 + l2) * c4[i] - c1 * (bi + l3[i])) / c), bi)

    def _get_matrices(self, f):
        if False:
            for i in range(10):
                print('nop')
        covarF = self._reduce_matrix(self.cov_matrix, f, f)
        meanF = self._reduce_matrix(self.mean, f, [0])
        b = self._get_b(f)
        covarFB = self._reduce_matrix(self.cov_matrix, f, b)
        wB = self._reduce_matrix(self.w[-1], b, [0])
        return (covarF, covarFB, meanF, wB)

    def _get_b(self, f):
        if False:
            for i in range(10):
                print('nop')
        return self._diff_lists(list(range(self.mean.shape[0])), f)

    @staticmethod
    def _diff_lists(list1, list2):
        if False:
            return 10
        return list(set(list1) - set(list2))

    @staticmethod
    def _reduce_matrix(matrix, listX, listY):
        if False:
            print('Hello World!')
        if len(listX) == 0 or len(listY) == 0:
            return
        matrix_ = matrix[:, listY[0]:listY[0] + 1]
        for i in listY[1:]:
            a = matrix[:, i:i + 1]
            matrix_ = np.append(matrix_, a, 1)
        matrix__ = matrix_[listX[0]:listX[0] + 1, :]
        for i in listX[1:]:
            a = matrix_[i:i + 1, :]
            matrix__ = np.append(matrix__, a, 0)
        return matrix__

    def _purge_num_err(self, tol):
        if False:
            i = 10
            return i + 15
        i = 0
        while True:
            flag = False
            if i == len(self.w):
                break
            if abs(sum(self.w[i]) - 1) > tol:
                flag = True
            else:
                for j in range(self.w[i].shape[0]):
                    if self.w[i][j] - self.lB[j] < -tol or self.w[i][j] - self.uB[j] > tol:
                        flag = True
                        break
            if flag is True:
                del self.w[i]
                del self.ls[i]
                del self.g[i]
                del self.f[i]
            else:
                i += 1

    def _purge_excess(self):
        if False:
            i = 10
            return i + 15
        (i, repeat) = (0, False)
        while True:
            if repeat is False:
                i += 1
            if i == len(self.w) - 1:
                break
            w = self.w[i]
            mu = np.dot(w.T, self.mean)[0, 0]
            (j, repeat) = (i + 1, False)
            while True:
                if j == len(self.w):
                    break
                w = self.w[j]
                mu_ = np.dot(w.T, self.mean)[0, 0]
                if mu < mu_:
                    del self.w[i]
                    del self.ls[i]
                    del self.g[i]
                    del self.f[i]
                    repeat = True
                    break
                else:
                    j += 1

    def _golden_section(self, obj, a, b, **kargs):
        if False:
            for i in range(10):
                print('nop')
        (tol, sign, args) = (1e-09, 1, None)
        if 'minimum' in kargs and kargs['minimum'] is False:
            sign = -1
        if 'args' in kargs:
            args = kargs['args']
        numIter = int(np.ceil(-2.078087 * np.log(tol / abs(b - a))))
        r = 0.618033989
        c = 1.0 - r
        x1 = r * a + c * b
        x2 = c * a + r * b
        f1 = sign * obj(x1, *args)
        f2 = sign * obj(x2, *args)
        for i in range(numIter):
            if f1 > f2:
                a = x1
                x1 = x2
                f1 = f2
                x2 = c * a + r * b
                f2 = sign * obj(x2, *args)
            else:
                b = x2
                x2 = x1
                f2 = f1
                x1 = r * a + c * b
                f1 = sign * obj(x1, *args)
        if f1 < f2:
            return (x1, sign * f1)
        else:
            return (x2, sign * f2)

    def _eval_sr(self, a, w0, w1):
        if False:
            while True:
                i = 10
        w = a * w0 + (1 - a) * w1
        b = np.dot(w.T, self.mean)[0, 0]
        c = np.dot(np.dot(w.T, self.cov_matrix), w)[0, 0] ** 0.5
        return b / c

    def _solve(self):
        if False:
            for i in range(10):
                print('nop')
        (f, w) = self._init_algo()
        self.w.append(np.copy(w))
        self.ls.append(None)
        self.g.append(None)
        self.f.append(f[:])
        while True:
            l_in = None
            if len(f) > 1:
                (covarF, covarFB, meanF, wB) = self._get_matrices(f)
                covarF_inv = np.linalg.inv(covarF)
                j = 0
                for i in f:
                    (l, bi) = self._compute_lambda(covarF_inv, covarFB, meanF, wB, j, [self.lB[i], self.uB[i]])
                    if CLA._infnone(l) > CLA._infnone(l_in):
                        (l_in, i_in, bi_in) = (l, i, bi)
                    j += 1
            l_out = None
            if len(f) < self.mean.shape[0]:
                b = self._get_b(f)
                for i in b:
                    (covarF, covarFB, meanF, wB) = self._get_matrices(f + [i])
                    covarF_inv = np.linalg.inv(covarF)
                    (l, bi) = self._compute_lambda(covarF_inv, covarFB, meanF, wB, meanF.shape[0] - 1, self.w[-1][i])
                    if (self.ls[-1] is None or l < self.ls[-1]) and l > CLA._infnone(l_out):
                        (l_out, i_out) = (l, i)
            if (l_in is None or l_in < 0) and (l_out is None or l_out < 0):
                self.ls.append(0)
                (covarF, covarFB, meanF, wB) = self._get_matrices(f)
                covarF_inv = np.linalg.inv(covarF)
                meanF = np.zeros(meanF.shape)
            else:
                if CLA._infnone(l_in) > CLA._infnone(l_out):
                    self.ls.append(l_in)
                    f.remove(i_in)
                    w[i_in] = bi_in
                else:
                    self.ls.append(l_out)
                    f.append(i_out)
                (covarF, covarFB, meanF, wB) = self._get_matrices(f)
                covarF_inv = np.linalg.inv(covarF)
            (wF, g) = self._compute_w(covarF_inv, covarFB, meanF, wB)
            for i in range(len(f)):
                w[f[i]] = wF[i]
            self.w.append(np.copy(w))
            self.g.append(g)
            self.f.append(f[:])
            if self.ls[-1] == 0:
                break
        self._purge_num_err(1e-09)
        self._purge_excess()

    def max_sharpe(self):
        if False:
            while True:
                i = 10
        '\n        Maximise the Sharpe ratio.\n\n        :return: asset weights for the max-sharpe portfolio\n        :rtype: OrderedDict\n        '
        if not self.w:
            self._solve()
        (w_sr, sr) = ([], [])
        for i in range(len(self.w) - 1):
            w0 = np.copy(self.w[i])
            w1 = np.copy(self.w[i + 1])
            kargs = {'minimum': False, 'args': (w0, w1)}
            (a, b) = self._golden_section(self._eval_sr, 0, 1, **kargs)
            w_sr.append(a * w0 + (1 - a) * w1)
            sr.append(b)
        self.weights = w_sr[sr.index(max(sr))].reshape((self.n_assets,))
        return self._make_output_weights()

    def min_volatility(self):
        if False:
            return 10
        '\n        Minimise volatility.\n\n        :return: asset weights for the volatility-minimising portfolio\n        :rtype: OrderedDict\n        '
        if not self.w:
            self._solve()
        var = []
        for w in self.w:
            a = np.dot(np.dot(w.T, self.cov_matrix), w)
            var.append(a)
        self.weights = self.w[var.index(min(var))].reshape((self.n_assets,))
        return self._make_output_weights()

    def efficient_frontier(self, points=100):
        if False:
            print('Hello World!')
        '\n        Efficiently compute the entire efficient frontier\n\n        :param points: rough number of points to evaluate, defaults to 100\n        :type points: int, optional\n        :raises ValueError: if weights have not been computed\n        :return: return list, std list, weight list\n        :rtype: (float list, float list, np.ndarray list)\n        '
        if not self.w:
            self._solve()
        (mu, sigma, weights) = ([], [], [])
        a = np.linspace(0, 1, points // len(self.w))[:-1]
        b = list(range(len(self.w) - 1))
        for i in b:
            (w0, w1) = (self.w[i], self.w[i + 1])
            if i == b[-1]:
                a = np.linspace(0, 1, points // len(self.w))
            for j in a:
                w = w1 * j + (1 - j) * w0
                weights.append(np.copy(w))
                mu.append(np.dot(w.T, self.mean)[0, 0])
                sigma.append(np.dot(np.dot(w.T, self.cov_matrix), w)[0, 0] ** 0.5)
        self.frontier_values = (mu, sigma, weights)
        return (mu, sigma, weights)

    def set_weights(self, _):
        if False:
            return 10
        raise NotImplementedError('set_weights does nothing for CLA')

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        if False:
            for i in range(10):
                print('nop')
        '\n        After optimising, calculate (and optionally print) the performance of the optimal\n        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.\n\n        :param verbose: whether performance should be printed, defaults to False\n        :type verbose: bool, optional\n        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02\n        :type risk_free_rate: float, optional\n        :raises ValueError: if weights have not been calculated yet\n        :return: expected return, volatility, Sharpe ratio.\n        :rtype: (float, float, float)\n        '
        return base_optimizer.portfolio_performance(self.weights, self.expected_returns, self.cov_matrix, verbose, risk_free_rate)