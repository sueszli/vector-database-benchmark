"""
The ``risk_models`` module provides functions for estimating the covariance matrix given
historical returns.

The format of the data input is the same as that in :ref:`expected-returns`.

**Currently implemented:**

- fix non-positive semidefinite matrices
- general risk matrix function, allowing you to run any risk model from one function.
- sample covariance
- semicovariance
- exponentially weighted covariance
- minimum covariance determinant
- shrunk covariance matrices:

    - manual shrinkage
    - Ledoit Wolf shrinkage
    - Oracle Approximating shrinkage

- covariance to correlation matrix
"""
import warnings
import numpy as np
import pandas as pd
from .expected_returns import returns_from_prices

def _is_positive_semidefinite(matrix):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function to check if a given matrix is positive semidefinite.\n    Any method that requires inverting the covariance matrix will struggle\n    with a non-positive semidefinite matrix\n\n    :param matrix: (covariance) matrix to test\n    :type matrix: np.ndarray, pd.DataFrame\n    :return: whether matrix is positive semidefinite\n    :rtype: bool\n    '
    try:
        np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
        return True
    except np.linalg.LinAlgError:
        return False

def fix_nonpositive_semidefinite(matrix, fix_method='spectral'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Check if a covariance matrix is positive semidefinite, and if not, fix it\n    with the chosen method.\n\n    The ``spectral`` method sets negative eigenvalues to zero then rebuilds the matrix,\n    while the ``diag`` method adds a small positive value to the diagonal.\n\n    :param matrix: raw covariance matrix (may not be PSD)\n    :type matrix: pd.DataFrame\n    :param fix_method: {"spectral", "diag"}, defaults to "spectral"\n    :type fix_method: str, optional\n    :raises NotImplementedError: if a method is passed that isn\'t implemented\n    :return: positive semidefinite covariance matrix\n    :rtype: pd.DataFrame\n    '
    if _is_positive_semidefinite(matrix):
        return matrix
    warnings.warn('The covariance matrix is non positive semidefinite. Amending eigenvalues.')
    (q, V) = np.linalg.eigh(matrix)
    if fix_method == 'spectral':
        q = np.where(q > 0, q, 0)
        fixed_matrix = V @ np.diag(q) @ V.T
    elif fix_method == 'diag':
        min_eig = np.min(q)
        fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
    else:
        raise NotImplementedError('Method {} not implemented'.format(fix_method))
    if not _is_positive_semidefinite(fixed_matrix):
        warnings.warn('Could not fix matrix. Please try a different risk model.', UserWarning)
    if isinstance(matrix, pd.DataFrame):
        tickers = matrix.index
        return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
    else:
        return fixed_matrix

def risk_matrix(prices, method='sample_cov', **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute a covariance matrix, using the risk model supplied in the ``method``\n    parameter.\n\n    :param prices: adjusted closing prices of the asset, each row is a date\n                   and each column is a ticker/id.\n    :type prices: pd.DataFrame\n    :param returns_data: if true, the first argument is returns instead of prices.\n    :type returns_data: bool, defaults to False.\n    :param method: the risk model to use. Should be one of:\n\n        - ``sample_cov``\n        - ``semicovariance``\n        - ``exp_cov``\n        - ``ledoit_wolf``\n        - ``ledoit_wolf_constant_variance``\n        - ``ledoit_wolf_single_factor``\n        - ``ledoit_wolf_constant_correlation``\n        - ``oracle_approximating``\n\n    :type method: str, optional\n    :raises NotImplementedError: if the supplied method is not recognised\n    :return: annualised sample covariance matrix\n    :rtype: pd.DataFrame\n    '
    if method == 'sample_cov':
        return sample_cov(prices, **kwargs)
    elif method == 'semicovariance' or method == 'semivariance':
        return semicovariance(prices, **kwargs)
    elif method == 'exp_cov':
        return exp_cov(prices, **kwargs)
    elif method == 'ledoit_wolf' or method == 'ledoit_wolf_constant_variance':
        return CovarianceShrinkage(prices, **kwargs).ledoit_wolf()
    elif method == 'ledoit_wolf_single_factor':
        return CovarianceShrinkage(prices, **kwargs).ledoit_wolf(shrinkage_target='single_factor')
    elif method == 'ledoit_wolf_constant_correlation':
        return CovarianceShrinkage(prices, **kwargs).ledoit_wolf(shrinkage_target='constant_correlation')
    elif method == 'oracle_approximating':
        return CovarianceShrinkage(prices, **kwargs).oracle_approximating()
    else:
        raise NotImplementedError('Risk model {} not implemented'.format(method))

def sample_cov(prices, returns_data=False, frequency=252, log_returns=False, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Calculate the annualised sample covariance matrix of (daily) asset returns.\n\n    :param prices: adjusted closing prices of the asset, each row is a date\n                   and each column is a ticker/id.\n    :type prices: pd.DataFrame\n    :param returns_data: if true, the first argument is returns instead of prices.\n    :type returns_data: bool, defaults to False.\n    :param frequency: number of time periods in a year, defaults to 252 (the number\n                      of trading days in a year)\n    :type frequency: int, optional\n    :param log_returns: whether to compute using log returns\n    :type log_returns: bool, defaults to False\n    :return: annualised sample covariance matrix\n    :rtype: pd.DataFrame\n    '
    if not isinstance(prices, pd.DataFrame):
        warnings.warn('data is not in a dataframe', RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    return fix_nonpositive_semidefinite(returns.cov() * frequency, kwargs.get('fix_method', 'spectral'))

def semicovariance(prices, returns_data=False, benchmark=7.9e-05, frequency=252, log_returns=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Estimate the semicovariance matrix, i.e the covariance given that\n    the returns are less than the benchmark.\n\n    .. semicov = E([min(r_i - B, 0)] . [min(r_j - B, 0)])\n\n    :param prices: adjusted closing prices of the asset, each row is a date\n                   and each column is a ticker/id.\n    :type prices: pd.DataFrame\n    :param returns_data: if true, the first argument is returns instead of prices.\n    :type returns_data: bool, defaults to False.\n    :param benchmark: the benchmark return, defaults to the daily risk-free rate, i.e\n                      :math:`1.02^{(1/252)} -1`.\n    :type benchmark: float\n    :param frequency: number of time periods in a year, defaults to 252 (the number\n                      of trading days in a year). Ensure that you use the appropriate\n                      benchmark, e.g if ``frequency=12`` use the monthly risk-free rate.\n    :type frequency: int, optional\n    :param log_returns: whether to compute using log returns\n    :type log_returns: bool, defaults to False\n    :return: semicovariance matrix\n    :rtype: pd.DataFrame\n    '
    if not isinstance(prices, pd.DataFrame):
        warnings.warn('data is not in a dataframe', RuntimeWarning)
        prices = pd.DataFrame(prices)
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    drops = np.fmin(returns - benchmark, 0)
    T = drops.shape[0]
    return fix_nonpositive_semidefinite(drops.T @ drops / T * frequency, kwargs.get('fix_method', 'spectral'))

def _pair_exp_cov(X, Y, span=180):
    if False:
        return 10
    '\n    Calculate the exponential covariance between two timeseries of returns.\n\n    :param X: first time series of returns\n    :type X: pd.Series\n    :param Y: second time series of returns\n    :type Y: pd.Series\n    :param span: the span of the exponential weighting function, defaults to 180\n    :type span: int, optional\n    :return: the exponential covariance between X and Y\n    :rtype: float\n    '
    covariation = (X - X.mean()) * (Y - Y.mean())
    if span < 10:
        warnings.warn('it is recommended to use a higher span, e.g 30 days')
    return covariation.ewm(span=span).mean().iloc[-1]

def exp_cov(prices, returns_data=False, span=180, frequency=252, log_returns=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Estimate the exponentially-weighted covariance matrix, which gives\n    greater weight to more recent data.\n\n    :param prices: adjusted closing prices of the asset, each row is a date\n                   and each column is a ticker/id.\n    :type prices: pd.DataFrame\n    :param returns_data: if true, the first argument is returns instead of prices.\n    :type returns_data: bool, defaults to False.\n    :param span: the span of the exponential weighting function, defaults to 180\n    :type span: int, optional\n    :param frequency: number of time periods in a year, defaults to 252 (the number\n                      of trading days in a year)\n    :type frequency: int, optional\n    :param log_returns: whether to compute using log returns\n    :type log_returns: bool, defaults to False\n    :return: annualised estimate of exponential covariance matrix\n    :rtype: pd.DataFrame\n    '
    if not isinstance(prices, pd.DataFrame):
        warnings.warn('data is not in a dataframe', RuntimeWarning)
        prices = pd.DataFrame(prices)
    assets = prices.columns
    if returns_data:
        returns = prices
    else:
        returns = returns_from_prices(prices, log_returns)
    N = len(assets)
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            S[i, j] = S[j, i] = _pair_exp_cov(returns.iloc[:, i], returns.iloc[:, j], span)
    cov = pd.DataFrame(S * frequency, columns=assets, index=assets)
    return fix_nonpositive_semidefinite(cov, kwargs.get('fix_method', 'spectral'))

def min_cov_determinant(prices, returns_data=False, frequency=252, random_state=None, log_returns=False, **kwargs):
    if False:
        i = 10
        return i + 15
    warnings.warn('min_cov_determinant is deprecated and will be removed in v1.5')
    if not isinstance(prices, pd.DataFrame):
        warnings.warn('data is not in a dataframe', RuntimeWarning)
        prices = pd.DataFrame(prices)
    try:
        import sklearn.covariance
    except (ModuleNotFoundError, ImportError):
        raise ImportError('Please install scikit-learn via pip or poetry')
    assets = prices.columns
    if returns_data:
        X = prices
    else:
        X = returns_from_prices(prices, log_returns)
    X = X.dropna().values
    raw_cov_array = sklearn.covariance.fast_mcd(X, random_state=random_state)[1]
    cov = pd.DataFrame(raw_cov_array, index=assets, columns=assets) * frequency
    return fix_nonpositive_semidefinite(cov, kwargs.get('fix_method', 'spectral'))

def cov_to_corr(cov_matrix):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a covariance matrix to a correlation matrix.\n\n    :param cov_matrix: covariance matrix\n    :type cov_matrix: pd.DataFrame\n    :return: correlation matrix\n    :rtype: pd.DataFrame\n    '
    if not isinstance(cov_matrix, pd.DataFrame):
        warnings.warn('cov_matrix is not a dataframe', RuntimeWarning)
        cov_matrix = pd.DataFrame(cov_matrix)
    Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
    corr = np.dot(Dinv, np.dot(cov_matrix, Dinv))
    return pd.DataFrame(corr, index=cov_matrix.index, columns=cov_matrix.index)

def corr_to_cov(corr_matrix, stdevs):
    if False:
        while True:
            i = 10
    '\n    Convert a correlation matrix to a covariance matrix\n\n    :param corr_matrix: correlation matrix\n    :type corr_matrix: pd.DataFrame\n    :param stdevs: vector of standard deviations\n    :type stdevs: array-like\n    :return: covariance matrix\n    :rtype: pd.DataFrame\n    '
    if not isinstance(corr_matrix, pd.DataFrame):
        warnings.warn('corr_matrix is not a dataframe', RuntimeWarning)
        corr_matrix = pd.DataFrame(corr_matrix)
    return corr_matrix * np.outer(stdevs, stdevs)

class CovarianceShrinkage:
    """
    Provide methods for computing shrinkage estimates of the covariance matrix, using the
    sample covariance matrix and choosing the structured estimator to be an identity matrix
    multiplied by the average sample variance. The shrinkage constant can be input manually,
    though there exist methods (notably Ledoit Wolf) to estimate the optimal value.

    Instance variables:

    - ``X`` - pd.DataFrame (returns)
    - ``S`` - np.ndarray (sample covariance matrix)
    - ``delta`` - float (shrinkage constant)
    - ``frequency`` - int
    """

    def __init__(self, prices, returns_data=False, frequency=252, log_returns=False):
        if False:
            return 10
        '\n        :param prices: adjusted closing prices of the asset, each row is a date and each column is a ticker/id.\n        :type prices: pd.DataFrame\n        :param returns_data: if true, the first argument is returns instead of prices.\n        :type returns_data: bool, defaults to False.\n        :param frequency: number of time periods in a year, defaults to 252 (the number of trading days in a year)\n        :type frequency: int, optional\n        :param log_returns: whether to compute using log returns\n        :type log_returns: bool, defaults to False\n        '
        try:
            from sklearn import covariance
            self.covariance = covariance
        except (ModuleNotFoundError, ImportError):
            raise ImportError('Please install scikit-learn via pip or poetry')
        if not isinstance(prices, pd.DataFrame):
            warnings.warn('data is not in a dataframe', RuntimeWarning)
            prices = pd.DataFrame(prices)
        self.frequency = frequency
        if returns_data:
            self.X = prices.dropna(how='all')
        else:
            self.X = returns_from_prices(prices, log_returns).dropna(how='all')
        self.S = self.X.cov().values
        self.delta = None

    def _format_and_annualize(self, raw_cov_array):
        if False:
            i = 10
            return i + 15
        '\n        Helper method which annualises the output of shrinkage calculations,\n        and formats the result into a dataframe\n\n        :param raw_cov_array: raw covariance matrix of daily returns\n        :type raw_cov_array: np.ndarray\n        :return: annualised covariance matrix\n        :rtype: pd.DataFrame\n        '
        assets = self.X.columns
        cov = pd.DataFrame(raw_cov_array, index=assets, columns=assets) * self.frequency
        return fix_nonpositive_semidefinite(cov, fix_method='spectral')

    def shrunk_covariance(self, delta=0.2):
        if False:
            i = 10
            return i + 15
        '\n        Shrink a sample covariance matrix to the identity matrix (scaled by the average\n        sample variance). This method does not estimate an optimal shrinkage parameter,\n        it requires manual input.\n\n        :param delta: shrinkage parameter, defaults to 0.2.\n        :type delta: float, optional\n        :return: shrunk sample covariance matrix\n        :rtype: np.ndarray\n        '
        self.delta = delta
        N = self.S.shape[1]
        mu = np.trace(self.S) / N
        F = np.identity(N) * mu
        shrunk_cov = delta * F + (1 - delta) * self.S
        return self._format_and_annualize(shrunk_cov)

    def ledoit_wolf(self, shrinkage_target='constant_variance'):
        if False:
            while True:
                i = 10
        '\n        Calculate the Ledoit-Wolf shrinkage estimate for a particular\n        shrinkage target.\n\n        :param shrinkage_target: choice of shrinkage target, either ``constant_variance``,\n                                 ``single_factor`` or ``constant_correlation``. Defaults to\n                                 ``constant_variance``.\n        :type shrinkage_target: str, optional\n        :raises NotImplementedError: if the shrinkage_target is unrecognised\n        :return: shrunk sample covariance matrix\n        :rtype: np.ndarray\n        '
        if shrinkage_target == 'constant_variance':
            X = np.nan_to_num(self.X.values)
            (shrunk_cov, self.delta) = self.covariance.ledoit_wolf(X)
        elif shrinkage_target == 'single_factor':
            (shrunk_cov, self.delta) = self._ledoit_wolf_single_factor()
        elif shrinkage_target == 'constant_correlation':
            (shrunk_cov, self.delta) = self._ledoit_wolf_constant_correlation()
        else:
            raise NotImplementedError('Shrinkage target {} not recognised'.format(shrinkage_target))
        return self._format_and_annualize(shrunk_cov)

    def _ledoit_wolf_single_factor(self):
        if False:
            print('Hello World!')
        '\n        Helper method to calculate the Ledoit-Wolf shrinkage estimate\n        with the Sharpe single-factor matrix as the shrinkage target.\n        See Ledoit and Wolf (2001).\n\n        :return: shrunk sample covariance matrix, shrinkage constant\n        :rtype: np.ndarray, float\n        '
        X = np.nan_to_num(self.X.values)
        (t, n) = np.shape(X)
        Xm = X - X.mean(axis=0)
        xmkt = Xm.mean(axis=1).reshape(t, 1)
        sample = np.cov(np.append(Xm, xmkt, axis=1), rowvar=False) * (t - 1) / t
        betas = sample[0:n, n].reshape(n, 1)
        varmkt = sample[n, n]
        sample = sample[:n, :n]
        F = np.dot(betas, betas.T) / varmkt
        F[np.eye(n) == 1] = np.diag(sample)
        c = np.linalg.norm(sample - F, 'fro') ** 2
        y = Xm ** 2
        p = 1 / t * np.sum(np.dot(y.T, y)) - np.sum(sample ** 2)
        rdiag = 1 / t * np.sum(y ** 2) - sum(np.diag(sample) ** 2)
        z = Xm * np.tile(xmkt, (n,))
        v1 = 1 / t * np.dot(y.T, z) - np.tile(betas, (n,)) * sample
        roff1 = np.sum(v1 * np.tile(betas, (n,)).T) / varmkt - np.sum(np.diag(v1) * betas.T) / varmkt
        v3 = 1 / t * np.dot(z.T, z) - varmkt * sample
        roff3 = np.sum(v3 * np.dot(betas, betas.T)) / varmkt ** 2 - np.sum(np.diag(v3).reshape(-1, 1) * betas ** 2) / varmkt ** 2
        roff = 2 * roff1 - roff3
        r = rdiag + roff
        k = (p - r) / c
        delta = max(0, min(1, k / t))
        shrunk_cov = delta * F + (1 - delta) * sample
        return (shrunk_cov, delta)

    def _ledoit_wolf_constant_correlation(self):
        if False:
            i = 10
            return i + 15
        '\n        Helper method to calculate the Ledoit-Wolf shrinkage estimate\n        with the constant correlation matrix as the shrinkage target.\n        See Ledoit and Wolf (2003)\n\n        :return: shrunk sample covariance matrix, shrinkage constant\n        :rtype: np.ndarray, float\n        '
        X = np.nan_to_num(self.X.values)
        (t, n) = np.shape(X)
        S = self.S
        var = np.diag(S).reshape(-1, 1)
        std = np.sqrt(var)
        _var = np.tile(var, (n,))
        _std = np.tile(std, (n,))
        r_bar = (np.sum(S / (_std * _std.T)) - n) / (n * (n - 1))
        F = r_bar * (_std * _std.T)
        F[np.eye(n) == 1] = var.reshape(-1)
        Xm = X - X.mean(axis=0)
        y = Xm ** 2
        pi_mat = np.dot(y.T, y) / t - 2 * np.dot(Xm.T, Xm) * S / t + S ** 2
        pi_hat = np.sum(pi_mat)
        term1 = np.dot((Xm ** 3).T, Xm) / t
        help_ = np.dot(Xm.T, Xm) / t
        help_diag = np.diag(help_)
        term2 = np.tile(help_diag, (n, 1)).T * S
        term3 = help_ * _var
        term4 = _var * S
        theta_mat = term1 - term2 - term3 + term4
        theta_mat[np.eye(n) == 1] = np.zeros(n)
        rho_hat = sum(np.diag(pi_mat)) + r_bar * np.sum(np.dot(1 / std, std.T) * theta_mat)
        gamma_hat = np.linalg.norm(S - F, 'fro') ** 2
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa_hat / t))
        shrunk_cov = delta * F + (1 - delta) * S
        return (shrunk_cov, delta)

    def oracle_approximating(self):
        if False:
            print('Hello World!')
        '\n        Calculate the Oracle Approximating Shrinkage estimate\n\n        :return: shrunk sample covariance matrix\n        :rtype: np.ndarray\n        '
        X = np.nan_to_num(self.X.values)
        (shrunk_cov, self.delta) = self.covariance.oas(X)
        return self._format_and_annualize(shrunk_cov)