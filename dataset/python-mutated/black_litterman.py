"""
The ``black_litterman`` module houses the BlackLittermanModel class, which
generates posterior estimates of expected returns given a prior estimate and user-supplied
views. In addition, two utility functions are defined, which calculate:

- market-implied prior estimate of returns
- market-implied risk-aversion parameter
"""
import sys
import warnings
import numpy as np
import pandas as pd
from . import base_optimizer

def market_implied_prior_returns(market_caps, risk_aversion, cov_matrix, risk_free_rate=0.02):
    if False:
        while True:
            i = 10
    "\n    Compute the prior estimate of returns implied by the market weights.\n    In other words, given each asset's contribution to the risk of the market\n    portfolio, how much are we expecting to be compensated?\n\n    .. math::\n\n        \\Pi = \\delta \\Sigma w_{mkt}\n\n    :param market_caps: market capitalisations of all assets\n    :type market_caps: {ticker: cap} dict or pd.Series\n    :param risk_aversion: risk aversion parameter\n    :type risk_aversion: positive float\n    :param cov_matrix: covariance matrix of asset returns\n    :type cov_matrix: pd.DataFrame\n    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.\n                           You should use the appropriate time period, corresponding\n                           to the covariance matrix.\n    :type risk_free_rate: float, optional\n    :return: prior estimate of returns as implied by the market caps\n    :rtype: pd.Series\n    "
    if not isinstance(cov_matrix, pd.DataFrame):
        warnings.warn('If cov_matrix is not a dataframe, market cap index must be aligned to cov_matrix', RuntimeWarning)
    mcaps = pd.Series(market_caps)
    mkt_weights = mcaps / mcaps.sum()
    return risk_aversion * cov_matrix.dot(mkt_weights) + risk_free_rate

def market_implied_risk_aversion(market_prices, frequency=252, risk_free_rate=0.02):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate the market-implied risk-aversion parameter (i.e market price of risk)\n    based on market prices. For example, if the market has excess returns of 10% a year\n    with 5% variance, the risk-aversion parameter is 2, i.e you have to be compensated 2x\n    the variance.\n\n    .. math::\n\n        \\delta = \\frac{R - R_f}{\\sigma^2}\n\n    :param market_prices: the (daily) prices of the market portfolio, e.g SPY.\n    :type market_prices: pd.Series with DatetimeIndex.\n    :param frequency: number of time periods in a year, defaults to 252 (the number\n                      of trading days in a year)\n    :type frequency: int, optional\n    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.\n                            The period of the risk-free rate should correspond to the\n                            frequency of expected returns.\n    :type risk_free_rate: float, optional\n    :raises TypeError: if market_prices cannot be parsed\n    :return: market-implied risk aversion\n    :rtype: float\n    '
    if not isinstance(market_prices, (pd.Series, pd.DataFrame)):
        raise TypeError('Please format market_prices as a pd.Series')
    rets = market_prices.pct_change().dropna()
    r = rets.mean() * frequency
    var = rets.var() * frequency
    return (r - risk_free_rate) / var

class BlackLittermanModel(base_optimizer.BaseOptimizer):
    """
    A BlackLittermanModel object (inheriting from BaseOptimizer) contains requires
    a specific input format, specifying the prior, the views, the uncertainty in views,
    and a picking matrix to map views to the asset universe. We can then compute
    posterior estimates of returns and covariance. Helper methods have been provided
    to supply defaults where possible.

    Instance variables:

    - Inputs:

        - ``cov_matrix`` - np.ndarray
        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``Q`` - np.ndarray
        - ``P`` - np.ndarray
        - ``pi`` - np.ndarray
        - ``omega`` - np.ndarray
        - ``tau`` - float

    - Output:

        - ``posterior_rets`` - pd.Series
        - ``posterior_cov`` - pd.DataFrame
        - ``weights`` - np.ndarray

    Public methods:

    - ``default_omega()`` - view uncertainty proportional to asset variance
    - ``idzorek_method()`` - convert views specified as percentages into BL uncertainties
    - ``bl_returns()`` - posterior estimate of returns
    - ``bl_cov()`` - posterior estimate of covariance
    - ``bl_weights()`` - weights implied by posterior returns
    - ``portfolio_performance()`` calculates the expected return, volatility
      and Sharpe ratio for the allocated portfolio.
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, cov_matrix, pi=None, absolute_views=None, Q=None, P=None, omega=None, view_confidences=None, tau=0.05, risk_aversion=1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param cov_matrix: NxN covariance matrix of returns\n        :type cov_matrix: pd.DataFrame or np.ndarray\n        :param pi: Nx1 prior estimate of returns, defaults to None.\n                   If pi="market", calculate a market-implied prior (requires market_caps\n                   to be passed).\n                   If pi="equal", use an equal-weighted prior.\n        :type pi: np.ndarray, pd.Series, optional\n        :param absolute_views: a collection of K absolute views on a subset of assets,\n                               defaults to None. If this is provided, we do not need P, Q.\n        :type absolute_views: pd.Series or dict, optional\n        :param Q: Kx1 views vector, defaults to None\n        :type Q: np.ndarray or pd.DataFrame, optional\n        :param P: KxN picking matrix, defaults to None\n        :type P: np.ndarray or pd.DataFrame, optional\n        :param omega: KxK view uncertainty matrix (diagonal), defaults to None\n                      Can instead pass "idzorek" to use Idzorek\'s method (requires\n                      you to pass view_confidences). If omega="default" or None,\n                      we set the uncertainty proportional to the variance.\n        :type omega: np.ndarray or Pd.DataFrame, or string, optional\n        :param view_confidences: Kx1 vector of percentage view confidences (between 0 and 1),\n                                required to compute omega via Idzorek\'s method.\n        :type view_confidences: np.ndarray, pd.Series, list, optional\n        :param tau: the weight-on-views scalar (default is 0.05)\n        :type tau: float, optional\n        :param risk_aversion: risk aversion parameter, defaults to 1\n        :type risk_aversion: positive float, optional\n        :param market_caps: (kwarg) market caps for the assets, required if pi="market"\n        :type market_caps: np.ndarray, pd.Series, optional\n        :param risk_free_rate: (kwarg) risk_free_rate is needed in some methods\n        :type risk_free_rate: float, defaults to 0.02\n        '
        if sys.version_info[1] == 5:
            warnings.warn('When using python 3.5 you must explicitly construct the Black-Litterman inputs')
        self._raw_cov_matrix = cov_matrix
        if isinstance(cov_matrix, np.ndarray):
            self.cov_matrix = cov_matrix
            super().__init__(len(cov_matrix), list(range(len(cov_matrix))))
        else:
            self.cov_matrix = cov_matrix.values
            super().__init__(len(cov_matrix), cov_matrix.columns)
        if absolute_views is not None:
            (self.Q, self.P) = self._parse_views(absolute_views)
        else:
            self._set_Q_P(Q, P)
        self._set_risk_aversion(risk_aversion)
        self._set_pi(pi, **kwargs)
        self._set_tau(tau)
        self._check_attribute_dimensions()
        self._set_omega(omega, view_confidences)
        self._tau_sigma_P = None
        self._A = None
        self.posterior_rets = None
        self.posterior_cov = None

    def _parse_views(self, absolute_views):
        if False:
            return 10
        '\n        Given a collection (dict or series) of absolute views, construct\n        the appropriate views vector and picking matrix. The views must\n        be a subset of the tickers in the covariance matrix.\n\n        {"AAPL": 0.20, "GOOG": 0.12, "XOM": -0.30}\n\n        :param absolute_views: absolute views on asset performances\n        :type absolute_views: dict, pd.Series\n        '
        if not isinstance(absolute_views, (dict, pd.Series)):
            raise TypeError('views should be a dict or pd.Series')
        views = pd.Series(absolute_views)
        k = len(views)
        Q = np.zeros((k, 1))
        P = np.zeros((k, self.n_assets))
        for (i, view_ticker) in enumerate(views.keys()):
            try:
                Q[i] = views[view_ticker]
                P[i, list(self.tickers).index(view_ticker)] = 1
            except ValueError:
                raise ValueError('Providing a view on an asset not in the universe')
        return (Q, P)

    def _set_Q_P(self, Q, P):
        if False:
            while True:
                i = 10
        if isinstance(Q, (pd.Series, pd.DataFrame)):
            self.Q = Q.values.reshape(-1, 1)
        elif isinstance(Q, np.ndarray):
            self.Q = Q.reshape(-1, 1)
        else:
            raise TypeError('Q must be an array or dataframe')
        if isinstance(P, pd.DataFrame):
            self.P = P.values
        elif isinstance(P, np.ndarray):
            self.P = P
        elif len(self.Q) == self.n_assets:
            self.P = np.eye(self.n_assets)
        else:
            raise TypeError('P must be an array or dataframe')

    def _set_pi(self, pi, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if pi is None:
            warnings.warn('Running Black-Litterman with no prior.')
            self.pi = np.zeros((self.n_assets, 1))
        elif isinstance(pi, (pd.Series, pd.DataFrame)):
            self.pi = pi.values.reshape(-1, 1)
        elif isinstance(pi, np.ndarray):
            self.pi = pi.reshape(-1, 1)
        elif pi == 'market':
            if 'market_caps' not in kwargs:
                raise ValueError('Please pass a series/array of market caps via the market_caps keyword argument')
            market_caps = kwargs.get('market_caps')
            risk_free_rate = kwargs.get('risk_free_rate', 0)
            market_prior = market_implied_prior_returns(market_caps, self.risk_aversion, self._raw_cov_matrix, risk_free_rate)
            self.pi = market_prior.values.reshape(-1, 1)
        elif pi == 'equal':
            self.pi = np.ones((self.n_assets, 1)) / self.n_assets
        else:
            raise TypeError('pi must be an array or series')

    def _set_tau(self, tau):
        if False:
            for i in range(10):
                print('nop')
        if tau <= 0 or tau > 1:
            raise ValueError('tau should be between 0 and 1')
        self.tau = tau

    def _set_risk_aversion(self, risk_aversion):
        if False:
            print('Hello World!')
        if risk_aversion <= 0:
            raise ValueError('risk_aversion should be a positive float')
        self.risk_aversion = risk_aversion

    def _set_omega(self, omega, view_confidences):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(omega, pd.DataFrame):
            self.omega = omega.values
        elif isinstance(omega, np.ndarray):
            self.omega = omega
        elif omega == 'idzorek':
            if view_confidences is None:
                raise ValueError("To use Idzorek's method, please supply a vector of percentage confidence levels for each view.")
            if not isinstance(view_confidences, np.ndarray):
                try:
                    view_confidences = np.array(view_confidences).reshape(-1, 1)
                    assert view_confidences.shape[0] == self.Q.shape[0]
                    assert np.issubdtype(view_confidences.dtype, np.number)
                except AssertionError:
                    raise ValueError('view_confidences should be a numpy 1D array or vector with the same length as the number of views.')
            self.omega = BlackLittermanModel.idzorek_method(view_confidences, self.cov_matrix, self.pi, self.Q, self.P, self.tau, self.risk_aversion)
        elif omega is None or omega == 'default':
            self.omega = BlackLittermanModel.default_omega(self.cov_matrix, self.P, self.tau)
        else:
            raise TypeError('self.omega must be a square array, dataframe, or string')
        K = len(self.Q)
        assert self.omega.shape == (K, K), 'omega must have dimensions KxK'

    def _check_attribute_dimensions(self):
        if False:
            print('Hello World!')
        '\n        Helper method to ensure that all of the attributes created by the initialiser\n        have the correct dimensions, to avoid linear algebra errors later on.\n\n        :raises ValueError: if there are incorrect dimensions.\n        '
        N = self.n_assets
        K = len(self.Q)
        assert self.pi.shape == (N, 1), 'pi must have dimensions Nx1'
        assert self.P.shape == (K, N), 'P must have dimensions KxN'
        assert self.cov_matrix.shape == (N, N), 'cov_matrix must have shape NxN'

    @staticmethod
    def default_omega(cov_matrix, P, tau):
        if False:
            while True:
                i = 10
        '\n        If the uncertainty matrix omega is not provided, we calculate using the method of\n        He and Litterman (1999), such that the ratio omega/tau is proportional to the\n        variance of the view portfolio.\n\n        :return: KxK diagonal uncertainty matrix\n        :rtype: np.ndarray\n        '
        return np.diag(np.diag(tau * P @ cov_matrix @ P.T))

    @staticmethod
    def idzorek_method(view_confidences, cov_matrix, pi, Q, P, tau, risk_aversion=1):
        if False:
            return 10
        "\n        Use Idzorek's method to create the uncertainty matrix given user-specified\n        percentage confidences. We use the closed-form solution described by\n        Jay Walters in The Black-Litterman Model in Detail (2014).\n\n        :param view_confidences: Kx1 vector of percentage view confidences (between 0 and 1),\n                                required to compute omega via Idzorek's method.\n        :type view_confidences: np.ndarray, pd.Series, list,, optional\n        :return: KxK diagonal uncertainty matrix\n        :rtype: np.ndarray\n        "
        view_omegas = []
        for view_idx in range(len(Q)):
            conf = view_confidences[view_idx]
            if conf < 0 or conf > 1:
                raise ValueError('View confidences must be between 0 and 1')
            if conf == 0:
                view_omegas.append(1000000.0)
                continue
            P_view = P[view_idx].reshape(1, -1)
            alpha = (1 - conf) / conf
            omega = tau * alpha * P_view @ cov_matrix @ P_view.T
            view_omegas.append(omega.item())
        return np.diag(view_omegas)

    def bl_returns(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculate the posterior estimate of the returns vector,\n        given views on some assets.\n\n        :return: posterior returns vector\n        :rtype: pd.Series\n        '
        if self._tau_sigma_P is None:
            self._tau_sigma_P = self.tau * self.cov_matrix @ self.P.T
        if self._A is None:
            self._A = self.P @ self._tau_sigma_P + self.omega
        b = self.Q - self.P @ self.pi
        post_rets = self.pi + self._tau_sigma_P @ np.linalg.solve(self._A, b)
        return pd.Series(post_rets.flatten(), index=self.tickers)

    def bl_cov(self):
        if False:
            return 10
        '\n        Calculate the posterior estimate of the covariance matrix,\n        given views on some assets. Based on He and Litterman (2002).\n        It is assumed that omega is diagonal. If this is not the case,\n        please manually set omega_inv.\n\n        :return: posterior covariance matrix\n        :rtype: pd.DataFrame\n        '
        if self._tau_sigma_P is None:
            self._tau_sigma_P = self.tau * self.cov_matrix @ self.P.T
        if self._A is None:
            self._A = self.P @ self._tau_sigma_P + self.omega
        b = self._tau_sigma_P.T
        M = self.tau * self.cov_matrix - self._tau_sigma_P @ np.linalg.solve(self._A, b)
        posterior_cov = self.cov_matrix + M
        return pd.DataFrame(posterior_cov, index=self.tickers, columns=self.tickers)

    def bl_weights(self, risk_aversion=None):
        if False:
            print('Hello World!')
        '\n        Compute the weights implied by the posterior returns, given the\n        market price of risk. Technically this can be applied to any\n        estimate of the expected returns, and is in fact a special case\n        of mean-variance optimization\n\n        .. math::\n\n            w = (\\delta \\Sigma)^{-1} E(R)\n\n        :param risk_aversion: risk aversion parameter, defaults to 1\n        :type risk_aversion: positive float, optional\n        :return: asset weights implied by returns\n        :rtype: OrderedDict\n        '
        if risk_aversion is None:
            risk_aversion = self.risk_aversion
        self.posterior_rets = self.bl_returns()
        A = risk_aversion * self.cov_matrix
        b = self.posterior_rets
        raw_weights = np.linalg.solve(A, b)
        self.weights = raw_weights / raw_weights.sum()
        return self._make_output_weights()

    def optimize(self, risk_aversion=None):
        if False:
            i = 10
            return i + 15
        '\n        Alias for bl_weights for consistency with other methods.\n        '
        return self.bl_weights(risk_aversion)

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        if False:
            while True:
                i = 10
        '\n        After optimising, calculate (and optionally print) the performance of the optimal\n        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.\n        This method uses the BL posterior returns and covariance matrix.\n\n        :param verbose: whether performance should be printed, defaults to False\n        :type verbose: bool, optional\n        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.\n                               The period of the risk-free rate should correspond to the\n                               frequency of expected returns.\n        :type risk_free_rate: float, optional\n        :raises ValueError: if weights have not been calculated yet\n        :return: expected return, volatility, Sharpe ratio.\n        :rtype: (float, float, float)\n        '
        if self.posterior_cov is None:
            self.posterior_cov = self.bl_cov()
        return base_optimizer.portfolio_performance(self.weights, self.posterior_rets, self.posterior_cov, verbose, risk_free_rate)