"""
The ``efficient_semivariance`` submodule houses the EfficientSemivariance class, which
generates portfolios along the mean-semivariance frontier.
"""
import cvxpy as cp
import numpy as np
from .. import objective_functions
from .efficient_frontier import EfficientFrontier

class EfficientSemivariance(EfficientFrontier):
    """
    EfficientSemivariance objects allow for optimization along the mean-semivariance frontier.
    This may be relevant for users who are more concerned about downside deviation.

    Instance variables:

    - Inputs:

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``bounds`` - float tuple OR (float tuple) list
        - ``returns`` - pd.DataFrame
        - ``expected_returns`` - np.ndarray
        - ``solver`` - str
        - ``solver_options`` - {str: str} dict


    - Output: ``weights`` - np.ndarray

    Public methods:

    - ``min_semivariance()`` minimises the portfolio semivariance (downside deviation)
    - ``max_quadratic_utility()`` maximises the "downside quadratic utility", given some risk aversion.
    - ``efficient_risk()`` maximises return for a given target semideviation
    - ``efficient_return()`` minimises semideviation for a given target return
    - ``add_objective()`` adds a (convex) objective to the optimization problem
    - ``add_constraint()`` adds a constraint to the optimization problem
    - ``convex_objective()`` solves for a generic convex objective with linear constraints

    - ``portfolio_performance()`` calculates the expected return, semideviation and Sortino ratio for
      the optimized portfolio.
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, expected_returns, returns, frequency=252, benchmark=0, weight_bounds=(0, 1), solver=None, verbose=False, solver_options=None):
        if False:
            i = 10
            return i + 15
        '\n        :param expected_returns: expected returns for each asset. Can be None if\n                                optimising for semideviation only.\n        :type expected_returns: pd.Series, list, np.ndarray\n        :param returns: (historic) returns for all your assets (no NaNs).\n                                 See ``expected_returns.returns_from_prices``.\n        :type returns: pd.DataFrame or np.array\n        :param frequency: number of time periods in a year, defaults to 252 (the number\n                          of trading days in a year). This must agree with the frequency\n                          parameter used in your ``expected_returns``.\n        :type frequency: int, optional\n        :param benchmark: the return threshold to distinguish "downside" and "upside".\n                          This should match the frequency of your ``returns``,\n                          i.e this should be a benchmark daily returns if your\n                          ``returns`` are also daily.\n        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair\n                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)\n                              for portfolios with shorting.\n        :type weight_bounds: tuple OR tuple list, optional\n        :param solver: name of solver. list available solvers with: `cvxpy.installed_solvers()`\n        :type solver: str\n        :param verbose: whether performance and debugging info should be printed, defaults to False\n        :type verbose: bool, optional\n        :param solver_options: parameters for the given solver\n        :type solver_options: dict, optional\n        :raises TypeError: if ``expected_returns`` is not a series, list or array\n        '
        super().__init__(expected_returns=expected_returns, cov_matrix=np.zeros((returns.shape[1],) * 2), weight_bounds=weight_bounds, solver=solver, verbose=verbose, solver_options=solver_options)
        self.returns = self._validate_returns(returns)
        self.benchmark = benchmark
        self.frequency = frequency
        self._T = self.returns.shape[0]

    def min_volatility(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('Please use min_semivariance instead.')

    def max_sharpe(self, risk_free_rate=0.02):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Method not available in EfficientSemivariance')

    def min_semivariance(self, market_neutral=False):
        if False:
            return 10
        '\n        Minimise portfolio semivariance (see docs for further explanation).\n\n        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),\n                               defaults to False. Requires negative lower weight bound.\n        :param market_neutral: bool, optional\n        :return: asset weights for the volatility-minimising portfolio\n        :rtype: OrderedDict\n        '
        p = cp.Variable(self._T, nonneg=True)
        n = cp.Variable(self._T, nonneg=True)
        self._objective = cp.sum(cp.square(n))
        for obj in self._additional_objectives:
            self._objective += obj
        B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
        self.add_constraint(lambda w: B @ w - p + n == 0)
        self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Maximise the given quadratic utility, using portfolio semivariance instead\n        of variance.\n\n        :param risk_aversion: risk aversion parameter (must be greater than 0),\n                              defaults to 1\n        :type risk_aversion: positive float\n        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),\n                               defaults to False. Requires negative lower weight bound.\n        :param market_neutral: bool, optional\n        :return: asset weights for the maximum-utility portfolio\n        :rtype: OrderedDict\n        '
        if risk_aversion <= 0:
            raise ValueError('risk aversion coefficient must be greater than zero')
        update_existing_parameter = self.is_parameter_defined('risk_aversion')
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value('risk_aversion', risk_aversion)
        else:
            p = cp.Variable(self._T, nonneg=True)
            n = cp.Variable(self._T, nonneg=True)
            mu = objective_functions.portfolio_return(self._w, self.expected_returns)
            mu /= self.frequency
            risk_aversion_par = cp.Parameter(value=risk_aversion, name='risk_aversion', nonneg=True)
            self._objective = mu + 0.5 * risk_aversion_par * cp.sum(cp.square(n))
            for obj in self._additional_objectives:
                self._objective += obj
            B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
            self.add_constraint(lambda w: B @ w - p + n == 0)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_risk(self, target_semideviation, market_neutral=False):
        if False:
            return 10
        '\n        Maximise return for a target semideviation (downside standard deviation).\n        The resulting portfolio will have a semideviation less than the target\n        (but not guaranteed to be equal).\n\n        :param target_semideviation: the desired maximum semideviation of the resulting portfolio.\n        :type target_semideviation: float\n        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),\n                               defaults to False. Requires negative lower weight bound.\n        :param market_neutral: bool, optional\n        :return: asset weights for the efficient risk portfolio\n        :rtype: OrderedDict\n        '
        update_existing_parameter = self.is_parameter_defined('target_semivariance')
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value('target_semivariance', target_semideviation ** 2)
        else:
            self._objective = objective_functions.portfolio_return(self._w, self.expected_returns)
            for obj in self._additional_objectives:
                self._objective += obj
            p = cp.Variable(self._T, nonneg=True)
            n = cp.Variable(self._T, nonneg=True)
            target_semivariance = cp.Parameter(value=target_semideviation ** 2, name='target_semivariance', nonneg=True)
            self.add_constraint(lambda _: self.frequency * cp.sum(cp.square(n)) <= target_semivariance)
            B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
            self.add_constraint(lambda w: B @ w - p + n == 0)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_return(self, target_return, market_neutral=False):
        if False:
            i = 10
            return i + 15
        '\n        Minimise semideviation for a given target return.\n\n        :param target_return: the desired return of the resulting portfolio.\n        :type target_return: float\n        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),\n                               defaults to False. Requires negative lower weight bound.\n        :type market_neutral: bool, optional\n        :raises ValueError: if ``target_return`` is not a positive float\n        :raises ValueError: if no portfolio can be found with return equal to ``target_return``\n        :return: asset weights for the optimal portfolio\n        :rtype: OrderedDict\n        '
        if not isinstance(target_return, float) or target_return < 0:
            raise ValueError('target_return should be a positive float')
        if target_return > np.abs(self.expected_returns).max():
            raise ValueError('target_return must be lower than the largest expected return')
        update_existing_parameter = self.is_parameter_defined('target_return')
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value('target_return', target_return)
        else:
            p = cp.Variable(self._T, nonneg=True)
            n = cp.Variable(self._T, nonneg=True)
            self._objective = cp.sum(cp.square(n))
            for obj in self._additional_objectives:
                self._objective += obj
            target_return_par = cp.Parameter(name='target_return', value=target_return)
            self.add_constraint(lambda w: cp.sum(w @ self.expected_returns) >= target_return_par)
            B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
            self.add_constraint(lambda w: B @ w - p + n == 0)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        if False:
            i = 10
            return i + 15
        '\n        After optimising, calculate (and optionally print) the performance of the optimal\n        portfolio, specifically: expected return, semideviation, Sortino ratio.\n\n        :param verbose: whether performance should be printed, defaults to False\n        :type verbose: bool, optional\n        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.\n                               The period of the risk-free rate should correspond to the\n                               frequency of expected returns.\n        :type risk_free_rate: float, optional\n        :raises ValueError: if weights have not been calculated yet\n        :return: expected return, semideviation, Sortino ratio.\n        :rtype: (float, float, float)\n        '
        mu = objective_functions.portfolio_return(self.weights, self.expected_returns, negative=False)
        portfolio_returns = self.returns @ self.weights
        drops = np.fmin(portfolio_returns - self.benchmark, 0)
        semivariance = np.sum(np.square(drops)) / self._T * self.frequency
        semi_deviation = np.sqrt(semivariance)
        sortino_ratio = (mu - risk_free_rate) / semi_deviation
        if verbose:
            print('Expected annual return: {:.1f}%'.format(100 * mu))
            print('Annual semi-deviation: {:.1f}%'.format(100 * semi_deviation))
            print('Sortino Ratio: {:.2f}'.format(sortino_ratio))
        return (mu, semi_deviation, sortino_ratio)