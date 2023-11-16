"""
The ``efficient_cdar`` submodule houses the EfficientCDaR class, which
generates portfolios along the mean-CDaR (conditional drawdown-at-risk) frontier.
"""
import warnings
import cvxpy as cp
import numpy as np
from .. import objective_functions
from .efficient_frontier import EfficientFrontier

class EfficientCDaR(EfficientFrontier):
    """
    The EfficientCDaR class allows for optimisation along the mean-CDaR frontier, using the
    formulation of Chekhlov, Ursayev and Zabarankin (2005).

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

    - ``min_cdar()`` minimises the CDaR
    - ``efficient_risk()`` maximises return for a given CDaR
    - ``efficient_return()`` minimises CDaR for a given target return
    - ``add_objective()`` adds a (convex) objective to the optimisation problem
    - ``add_constraint()`` adds a (linear) constraint to the optimisation problem

    - ``portfolio_performance()`` calculates the expected return and CDaR of the portfolio
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, expected_returns, returns, beta=0.95, weight_bounds=(0, 1), solver=None, verbose=False, solver_options=None):
        if False:
            while True:
                i = 10
        '\n        :param expected_returns: expected returns for each asset. Can be None if\n                                optimising for CDaR only.\n        :type expected_returns: pd.Series, list, np.ndarray\n        :param returns: (historic) returns for all your assets (no NaNs).\n                                 See ``expected_returns.returns_from_prices``.\n        :type returns: pd.DataFrame or np.array\n        :param beta: confidence level, defaults to 0.95 (i.e expected drawdown on the worst (1-beta) days).\n        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair\n                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)\n                              for portfolios with shorting.\n        :type weight_bounds: tuple OR tuple list, optional\n        :param solver: name of solver. list available solvers with: `cvxpy.installed_solvers()`\n        :type solver: str\n        :param verbose: whether performance and debugging info should be printed, defaults to False\n        :type verbose: bool, optional\n        :param solver_options: parameters for the given solver\n        :type solver_options: dict, optional\n        :raises TypeError: if ``expected_returns`` is not a series, list or array\n        '
        super().__init__(expected_returns=expected_returns, cov_matrix=np.zeros((len(expected_returns),) * 2), weight_bounds=weight_bounds, solver=solver, verbose=verbose, solver_options=solver_options)
        self.returns = self._validate_returns(returns)
        self._beta = self._validate_beta(beta)
        self._alpha = cp.Variable()
        self._u = cp.Variable(len(self.returns) + 1)
        self._z = cp.Variable(len(self.returns))

    def set_weights(self, input_weights):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Method not available in EfficientCDaR.')

    @staticmethod
    def _validate_beta(beta):
        if False:
            print('Hello World!')
        if not 0 <= beta < 1:
            raise ValueError('beta must be between 0 and 1')
        if beta <= 0.2:
            warnings.warn('Warning: beta is the confidence-level, not the quantile. Typical values are 80%, 90%, 95%.', UserWarning)
        return beta

    def min_volatility(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Please use min_cdar instead.')

    def max_sharpe(self, risk_free_rate=0.02):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Method not available in EfficientCDaR.')

    def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
        if False:
            return 10
        raise NotImplementedError('Method not available in EfficientCDaR.')

    def min_cdar(self, market_neutral=False):
        if False:
            while True:
                i = 10
        '\n        Minimise portfolio CDaR (see docs for further explanation).\n\n        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),\n                               defaults to False. Requires negative lower weight bound.\n        :param market_neutral: bool, optional\n        :return: asset weights for the volatility-minimising portfolio\n        :rtype: OrderedDict\n        '
        self._objective = self._alpha + 1.0 / (len(self.returns) * (1 - self._beta)) * cp.sum(self._z)
        for obj in self._additional_objectives:
            self._objective += obj
        self._add_cdar_constraints()
        self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_return(self, target_return, market_neutral=False):
        if False:
            print('Hello World!')
        '\n        Minimise CDaR for a given target return.\n\n        :param target_return: the desired return of the resulting portfolio.\n        :type target_return: float\n        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),\n                               defaults to False. Requires negative lower weight bound.\n        :type market_neutral: bool, optional\n        :raises ValueError: if ``target_return`` is not a positive float\n        :raises ValueError: if no portfolio can be found with return equal to ``target_return``\n        :return: asset weights for the optimal portfolio\n        :rtype: OrderedDict\n        '
        update_existing_parameter = self.is_parameter_defined('target_return')
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value('target_return', target_return)
            return self._solve_cvxpy_opt_problem()
        else:
            ret = self.expected_returns.T @ self._w
            target_return_par = cp.Parameter(value=target_return, name='target_return', nonneg=True)
            self.add_constraint(lambda _: ret >= target_return_par)
            return self.min_cdar(market_neutral)

    def efficient_risk(self, target_cdar, market_neutral=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Maximise return for a target CDaR.\n        The resulting portfolio will have a CDaR less than the target\n        (but not guaranteed to be equal).\n\n        :param target_cdar: the desired maximum CDaR of the resulting portfolio.\n        :type target_cdar: float\n        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),\n                               defaults to False. Requires negative lower weight bound.\n        :param market_neutral: bool, optional\n        :return: asset weights for the efficient risk portfolio\n        :rtype: OrderedDict\n        '
        update_existing_parameter = self.is_parameter_defined('target_cdar')
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value('target_cdar', target_cdar)
        else:
            self._objective = objective_functions.portfolio_return(self._w, self.expected_returns)
            for obj in self._additional_objectives:
                self._objective += obj
            cdar = self._alpha + 1.0 / (len(self.returns) * (1 - self._beta)) * cp.sum(self._z)
            target_cdar_par = cp.Parameter(value=target_cdar, name='target_cdar', nonneg=True)
            self.add_constraint(lambda _: cdar <= target_cdar_par)
            self._add_cdar_constraints()
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def _add_cdar_constraints(self) -> None:
        if False:
            while True:
                i = 10
        self.add_constraint(lambda _: self._z >= self._u[1:] - self._alpha)
        self.add_constraint(lambda w: self._u[1:] >= self._u[:-1] - self.returns.values @ w)
        self.add_constraint(lambda _: self._u[0] == 0)
        self.add_constraint(lambda _: self._z >= 0)
        self.add_constraint(lambda _: self._u[1:] >= 0)

    def portfolio_performance(self, verbose=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        After optimising, calculate (and optionally print) the performance of the optimal\n        portfolio, specifically: expected return, CDaR\n\n        :param verbose: whether performance should be printed, defaults to False\n        :type verbose: bool, optional\n        :raises ValueError: if weights have not been calculated yet\n        :return: expected return, CDaR.\n        :rtype: (float, float)\n        '
        mu = objective_functions.portfolio_return(self.weights, self.expected_returns, negative=False)
        cdar = self._alpha + 1.0 / (len(self.returns) * (1 - self._beta)) * cp.sum(self._z)
        cdar_val = cdar.value
        if verbose:
            print('Expected annual return: {:.1f}%'.format(100 * mu))
            print('Conditional Drawdown at Risk: {:.2f}%'.format(100 * cdar_val))
        return (mu, cdar_val)