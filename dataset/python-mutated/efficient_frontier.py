"""
The ``efficient_frontier`` submodule houses the EfficientFrontier class, which generates
classical mean-variance optimal portfolios for a variety of objectives and constraints
"""
import warnings
import cvxpy as cp
import numpy as np
import pandas as pd
from .. import base_optimizer, exceptions, objective_functions

class EfficientFrontier(base_optimizer.BaseConvexOptimizer):
    """
    An EfficientFrontier object (inheriting from BaseConvexOptimizer) contains multiple
    optimization methods that can be called (corresponding to different objective
    functions) with various parameters. Note: a new EfficientFrontier object should
    be instantiated if you want to make any change to objectives/constraints/bounds/parameters.

    Instance variables:

    - Inputs:

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``bounds`` - float tuple OR (float tuple) list
        - ``cov_matrix`` - np.ndarray
        - ``expected_returns`` - np.ndarray
        - ``solver`` - str
        - ``solver_options`` - {str: str} dict

    - Output: ``weights`` - np.ndarray

    Public methods:

    - ``min_volatility()`` optimizes for minimum volatility
    - ``max_sharpe()`` optimizes for maximal Sharpe ratio (a.k.a the tangency portfolio)
    - ``max_quadratic_utility()`` maximises the quadratic utility, given some risk aversion.
    - ``efficient_risk()`` maximises return for a given target risk
    - ``efficient_return()`` minimises risk for a given target return

    - ``add_objective()`` adds a (convex) objective to the optimization problem
    - ``add_constraint()`` adds a constraint to the optimization problem
    - ``convex_objective()`` solves for a generic convex objective with linear constraints

    - ``portfolio_performance()`` calculates the expected return, volatility and Sharpe ratio for
      the optimized portfolio.
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, expected_returns, cov_matrix, weight_bounds=(0, 1), solver=None, verbose=False, solver_options=None):
        if False:
            print('Hello World!')
        '\n        :param expected_returns: expected returns for each asset. Can be None if\n                                optimising for volatility only (but not recommended).\n        :type expected_returns: pd.Series, list, np.ndarray\n        :param cov_matrix: covariance of returns for each asset. This **must** be\n                           positive semidefinite, otherwise optimization will fail.\n        :type cov_matrix: pd.DataFrame or np.array\n        :param weight_bounds: minimum and maximum weight of each asset OR single min/max pair\n                              if all identical, defaults to (0, 1). Must be changed to (-1, 1)\n                              for portfolios with shorting.\n        :type weight_bounds: tuple OR tuple list, optional\n        :param solver: name of solver. list available solvers with: `cvxpy.installed_solvers()`\n        :type solver: str\n        :param verbose: whether performance and debugging info should be printed, defaults to False\n        :type verbose: bool, optional\n        :param solver_options: parameters for the given solver\n        :type solver_options: dict, optional\n        :raises TypeError: if ``expected_returns`` is not a series, list or array\n        :raises TypeError: if ``cov_matrix`` is not a dataframe or array\n        '
        self.cov_matrix = self._validate_cov_matrix(cov_matrix)
        self.expected_returns = self._validate_expected_returns(expected_returns)
        self._max_return_value = None
        self._market_neutral = None
        if self.expected_returns is None:
            num_assets = len(cov_matrix)
        else:
            num_assets = len(expected_returns)
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:
            tickers = list(range(num_assets))
        if expected_returns is not None and cov_matrix is not None:
            if cov_matrix.shape != (num_assets, num_assets):
                raise ValueError('Covariance matrix does not match expected returns')
        super().__init__(len(tickers), tickers, weight_bounds, solver=solver, verbose=verbose, solver_options=solver_options)

    @staticmethod
    def _validate_expected_returns(expected_returns):
        if False:
            i = 10
            return i + 15
        if expected_returns is None:
            return None
        elif isinstance(expected_returns, pd.Series):
            return expected_returns.values
        elif isinstance(expected_returns, list):
            return np.array(expected_returns)
        elif isinstance(expected_returns, np.ndarray):
            return expected_returns.ravel()
        else:
            raise TypeError('expected_returns is not a series, list or array')

    @staticmethod
    def _validate_cov_matrix(cov_matrix):
        if False:
            return 10
        if cov_matrix is None:
            raise ValueError('cov_matrix must be provided')
        elif isinstance(cov_matrix, pd.DataFrame):
            return cov_matrix.values
        elif isinstance(cov_matrix, np.ndarray):
            return cov_matrix
        else:
            raise TypeError('cov_matrix is not a dataframe or array')

    def _validate_returns(self, returns):
        if False:
            i = 10
            return i + 15
        '\n        Helper method to validate daily returns (needed for some efficient frontiers)\n        '
        if not isinstance(returns, (pd.DataFrame, np.ndarray)):
            raise TypeError('returns should be a pd.Dataframe or np.ndarray')
        returns_df = pd.DataFrame(returns)
        if returns_df.isnull().values.any():
            warnings.warn('Removing NaNs from returns', UserWarning)
            returns_df = returns_df.dropna(axis=0, how='any')
        if self.expected_returns is not None:
            if returns_df.shape[1] != len(self.expected_returns):
                raise ValueError('returns columns do not match expected_returns. Please check your tickers.')
        return returns_df

    def _make_weight_sum_constraint(self, is_market_neutral):
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper method to make the weight sum constraint. If market neutral,\n        validate the weights proided in the constructor.\n        '
        if is_market_neutral:
            portfolio_possible = np.any(self._lower_bounds < 0)
            if not portfolio_possible:
                warnings.warn('Market neutrality requires shorting - bounds have been amended', RuntimeWarning)
                self._map_bounds_to_constraints((-1, 1))
                del self._constraints[0]
                del self._constraints[0]
            self.add_constraint(lambda w: cp.sum(w) == 0)
        else:
            self.add_constraint(lambda w: cp.sum(w) == 1)
        self._market_neutral = is_market_neutral

    def min_volatility(self):
        if False:
            while True:
                i = 10
        '\n        Minimise volatility.\n\n        :return: asset weights for the volatility-minimising portfolio\n        :rtype: OrderedDict\n        '
        self._objective = objective_functions.portfolio_variance(self._w, self.cov_matrix)
        for obj in self._additional_objectives:
            self._objective += obj
        self.add_constraint(lambda w: cp.sum(w) == 1)
        return self._solve_cvxpy_opt_problem()

    def _max_return(self, return_value=True):
        if False:
            i = 10
            return i + 15
        '\n        Helper method to maximise return. This should not be used to optimize a portfolio.\n\n        :return: asset weights for the return-minimising portfolio\n        :rtype: OrderedDict\n        '
        if self.expected_returns is None:
            raise ValueError('no expected returns provided')
        self._objective = objective_functions.portfolio_return(self._w, self.expected_returns)
        self.add_constraint(lambda w: cp.sum(w) == 1)
        res = self._solve_cvxpy_opt_problem()
        if return_value:
            return -self._opt.value
        else:
            return res

    def max_sharpe(self, risk_free_rate=0.02):
        if False:
            for i in range(10):
                print('nop')
        '\n        Maximise the Sharpe Ratio. The result is also referred to as the tangency portfolio,\n        as it is the portfolio for which the capital market line is tangent to the efficient frontier.\n\n        This is a convex optimization problem after making a certain variable substitution. See\n        `Cornuejols and Tutuncu (2006) <http://web.math.ku.dk/~rolf/CT_FinOpt.pdf>`_ for more.\n\n        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.\n                               The period of the risk-free rate should correspond to the\n                               frequency of expected returns.\n        :type risk_free_rate: float, optional\n        :raises ValueError: if ``risk_free_rate`` is non-numeric\n        :return: asset weights for the Sharpe-maximising portfolio\n        :rtype: OrderedDict\n        '
        if not isinstance(risk_free_rate, (int, float)):
            raise ValueError('risk_free_rate should be numeric')
        if max(self.expected_returns) <= risk_free_rate:
            raise ValueError('at least one of the assets must have an expected return exceeding the risk-free rate')
        self._risk_free_rate = risk_free_rate
        self._objective = cp.quad_form(self._w, self.cov_matrix)
        k = cp.Variable()
        if len(self._additional_objectives) > 0:
            warnings.warn('max_sharpe transforms the optimization problem so additional objectives may not work as expected.')
        for obj in self._additional_objectives:
            self._objective += obj
        new_constraints = []
        for constr in self._constraints:
            if isinstance(constr, cp.constraints.nonpos.Inequality):
                if isinstance(constr.args[0], cp.expressions.constants.constant.Constant):
                    new_constraints.append(constr.args[1] >= constr.args[0] * k)
                else:
                    new_constraints.append(constr.args[0] <= constr.args[1] * k)
            elif isinstance(constr, cp.constraints.zero.Equality):
                new_constraints.append(constr.args[0] == constr.args[1] * k)
            else:
                raise TypeError('Please check that your constraints are in a suitable format')
        self._constraints = [(self.expected_returns - risk_free_rate).T @ self._w == 1, cp.sum(self._w) == k, k >= 0] + new_constraints
        self._solve_cvxpy_opt_problem()
        self.weights = (self._w.value / k.value).round(16) + 0.0
        return self._make_output_weights()

    def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
        if False:
            while True:
                i = 10
        '\n        Maximise the given quadratic utility, i.e:\n\n        .. math::\n\n            \\max_w w^T \\mu - \\frac \\delta 2 w^T \\Sigma w\n\n        :param risk_aversion: risk aversion parameter (must be greater than 0),\n                              defaults to 1\n        :type risk_aversion: positive float\n        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),\n                               defaults to False. Requires negative lower weight bound.\n        :param market_neutral: bool, optional\n        :return: asset weights for the maximum-utility portfolio\n        :rtype: OrderedDict\n        '
        if risk_aversion <= 0:
            raise ValueError('risk aversion coefficient must be greater than zero')
        update_existing_parameter = self.is_parameter_defined('risk_aversion')
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value('risk_aversion', risk_aversion)
        else:
            self._objective = objective_functions.quadratic_utility(self._w, self.expected_returns, self.cov_matrix, risk_aversion=risk_aversion)
            for obj in self._additional_objectives:
                self._objective += obj
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_risk(self, target_volatility, market_neutral=False):
        if False:
            i = 10
            return i + 15
        '\n        Maximise return for a target risk. The resulting portfolio will have a volatility\n        less than the target (but not guaranteed to be equal).\n\n        :param target_volatility: the desired maximum volatility of the resulting portfolio.\n        :type target_volatility: float\n        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),\n                               defaults to False. Requires negative lower weight bound.\n        :param market_neutral: bool, optional\n        :raises ValueError: if ``target_volatility`` is not a positive float\n        :raises ValueError: if no portfolio can be found with volatility equal to ``target_volatility``\n        :raises ValueError: if ``risk_free_rate`` is non-numeric\n        :return: asset weights for the efficient risk portfolio\n        :rtype: OrderedDict\n        '
        if not isinstance(target_volatility, (float, int)) or target_volatility < 0:
            raise ValueError('target_volatility should be a positive float')
        global_min_volatility = np.sqrt(1 / np.sum(np.linalg.pinv(self.cov_matrix)))
        if target_volatility < global_min_volatility:
            raise ValueError('The minimum volatility is {:.3f}. Please use a higher target_volatility'.format(global_min_volatility))
        update_existing_parameter = self.is_parameter_defined('target_variance')
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value('target_variance', target_volatility ** 2)
        else:
            self._objective = objective_functions.portfolio_return(self._w, self.expected_returns)
            variance = objective_functions.portfolio_variance(self._w, self.cov_matrix)
            for obj in self._additional_objectives:
                self._objective += obj
            target_variance = cp.Parameter(name='target_variance', value=target_volatility ** 2, nonneg=True)
            self.add_constraint(lambda _: variance <= target_variance)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_return(self, target_return, market_neutral=False):
        if False:
            i = 10
            return i + 15
        "\n        Calculate the 'Markowitz portfolio', minimising volatility for a given target return.\n\n        :param target_return: the desired return of the resulting portfolio.\n        :type target_return: float\n        :param market_neutral: whether the portfolio should be market neutral (weights sum to zero),\n                               defaults to False. Requires negative lower weight bound.\n        :type market_neutral: bool, optional\n        :raises ValueError: if ``target_return`` is not a positive float\n        :raises ValueError: if no portfolio can be found with return equal to ``target_return``\n        :return: asset weights for the Markowitz portfolio\n        :rtype: OrderedDict\n        "
        if not isinstance(target_return, float):
            raise ValueError('target_return should be a float')
        if not self._max_return_value:
            a = self.deepcopy()
            self._max_return_value = a._max_return()
        if target_return > self._max_return_value:
            raise ValueError('target_return must be lower than the maximum possible return')
        update_existing_parameter = self.is_parameter_defined('target_return')
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value('target_return', target_return)
        else:
            self._objective = objective_functions.portfolio_variance(self._w, self.cov_matrix)
            ret = objective_functions.portfolio_return(self._w, self.expected_returns, negative=False)
            for obj in self._additional_objectives:
                self._objective += obj
            target_return_par = cp.Parameter(name='target_return', value=target_return)
            self.add_constraint(lambda _: ret >= target_return_par)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02):
        if False:
            while True:
                i = 10
        '\n        After optimising, calculate (and optionally print) the performance of the optimal\n        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio.\n\n        :param verbose: whether performance should be printed, defaults to False\n        :type verbose: bool, optional\n        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.\n                               The period of the risk-free rate should correspond to the\n                               frequency of expected returns.\n        :type risk_free_rate: float, optional\n        :raises ValueError: if weights have not been calculated yet\n        :return: expected return, volatility, Sharpe ratio.\n        :rtype: (float, float, float)\n        '
        if self._risk_free_rate is not None:
            if risk_free_rate != self._risk_free_rate:
                warnings.warn('The risk_free_rate provided to portfolio_performance is different to the one used by max_sharpe. Using the previous value.', UserWarning)
            risk_free_rate = self._risk_free_rate
        return base_optimizer.portfolio_performance(self.weights, self.expected_returns, self.cov_matrix, verbose, risk_free_rate)

    def _validate_market_neutral(self, market_neutral: bool) -> None:
        if False:
            print('Hello World!')
        if self._market_neutral != market_neutral:
            raise exceptions.InstantiationError('A new instance must be created when changing market_neutral.')