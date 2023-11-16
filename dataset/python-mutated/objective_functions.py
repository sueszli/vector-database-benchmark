"""
The ``objective_functions`` module provides optimization objectives, including the actual
objective functions called by the ``EfficientFrontier`` object's optimization methods.
These methods are primarily designed for internal use during optimization and each requires
a different signature (which is why they have not been factored into a class).
For obvious reasons, any objective function must accept ``weights``
as an argument, and must also have at least one of ``expected_returns`` or ``cov_matrix``.

The objective functions either compute the objective given a numpy array of weights, or they
return a cvxpy *expression* when weights are a ``cp.Variable``. In this way, the same objective
function can be used both internally for optimization and externally for computing the objective
given weights. ``_objective_value()`` automatically chooses between the two behaviours.

``objective_functions`` defaults to objectives for minimisation. In the cases of objectives
that clearly should be maximised (e.g Sharpe Ratio, portfolio return), the objective function
actually returns the negative quantity, since minimising the negative is equivalent to maximising
the positive. This behaviour is controlled by the ``negative=True`` optional argument.

Currently implemented:

- Portfolio variance (i.e square of volatility)
- Portfolio return
- Sharpe ratio
- L2 regularisation (minimising this reduces nonzero weights)
- Quadratic utility
- Transaction cost model (a simple one)
- Ex-ante (squared) tracking error
- Ex-post (squared) tracking error
"""
import cvxpy as cp
import numpy as np

def _objective_value(w, obj):
    if False:
        print('Hello World!')
    '\n    Helper method to return either the value of the objective function\n    or the objective function as a cvxpy object depending on whether\n    w is a cvxpy variable or np array.\n\n    :param w: weights\n    :type w: np.ndarray OR cp.Variable\n    :param obj: objective function expression\n    :type obj: cp.Expression\n    :return: value of the objective function OR objective function expression\n    :rtype: float OR cp.Expression\n    '
    if isinstance(w, np.ndarray):
        if np.isscalar(obj):
            return obj
        elif np.isscalar(obj.value):
            return obj.value
        else:
            return obj.value.item()
    else:
        return obj

def portfolio_variance(w, cov_matrix):
    if False:
        print('Hello World!')
    '\n    Calculate the total portfolio variance (i.e square volatility).\n\n    :param w: asset weights in the portfolio\n    :type w: np.ndarray OR cp.Variable\n    :param cov_matrix: covariance matrix\n    :type cov_matrix: np.ndarray\n    :return: value of the objective function OR objective function expression\n    :rtype: float OR cp.Expression\n    '
    variance = cp.quad_form(w, cov_matrix)
    return _objective_value(w, variance)

def portfolio_return(w, expected_returns, negative=True):
    if False:
        print('Hello World!')
    '\n    Calculate the (negative) mean return of a portfolio\n\n    :param w: asset weights in the portfolio\n    :type w: np.ndarray OR cp.Variable\n    :param expected_returns: expected return of each asset\n    :type expected_returns: np.ndarray\n    :param negative: whether quantity should be made negative (so we can minimise)\n    :type negative: boolean\n    :return: negative mean return\n    :rtype: float\n    '
    sign = -1 if negative else 1
    mu = w @ expected_returns
    return _objective_value(w, sign * mu)

def sharpe_ratio(w, expected_returns, cov_matrix, risk_free_rate=0.02, negative=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate the (negative) Sharpe ratio of a portfolio\n\n    :param w: asset weights in the portfolio\n    :type w: np.ndarray OR cp.Variable\n    :param expected_returns: expected return of each asset\n    :type expected_returns: np.ndarray\n    :param cov_matrix: covariance matrix\n    :type cov_matrix: np.ndarray\n    :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.\n                           The period of the risk-free rate should correspond to the\n                           frequency of expected returns.\n    :type risk_free_rate: float, optional\n    :param negative: whether quantity should be made negative (so we can minimise)\n    :type negative: boolean\n    :return: (negative) Sharpe ratio\n    :rtype: float\n    '
    mu = w @ expected_returns
    sigma = cp.sqrt(cp.quad_form(w, cov_matrix))
    sign = -1 if negative else 1
    sharpe = (mu - risk_free_rate) / sigma
    return _objective_value(w, sign * sharpe)

def L2_reg(w, gamma=1):
    if False:
        return 10
    '\n    L2 regularisation, i.e :math:`\\gamma ||w||^2`, to increase the number of nonzero weights.\n\n    Example::\n\n        ef = EfficientFrontier(mu, S)\n        ef.add_objective(objective_functions.L2_reg, gamma=2)\n        ef.min_volatility()\n\n    :param w: asset weights in the portfolio\n    :type w: np.ndarray OR cp.Variable\n    :param gamma: L2 regularisation parameter, defaults to 1. Increase if you want more\n                    non-negligible weights\n    :type gamma: float, optional\n    :return: value of the objective function OR objective function expression\n    :rtype: float OR cp.Expression\n    '
    L2_reg = gamma * cp.sum_squares(w)
    return _objective_value(w, L2_reg)

def quadratic_utility(w, expected_returns, cov_matrix, risk_aversion, negative=True):
    if False:
        return 10
    '\n    Quadratic utility function, i.e :math:`\\mu - \\frac 1 2 \\delta  w^T \\Sigma w`.\n\n    :param w: asset weights in the portfolio\n    :type w: np.ndarray OR cp.Variable\n    :param expected_returns: expected return of each asset\n    :type expected_returns: np.ndarray\n    :param cov_matrix: covariance matrix\n    :type cov_matrix: np.ndarray\n    :param risk_aversion: risk aversion coefficient. Increase to reduce risk.\n    :type risk_aversion: float\n    :param negative: whether quantity should be made negative (so we can minimise).\n    :type negative: boolean\n    :return: value of the objective function OR objective function expression\n    :rtype: float OR cp.Expression\n    '
    sign = -1 if negative else 1
    mu = w @ expected_returns
    variance = cp.quad_form(w, cov_matrix)
    risk_aversion_par = cp.Parameter(value=risk_aversion, name='risk_aversion', nonneg=True)
    utility = mu - 0.5 * risk_aversion_par * variance
    return _objective_value(w, sign * utility)

def transaction_cost(w, w_prev, k=0.001):
    if False:
        return 10
    '\n    A very simple transaction cost model: sum all the weight changes\n    and multiply by a given fraction (default to 10bps). This simulates\n    a fixed percentage commission from your broker.\n\n    :param w: asset weights in the portfolio\n    :type w: np.ndarray OR cp.Variable\n    :param w_prev: previous weights\n    :type w_prev: np.ndarray\n    :param k: fractional cost per unit weight exchanged\n    :type k: float\n    :return: value of the objective function OR objective function expression\n    :rtype: float OR cp.Expression\n    '
    return _objective_value(w, k * cp.norm(w - w_prev, 1))

def ex_ante_tracking_error(w, cov_matrix, benchmark_weights):
    if False:
        return 10
    '\n    Calculate the (square of) the ex-ante Tracking Error, i.e\n    :math:`(w - w_b)^T \\Sigma (w-w_b)`.\n\n    :param w: asset weights in the portfolio\n    :type w: np.ndarray OR cp.Variable\n    :param cov_matrix: covariance matrix\n    :type cov_matrix: np.ndarray\n    :param benchmark_weights: asset weights in the benchmark\n    :type benchmark_weights: np.ndarray\n    :return: value of the objective function OR objective function expression\n    :rtype: float OR cp.Expression\n    '
    relative_weights = w - benchmark_weights
    tracking_error = cp.quad_form(relative_weights, cov_matrix)
    return _objective_value(w, tracking_error)

def ex_post_tracking_error(w, historic_returns, benchmark_returns):
    if False:
        return 10
    '\n    Calculate the (square of) the ex-post Tracking Error, i.e :math:`Var(r - r_b)`.\n\n    :param w: asset weights in the portfolio\n    :type w: np.ndarray OR cp.Variable\n    :param historic_returns: historic asset returns\n    :type historic_returns: np.ndarray\n    :param benchmark_returns: historic benchmark returns\n    :type benchmark_returns: pd.Series or np.ndarray\n    :return: value of the objective function OR objective function expression\n    :rtype: float OR cp.Expression\n    '
    if not isinstance(historic_returns, np.ndarray):
        historic_returns = np.array(historic_returns)
    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)
    x_i = w @ historic_returns.T - benchmark_returns
    mean = cp.sum(x_i) / len(benchmark_returns)
    tracking_error = cp.sum_squares(x_i - mean)
    return _objective_value(w, tracking_error)