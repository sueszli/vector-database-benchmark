"""
The ``discrete_allocation`` module contains the ``DiscreteAllocation`` class, which
offers multiple methods to generate a discrete portfolio allocation from continuous weights.
"""
import collections
import cvxpy as cp
import numpy as np
import pandas as pd
from . import exceptions

def get_latest_prices(prices):
    if False:
        i = 10
        return i + 15
    '\n    A helper tool which retrieves the most recent asset prices from a dataframe of\n    asset prices, required in order to generate a discrete allocation.\n\n    :param prices: historical asset prices\n    :type prices: pd.DataFrame\n    :raises TypeError: if prices are not in a dataframe\n    :return: the most recent price of each asset\n    :rtype: pd.Series\n    '
    if not isinstance(prices, pd.DataFrame):
        raise TypeError('prices not in a dataframe')
    return prices.ffill().iloc[-1]

class DiscreteAllocation:
    """
    Generate a discrete portfolio allocation from continuous weights

    Instance variables:

    - Inputs:

        - ``weights`` - dict
        - ``latest_prices`` - pd.Series or dict
        - ``total_portfolio_value`` - int/float
        - ``short_ratio``- float

    - Output: ``allocation`` - dict

    Public methods:

    - ``greedy_portfolio()`` - uses a greedy algorithm
    - ``lp_portfolio()`` - uses linear programming
    """

    def __init__(self, weights, latest_prices, total_portfolio_value=10000, short_ratio=None):
        if False:
            return 10
        "\n        :param weights: continuous weights generated from the ``efficient_frontier`` module\n        :type weights: dict\n        :param latest_prices: the most recent price for each asset\n        :type latest_prices: pd.Series\n        :param total_portfolio_value: the desired total value of the portfolio, defaults to 10000\n        :type total_portfolio_value: int/float, optional\n        :param short_ratio: the short ratio, e.g 0.3 corresponds to 130/30. If None,\n                            defaults to the input weights.\n        :type short_ratio: float, defaults to None.\n        :raises TypeError: if ``weights`` is not a dict\n        :raises TypeError: if ``latest_prices`` isn't a series\n        :raises ValueError: if ``short_ratio < 0``\n        "
        if not isinstance(weights, dict):
            raise TypeError('weights should be a dictionary of {ticker: weight}')
        if any((np.isnan(val) for val in weights.values())):
            raise ValueError('weights should have no NaNs')
        if not isinstance(latest_prices, pd.Series) or any(np.isnan(latest_prices)):
            raise TypeError('latest_prices should be a pd.Series with no NaNs')
        if total_portfolio_value <= 0:
            raise ValueError('total_portfolio_value must be greater than zero')
        if short_ratio is not None and short_ratio < 0:
            raise ValueError('short_ratio must be non-negative')
        self.weights = list(weights.items())
        self.latest_prices = latest_prices
        self.total_portfolio_value = total_portfolio_value
        if short_ratio is None:
            self.short_ratio = sum((-x[1] for x in self.weights if x[1] < 0))
        else:
            self.short_ratio = short_ratio

    @staticmethod
    def _remove_zero_positions(allocation):
        if False:
            print('Hello World!')
        '\n        Utility function to remove zero positions (i.e with no shares being bought)\n\n        :type allocation: dict\n        '
        return {k: v for (k, v) in allocation.items() if v != 0}

    def _allocation_rmse_error(self, verbose=True):
        if False:
            i = 10
            return i + 15
        '\n        Utility function to calculate and print RMSE error between discretised\n        weights and continuous weights. RMSE was used instead of MAE because we\n        want to penalise large variations.\n\n        :param verbose: print weight discrepancies?\n        :type verbose: bool\n        :return: rmse error\n        :rtype: float\n        '
        portfolio_val = 0
        for (ticker, num) in self.allocation.items():
            portfolio_val += num * self.latest_prices[ticker]
        sse = 0
        for (ticker, weight) in self.weights:
            if ticker in self.allocation:
                allocation_weight = self.allocation[ticker] * self.latest_prices[ticker] / portfolio_val
            else:
                allocation_weight = 0
            sse += (weight - allocation_weight) ** 2
            if verbose:
                print('{}: allocated {:.3f}, desired {:.3f}'.format(ticker, allocation_weight, weight))
        rmse = np.sqrt(sse / len(self.weights))
        print('Allocation has RMSE: {:.3f}'.format(rmse))
        return rmse

    def greedy_portfolio(self, reinvest=False, verbose=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert continuous weights into a discrete portfolio allocation\n        using a greedy iterative approach.\n\n        :param reinvest: whether or not to reinvest cash gained from shorting\n        :type reinvest: bool, defaults to False\n        :param verbose: print error analysis?\n        :type verbose: bool, defaults to False\n        :return: the number of shares of each ticker that should be purchased,\n                 along with the amount of funds leftover.\n        :rtype: (dict, float)\n        '
        self.weights.sort(key=lambda x: x[1], reverse=True)
        if self.weights[-1][1] < 0:
            longs = {t: w for (t, w) in self.weights if w >= 0}
            shorts = {t: -w for (t, w) in self.weights if w < 0}
            long_total_weight = sum(longs.values())
            short_total_weight = sum(shorts.values())
            longs = {t: w / long_total_weight for (t, w) in longs.items()}
            shorts = {t: w / short_total_weight for (t, w) in shorts.items()}
            short_val = self.total_portfolio_value * self.short_ratio
            long_val = self.total_portfolio_value
            if reinvest:
                long_val += short_val
            if verbose:
                print('\nAllocating long sub-portfolio...')
            da1 = DiscreteAllocation(longs, self.latest_prices[longs.keys()], total_portfolio_value=long_val)
            (long_alloc, long_leftover) = da1.greedy_portfolio()
            if verbose:
                print('\nAllocating short sub-portfolio...')
            da2 = DiscreteAllocation(shorts, self.latest_prices[shorts.keys()], total_portfolio_value=short_val)
            (short_alloc, short_leftover) = da2.greedy_portfolio()
            short_alloc = {t: -w for (t, w) in short_alloc.items()}
            self.allocation = long_alloc.copy()
            self.allocation.update(short_alloc)
            self.allocation = self._remove_zero_positions(self.allocation)
            return (self.allocation, long_leftover + short_leftover)
        available_funds = self.total_portfolio_value
        shares_bought = []
        buy_prices = []
        for (ticker, weight) in self.weights:
            price = self.latest_prices[ticker]
            n_shares = int(weight * self.total_portfolio_value / price)
            cost = n_shares * price
            assert cost <= available_funds, 'Unexpectedly insufficient funds.'
            available_funds -= cost
            shares_bought.append(n_shares)
            buy_prices.append(price)
        while available_funds > 0:
            current_weights = np.array(buy_prices) * np.array(shares_bought)
            current_weights /= current_weights.sum()
            ideal_weights = np.array([i[1] for i in self.weights])
            deficit = ideal_weights - current_weights
            idx = np.argmax(deficit)
            (ticker, weight) = self.weights[idx]
            price = self.latest_prices[ticker]
            counter = 0
            while price > available_funds:
                deficit[idx] = 0
                idx = np.argmax(deficit)
                if deficit[idx] < 0 or counter == 10:
                    break
                (ticker, weight) = self.weights[idx]
                price = self.latest_prices[ticker]
                counter += 1
            if deficit[idx] <= 0 or counter == 10:
                break
            shares_bought[idx] += 1
            available_funds -= price
        self.allocation = self._remove_zero_positions(collections.OrderedDict(zip([i[0] for i in self.weights], shares_bought)))
        if verbose:
            print('Funds remaining: {:.2f}'.format(available_funds))
            self._allocation_rmse_error(verbose)
        return (self.allocation, available_funds)

    def lp_portfolio(self, reinvest=False, verbose=False, solver='ECOS_BB'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert continuous weights into a discrete portfolio allocation\n        using integer programming.\n\n        :param reinvest: whether or not to reinvest cash gained from shorting\n        :type reinvest: bool, defaults to False\n        :param verbose: print error analysis?\n        :type verbose: bool\n        :param solver: the CVXPY solver to use (must support mixed-integer programs)\n        :type solver: str, defaults to "ECOS_BB"\n        :return: the number of shares of each ticker that should be purchased, along with the amount\n                of funds leftover.\n        :rtype: (dict, float)\n        '
        if any([w < 0 for (_, w) in self.weights]):
            longs = {t: w for (t, w) in self.weights if w >= 0}
            shorts = {t: -w for (t, w) in self.weights if w < 0}
            long_total_weight = sum(longs.values())
            short_total_weight = sum(shorts.values())
            longs = {t: w / long_total_weight for (t, w) in longs.items()}
            shorts = {t: w / short_total_weight for (t, w) in shorts.items()}
            short_val = self.total_portfolio_value * self.short_ratio
            long_val = self.total_portfolio_value
            if reinvest:
                long_val += short_val
            if verbose:
                print('\nAllocating long sub-portfolio:')
            da1 = DiscreteAllocation(longs, self.latest_prices[longs.keys()], total_portfolio_value=long_val)
            (long_alloc, long_leftover) = da1.lp_portfolio(solver=solver)
            if verbose:
                print('\nAllocating short sub-portfolio:')
            da2 = DiscreteAllocation(shorts, self.latest_prices[shorts.keys()], total_portfolio_value=short_val)
            (short_alloc, short_leftover) = da2.lp_portfolio(solver=solver)
            short_alloc = {t: -w for (t, w) in short_alloc.items()}
            self.allocation = long_alloc.copy()
            self.allocation.update(short_alloc)
            self.allocation = self._remove_zero_positions(self.allocation)
            return (self.allocation, long_leftover + short_leftover)
        p = self.latest_prices.values
        n = len(p)
        w = np.fromiter([i[1] for i in self.weights], dtype=float)
        x = cp.Variable(n, integer=True)
        r = self.total_portfolio_value - p.T @ x
        eta = w * self.total_portfolio_value - cp.multiply(x, p)
        u = cp.Variable(n)
        constraints = [eta <= u, eta >= -u, x >= 0, r >= 0]
        objective = cp.sum(u) + r
        opt = cp.Problem(cp.Minimize(objective), constraints)
        opt.solve(solver=solver)
        if opt.status not in {'optimal', 'optimal_inaccurate'}:
            raise exceptions.OptimizationError('Please try greedy_portfolio')
        vals = np.rint(x.value).astype(int)
        self.allocation = self._remove_zero_positions(collections.OrderedDict(zip([i[0] for i in self.weights], vals)))
        if verbose:
            print('Funds remaining: {:.2f}'.format(r.value))
            self._allocation_rmse_error()
        return (self.allocation, r.value)