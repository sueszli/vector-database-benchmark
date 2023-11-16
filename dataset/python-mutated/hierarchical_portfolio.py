"""
The ``hierarchical_portfolio`` module seeks to implement one of the recent advances in
portfolio optimization – the application of hierarchical clustering models in allocation.

All of the hierarchical classes have a similar API to ``EfficientFrontier``, though since
many hierarchical models currently don't support different objectives, the actual allocation
happens with a call to `optimize()`.

Currently implemented:

- ``HRPOpt`` implements the Hierarchical Risk Parity (HRP) portfolio. Code reproduced with
  permission from Marcos Lopez de Prado (2016).
"""
import collections
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from . import base_optimizer, risk_models

class HRPOpt(base_optimizer.BaseOptimizer):
    """
    A HRPOpt object (inheriting from BaseOptimizer) constructs a hierarchical
    risk parity portfolio.

    Instance variables:

    - Inputs

        - ``n_assets`` - int
        - ``tickers`` - str list
        - ``returns`` - pd.DataFrame

    - Output:

        - ``weights`` - np.ndarray
        - ``clusters`` - linkage matrix corresponding to clustered assets.

    Public methods:

    - ``optimize()`` calculates weights using HRP
    - ``portfolio_performance()`` calculates the expected return, volatility and Sharpe ratio for
      the optimized portfolio.
    - ``set_weights()`` creates self.weights (np.ndarray) from a weights dict
    - ``clean_weights()`` rounds the weights and clips near-zeros.
    - ``save_weights_to_file()`` saves the weights to csv, json, or txt.
    """

    def __init__(self, returns=None, cov_matrix=None):
        if False:
            return 10
        '\n        :param returns: asset historical returns\n        :type returns: pd.DataFrame\n        :param cov_matrix: covariance of asset returns\n        :type cov_matrix: pd.DataFrame.\n        :raises TypeError: if ``returns`` is not a dataframe\n        '
        if returns is None and cov_matrix is None:
            raise ValueError('Either returns or cov_matrix must be provided')
        if returns is not None and (not isinstance(returns, pd.DataFrame)):
            raise TypeError('returns are not a dataframe')
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.clusters = None
        if returns is None:
            tickers = list(cov_matrix.columns)
        else:
            tickers = list(returns.columns)
        super().__init__(len(tickers), tickers)

    @staticmethod
    def _get_cluster_var(cov, cluster_items):
        if False:
            i = 10
            return i + 15
        '\n        Compute the variance per cluster\n\n        :param cov: covariance matrix\n        :type cov: np.ndarray\n        :param cluster_items: tickers in the cluster\n        :type cluster_items: list\n        :return: the variance per cluster\n        :rtype: float\n        '
        cov_slice = cov.loc[cluster_items, cluster_items]
        weights = 1 / np.diag(cov_slice)
        weights /= weights.sum()
        return np.linalg.multi_dot((weights, cov_slice, weights))

    @staticmethod
    def _get_quasi_diag(link):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sort clustered items by distance\n\n        :param link: linkage matrix after clustering\n        :type link: np.ndarray\n        :return: sorted list of indices\n        :rtype: list\n        '
        return sch.to_tree(link, rd=False).pre_order()

    @staticmethod
    def _raw_hrp_allocation(cov, ordered_tickers):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given the clusters, compute the portfolio that minimises risk by\n        recursively traversing the hierarchical tree from the top.\n\n        :param cov: covariance matrix\n        :type cov: np.ndarray\n        :param ordered_tickers: list of tickers ordered by distance\n        :type ordered_tickers: str list\n        :return: raw portfolio weights\n        :rtype: pd.Series\n        '
        w = pd.Series(1, index=ordered_tickers)
        cluster_items = [ordered_tickers]
        while len(cluster_items) > 0:
            cluster_items = [i[j:k] for i in cluster_items for (j, k) in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            for i in range(0, len(cluster_items), 2):
                first_cluster = cluster_items[i]
                second_cluster = cluster_items[i + 1]
                first_variance = HRPOpt._get_cluster_var(cov, first_cluster)
                second_variance = HRPOpt._get_cluster_var(cov, second_cluster)
                alpha = 1 - first_variance / (first_variance + second_variance)
                w[first_cluster] *= alpha
                w[second_cluster] *= 1 - alpha
        return w

    def optimize(self, linkage_method='single'):
        if False:
            print('Hello World!')
        '\n        Construct a hierarchical risk parity portfolio, using Scipy hierarchical clustering\n        (see `here <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_)\n\n        :param linkage_method: which scipy linkage method to use\n        :type linkage_method: str\n        :return: weights for the HRP portfolio\n        :rtype: OrderedDict\n        '
        if linkage_method not in sch._LINKAGE_METHODS:
            raise ValueError('linkage_method must be one recognised by scipy')
        if self.returns is None:
            cov = self.cov_matrix
            corr = risk_models.cov_to_corr(self.cov_matrix).round(6)
        else:
            (corr, cov) = (self.returns.corr(), self.returns.cov())
        matrix = np.sqrt(np.clip((1.0 - corr) / 2.0, a_min=0.0, a_max=1.0))
        dist = ssd.squareform(matrix, checks=False)
        self.clusters = sch.linkage(dist, linkage_method)
        sort_ix = HRPOpt._get_quasi_diag(self.clusters)
        ordered_tickers = corr.index[sort_ix].tolist()
        hrp = HRPOpt._raw_hrp_allocation(cov, ordered_tickers)
        weights = collections.OrderedDict(hrp.sort_index())
        self.set_weights(weights)
        return weights

    def portfolio_performance(self, verbose=False, risk_free_rate=0.02, frequency=252):
        if False:
            for i in range(10):
                print('nop')
        '\n        After optimising, calculate (and optionally print) the performance of the optimal\n        portfolio. Currently calculates expected return, volatility, and the Sharpe ratio\n        assuming returns are daily\n\n        :param verbose: whether performance should be printed, defaults to False\n        :type verbose: bool, optional\n        :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.\n                               The period of the risk-free rate should correspond to the\n                               frequency of expected returns.\n        :type risk_free_rate: float, optional\n        :param frequency: number of time periods in a year, defaults to 252 (the number\n                            of trading days in a year)\n        :type frequency: int, optional\n        :raises ValueError: if weights have not been calculated yet\n        :return: expected return, volatility, Sharpe ratio.\n        :rtype: (float, float, float)\n        '
        if self.returns is None:
            cov = self.cov_matrix
            mu = None
        else:
            cov = self.returns.cov() * frequency
            mu = self.returns.mean() * frequency
        return base_optimizer.portfolio_performance(self.weights, mu, cov, verbose, risk_free_rate)