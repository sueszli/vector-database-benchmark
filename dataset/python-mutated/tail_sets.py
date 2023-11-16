import numpy as np
import pandas as pd

class TailSetLabels:
    """
    Tail set labels are a classification labeling technique introduced in the following paper: Nonlinear support vector
    machines can systematically identify stocks with high and low future returns. Algorithmic Finance, 2(1), pp.45-58.

    A tail set is defined to be a group of stocks whose volatility-adjusted return is in the highest or lowest
    quantile, for example the highest or lowest 5%.

    A classification model is then fit using these labels to determine which stocks to buy and sell in a long / short
    portfolio.
    """

    def __init__(self, prices, n_bins, vol_adj=None, window=None):
        if False:
            while True:
                i = 10
        '\n        :param prices: (pd.DataFrame) Asset prices.\n        :param n_bins: (int) Number of bins to determine the quantiles for defining the tail sets. The top and\n                        bottom quantiles are considered to be the positive and negative tail sets, respectively.\n        :param vol_adj: (str) Whether to take volatility adjusted returns. Allowable inputs are ``None``,\n                        ``mean_abs_dev``, and ``stdev``.\n        :param window: (int) Window period used in the calculation of the volatility adjusted returns, if vol_adj is not\n                        None. Has no impact if vol_adj is None.\n        '
        pass

    def get_tail_sets(self):
        if False:
            print('Hello World!')
        '\n        Computes the tail sets (positive and negative) and then returns a tuple with 3 elements, positive set, negative\n        set, full matrix set.\n\n        The positive and negative sets are each a series of lists with the names of the securities that fall within each\n        set at a specific timestamp.\n\n        For the full matrix a value of 1 indicates the volatility adjusted returns were in the top quantile, a value of\n        -1 for the bottom quantile.\n        :return: (tuple) positive set, negative set, full matrix set.\n        '
        pass

    def _vol_adjusted_rets(self):
        if False:
            return 10
        '\n        Computes the volatility adjusted returns. This is simply the log returns divided by a volatility estimate. We\n        have provided 2 techniques for volatility estimation: an exponential moving average and the traditional standard\n        deviation.\n        '
        pass

    def _extract_tail_sets(self, row):
        if False:
            for i in range(10):
                print('nop')
        '\n        Method used in a .apply() setting to transform each row in a DataFrame to the positive and negative tail sets.\n\n        This method splits the data into quantiles determined by the user, with n_bins.\n\n        :param row: (pd.Series) Vol adjusted returns for a given date.\n        :return: (pd.Series) Tail set with positive and negative labels.\n        '
        pass

    @staticmethod
    def _positive_tail_set(row):
        if False:
            return 10
        '\n        Takes as input a row from the vol_adj_ret DataFrame and then returns a list of names of the securities in the\n        positive tail set, for this specific row date.\n\n        This method is used in an apply() setting.\n\n        :param row: (pd.Series) Labeled row of several stocks where each is already labeled with +1 (positive tail set),\n                    -1 (negative tail set), or 0.\n        :return: (list) Securities in the positive tail set.\n        '
        pass

    @staticmethod
    def _negative_tail_set(row):
        if False:
            return 10
        '\n        Takes as input a row from the vol_adj_ret DataFrame and then returns a list of names of the securities in the\n        negative tail set, for this specific row date.\n\n        This method is used in an apply() setting.\n\n        :param row: (pd.Series) Labeled row of several stocks where each is already labeled with +1 (positive tail set),\n                    -1 (negative tail set), or 0.\n        :return: (list) Securities in the negative tail set.\n        '
        pass