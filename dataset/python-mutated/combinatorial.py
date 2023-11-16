"""
Implements the following classes from Chapter 12 of AFML:

- Combinatorial Purged Cross-Validation class.
- Stacked Combinatorial Purged Cross-Validation class.
"""
from itertools import combinations
from typing import List
import pandas as pd
import numpy as np
from scipy.special import comb
from sklearn.model_selection import KFold
from mlfinlab.cross_validation.cross_validation import ml_get_train_times

def _get_number_of_backtest_paths(n_train_splits: int, n_test_splits: int) -> int:
    if False:
        i = 10
        return i + 15
    '\n    Number of combinatorial paths for CPCV(N,K).\n\n    :param n_train_splits: (int) Number of train splits.\n    :param n_test_splits: (int) Number of test splits.\n    :return: (int) Number of backtest paths for CPCV(N,k).\n    '
    pass

class CombinatorialPurgedKFold(KFold):
    """
    Advances in Financial Machine Learning, Chapter 12.

    Implements Combinatorial Purged Cross Validation (CPCV).

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(self, n_splits: int=3, n_test_splits: int=2, samples_info_sets: pd.Series=None, pct_embargo: float=0.0):
        if False:
            i = 10
            return i + 15
        '\n        Initialize.\n\n        :param n_splits: (int) The number of splits. Default to 3\n        :param samples_info_sets: (pd.Series) The information range on which each record is constructed from\n            *samples_info_sets.index*: Time when the information extraction started.\n            *samples_info_sets.value*: Time when the information extraction ended.\n        :param pct_embargo: (float) Percent that determines the embargo size.\n        '
        pass

    def _generate_combinatorial_test_ranges(self, splits_indices: dict) -> List:
        if False:
            print('Hello World!')
        '\n        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),\n        generates combinatorial test ranges splits.\n\n        :param splits_indices: (dict) Test fold integer index: [start test index, end test index].\n        :return: (list) Combinatorial test splits ([start index, end index]).\n        '
        pass

    def _fill_backtest_paths(self, train_indices: list, test_splits: list):
        if False:
            for i in range(10):
                print('nop')
        '\n        Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and\n        place in the path where these indices should be used.\n\n        :param test_splits: (list) List of lists with first element corresponding to test start index and second - test end.\n        '
        pass

    def split(self, X: pd.DataFrame, y: pd.Series=None, groups=None) -> tuple:
        if False:
            print('Hello World!')
        '\n        The main method to call for the PurgedKFold class.\n\n        :param X: (pd.DataFrame) Samples dataset that is to be split.\n        :param y: (pd.Series) Sample labels series.\n        :param groups: (array-like), with shape (n_samples,), optional\n            Group labels for the samples used while splitting the dataset into\n            train/test set.\n        :return: (tuple) [train list of sample indices, and test list of sample indices].\n        '
        pass

class StackedCombinatorialPurgedKFold(KFold):
    """
    Advances in Financial Machine Learning, Chapter 12.

    Implements Stacked Combinatorial Purged Cross Validation (CPCV). It implements CPCV for multiasset dataset.

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(self, n_splits: int=3, n_test_splits: int=2, samples_info_sets_dict: dict=None, pct_embargo: float=0.0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize.\n\n        :param n_splits: (int) The number of splits. Default to 3\n        :param samples_info_sets_dict: (dict) Dictionary of samples info sets.\n                                        ASSET_1: SAMPLE_INFO_SETS, ASSET_2:...\n\n            *samples_info_sets.index*: Time when the information extraction started.\n            *samples_info_sets.value*: Time when the information extraction ended.\n        :param pct_embargo: (float) Percent that determines the embargo size.\n        '
        pass

    def _fill_backtest_paths(self, asset, train_indices: list, test_splits: list):
        if False:
            while True:
                i = 10
        '\n        Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and\n        place in the path where these indices should be used.\n\n        :param asset: (str) Asset for which backtest paths are filled.\n        :param train_indices: (list) List of lists with first element corresponding to train start index, second - test end.\n        :param test_splits: (list) List of lists with first element corresponding to test start index and second - test end.\n        '
        pass

    def _generate_combinatorial_test_ranges(self, splits_indices: dict) -> List:
        if False:
            return 10
        '\n        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),\n        generates combinatorial test ranges splits.\n\n        :param splits_indices: (dict) Test fold integer index: [start test index, end test index].\n        :return: (list) Combinatorial test splits ([start index, end index]).\n        '
        pass

    def split(self, X_dict: dict, y_dict: dict=None, groups=None) -> tuple:
        if False:
            return 10
        '\n        The main method to call for the PurgedKFold class.\n\n        :param X_dict: (dict) Dictionary of asset : X_{asset}.\n        :param y_dict: (dict) Dictionary of asset : y_{asset}.\n        :param groups: (array-like), with shape (n_samples,), optional\n            Group labels for the samples used while splitting the dataset into\n            train/test set.\n        :return: (tuple) [train list of sample indices, and test list of sample indices].\n        '
        pass