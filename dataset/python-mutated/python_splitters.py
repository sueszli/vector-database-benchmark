import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_split
from recommenders.utils.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL
from recommenders.datasets.split_utils import process_split_ratio, min_rating_filter_pandas, split_pandas_data_with_ratios

def python_random_split(data, ratio=0.75, seed=42):
    if False:
        print('Hello World!')
    'Pandas random splitter.\n\n    The splitter randomly splits the input data.\n\n    Args:\n        data (pandas.DataFrame): Pandas DataFrame to be split.\n        ratio (float or list): Ratio for splitting data. If it is a single float number\n            it splits data into two halves and the ratio argument indicates the ratio\n            of training data set; if it is a list of float numbers, the splitter splits\n            data into several portions corresponding to the split ratios. If a list is\n            provided and the ratios are not summed to 1, they will be normalized.\n        seed (int): Seed.\n\n    Returns:\n        list: Splits of the input data as pandas.DataFrame.\n    '
    (multi_split, ratio) = process_split_ratio(ratio)
    if multi_split:
        splits = split_pandas_data_with_ratios(data, ratio, shuffle=True, seed=seed)
        splits_new = [x.drop('split_index', axis=1) for x in splits]
        return splits_new
    else:
        return sk_split(data, test_size=None, train_size=ratio, random_state=seed)

def _do_stratification(data, ratio=0.75, min_rating=1, filter_by='user', is_random=True, seed=42, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_timestamp=DEFAULT_TIMESTAMP_COL):
    if False:
        print('Hello World!')
    if not (filter_by == 'user' or filter_by == 'item'):
        raise ValueError("filter_by should be either 'user' or 'item'.")
    if min_rating < 1:
        raise ValueError('min_rating should be integer and larger than or equal to 1.')
    if col_user not in data.columns:
        raise ValueError('Schema of data not valid. Missing User Col')
    if col_item not in data.columns:
        raise ValueError('Schema of data not valid. Missing Item Col')
    if not is_random:
        if col_timestamp not in data.columns:
            raise ValueError('Schema of data not valid. Missing Timestamp Col')
    (multi_split, ratio) = process_split_ratio(ratio)
    split_by_column = col_user if filter_by == 'user' else col_item
    ratio = ratio if multi_split else [ratio, 1 - ratio]
    if min_rating > 1:
        data = min_rating_filter_pandas(data, min_rating=min_rating, filter_by=filter_by, col_user=col_user, col_item=col_item)
    if is_random:
        np.random.seed(seed)
        data['random'] = np.random.rand(data.shape[0])
        order_by = 'random'
    else:
        order_by = col_timestamp
    data = data.sort_values([split_by_column, order_by])
    groups = data.groupby(split_by_column)
    data['count'] = groups[split_by_column].transform('count')
    data['rank'] = groups.cumcount() + 1
    if is_random:
        data = data.drop('random', axis=1)
    splits = []
    prev_threshold = None
    for threshold in np.cumsum(ratio):
        condition = data['rank'] <= round(threshold * data['count'])
        if prev_threshold is not None:
            condition &= data['rank'] > round(prev_threshold * data['count'])
        splits.append(data[condition].drop(['rank', 'count'], axis=1))
        prev_threshold = threshold
    return splits

def python_chrono_split(data, ratio=0.75, min_rating=1, filter_by='user', col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_timestamp=DEFAULT_TIMESTAMP_COL):
    if False:
        i = 10
        return i + 15
    'Pandas chronological splitter.\n\n    This function splits data in a chronological manner. That is, for each user / item, the\n    split function takes proportions of ratings which is specified by the split ratio(s).\n    The split is stratified.\n\n    Args:\n        data (pandas.DataFrame): Pandas DataFrame to be split.\n        ratio (float or list): Ratio for splitting data. If it is a single float number\n            it splits data into two halves and the ratio argument indicates the ratio of\n            training data set; if it is a list of float numbers, the splitter splits\n            data into several portions corresponding to the split ratios. If a list is\n            provided and the ratios are not summed to 1, they will be normalized.\n        seed (int): Seed.\n        min_rating (int): minimum number of ratings for user or item.\n        filter_by (str): either "user" or "item", depending on which of the two is to\n            filter with min_rating.\n        col_user (str): column name of user IDs.\n        col_item (str): column name of item IDs.\n        col_timestamp (str): column name of timestamps.\n\n    Returns:\n        list: Splits of the input data as pandas.DataFrame.\n    '
    return _do_stratification(data, ratio=ratio, min_rating=min_rating, filter_by=filter_by, col_user=col_user, col_item=col_item, col_timestamp=col_timestamp, is_random=False)

def python_stratified_split(data, ratio=0.75, min_rating=1, filter_by='user', col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, seed=42):
    if False:
        while True:
            i = 10
    'Pandas stratified splitter.\n\n    For each user / item, the split function takes proportions of ratings which is\n    specified by the split ratio(s). The split is stratified.\n\n    Args:\n        data (pandas.DataFrame): Pandas DataFrame to be split.\n        ratio (float or list): Ratio for splitting data. If it is a single float number\n            it splits data into two halves and the ratio argument indicates the ratio of\n            training data set; if it is a list of float numbers, the splitter splits\n            data into several portions corresponding to the split ratios. If a list is\n            provided and the ratios are not summed to 1, they will be normalized.\n        seed (int): Seed.\n        min_rating (int): minimum number of ratings for user or item.\n        filter_by (str): either "user" or "item", depending on which of the two is to\n            filter with min_rating.\n        col_user (str): column name of user IDs.\n        col_item (str): column name of item IDs.\n\n    Returns:\n        list: Splits of the input data as pandas.DataFrame.\n    '
    return _do_stratification(data, ratio=ratio, min_rating=min_rating, filter_by=filter_by, col_user=col_user, col_item=col_item, is_random=True, seed=seed)

def numpy_stratified_split(X, ratio=0.75, seed=42):
    if False:
        return 10
    'Split the user/item affinity matrix (sparse matrix) into train and test set matrices while maintaining\n    local (i.e. per user) ratios.\n\n    Main points :\n\n    1. In a typical recommender problem, different users rate a different number of items,\n    and therefore the user/affinity matrix has a sparse structure with variable number\n    of zeroes (unrated items) per row (user). Cutting a total amount of ratings will\n    result in a non-homogeneous distribution between train and test set, i.e. some test\n    users may have many ratings while other very little if none.\n\n    2. In an unsupervised learning problem, no explicit answer is given. For this reason\n    the split needs to be implemented in a different way then in supervised learningself.\n    In the latter, one typically split the dataset by rows (by examples), ending up with\n    the same number of features but different number of examples in the train/test setself.\n    This scheme does not work in the unsupervised case, as part of the rated items needs to\n    be used as a test set for fixed number of users.\n\n    Solution:\n\n    1. Instead of cutting a total percentage, for each user we cut a relative ratio of the rated\n    items. For example, if user1 has rated 4 items and user2 10, cutting 25% will correspond to\n    1 and 2.6 ratings in the test set, approximated as 1 and 3 according to the round() function.\n    In this way, the 0.75 ratio is satisfied both locally and globally, preserving the original\n    distribution of ratings across the train and test set.\n\n    2. It is easy (and fast) to satisfy this requirements by creating the test via element subtraction\n    from the original dataset X. We first create two copies of X; for each user we select a random\n    sample of local size ratio (point 1) and erase the remaining ratings, obtaining in this way the\n    train set matrix Xtst. The train set matrix is obtained in the opposite way.\n\n    Args:\n        X (numpy.ndarray, int): a sparse matrix to be split\n        ratio (float): fraction of the entire dataset to constitute the train set\n        seed (int): random seed\n\n    Returns:\n        numpy.ndarray, numpy.ndarray:\n        - Xtr: The train set user/item affinity matrix.\n        - Xtst: The test set user/item affinity matrix.\n    '
    np.random.seed(seed)
    test_cut = int((1 - ratio) * 100)
    Xtr = X.copy()
    Xtst = X.copy()
    rated = np.sum(Xtr != 0, axis=1)
    tst = np.around(rated * test_cut / 100).astype(int)
    for u in range(X.shape[0]):
        idx = np.asarray(np.where(Xtr[u] != 0))[0].tolist()
        idx_tst = np.random.choice(idx, tst[u], replace=False)
        idx_train = list(set(idx).difference(set(idx_tst)))
        Xtr[u, idx_tst] = 0
        Xtst[u, idx_train] = 0
    del idx, idx_train, idx_tst
    return (Xtr, Xtst)