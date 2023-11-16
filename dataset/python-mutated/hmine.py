import math
import numpy as np
import pandas as pd
from ..frequent_patterns import fpcommon as fpc

def hmine(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0) -> pd.DataFrame:
    if False:
        while True:
            i = 10
    "\n    Get frequent itemsets from a one-hot DataFrame\n\n    Parameters\n    -----------\n    df : pandas DataFrame\n      pandas DataFrame the encoded format. Also supports\n      DataFrames with sparse data; for more info, please\n      see https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html#sparse-data-structures.\n\n      Please note that the old pandas SparseDataFrame format\n      is no longer supported in mlxtend >= 0.17.2.\n\n      The allowed values are either 0/1 or True/False.\n      For example,\n\n    ```\n           Apple  Bananas   Beer  Chicken   Milk   Rice\n        0   True    False   True     True  False   True\n        1   True    False   True    False  False   True\n        2   True    False   True    False  False  False\n        3   True     True  False    False  False  False\n        4  False    False   True     True   True   True\n        5  False    False   True    False   True   True\n        6  False    False   True    False   True  False\n        7   True     True  False    False  False  False\n    ```\n\n    min_support : float (default: 0.5)\n      A float between 0 and 1 for minimum support of the itemsets returned.\n      The support is computed as the fraction\n      transactions_where_item(s)_occur / total_transactions.\n\n    use_colnames : bool (default: False)\n      If true, uses the DataFrames' column names in the returned DataFrame\n      instead of column indices.\n\n    max_len : int (default: None)\n      Maximum length of the itemsets generated. If `None` (default) all\n      possible itemsets lengths are evaluated.\n\n    verbose : int (default: 0)\n      Shows the stages of conditional tree generation.\n\n    Returns\n    -----------\n    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets\n      that are >= `min_support` and < than `max_len`\n      (if `max_len` is not None).\n      Each itemset in the 'itemsets' column is of type `frozenset`,\n      which is a Python built-in type that behaves similarly to\n      sets except that it is immutable\n      (For more info, see\n      https://docs.python.org/3.6/library/stdtypes.html#frozenset).\n\n    Examples\n    ----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/hmine/\n\n    "
    fpc.valid_input_check(df)
    if min_support <= 0.0:
        raise ValueError('`min_support` must be a positive number within the interval `(0, 1]`. Got %s.' % min_support)
    minsup = math.ceil(min_support * len(df))
    is_sparse = False
    if hasattr(df, 'sparse'):
        if df.size == 0:
            itemsets = df.values
        else:
            itemsets = df.sparse.to_coo().tocsr()
            is_sparse = True
    else:
        itemsets = df.values
    if is_sparse:
        is_sparse
    single_items = np.array(df.columns)
    itemsets_shape = itemsets.shape[0]
    (itemsets, single_items, single_items_support) = itemset_optimisation(itemsets, single_items, minsup)
    numeric_single_items = np.arange(len(single_items))
    frequent_itemsets = {}
    for item in numeric_single_items:
        if single_items_support[item] >= minsup:
            supp = single_items_support[item] / itemsets_shape
            frequent_itemsets[frozenset([single_items[item]])] = supp
        if max_len == 1:
            continue
        frequent_itemsets = hmine_driver([item], itemsets, minsup, itemsets_shape, max_len, verbose, single_items, frequent_itemsets)
    res_df = pd.DataFrame([frequent_itemsets.values(), frequent_itemsets.keys()]).T
    res_df.columns = ['support', 'itemsets']
    if not use_colnames:
        mapping = {item: idx for (idx, item) in enumerate(df.columns)}
        res_df['itemsets'] = res_df['itemsets'].apply(lambda x: frozenset([mapping[i] for i in x]))
    return res_df

def itemset_optimisation(itemsets: np.array, single_items: np.array, minsup: int) -> tuple:
    if False:
        return 10
    '\n    Downward-closure property of H-Mine algorithm.\n        Optimizes the itemsets matrix by removing items that do not\n        meet the minimum support. (For more info, see\n        https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/hmine/)\n\n    Args:\n        itemsets (np.array): matrix of bools or binary\n        single_items (np.array): array of single items\n        minsup (int): minimum absolute support\n\n    Returns:\n        itemsets (np.array): reduced itemsets matrix of bools or binary\n        single_items (np.array): reduced array of single items\n        single_items_support (np.array): reduced single items support\n    '
    single_items_support = np.array(np.sum(itemsets, axis=0)).reshape(-1)
    items = np.nonzero(single_items_support >= minsup)[0]
    itemsets = itemsets[:, items]
    single_items = single_items[items]
    single_items_support = single_items_support[items]
    return (itemsets, single_items, single_items_support)

def hmine_driver(item: list, itemsets: np.array, minsup: int, itemsets_shape: int, max_len: int, verbose: int, single_items: np.array, frequent_itemsets: dict) -> dict:
    if False:
        print('Hello World!')
    '\n    Driver function for the hmine algorithm.\n    Recursively generates frequent itemsets.\n    Also works for sparse matrix.\n    egg: item = [1] -> [1,2] -> [1,2,3] -> [1,2,4] -> [1,2,5]\n    For more info, see\n    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/hmine/\n\n    Args:\n        item (list): list of items\n        itemsets (np.array): matrix of bools or binary\n        minsup (int): minimum absolute support\n        itemsets_shape (int): number of transactions\n        single_items (np.array): array of single items\n        max_len (int): maximum length of frequent itemsets\n        verbose (int): verbose mode\n        frequent_itemsets (dict): dictionary of frequent itemsets\n\n    Returns:\n        frequent_itemsets(dict): dictionary of frequent itemsets\n    '
    if max_len and len(item) >= max_len:
        return frequent_itemsets
    projected_itemsets = create_projected_itemsets(item, itemsets)
    initial_supports = np.array(np.sum(projected_itemsets, axis=0)).reshape(-1)
    suffixes = np.nonzero(initial_supports >= minsup)[0]
    suffixes = suffixes[np.nonzero(suffixes > item[-1])[0]]
    if verbose:
        print(f"{len(suffixes)} itemset(s) from the suffixes on item(s) ({', '.join(single_items[item])})")
    for suffix in suffixes:
        new_item = item.copy()
        new_item.append(suffix)
        supp = initial_supports[suffix] / itemsets_shape
        frequent_itemsets[frozenset(single_items[new_item])] = supp
        frequent_itemsets = hmine_driver(new_item, projected_itemsets, minsup, itemsets_shape, max_len, verbose, single_items, frequent_itemsets)
    return frequent_itemsets

def create_projected_itemsets(item: list, itemsets: np.array) -> np.array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates the projected itemsets for the given item. (For more info, see\n    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/hmine/)\n\n    Args:\n        item (list): list of items\n        itemsets (np.array): matrix of bools or binary\n\n    Returns:\n        projected_itemsets(np.array): projected itemsets for the given item\n    '
    indices = np.nonzero(np.sum(itemsets[:, item], axis=1) == len(item))[0]
    projected_itemsets = itemsets[indices, :]
    return projected_itemsets