import itertools
import math
from ..frequent_patterns import fpcommon as fpc

def fpgrowth(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0):
    if False:
        i = 10
        return i + 15
    "Get frequent itemsets from a one-hot DataFrame\n\n    Parameters\n    -----------\n    df : pandas DataFrame\n      pandas DataFrame the encoded format. Also supports\n      DataFrames with sparse data; for more info, please\n      see https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html#sparse-data-structures.\n\n      Please note that the old pandas SparseDataFrame format\n      is no longer supported in mlxtend >= 0.17.2.\n\n      The allowed values are either 0/1 or True/False.\n      For example,\n\n    ```\n           Apple  Bananas   Beer  Chicken   Milk   Rice\n        0   True    False   True     True  False   True\n        1   True    False   True    False  False   True\n        2   True    False   True    False  False  False\n        3   True     True  False    False  False  False\n        4  False    False   True     True   True   True\n        5  False    False   True    False   True   True\n        6  False    False   True    False   True  False\n        7   True     True  False    False  False  False\n    ```\n\n    min_support : float (default: 0.5)\n      A float between 0 and 1 for minimum support of the itemsets returned.\n      The support is computed as the fraction\n      transactions_where_item(s)_occur / total_transactions.\n\n    use_colnames : bool (default: False)\n      If true, uses the DataFrames' column names in the returned DataFrame\n      instead of column indices.\n\n    max_len : int (default: None)\n      Maximum length of the itemsets generated. If `None` (default) all\n      possible itemsets lengths are evaluated.\n\n    verbose : int (default: 0)\n      Shows the stages of conditional tree generation.\n\n    Returns\n    -----------\n    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets\n      that are >= `min_support` and < than `max_len`\n      (if `max_len` is not None).\n      Each itemset in the 'itemsets' column is of type `frozenset`,\n      which is a Python built-in type that behaves similarly to\n      sets except that it is immutable\n      (For more info, see\n      https://docs.python.org/3.6/library/stdtypes.html#frozenset).\n\n    Examples\n    ----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/\n\n    "
    fpc.valid_input_check(df)
    if min_support <= 0.0:
        raise ValueError('`min_support` must be a positive number within the interval `(0, 1]`. Got %s.' % min_support)
    colname_map = None
    if use_colnames:
        colname_map = {idx: item for (idx, item) in enumerate(df.columns)}
    (tree, _) = fpc.setup_fptree(df, min_support)
    minsup = math.ceil(min_support * len(df.index))
    generator = fpg_step(tree, minsup, colname_map, max_len, verbose)
    return fpc.generate_itemsets(generator, len(df.index), colname_map)

def fpg_step(tree, minsup, colnames, max_len, verbose):
    if False:
        for i in range(10):
            print('nop')
    '\n    Performs a recursive step of the fpgrowth algorithm.\n\n    Parameters\n    ----------\n    tree : FPTree\n    minsup : int\n\n    Yields\n    ------\n    lists of strings\n        Set of items that has occurred in minsup itemsets.\n    '
    count = 0
    items = tree.nodes.keys()
    if tree.is_path():
        size_remain = len(items) + 1
        if max_len:
            size_remain = max_len - len(tree.cond_items) + 1
        for i in range(1, size_remain):
            for itemset in itertools.combinations(items, i):
                count += 1
                support = min([tree.nodes[i][0].count for i in itemset])
                yield (support, tree.cond_items + list(itemset))
    elif not max_len or max_len > len(tree.cond_items):
        for item in items:
            count += 1
            support = sum([node.count for node in tree.nodes[item]])
            yield (support, tree.cond_items + [item])
    if verbose:
        tree.print_status(count, colnames)
    if not tree.is_path() and (not max_len or max_len > len(tree.cond_items)):
        for item in items:
            cond_tree = tree.conditional_tree(item, minsup)
            for (sup, iset) in fpg_step(cond_tree, minsup, colnames, max_len, verbose):
                yield (sup, iset)