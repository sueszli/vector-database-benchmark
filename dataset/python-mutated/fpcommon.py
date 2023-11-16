import collections
import warnings
from distutils.version import LooseVersion as Version
import numpy as np
import pandas as pd
from pandas import __version__ as pandas_version
warnings.simplefilter('always', DeprecationWarning)

def setup_fptree(df, min_support):
    if False:
        i = 10
        return i + 15
    num_itemsets = len(df.index)
    is_sparse = False
    if hasattr(df, 'sparse'):
        if df.size == 0:
            itemsets = df.values
        else:
            itemsets = df.sparse.to_coo().tocsr()
            is_sparse = True
    else:
        itemsets = df.values
    item_support = np.array(np.sum(itemsets, axis=0) / float(num_itemsets))
    item_support = item_support.reshape(-1)
    items = np.nonzero(item_support >= min_support)[0]
    indices = item_support[items].argsort()
    rank = {item: i for (i, item) in enumerate(items[indices])}
    if is_sparse:
        itemsets.eliminate_zeros()
    tree = FPTree(rank)
    for i in range(num_itemsets):
        if is_sparse:
            nonnull = itemsets.indices[itemsets.indptr[i]:itemsets.indptr[i + 1]]
        else:
            nonnull = np.where(itemsets[i, :])[0]
        itemset = [item for item in nonnull if item in rank]
        itemset.sort(key=rank.get, reverse=True)
        tree.insert_itemset(itemset)
    return (tree, rank)

def generate_itemsets(generator, num_itemsets, colname_map):
    if False:
        while True:
            i = 10
    itemsets = []
    supports = []
    for (sup, iset) in generator:
        itemsets.append(frozenset(iset))
        supports.append(sup / num_itemsets)
    res_df = pd.DataFrame({'support': supports, 'itemsets': itemsets})
    if colname_map is not None:
        res_df['itemsets'] = res_df['itemsets'].apply(lambda x: frozenset([colname_map[i] for i in x]))
    return res_df

def valid_input_check(df):
    if False:
        for i in range(10):
            print('nop')
    if f'{type(df)}' == "<class 'pandas.core.frame.SparseDataFrame'>":
        msg = 'SparseDataFrame support has been deprecated in pandas 1.0, and is no longer supported in mlxtend.  Please see the pandas migration guide at https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html#sparse-data-structures for supporting sparse data in DataFrames.'
        raise TypeError(msg)
    if df.size == 0:
        return
    if hasattr(df, 'sparse'):
        if not isinstance(df.columns[0], str) and df.columns[0] != 0:
            raise ValueError('Due to current limitations in Pandas, if the sparse format has integer column names,names, please make sure they either start with `0` or cast them as string column names: `df.columns = [str(i) for i in df.columns`].')
    all_bools = df.dtypes.apply(pd.api.types.is_bool_dtype).all()
    if not all_bools:
        warnings.warn('DataFrames with non-bool types result in worse computationalperformance and their support might be discontinued in the future.Please use a DataFrame with bool type', DeprecationWarning)
        if hasattr(df, 'sparse'):
            if df.size == 0:
                values = df.values
            else:
                values = df.sparse.to_coo().tocoo().data
        else:
            values = df.values
        idxs = np.where((values != 1) & (values != 0))
        if len(idxs[0]) > 0:
            val = values[tuple((loc[0] for loc in idxs))]
            s = 'The allowed values for a DataFrame are True, False, 0, 1. Found value %s' % val
            raise ValueError(s)

class FPTree(object):

    def __init__(self, rank=None):
        if False:
            for i in range(10):
                print('nop')
        self.root = FPNode(None)
        self.nodes = collections.defaultdict(list)
        self.cond_items = []
        self.rank = rank

    def conditional_tree(self, cond_item, minsup):
        if False:
            return 10
        '\n        Creates and returns the subtree of self conditioned on cond_item.\n\n        Parameters\n        ----------\n        cond_item : int | str\n            Item that the tree (self) will be conditioned on.\n        minsup : int\n            Minimum support threshold.\n\n        Returns\n        -------\n        cond_tree : FPtree\n        '
        branches = []
        count = collections.defaultdict(int)
        for node in self.nodes[cond_item]:
            branch = node.itempath_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count
        items = [item for item in count if count[item] >= minsup]
        items.sort(key=count.get)
        rank = {item: i for (i, item) in enumerate(items)}
        cond_tree = FPTree(rank)
        for (idx, branch) in enumerate(branches):
            branch = sorted([i for i in branch if i in rank], key=rank.get, reverse=True)
            cond_tree.insert_itemset(branch, self.nodes[cond_item][idx].count)
        cond_tree.cond_items = self.cond_items + [cond_item]
        return cond_tree

    def insert_itemset(self, itemset, count=1):
        if False:
            i = 10
            return i + 15
        '\n        Inserts a list of items into the tree.\n\n        Parameters\n        ----------\n        itemset : list\n            Items that will be inserted into the tree.\n        count : int\n            The number of occurrences of the itemset.\n        '
        self.root.count += count
        if len(itemset) == 0:
            return
        index = 0
        node = self.root
        for item in itemset:
            if item in node.children:
                child = node.children[item]
                child.count += count
                node = child
                index += 1
            else:
                break
        for item in itemset[index:]:
            child_node = FPNode(item, count, node)
            self.nodes[item].append(child_node)
            node = child_node

    def is_path(self):
        if False:
            return 10
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if len(self.nodes[i]) > 1 or len(self.nodes[i][0].children) > 1:
                return False
        return True

    def print_status(self, count, colnames):
        if False:
            print('Hello World!')
        cond_items = [str(i) for i in self.cond_items]
        if colnames:
            cond_items = [str(colnames[i]) for i in self.cond_items]
        cond_items = ', '.join(cond_items)
        print('\r%d itemset(s) from tree conditioned on items (%s)' % (count, cond_items), end='\n')

class FPNode(object):

    def __init__(self, item, count=0, parent=None):
        if False:
            print('Hello World!')
        self.item = item
        self.count = count
        self.parent = parent
        self.children = collections.defaultdict(FPNode)
        if parent is not None:
            parent.children[item] = self

    def itempath_from_root(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the top-down sequence of items from self to\n        (but not including) the root node.'
        path = []
        if self.item is None:
            return path
        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent
        path.reverse()
        return path