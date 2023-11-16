import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

class TransactionEncoder(BaseEstimator, TransformerMixin):
    """Encoder class for transaction data in Python lists

    Parameters
    ------------
    None

    Attributes
    ------------
    columns_: list
      List of unique names in the `X` input list of lists

    Examples
    ------------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/

    """

    def __init__(self):
        if False:
            print('Hello World!')
        return None

    def fit(self, X):
        if False:
            return 10
        "Learn unique column names from transaction DataFrame\n\n        Parameters\n        ------------\n        X : list of lists\n          A python list of lists, where the outer list stores the\n          n transactions and the inner list stores the items in each\n          transaction.\n\n          For example,\n          [['Apple', 'Beer', 'Rice', 'Chicken'],\n           ['Apple', 'Beer', 'Rice'],\n           ['Apple', 'Beer'],\n           ['Apple', 'Bananas'],\n           ['Milk', 'Beer', 'Rice', 'Chicken'],\n           ['Milk', 'Beer', 'Rice'],\n           ['Milk', 'Beer'],\n           ['Apple', 'Bananas']]\n\n        "
        unique_items = set()
        for transaction in X:
            for item in transaction:
                unique_items.add(item)
        self.columns_ = sorted(unique_items)
        columns_mapping = {}
        for (col_idx, item) in enumerate(self.columns_):
            columns_mapping[item] = col_idx
        self.columns_mapping_ = columns_mapping
        return self

    def transform(self, X, sparse=False):
        if False:
            return 10
        "Transform transactions into a one-hot encoded NumPy array.\n\n        Parameters\n        ------------\n        X : list of lists\n          A python list of lists, where the outer list stores the\n          n transactions and the inner list stores the items in each\n          transaction.\n\n          For example,\n          [['Apple', 'Beer', 'Rice', 'Chicken'],\n           ['Apple', 'Beer', 'Rice'],\n           ['Apple', 'Beer'],\n           ['Apple', 'Bananas'],\n           ['Milk', 'Beer', 'Rice', 'Chicken'],\n           ['Milk', 'Beer', 'Rice'],\n           ['Milk', 'Beer'],\n           ['Apple', 'Bananas']]\n\n        sparse: bool (default=False)\n          If True, transform will return Compressed Sparse Row matrix\n          instead of the regular one.\n\n        Returns\n        ------------\n        array : NumPy array [n_transactions, n_unique_items]\n           if sparse=False (default).\n           Compressed Sparse Row matrix otherwise\n           The one-hot encoded boolean array of the input transactions,\n           where the columns represent the unique items found in the input\n           array in alphabetic order. Exact representation depends\n           on the sparse argument\n\n           For example,\n           array([[True , False, True , True , False, True ],\n                  [True , False, True , False, False, True ],\n                  [True , False, True , False, False, False],\n                  [True , True , False, False, False, False],\n                  [False, False, True , True , True , True ],\n                  [False, False, True , False, True , True ],\n                  [False, False, True , False, True , False],\n                  [True , True , False, False, False, False]])\n          The corresponding column labels are available as self.columns_, e.g.,\n          ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']\n        "
        if sparse:
            indptr = [0]
            indices = []
            for transaction in X:
                for item in set(transaction):
                    col_idx = self.columns_mapping_[item]
                    indices.append(col_idx)
                indptr.append(len(indices))
            non_sparse_values = [True] * len(indices)
            array = csr_matrix((non_sparse_values, indices, indptr), dtype=bool)
        else:
            array = np.zeros((len(X), len(self.columns_)), dtype=bool)
            for (row_idx, transaction) in enumerate(X):
                for item in transaction:
                    col_idx = self.columns_mapping_[item]
                    array[row_idx, col_idx] = True
        return array

    def inverse_transform(self, array):
        if False:
            while True:
                i = 10
        "Transforms an encoded NumPy array back into transactions.\n\n        Parameters\n        ------------\n        array : NumPy array [n_transactions, n_unique_items]\n            The NumPy one-hot encoded boolean array of the input transactions,\n            where the columns represent the unique items found in the input\n            array in alphabetic order\n\n            For example,\n            ```\n            array([[True , False, True , True , False, True ],\n                  [True , False, True , False, False, True ],\n                  [True , False, True , False, False, False],\n                  [True , True , False, False, False, False],\n                  [False, False, True , True , True , True ],\n                  [False, False, True , False, True , True ],\n                  [False, False, True , False, True , False],\n                  [True , True , False, False, False, False]])\n            ```\n            The corresponding column labels are available as self.columns_,\n            e.g., ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']\n\n        Returns\n        ------------\n        X : list of lists\n            A python list of lists, where the outer list stores the\n            n transactions and the inner list stores the items in each\n            transaction.\n\n          For example,\n          ```\n          [['Apple', 'Beer', 'Rice', 'Chicken'],\n           ['Apple', 'Beer', 'Rice'],\n           ['Apple', 'Beer'],\n           ['Apple', 'Bananas'],\n           ['Milk', 'Beer', 'Rice', 'Chicken'],\n           ['Milk', 'Beer', 'Rice'],\n           ['Milk', 'Beer'],\n           ['Apple', 'Bananas']]\n          ```\n\n        "
        return [[self.columns_[idx] for (idx, cell) in enumerate(row) if cell] for row in array]

    def fit_transform(self, X, sparse=False):
        if False:
            while True:
                i = 10
        'Fit a TransactionEncoder encoder and transform a dataset.'
        return self.fit(X).transform(X, sparse=sparse)