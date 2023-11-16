"""
Dataset splitting functions.
"""
import numpy as np
import scipy.sparse as sp

def _shuffle(uids, iids, data, random_state):
    if False:
        i = 10
        return i + 15
    shuffle_indices = np.arange(len(uids))
    random_state.shuffle(shuffle_indices)
    return (uids[shuffle_indices], iids[shuffle_indices], data[shuffle_indices])

def random_train_test_split(interactions, test_percentage=0.2, random_state=None):
    if False:
        while True:
            i = 10
    '\n    Randomly split interactions between training and testing.\n\n    This function takes an interaction set and splits it into\n    two disjoint sets, a training set and a test set. Note that\n    no effort is made to make sure that all items and users with\n    interactions in the test set also have interactions in the\n    training set; this may lead to a partial cold-start problem\n    in the test set.\n    To split a sample_weight matrix along the same lines, pass it\n    into this function with the same random_state seed as was used\n    for splitting the interactions.\n\n    Parameters\n    ----------\n\n    interactions: a scipy sparse matrix containing interactions\n        The interactions to split.\n    test_percentage: float, optional\n        The fraction of interactions to place in the test set.\n    random_state: int or numpy.random.RandomState, optional\n        Random seed used to initialize the numpy.random.RandomState number generator.\n        Accepts an instance of numpy.random.RandomState for backwards compatibility.\n\n    Returns\n    -------\n\n    (train, test): (scipy.sparse.COOMatrix,\n                    scipy.sparse.COOMatrix)\n         A tuple of (train data, test data)\n    '
    if not sp.issparse(interactions):
        raise ValueError('Interactions must be a scipy.sparse matrix.')
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)
    interactions = interactions.tocoo()
    shape = interactions.shape
    (uids, iids, data) = (interactions.row, interactions.col, interactions.data)
    (uids, iids, data) = _shuffle(uids, iids, data, random_state)
    cutoff = int((1.0 - test_percentage) * len(uids))
    train_idx = slice(None, cutoff)
    test_idx = slice(cutoff, None)
    train = sp.coo_matrix((data[train_idx], (uids[train_idx], iids[train_idx])), shape=shape, dtype=interactions.dtype)
    test = sp.coo_matrix((data[test_idx], (uids[test_idx], iids[test_idx])), shape=shape, dtype=interactions.dtype)
    return (train, test)