import pytest
from lightfm.cross_validation import random_train_test_split
from lightfm.datasets import fetch_movielens

def _assert_disjoint(x, y):
    if False:
        return 10
    x = x.tocsr()
    y = y.tocoo()
    for (i, j) in zip(y.row, y.col):
        assert x[i, j] == 0.0

@pytest.mark.parametrize('test_percentage', [0.2, 0.5, 0.7])
def test_random_train_test_split(test_percentage):
    if False:
        for i in range(10):
            print('nop')
    data = fetch_movielens()['train']
    (train, test) = random_train_test_split(data, test_percentage=test_percentage)
    assert test.nnz / float(data.nnz) == test_percentage
    _assert_disjoint(train, test)