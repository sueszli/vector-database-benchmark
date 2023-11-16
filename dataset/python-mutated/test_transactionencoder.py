import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import clone
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.utils import assert_raises
dataset = [['Apple', 'Beer', 'Rice', 'Chicken'], ['Apple', 'Beer', 'Rice'], ['Apple', 'Beer'], ['Apple', 'Bananas'], ['Milk', 'Beer', 'Rice', 'Chicken'], ['Milk', 'Beer', 'Rice'], ['Milk', 'Beer'], ['Apple', 'Bananas']]
data_sorted = [['Apple', 'Beer', 'Chicken', 'Rice'], ['Apple', 'Beer', 'Rice'], ['Apple', 'Beer'], ['Apple', 'Bananas'], ['Beer', 'Chicken', 'Milk', 'Rice'], ['Beer', 'Milk', 'Rice'], ['Beer', 'Milk'], ['Apple', 'Bananas']]
expect = np.array([[1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1], [0, 0, 1, 0, 1, 1], [0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 0]])

def test_fit():
    if False:
        return 10
    oht = TransactionEncoder()
    oht.fit(dataset)
    assert oht.columns_ == ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']

def test_transform():
    if False:
        return 10
    oht = TransactionEncoder()
    oht.fit(dataset)
    trans = oht.transform(dataset)
    np.testing.assert_array_equal(expect, trans)

def test_transform_sparse():
    if False:
        return 10
    oht = TransactionEncoder()
    oht.fit(dataset)
    trans = oht.transform(dataset, sparse=True)
    assert isinstance(trans, csr_matrix)
    np.testing.assert_array_equal(expect, trans.todense())

def test_fit_transform():
    if False:
        return 10
    oht = TransactionEncoder()
    trans = oht.fit_transform(dataset)
    np.testing.assert_array_equal(expect, trans)

def test_inverse_transform():
    if False:
        i = 10
        return i + 15
    oht = TransactionEncoder()
    oht.fit(dataset)
    np.testing.assert_array_equal(np.array(data_sorted), np.array(oht.inverse_transform(expect)))

def test_cloning():
    if False:
        print('Hello World!')
    oht = TransactionEncoder()
    oht.fit(dataset)
    oht2 = clone(oht)
    msg = "'TransactionEncoder' object has no attribute 'columns_'"
    assert_raises(AttributeError, msg, oht2.transform, dataset)
    trans = oht2.fit_transform(dataset)
    np.testing.assert_array_equal(expect, trans)