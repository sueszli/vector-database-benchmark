from numpy.testing import assert_array_equal
import pytest
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import random
from scipy.sparse._sputils import matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse.csgraph.tests.test_matching import linear_sum_assignment_assertions, linear_sum_assignment_test_cases

def test_linear_sum_assignment_input_shape():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError, match='expected a matrix'):
        linear_sum_assignment([1, 2, 3])

def test_linear_sum_assignment_input_object():
    if False:
        for i in range(10):
            print('nop')
    C = [[1, 2, 3], [4, 5, 6]]
    assert_array_equal(linear_sum_assignment(C), linear_sum_assignment(np.asarray(C)))
    assert_array_equal(linear_sum_assignment(C), linear_sum_assignment(matrix(C)))

def test_linear_sum_assignment_input_bool():
    if False:
        i = 10
        return i + 15
    I = np.identity(3)
    assert_array_equal(linear_sum_assignment(I.astype(np.bool_)), linear_sum_assignment(I))

def test_linear_sum_assignment_input_string():
    if False:
        i = 10
        return i + 15
    I = np.identity(3)
    with pytest.raises(TypeError, match='Cannot cast array data'):
        linear_sum_assignment(I.astype(str))

def test_linear_sum_assignment_input_nan():
    if False:
        for i in range(10):
            print('nop')
    I = np.diag([np.nan, 1, 1])
    with pytest.raises(ValueError, match='contains invalid numeric entries'):
        linear_sum_assignment(I)

def test_linear_sum_assignment_input_neginf():
    if False:
        for i in range(10):
            print('nop')
    I = np.diag([1, -np.inf, 1])
    with pytest.raises(ValueError, match='contains invalid numeric entries'):
        linear_sum_assignment(I)

def test_linear_sum_assignment_input_inf():
    if False:
        i = 10
        return i + 15
    I = np.identity(3)
    I[:, 0] = np.inf
    with pytest.raises(ValueError, match='cost matrix is infeasible'):
        linear_sum_assignment(I)

def test_constant_cost_matrix():
    if False:
        print('Hello World!')
    n = 8
    C = np.ones((n, n))
    (row_ind, col_ind) = linear_sum_assignment(C)
    assert_array_equal(row_ind, np.arange(n))
    assert_array_equal(col_ind, np.arange(n))

@pytest.mark.parametrize('num_rows,num_cols', [(0, 0), (2, 0), (0, 3)])
def test_linear_sum_assignment_trivial_cost(num_rows, num_cols):
    if False:
        for i in range(10):
            print('nop')
    C = np.empty(shape=(num_cols, num_rows))
    (row_ind, col_ind) = linear_sum_assignment(C)
    assert len(row_ind) == 0
    assert len(col_ind) == 0

@pytest.mark.parametrize('sign,test_case', linear_sum_assignment_test_cases)
def test_linear_sum_assignment_small_inputs(sign, test_case):
    if False:
        while True:
            i = 10
    linear_sum_assignment_assertions(linear_sum_assignment, np.array, sign, test_case)

def test_two_methods_give_same_result_on_many_sparse_inputs():
    if False:
        print('Hello World!')
    np.random.seed(1234)
    for _ in range(100):
        lsa_raises = False
        mwfbm_raises = False
        sparse = random(100, 100, density=0.06, data_rvs=lambda size: np.random.randint(1, 100, size))
        dense = np.full(sparse.shape, np.inf)
        dense[sparse.row, sparse.col] = sparse.data
        sparse = sparse.tocsr()
        try:
            (row_ind, col_ind) = linear_sum_assignment(dense)
            lsa_cost = dense[row_ind, col_ind].sum()
        except ValueError:
            lsa_raises = True
        try:
            (row_ind, col_ind) = min_weight_full_bipartite_matching(sparse)
            mwfbm_cost = sparse[row_ind, col_ind].sum()
        except ValueError:
            mwfbm_raises = True
        assert lsa_raises == mwfbm_raises
        if not lsa_raises:
            assert lsa_cost == mwfbm_cost