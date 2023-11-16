import ivy
import numpy as np
import pytest

@pytest.mark.parametrize('n_pad', [1, 2])
def test_pad_tt_rank(n_pad):
    if False:
        i = 10
        return i + 15
    rank = (1, 2, 2, 2, 1)
    tt = ivy.random_tt((4, 3, 5, 2), rank)
    padded_tt = ivy.TTTensor(ivy.TTTensor.pad_tt_rank(tt, n_padding=n_pad, pad_boundaries=False))
    rec = tt.to_tensor()
    rec_padded = padded_tt.to_tensor()
    np.testing.assert_array_almost_equal(rec, rec_padded, decimal=4)
    np.testing.assert_(padded_tt.rank == (1, *[i + n_pad for i in rank[1:-1]], 1))

@pytest.mark.parametrize(('shape', 'rank'), [((4, 5, 4, 8, 5), (1, 3, 2, 2, 4, 1))])
def test_tt_n_param(shape, rank):
    if False:
        return 10
    factors = ivy.random_tt(shape, rank)
    true_n_param = ivy.sum([ivy.prod(f.shape) for f in factors])
    n_param = ivy.TTTensor._tt_n_param(shape, rank)
    np.testing.assert_equal(n_param, true_n_param)

@pytest.mark.parametrize(('n1', 'n2', 'n3', 'shape1', 'shape2', 'shape3'), [(3, 4, 2, (1, 3, 2), (2, 4, 2), (2, 2, 1))])
def test_tt_to_tensor(n1, n2, n3, shape1, shape2, shape3):
    if False:
        while True:
            i = 10
    tensor = ivy.zeros((n1, n2, n3))
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                tensor[i][j][k] = i + 1 + (j + 1) + (k + 1)
    tensor = ivy.array(tensor)
    factors = [None] * 3
    factors[0] = ivy.zeros(shape1)
    factors[1] = ivy.zeros(shape2)
    factors[2] = ivy.zeros(shape3)
    for i in range(3):
        for j in range(4):
            for k in range(2):
                factors[0][0][i][0] = i + 1
                factors[0][0][i][1] = 1
                factors[1][0][j][0] = 1
                factors[1][0][j][1] = 0
                factors[1][1][j][0] = j + 1
                factors[1][1][j][1] = 1
                factors[2][0][k][0] = 1
                factors[2][1][k][0] = k + 1
    factors = [ivy.array(f) for f in factors]
    np.testing.assert_array_almost_equal(tensor, ivy.TTTensor.tt_to_tensor(factors))

@pytest.mark.parametrize('coef', [0.2])
def test_validate_tt_rank(coef):
    if False:
        i = 10
        return i + 15
    tensor_shape = tuple(ivy.random.randint(5, 10, shape=(4,)))
    n_param_tensor = ivy.prod(tensor_shape)
    rank = ivy.TTTensor.validate_tt_rank(tensor_shape, coef, rounding='floor')
    n_param = ivy.TTTensor._tt_n_param(tensor_shape, rank)
    np.testing.assert_(n_param <= n_param_tensor * coef)
    rank = ivy.TTTensor.validate_tt_rank(tensor_shape, coef, rounding='ceil')
    n_param = ivy.TTTensor._tt_n_param(tensor_shape, rank)
    np.testing.assert_(n_param >= n_param_tensor * coef)

@pytest.mark.parametrize(('true_shape', 'true_rank'), [((3, 4, 5), (1, 3, 2, 1))])
def test_validate_tt_tensor(true_shape, true_rank):
    if False:
        i = 10
        return i + 15
    factors = ivy.random_tt(true_shape, true_rank).factors
    (shape, rank) = ivy.TTTensor.validate_tt_tensor(factors)
    np.testing.assert_equal(shape, true_shape, err_msg=f'Returned incorrect shape (got {shape}, expected {true_shape})')
    np.testing.assert_equal(rank, true_rank, err_msg=f'Returned incorrect rank (got {rank}, expected {true_rank})')
    factors[0] = ivy.random_uniform(shape=(4, 4))
    with np.testing.assert_raises(ValueError):
        ivy.TTTensor.validate_tt_tensor(factors)
    factors[0] = ivy.random_uniform(shape=(3, 3, 2))
    with np.testing.assert_raises(ValueError):
        ivy.TTTensor.validate_tt_tensor(factors)