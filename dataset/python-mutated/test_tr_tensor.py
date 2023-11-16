import ivy
import numpy as np
import pytest

@pytest.mark.parametrize(('shape1', 'shape2', 'shape3'), [((2, 4, 3), (3, 5, 2), (2, 6, 2))])
def test_tr_to_tensor(shape1, shape2, shape3):
    if False:
        return 10
    factors = [ivy.random_uniform(shape=shape1), ivy.random_uniform(shape=shape2), ivy.random_uniform(shape=shape3)]
    tensor = ivy.einsum('iaj,jbk,kci->abc', *factors)
    assert np.allclose(tensor, ivy.TRTensor.tr_to_tensor(factors), atol=1e-06, rtol=1e-06)

@pytest.mark.parametrize(('rank1', 'rank2'), [((2, 3, 4, 2), (2, 3, 4, 2, 3))])
def test_validate_tr_rank(rank1, rank2):
    if False:
        print('Hello World!')
    tensor_shape = tuple(np.random.randint(1, 100, size=4))
    n_param_tensor = np.prod(tensor_shape)
    rank = ivy.TRTensor.validate_tr_rank(tensor_shape, rank='same', rounding='floor')
    n_param = ivy.TRTensor.tr_n_param(tensor_shape, rank)
    assert n_param <= n_param_tensor
    rank = ivy.TRTensor.validate_tr_rank(tensor_shape, rank='same', rounding='ceil')
    n_param = ivy.TRTensor.tr_n_param(tensor_shape, rank)
    assert n_param >= n_param_tensor
    with np.testing.assert_raises(ValueError):
        ivy.TRTensor.validate_tr_rank(tensor_shape, rank=rank1)
    with np.testing.assert_raises(ValueError):
        ivy.TRTensor.validate_tr_rank(tensor_shape, rank=rank2)

@pytest.mark.parametrize(('true_shape', 'true_rank'), [((6, 4, 5), (3, 2, 2, 3))])
def test_validate_tr_tensor(true_shape, true_rank):
    if False:
        return 10
    factors = ivy.random_tr(true_shape, true_rank).factors
    (shape, rank) = ivy.TRTensor.validate_tr_tensor(factors)
    np.testing.assert_equal(shape, true_shape, err_msg=f'Returned incorrect shape (got {shape}, expected {true_shape})')
    np.testing.assert_equal(rank, true_rank, err_msg=f'Returned incorrect rank (got {rank}, expected {true_rank})')
    factors[0] = ivy.random_uniform(shape=(4, 4))
    with np.testing.assert_raises(ValueError):
        ivy.TRTensor.validate_tr_tensor(factors)
    factors[0] = ivy.random_uniform(shape=(3, 6, 4))
    with np.testing.assert_raises(ValueError):
        ivy.TRTensor.validate_tr_tensor(factors)
    factors[0] = ivy.random_uniform(shape=(2, 6, 2))
    with np.testing.assert_raises(ValueError):
        ivy.TRTensor.validate_tr_tensor(factors)