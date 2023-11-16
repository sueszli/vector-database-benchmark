"""
Basic tests to assert and illustrate the  behavior around the decision to use 0D
arrays in place of array scalars.

Extensive tests of this sort of functionality is in numpy_tests/core/*scalar*

Also test the isscalar function (which is deliberately a bit more lax).
"""
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests, subtest, TEST_WITH_TORCHDYNAMO, TestCase, xfailIfTorchDynamo
if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import assert_equal
else:
    import torch._numpy as np
    from torch._numpy.testing import assert_equal
parametrize_value = parametrize('value', [subtest(np.int64(42), name='int64'), subtest(np.array(42), name='array'), subtest(np.asarray(42), name='asarray'), subtest(np.asarray(np.int64(42)), name='asarray_int')])

@instantiate_parametrized_tests
class TestArrayScalars(TestCase):

    @parametrize_value
    def test_array_scalar_basic(self, value):
        if False:
            for i in range(10):
                print('nop')
        assert value.ndim == 0
        assert value.shape == ()
        assert value.size == 1
        assert value.dtype == np.dtype('int64')

    @parametrize_value
    def test_conversion_to_int(self, value):
        if False:
            i = 10
            return i + 15
        py_scalar = int(value)
        assert py_scalar == 42
        assert isinstance(py_scalar, int)
        assert not isinstance(value, int)

    @parametrize_value
    def test_decay_to_py_scalar(self, value):
        if False:
            for i in range(10):
                print('nop')
        lst = [1, 2, 3]
        product = value * lst
        assert isinstance(product, np.ndarray)
        assert product.shape == (3,)
        assert_equal(product, [42, 42 * 2, 42 * 3])
        product = lst * value
        assert isinstance(product, np.ndarray)
        assert product.shape == (3,)
        assert_equal(product, [42, 42 * 2, 42 * 3])

    def test_scalar_comparisons(self):
        if False:
            while True:
                i = 10
        scalar = np.int64(42)
        arr = np.array(42)
        assert arr == scalar
        assert arr >= scalar
        assert arr <= scalar
        assert scalar == 42
        assert arr == 42

@instantiate_parametrized_tests
class TestIsScalar(TestCase):
    scalars = [subtest(42, 'literal'), subtest(int(42.0), 'int'), subtest(np.float32(42), 'float32'), subtest(np.array(42), 'array_0D', decorators=[xfailIfTorchDynamo]), subtest([42], 'list', decorators=[xfailIfTorchDynamo]), subtest([[42]], 'list-list', decorators=[xfailIfTorchDynamo]), subtest(np.array([42]), 'array_1D', decorators=[xfailIfTorchDynamo]), subtest(np.array([[42]]), 'array_2D', decorators=[xfailIfTorchDynamo])]
    import math
    not_scalars = [int, np.float32, subtest('s', decorators=[xfailIfTorchDynamo]), subtest('string', decorators=[xfailIfTorchDynamo]), (), [], math.sin, np, np.transpose, [1, 2], np.asarray([1, 2]), np.float32([1, 2])]

    @parametrize('value', scalars)
    def test_is_scalar(self, value):
        if False:
            return 10
        assert np.isscalar(value)

    @parametrize('value', not_scalars)
    def test_is_not_scalar(self, value):
        if False:
            while True:
                i = 10
        assert not np.isscalar(value)
if __name__ == '__main__':
    run_tests()