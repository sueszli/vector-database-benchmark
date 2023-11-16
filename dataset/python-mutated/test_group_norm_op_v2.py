import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core

def group_norm_naive_for_general_dimension(x, scale, bias, epsilon, groups):
    if False:
        return 10
    input_shape = x.shape
    (N, C) = (x.shape[0], x.shape[1])
    G = groups
    x = x.reshape((N * G, -1))
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    output = (x - mean) / np.sqrt(var + epsilon)
    output = output.reshape(input_shape) * scale.reshape((-1, 1, 1)) + bias.reshape((-1, 1, 1))
    return output

class TestGroupNormAPIV2_With_General_Dimensions(unittest.TestCase):

    def test_numerical_accuracy(self):
        if False:
            while True:
                i = 10
        paddle.disable_static()
        shapes = [(2, 6), (2, 6, 4), (2, 6, 4, 4), (2, 6, 6, 6, 2), (2, 6, 6, 6, 2, 3)]
        np.random.seed(10)
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu('group_norm'):
            places.append(base.CUDAPlace(0))
        for place in places:
            for shape in shapes:
                scale = np.array([1]).astype('float32')
                bias = np.array([0]).astype('float32')
                data = np.random.random(shape).astype('float32')
                expect_res1 = group_norm_naive_for_general_dimension(data, scale, bias, epsilon=1e-05, groups=6)
                expect_res2 = group_norm_naive_for_general_dimension(data, scale, bias, epsilon=1e-05, groups=2)
                gn1 = paddle.nn.GroupNorm(num_channels=6, num_groups=6)
                gn2 = paddle.nn.GroupNorm(num_channels=6, num_groups=2)
                data_pd = paddle.to_tensor(data)
                result1 = gn1(data_pd).numpy()
                result2 = gn2(data_pd).numpy()
                self.assertTrue(np.allclose(result1, expect_res1, atol=1e-05))
                self.assertTrue(np.allclose(result2, expect_res2, atol=1e-05))

class TestGroupNormAPIV2_With_General_Dimensions_fp16(unittest.TestCase):

    def test_numerical_accuracy(self):
        if False:
            for i in range(10):
                print('nop')
        if not core.is_compiled_with_cuda():
            return
        paddle.disable_static()
        shapes = [(2, 6, 4), (2, 6, 4, 4), (2, 6, 6, 6, 2), (2, 6, 6, 6, 2, 3), (2, 6, 6, 6, 256, 3)]
        np.random.seed(10)
        places = [base.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu('group_norm'):
            places.append(base.CUDAPlace(0))
        for place in places:
            for shape in shapes:
                scale = np.array([1]).astype('float32')
                bias = np.array([0]).astype('float32')
                data = np.random.random(shape).astype('float32')
                expect_res1 = group_norm_naive_for_general_dimension(data, scale, bias, epsilon=1e-05, groups=6)
                expect_res2 = group_norm_naive_for_general_dimension(data, scale, bias, epsilon=1e-05, groups=2)
                gn1 = paddle.nn.GroupNorm(num_channels=6, num_groups=6)
                gn2 = paddle.nn.GroupNorm(num_channels=6, num_groups=2)
                paddle.assign(paddle.cast(gn1.weight, 'float16'), gn1.weight)
                paddle.assign(paddle.cast(gn1.bias, 'float16'), gn1.bias)
                paddle.assign(paddle.cast(gn2.weight, 'float16'), gn2.weight)
                paddle.assign(paddle.cast(gn2.bias, 'float16'), gn2.bias)
                data_pd = paddle.to_tensor(data.astype('float16'))
                result1 = gn1(data_pd).numpy()
                result2 = gn2(data_pd).numpy()
                np.testing.assert_allclose(result1, expect_res1, rtol=0.01, atol=0.001)
                np.testing.assert_allclose(result2, expect_res2, rtol=0.01, atol=0.001)

class TestGroupNormDimException(unittest.TestCase):

    def test_exception(self):
        if False:
            while True:
                i = 10

        def test_empty_input_static_API():
            if False:
                i = 10
                return i + 15
            x = paddle.to_tensor([], dtype='float32')
            paddle.static.nn.group_norm(x, 3)
        self.assertRaises(ValueError, test_empty_input_static_API)

        def test_one_dim_input_static_API():
            if False:
                for i in range(10):
                    print('nop')
            x = paddle.randn((3,), dtype='float32')
            paddle.static.nn.group_norm(x, 3)
        self.assertRaises(ValueError, test_one_dim_input_static_API)
if __name__ == '__main__':
    unittest.main()