import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only, test_legacy_and_pir
import paddle

class TestCpuCuda(Dy2StTestBase):

    def test_cpu_cuda(self):
        if False:
            i = 10
            return i + 15

        def func(x):
            if False:
                i = 10
                return i + 15
            x = paddle.to_tensor([1, 2, 3, 4])
            x = x.cuda()
            x = x.cpu()
            return x
        x = paddle.to_tensor([3])

class TestToTensor(Dy2StTestBase):

    @test_legacy_and_pir
    def test_to_tensor_with_variable_list(self):
        if False:
            while True:
                i = 10

        def func(x):
            if False:
                print('Hello World!')
            ones = paddle.to_tensor(1)
            twos = paddle.to_tensor(2)
            x = paddle.to_tensor([ones, twos, 3, 4])
            return x
        x = paddle.to_tensor([3])
        np.testing.assert_allclose(paddle.jit.to_static(func)(x).numpy(), np.array([1, 2, 3, 4]), rtol=1e-05)

class TestToTensor1(Dy2StTestBase):

    @test_ast_only
    @test_legacy_and_pir
    def test_to_tensor_with_variable_list(self):
        if False:
            i = 10
            return i + 15

        def func(x):
            if False:
                print('Hello World!')
            ones = paddle.to_tensor([1])
            twos = paddle.to_tensor([2])
            ' we ignore the [3] and [4], they will be assign to a variable, and is regard as scalar.\n                TODO: deal with this case after 0-dim tensor is developed.\n            '
            x = paddle.to_tensor([ones, twos, [3], [4]])
            return x
        x = paddle.to_tensor([3])
        np.testing.assert_allclose(paddle.jit.to_static(func)(x).numpy(), np.array([[1], [2], [3], [4]]), rtol=1e-05)

    @test_ast_only
    @test_legacy_and_pir
    def test_to_tensor_with_variable_list_sot(self):
        if False:
            print('Hello World!')

        def func(x):
            if False:
                while True:
                    i = 10
            ones = paddle.to_tensor([1])
            twos = paddle.to_tensor([2])
            ' we ignore the [3] and [4], they will be assign to a variable, and is regard as scalar.\n                TODO: deal with this case after 0-dim tensor is developed.\n            '
            x = paddle.to_tensor([ones, twos, [3], [4]])
            return x
        x = paddle.to_tensor([3])
        np.testing.assert_allclose(paddle.jit.to_static(func)(x), np.array([[1], [2], [3], [4]]), rtol=1e-05)

class TestToTensor2(Dy2StTestBase):

    @test_ast_only
    @test_legacy_and_pir
    def test_to_tensor_with_variable_list(self):
        if False:
            i = 10
            return i + 15

        def func(x):
            if False:
                for i in range(10):
                    print('nop')
            x = paddle.to_tensor([[1], [2], [3], [4]])
            return x
        x = paddle.to_tensor([3])
        np.testing.assert_allclose(paddle.jit.to_static(func)(x).numpy(), np.array([[1], [2], [3], [4]]), rtol=1e-05)

    @test_ast_only
    @test_legacy_and_pir
    def test_to_tensor_with_variable_list_sot(self):
        if False:
            i = 10
            return i + 15

        def func(x):
            if False:
                for i in range(10):
                    print('nop')
            x = paddle.to_tensor([[1], [2], [3], [4]])
            return x
        x = paddle.to_tensor([3])
        np.testing.assert_allclose(paddle.jit.to_static(func)(x), np.array([[1], [2], [3], [4]]), rtol=1e-05)
if __name__ == '__main__':
    unittest.main()