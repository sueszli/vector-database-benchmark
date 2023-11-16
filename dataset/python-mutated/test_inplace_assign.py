import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_ast_only, test_legacy_and_pir
import paddle

class TestInplaceAssign(Dy2StTestBase):

    @test_ast_only
    def test_case0(self):
        if False:
            print('Hello World!')
        a = paddle.ones((1024, 2)) * 1
        b = paddle.ones((1024, 3)) * 2
        c = paddle.ones((1024, 4)) * 3
        a._inplace_assign(b)
        np.testing.assert_array_equal(a.numpy(), b.numpy())
        b._inplace_assign(c)
        np.testing.assert_array_equal(b.numpy(), c.numpy())

    @test_ast_only
    def test_case1(self):
        if False:
            i = 10
            return i + 15

        def func(x):
            if False:
                i = 10
                return i + 15
            a = 1 * x
            b = 2 * x
            a._inplace_assign(b)
            return a
        x = paddle.ones((1,))
        a = paddle.randn((1,))
        x.stop_gradient = False
        a.stop_gradient = False
        y = func(x)
        y.mean().backward()
        np.testing.assert_array_equal(x.grad.numpy(), np.array([2.0]))

    @test_legacy_and_pir
    def test_case2(self):
        if False:
            i = 10
            return i + 15

        def func(a, x):
            if False:
                print('Hello World!')
            x[:] = a * 2.0
            return x

        def forward(a, x):
            if False:
                for i in range(10):
                    print('nop')
            output = paddle.jit.to_static(func)(a, x)
            x._inplace_assign(output)
            return x
        x = paddle.ones((1,))
        a = paddle.randn((1,))
        x.stop_gradient = False
        a.stop_gradient = False
        y = forward(a, x)
        y.mean().backward()
        np.testing.assert_array_equal(a.grad.numpy(), np.array([2.0]))
if __name__ == '__main__':
    unittest.main()