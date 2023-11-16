import unittest
import paddle
from paddle import base, jit, nn
paddle.jit.enable_to_static(True)
base.core._set_prim_all_enabled(True)
x = paddle.randn([4, 1])
y = paddle.randn([4, 1])
x.stop_gradient = False
y.stop_gradient = False
model = nn.Sequential(nn.Linear(1, 1), nn.Tanh())
model2 = nn.Sequential(nn.Linear(1, 1))

class TestPaddleSciencemodel(unittest.TestCase):

    def test_concat(self):
        if False:
            for i in range(10):
                print('nop')

        @jit.to_static
        def concat(x, y):
            if False:
                return 10
            'abc'
            z = paddle.concat([x, y], 0)
            out = model(z)
            (out0, out1) = paddle.split(out, 2, axis=0)
            g0 = paddle.grad(out0, x)[0]
            g1 = paddle.grad(out1, y)[0]
            return (g0, g1)
        (g0, g1) = concat(x, y)
        loss = g0.sum() + g1.sum()
        loss.backward()

class TestEularBeam(unittest.TestCase):

    def test_eular_beam(self):
        if False:
            print('Hello World!')

        @jit.to_static
        def eular_beam(x):
            if False:
                print('Hello World!')
            'abc'
            z_ = model(x)
            out = model2(z_)
            g0 = paddle.grad(out, x)[0]
            g1 = paddle.grad(g0, x)[0]
            g2 = paddle.grad(g1, x)[0]
            g3 = paddle.grad(g2, x)[0]
            return g3
        g3 = eular_beam(x)
        loss = g3.sum()
        loss.backward()
if __name__ == '__main__':
    unittest.main()