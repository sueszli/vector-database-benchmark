import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir_exe_and_pir_api
import paddle
np.random.seed(1)
if paddle.base.is_compiled_with_cuda():
    place = paddle.base.CUDAPlace(0)
else:
    place = paddle.base.CPUPlace()

class SimpleNet(paddle.nn.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._linear = paddle.nn.Linear(1, 1)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        'forward with duplicate outputs.'
        x = self._linear(x)
        return (x, x)

class TestDuplicateOutput(Dy2StTestBase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `base.layers.cond`.
    """

    def _run_static(self):
        if False:
            for i in range(10):
                print('nop')
        net = paddle.jit.to_static(SimpleNet())
        x = paddle.to_tensor([1.0])
        param = net.parameters()
        param[0].clear_grad()
        (loss0, loss1) = net(x)
        loss0.backward()
        self.assertEqual(param[0].grad.numpy(), 1.0)

    @test_legacy_and_pir_exe_and_pir_api
    def test_ast_to_func(self):
        if False:
            return 10
        self._run_static()
if __name__ == '__main__':
    unittest.main()