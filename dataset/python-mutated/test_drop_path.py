import unittest
import numpy as np
from dygraph_to_static_utils_new import Dy2StTestBase, test_legacy_and_pir_exe_and_pir_api
import paddle

def drop_path(x, training=False):
    if False:
        while True:
            i = 10
    if not training:
        return x
    else:
        return 2 * x

class DropPath(paddle.nn.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return drop_path(x, self.training)

class TestTrainEval(Dy2StTestBase):

    @test_legacy_and_pir_exe_and_pir_api
    def test_train_and_eval(self):
        if False:
            print('Hello World!')
        model = paddle.jit.to_static(DropPath())
        x = paddle.to_tensor([1, 2, 3]).astype('int64')
        eval_out = x.numpy()
        train_out = x.numpy() * 2
        model.train()
        np.testing.assert_allclose(model(x).numpy(), train_out, rtol=1e-05)
        model.eval()
        np.testing.assert_allclose(model(x).numpy(), eval_out, rtol=1e-05)
if __name__ == '__main__':
    unittest.main()