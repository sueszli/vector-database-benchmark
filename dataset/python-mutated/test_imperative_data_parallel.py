import unittest
import numpy as np
import paddle
from paddle import base
from paddle.base import core
from paddle.nn import Linear

class MLP(paddle.nn.Layer):

    def __init__(self, param_attr=None, bias_attr=None):
        if False:
            print('Hello World!')
        super().__init__()
        self._linear1 = Linear(784, 10)
        self._linear2 = Linear(10, 10)

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        y = self._linear1(inputs)
        y = self._linear2(y)
        return y

class TestDataParallelStateDict(unittest.TestCase):

    def test_data_parallel_state_dict(self):
        if False:
            i = 10
            return i + 15
        with base.dygraph.guard():
            paddle.distributed.init_parallel_env()
            mlp = MLP()
            parallel_mlp = paddle.DataParallel(mlp)
            single_state = mlp.state_dict()
            parallel_state = parallel_mlp.state_dict()
            base_para = {}
            place = base.CPUPlace() if not core.is_compiled_with_cuda() else base.CUDAPlace(0)
            for (k, v) in single_state.items():
                self.assertTrue(k in parallel_state)
                np.testing.assert_array_equal(v.numpy(), parallel_state[k].numpy())
                base_para[k] = v.numpy()
            for (k, v) in parallel_state.items():
                np_t = v.numpy()
                var = v.value().get_tensor()
                var.set(np.zeros_like(np_t), place)
                self.assertTrue(np.sum(np.abs(v.numpy())) == 0)
            parallel_mlp.set_dict(base_para)
            parallel_state = parallel_mlp.state_dict()
            for (k, v) in parallel_state.items():
                np.testing.assert_array_equal(v.numpy(), base_para[k])
            parallel_mlp.load_dict(base_para)
if __name__ == '__main__':
    unittest.main()