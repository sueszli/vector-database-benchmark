import unittest
import paddle
from paddle.distributed.fleet.utils.tensor_fusion_helper import HOOK_ACTION, FusedCommBuffer

class TestFusedCommBufferGradChecker(unittest.TestCase):

    def test_fused_comm_buffer_grad_checker(self):
        if False:
            return 10
        linear = paddle.nn.Linear(10, 10)
        w = linear.weight
        b = linear.bias
        w.main_grad = None
        b.main_grad = None
        buffer = FusedCommBuffer(id=0, params=[w, b], comm_group=None, acc_steps=10, act=HOOK_ACTION.ALL_REDUCE)
        assert buffer.use_main_grad
        buffer.add_grad(w)
        buffer.add_grad(b)
        w.main_grad = paddle.to_tensor([1], stop_gradient=True, dtype='float32')
        try:
            buffer.add_grad(w)
            raise AssertionError('Above add_grad should raise value error, this assertion should be unreachable.')
        except ValueError:
            pass
if __name__ == '__main__':
    unittest.main()