import unittest
from test_dist_sparse_tensor_load_sgd import TestSparseLoadProgram
import paddle
from paddle import base
from paddle.distributed.fleet import fleet

class TestSparseLoadProgramRmsprop(TestSparseLoadProgram):
    """
    Test Sparse load operator.
    """

    def test_server_init(self):
        if False:
            i = 10
            return i + 15
        (scope, train_program, startup_program, loss) = self.net()
        with base.scope_guard(scope):
            with base.program_guard(train_program, startup_program):
                optimizer = paddle.optimizer.SGD(0.001)
                optimizer = fleet.distributed_optimizer(optimizer, self.strategy)
                optimizer.minimize(loss)
                fleet.init_server()
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()