import os
import unittest
import paddle
from paddle import base
from paddle.distributed.fleet import fleet
from paddle.distributed.fleet.base import role_maker

class TestSparseLoadProgram(unittest.TestCase):
    """
    Test Sparse load operator.
    """

    def setUp(self):
        if False:
            return 10
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:4001,127.0.0.1:4002'
        os.environ['PADDLE_TRAINERS_NUM'] = str(2)
        os.environ['TRAINING_ROLE'] = 'PSERVER'
        os.environ['PADDLE_PORT'] = '4001'
        os.environ['POD_IP'] = '127.0.0.1'
        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        self.strategy = paddle.distributed.fleet.DistributedStrategy()
        self.strategy.a_sync = True

    def net(self):
        if False:
            for i in range(10):
                print('nop')
        train_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        with base.scope_guard(scope):
            with base.program_guard(train_program, startup_program):
                with base.unique_name.guard():
                    inputs = paddle.static.data('input', shape=[None, 1], dtype='int64')
                    emb = paddle.static.nn.embedding(inputs, is_sparse=True, size=[10000, 128])
                    fc1 = paddle.static.nn.fc(x=emb, size=128, activation='relu')
                    fc2 = paddle.static.nn.fc(x=fc1, size=64, activation='relu')
                    loss = paddle.mean(fc2)
            return (scope, train_program, startup_program, loss)

class TestSparseLoadProgramSGD(TestSparseLoadProgram):

    def test_server_init(self):
        if False:
            return 10
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