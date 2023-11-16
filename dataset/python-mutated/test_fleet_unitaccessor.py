"""Test fleet."""
import os
import unittest
import paddle

class TestFleet1(unittest.TestCase):
    """
    Test cases for fleet minimize.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Set up, set envs.'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:36001,127.0.0.2:36001'

    def test_pslib_1(self):
        if False:
            i = 10
            return i + 15
        'Test cases for pslib.'
        from paddle import base
        from paddle.incubate.distributed.fleet.parameter_server.pslib import fleet
        from paddle.incubate.distributed.fleet.role_maker import GeneralRoleMaker
        os.environ['POD_IP'] = '127.0.0.1'
        os.environ['PADDLE_PORT'] = '36001'
        os.environ['TRAINING_ROLE'] = 'TRAINER'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '127.0.0.1:36001'
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:36002'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        role_maker = GeneralRoleMaker()
        place = base.CPUPlace()
        exe = base.Executor(place)
        train_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        with base.program_guard(train_program, startup_program):
            show = paddle.static.data(name='show', shape=[-1, 1], dtype='int64', lod_level=1)
            emb = paddle.static.nn.embedding(input=show, size=[1, 1], is_sparse=True, is_distributed=True, param_attr=base.ParamAttr(name='embedding'))
            fc = paddle.static.nn.fc(x=emb, size=1, activation=None)
            label = paddle.static.data(name='click', shape=[-1, 1], dtype='int64', lod_level=1)
            label_cast = paddle.cast(label, dtype='float32')
            cost = paddle.nn.functional.log_loss(fc, label_cast)
        strategy = {}
        strategy['embedding'] = {}
        strategy['embedding']['sparse_accessor_class'] = 'DownpourUnitAccessor'
        strategy['embedding']['embed_sparse_optimizer'] = 'naive'
        try:
            adam1 = paddle.optimizer.Adam(learning_rate=5e-06)
            adam1 = fleet.distributed_optimizer(adam1, strategy=strategy)
            adam1.minimize([cost], [scope])
            strategy['embedding']['embed_sparse_optimizer'] = 'adagrad'
            adam2 = paddle.optimizer.Adam(learning_rate=5e-06)
            adam2 = fleet.distributed_optimizer(adam2, strategy=strategy)
            adam2.minimize([cost], [scope])
            strategy['embedding']['embed_sparse_optimizer'] = 'adam'
            adam3 = paddle.optimizer.Adam(learning_rate=5e-06)
            adam3 = fleet.distributed_optimizer(adam3, strategy=strategy)
            adam3.minimize([cost], [scope])
        except:
            print('do not support pslib test, skip')
            return
if __name__ == '__main__':
    unittest.main()