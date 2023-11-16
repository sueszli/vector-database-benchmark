"""Test cloud role maker."""
import os
import unittest
import paddle

class TestCloudRoleMaker(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMaker.
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
            while True:
                i = 10
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
        role_maker = GeneralRoleMaker(init_timeout_seconds=100, run_timeout_seconds=100, http_ip_port='127.0.0.1:36003')
        place = base.CPUPlace()
        exe = base.Executor(place)
        train_program = base.Program()
        startup_program = base.Program()
        scope = base.Scope()
        with base.program_guard(train_program, startup_program):
            show = paddle.static.data(name='show', shape=[-1, 1], dtype='float32', lod_level=1)
            fc = paddle.static.nn.fc(x=show, size=1, activation=None)
            label = paddle.static.data(name='click', shape=[-1, 1], dtype='int64', lod_level=1)
            label_cast = paddle.cast(label, dtype='float32')
            cost = paddle.nn.functional.log_loss(fc, label_cast)
        try:
            adam = paddle.optimizer.Adam(learning_rate=5e-06)
            adam = fleet.distributed_optimizer(adam)
            adam.minimize([cost], [scope])
            fleet.run_server()
            http_server_d = {}
            http_server_d['running'] = False
            size_d = {}
            role_maker._GeneralRoleMaker__start_kv_server(http_server_d, size_d)
        except:
            print('do not support pslib test, skip')
            return
        from paddle.incubate.distributed.fleet.role_maker import MockBarrier
        mb = MockBarrier()
        mb.barrier()
        mb.barrier_all()
        mb.all_reduce(1)
        mb.all_gather(1)
        os.environ['POD_IP'] = '127.0.0.1'
        os.environ['PADDLE_PORT'] = '36005'
        os.environ['TRAINING_ROLE'] = 'TRAINER'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '127.0.0.1:36005'
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:36006'
        os.environ['PADDLE_IS_BARRIER_ALL_ROLE'] = '0'
        role_maker = GeneralRoleMaker(path='test_mock1')
        role_maker.generate_role()
if __name__ == '__main__':
    unittest.main()