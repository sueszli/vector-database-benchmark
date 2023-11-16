import os
import unittest
import numpy as np
import paddle
from paddle import base
from paddle.distributed import fleet
from paddle.distributed.fleet.base import role_maker

class TestFleetBase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        os.environ['POD_IP'] = '127.0.0.1'
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '127.0.0.1:36000'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:36001,127.0.0.2:36002'

    def test_init(self):
        if False:
            i = 10
            return i + 15
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)

    def test_is_first_worker(self):
        if False:
            while True:
                i = 10
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_first_worker():
            print('test fleet first worker done.')

    def test_worker_index(self):
        if False:
            while True:
                i = 10
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        print(fleet.worker_index())

    def test_worker_num(self):
        if False:
            for i in range(10):
                print('nop')
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        print(fleet.worker_num())

    def test_is_worker(self):
        if False:
            print('Hello World!')
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_worker():
            print('test fleet is worker')

    def test_worker_endpoints(self):
        if False:
            i = 10
            return i + 15
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        self.assertEqual('127.0.0.1:36000', fleet.worker_endpoints(to_string=True))
        self.assertEqual(['127.0.0.1:36000'], fleet.worker_endpoints())

    def test_server_num(self):
        if False:
            i = 10
            return i + 15
        os.environ['TRAINING_ROLE'] = 'PSERVER'
        os.environ['PADDLE_PORT'] = '36001'
        os.environ['POD_IP'] = '127.0.0.1'
        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        self.assertEqual(2, fleet.server_num())

    def test_server_index(self):
        if False:
            for i in range(10):
                print('nop')
        os.environ['TRAINING_ROLE'] = 'PSERVER'
        os.environ['PADDLE_PORT'] = '36001'
        os.environ['POD_IP'] = '127.0.0.1'
        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        self.assertEqual(0, fleet.server_index())

    def test_server_endpoints(self):
        if False:
            print('Hello World!')
        os.environ['TRAINING_ROLE'] = 'PSERVER'
        os.environ['PADDLE_PORT'] = '36001'
        os.environ['POD_IP'] = '127.0.0.1'
        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        if fleet.is_server():
            self.assertEqual('127.0.0.1:36001,127.0.0.2:36002', fleet.server_endpoints(to_string=True))
            self.assertEqual(['127.0.0.1:36001', '127.0.0.2:36002'], fleet.server_endpoints())

    def test_is_server(self):
        if False:
            i = 10
            return i + 15
        os.environ['TRAINING_ROLE'] = 'PSERVER'
        os.environ['PADDLE_PORT'] = '36001'
        os.environ['POD_IP'] = '127.0.0.1'
        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        self.assertTrue(fleet.is_server())

    def test_util(self):
        if False:
            while True:
                i = 10
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        self.assertIsNotNone(fleet.util)

    def test_barrier_worker(self):
        if False:
            i = 10
            return i + 15
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        if fleet.is_worker():
            fleet.barrier_worker()

    def test_init_worker(self):
        if False:
            return 10
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        with self.assertRaises(ValueError):
            if fleet.is_worker():
                fleet.init_worker()

    def test_stop_worker(self):
        if False:
            for i in range(10):
                print('nop')
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        with self.assertRaises(ValueError):
            if fleet.is_worker():
                fleet.stop_worker()

    def test_distributed_optimizer(self):
        if False:
            print('Hello World!')
        role = role_maker.PaddleCloudRoleMaker(is_collective=True)
        fleet.init(role)
        optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        optimizer = fleet.distributed_optimizer(optimizer)

    def test_exception(self):
        if False:
            return 10
        from paddle.distributed import fleet
        self.assertRaises(Exception, fleet.init_worker)

class TestFleetDygraph(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        os.environ['PADDLE_TRAINER_ENDPOINTS'] = '127.0.0.1:36213,127.0.0.1:36214'
        os.environ['PADDLE_CURRENT_ENDPOINTS'] = '127.0.0.1:36213'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['PADDLE_TRAINER_ID'] = '0'

    def test_dygraph_method(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.disable_static()
        value = np.arange(26).reshape(2, 13).astype('float32')
        a = base.dygraph.to_variable(value)
        layer = paddle.nn.Linear(13, 5)
        adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())
        adam = fleet.distributed_optimizer(adam)
        try:
            dp_layer = fleet.distributed_model(layer)
        except Exception as e:
            lr = 0.001
            adam.set_lr(lr)
            cur_lr = adam.get_lr()
            assert lr == cur_lr
            state_dict = adam.state_dict()
            adam.set_state_dict(state_dict)
            final_strategy = fleet._final_strategy()

class TestFleetBaseSingleError(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        os.environ.pop('PADDLE_TRAINER_ENDPOINTS')

    def gen_data(self):
        if False:
            for i in range(10):
                print('nop')
        return {'x': np.random.random(size=(128, 32)).astype('float32'), 'y': np.random.randint(2, size=(128, 1)).astype('int64')}

    def test_single_run_collective_minimize(self):
        if False:
            while True:
                i = 10

        def test_single_error():
            if False:
                while True:
                    i = 10
            input_x = paddle.static.data(name='x', shape=[-1, 32], dtype='float32')
            input_y = paddle.static.data(name='y', shape=[-1, 1], dtype='int64')
            fc_1 = paddle.static.nn.fc(x=input_x, size=64, activation='tanh')
            prediction = paddle.static.nn.fc(x=fc_1, size=2, activation='softmax')
            cost = paddle.nn.functional.cross_entropy(input=prediction, label=input_y, reduction='none', use_softmax=False)
            avg_cost = paddle.mean(x=cost)
            fleet.init(is_collective=True)
        if base.core.is_compiled_with_cuda() and base.core.get_cuda_device_count() > 1:
            self.assertRaises(ValueError, test_single_error)
        else:
            test_single_error()
if __name__ == '__main__':
    unittest.main()