import tempfile
import unittest
import paddle
paddle.enable_static()
import os
from paddle import base

class TestFleetBase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        os.environ['POD_IP'] = '127.0.0.1'
        os.environ['PADDLE_PORT'] = '36000'
        os.environ['PADDLE_TRAINERS_NUM'] = '1'

    def test_ps_minimize(self):
        if False:
            print('Hello World!')
        import paddle
        from paddle.distributed import fleet
        os.environ['TRAINING_ROLE'] = 'TRAINER'
        os.environ['PADDLE_TRAINER_ID'] = '1'
        input_x = paddle.static.data(name='x', shape=[-1, 32], dtype='float32')
        input_slot = paddle.static.data(name='slot', shape=[-1, 1], dtype='int64')
        input_y = paddle.static.data(name='y', shape=[-1, 1], dtype='int64')
        emb = paddle.static.nn.sparse_embedding(input=input_slot, size=[10, 9])
        input_x = paddle.concat(x=[input_x, emb], axis=1)
        fc_1 = paddle.static.nn.fc(x=input_x, size=64, activation='tanh')
        fc_2 = paddle.static.nn.fc(x=fc_1, size=64, activation='tanh')
        prediction = paddle.static.nn.fc(x=[fc_2], size=2, activation='softmax')
        cost = paddle.nn.functional.cross_entropy(input=prediction, label=input_y, reduction='none', use_softmax=False)
        avg_cost = paddle.mean(x=cost)
        role = fleet.PaddleCloudRoleMaker(is_collective=False)
        fleet.init(role)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = False
        strategy.a_sync_configs = {'launch_barrier': False}
        optimizer = paddle.optimizer.SGD(learning_rate=0.001)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(avg_cost)
        place = base.CPUPlace()
        exe = base.Executor(place)
        exe.run(paddle.static.default_startup_program())
        compiled_prog = base.compiler.CompiledProgram(base.default_main_program())
        temp_dir = tempfile.TemporaryDirectory()
        fleet.init_worker()
        fleet.fleet.save(dirname=temp_dir.name, feed=['x', 'y'], fetch=[avg_cost])
        fleet.fleet.save(dirname=temp_dir.name, feed=[input_x, input_y], fetch=[avg_cost])
        fleet.fleet.save(dirname=temp_dir.name)
        fleet.load_model(path=temp_dir.name, mode=0)
        fleet.load_model(path=temp_dir.name, mode=1)
        temp_dir.cleanup()
if __name__ == '__main__':
    unittest.main()