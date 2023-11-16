import os
import shutil
import unittest
import numpy as np
from test_dist_sparse_load_ps0 import SparseLoadOp
import paddle
from paddle import base
from paddle.distributed.fleet import fleet
from paddle.distributed.fleet.base import role_maker

@unittest.skip(reason='Skip unstable ut, need rewrite with new implement')
class TestSparseLoadOpCase2(SparseLoadOp):

    def test_2ps_0_load(self):
        if False:
            print('Hello World!')
        env = {}
        env['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:4001,127.0.0.1:4002'
        env['PADDLE_TRAINERS_NUM'] = str(2)
        env['TRAINING_ROLE'] = 'PSERVER'
        env['PADDLE_PORT'] = '4002'
        env['POD_IP'] = '127.0.0.1'
        for (k, v) in env.items():
            os.environ[k] = str(v)
        '\n        array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],\n                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],\n                [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],\n                [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],\n                [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],\n                [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],\n                [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],\n                [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],\n                [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],\n                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]])\n        '
        emb_array = np.arange(0, 1, 0.1).repeat(10).reshape(10, 10)
        fc_array = np.arange(0, 1, 0.1).repeat(10).reshape(10, 10)
        model_path = self.save_origin_model(emb_array, fc_array)
        startup_program = base.framework.Program()
        test_program = base.framework.Program()
        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        loss = self.net(emb_array, fc_array)
        strategy = paddle.distributed.fleet.DistributedStrategy()
        strategy.a_sync = True
        optimizer = paddle.optimizer.Adam(0.001)
        optimizer = fleet.distributed_optimizer(optimizer, strategy)
        optimizer.minimize(loss)
        fleet.init_server(model_path)
        emb = np.array(base.global_scope().find_var('embedding.block1').get_tensor())
        assert emb.all() == emb_array[1::2].all()
        shutil.rmtree(model_path)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()