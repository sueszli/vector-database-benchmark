import os
os.environ['WITH_DISTRIBUTE'] = 'ON'
import unittest
import paddle
from paddle import base
from paddle.distributed import fleet
from paddle.distributed.fleet.base import role_maker
paddle.enable_static()
base_lr = 0.2
emb_lr = base_lr * 3
dict_dim = 1500
emb_dim = 128
hid_dim = 128
margin = 0.1
sample_rate = 1
batch_size = 4

class TestPSPassWithBow(unittest.TestCase):

    def net(self):
        if False:
            print('Hello World!')

        def get_acc(cos_q_nt, cos_q_pt, batch_size):
            if False:
                for i in range(10):
                    print('nop')
            cond = paddle.less_than(cos_q_nt, cos_q_pt)
            cond = paddle.cast(cond, dtype='float64')
            cond_3 = paddle.sum(cond)
            acc = paddle.divide(cond_3, paddle.tensor.fill_constant(shape=[1], value=batch_size * 1.0, dtype='float64'), name='simnet_acc')
            return acc

        def get_loss(cos_q_pt, cos_q_nt):
            if False:
                while True:
                    i = 10
            fill_shape = [-1, 1]
            fill_shape[0] = paddle.shape(cos_q_pt)[0].item()
            loss_op1 = paddle.subtract(paddle.full(shape=fill_shape, fill_value=margin, dtype='float32'), cos_q_pt)
            loss_op2 = paddle.add(loss_op1, cos_q_nt)
            fill_shape = [-1, 1]
            fill_shape[0] = paddle.shape(loss_op2)[0].item()
            loss_op3 = paddle.maximum(paddle.full(shape=fill_shape, fill_value=0.0, dtype='float32'), loss_op2)
            avg_cost = paddle.mean(loss_op3)
            return avg_cost
        is_distributed = False
        is_sparse = True
        q = paddle.static.data(name='1', shape=[-1, 1], dtype='int64', lod_level=1)
        q_emb = paddle.static.nn.sparse_embedding(input=q, size=[dict_dim, emb_dim], param_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01), name='__emb__', learning_rate=emb_lr))
        q_emb = paddle.reshape(q_emb, [-1, emb_dim])
        q_sum = paddle.static.nn.sequence_lod.sequence_pool(input=q_emb, pool_type='sum')
        q_ss = paddle.nn.functional.softsign(q_sum)
        q_fc = paddle.static.nn.fc(x=q_ss, size=hid_dim, weight_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01), name='__q_fc__', learning_rate=base_lr))
        label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
        pt = paddle.static.data(name='2', shape=[-1, 1], dtype='int64', lod_level=1)
        pt_emb = paddle.static.nn.sparse_embedding(input=pt, size=[dict_dim, emb_dim], param_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01), name='__emb__', learning_rate=emb_lr))
        pt_emb = paddle.reshape(pt_emb, [-1, emb_dim])
        pt_sum = paddle.static.nn.sequence_lod.sequence_pool(input=pt_emb, pool_type='sum')
        pt_ss = paddle.nn.functional.softsign(pt_sum)
        pt_fc = paddle.static.nn.fc(x=pt_ss, size=hid_dim, weight_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01), name='__fc__', learning_rate=base_lr), bias_attr=base.ParamAttr(name='__fc_b__'))
        nt = paddle.static.data(name='3', shape=[-1, 1], dtype='int64', lod_level=1)
        nt_emb = paddle.static.nn.sparse_embedding(input=nt, size=[dict_dim, emb_dim], param_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01), name='__emb__', learning_rate=emb_lr))
        nt_emb = paddle.reshape(nt_emb, [-1, emb_dim])
        nt_sum = paddle.static.nn.sequence_lod.sequence_pool(input=nt_emb, pool_type='sum')
        nt_ss = paddle.nn.functional.softsign(nt_sum)
        nt_fc = paddle.static.nn.fc(x=nt_ss, size=hid_dim, weight_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01), name='__fc__', learning_rate=base_lr), bias_attr=base.ParamAttr(name='__fc_b__'))
        cos_q_pt = paddle.nn.functional.cosine_similarity(q_fc, pt_fc)
        cos_q_nt = paddle.nn.functional.cosine_similarity(q_fc, nt_fc)
        avg_cost = get_loss(cos_q_pt, cos_q_nt)
        acc = get_acc(cos_q_nt, cos_q_pt, batch_size)
        return [avg_cost, acc, cos_q_pt]

    def test(self):
        if False:
            return 10
        os.environ['PADDLE_PSERVER_NUMS'] = '2'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['POD_IP'] = '127.0.0.1'
        os.environ['PADDLE_PORT'] = '36001'
        os.environ['PADDLE_TRAINER_ID'] = '0'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:36001,127.0.0.2:36001'
        os.environ['TRAINING_ROLE'] = 'PSERVER'
        role = role_maker.PaddleCloudRoleMaker()
        fleet.init(role)
        (loss, acc, _) = self.net()
        strategy = paddle.distributed.fleet.DistributedStrategy()
        configs = {'use_ps_gpu': 1}
        strategy.a_sync_configs = configs
        strategy.a_sync = True
        optimizer = paddle.optimizer.Adam(learning_rate=0.01)
        optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)
        optimizer.minimize(loss)
        fleet.init_server()
if __name__ == '__main__':
    unittest.main()