import os
import time
import numpy as np
from test_dist_fleet_base import FleetDistRunnerBase, runtime_main
import paddle
from paddle import base
paddle.enable_static()
DTYPE = 'int64'
DATA_URL = 'http://paddle-dist-ce-data.bj.bcebos.com/simnet.train.1000'
DATA_MD5 = '24e49366eb0611c552667989de2f57d5'
base_lr = 0.2
emb_lr = base_lr * 3
dict_dim = 1500
emb_dim = 128
hid_dim = 128
margin = 0.1
sample_rate = 1
base.default_startup_program().random_seed = 1
base.default_main_program().random_seed = 1

def fake_simnet_reader():
    if False:
        while True:
            i = 10

    def reader():
        if False:
            print('Hello World!')
        for _ in range(1000):
            q = np.random.random_integers(0, 1500 - 1, size=1).tolist()
            label = np.random.random_integers(0, 1, size=1).tolist()
            pt = np.random.random_integers(0, 1500 - 1, size=1).tolist()
            nt = np.random.random_integers(0, 1500 - 1, size=1).tolist()
            yield [q, label, pt, nt]
    return reader

def get_acc(cos_q_nt, cos_q_pt, batch_size):
    if False:
        i = 10
        return i + 15
    cond = paddle.less_than(cos_q_nt, cos_q_pt)
    cond = paddle.cast(cond, dtype='float64')
    cond_3 = paddle.sum(cond)
    acc = paddle.divide(cond_3, paddle.tensor.fill_constant(shape=[1], value=batch_size * 1.0, dtype='float64'), name='simnet_acc')
    return acc

def get_loss(cos_q_pt, cos_q_nt):
    if False:
        i = 10
        return i + 15
    fill_shape = [-1, 1]
    fill_shape[0] = paddle.shape(cos_q_pt)[0].item()
    loss_op1 = paddle.subtract(paddle.full(shape=fill_shape, fill_value=margin, dtype='float32'), cos_q_pt)
    loss_op2 = paddle.add(loss_op1, cos_q_nt)
    fill_shape[0] = paddle.shape(cos_q_pt)[0].item()
    loss_op3 = paddle.maximum(paddle.full(shape=fill_shape, fill_value=0.0, dtype='float32'), loss_op2)
    avg_cost = paddle.mean(loss_op3)
    return avg_cost

def train_network(batch_size, is_distributed=False, is_sparse=False, is_self_contained_lr=False, is_pyreader=False):
    if False:
        while True:
            i = 10
    q = paddle.static.data(name='query_ids', shape=[-1, 1], dtype='int64', lod_level=1)
    label = paddle.static.data(name='label', shape=[-1, 1], dtype='int64')
    pt = paddle.static.data(name='pos_title_ids', shape=[-1, 1], dtype='int64', lod_level=1)
    nt = paddle.static.data(name='neg_title_ids', shape=[-1, 1], dtype='int64', lod_level=1)
    datas = [q, label, pt, nt]
    reader = None
    if is_pyreader:
        reader = base.io.PyReader(feed_list=datas, capacity=64, iterable=False, use_double_buffer=False)
    q_emb = paddle.static.nn.embedding(input=q, is_distributed=is_distributed, size=[dict_dim, emb_dim], param_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01), name='__emb__'), is_sparse=is_sparse)
    q_emb = paddle.reshape(q_emb, [-1, emb_dim])
    q_sum = paddle.static.nn.sequence_lod.sequence_pool(input=q_emb, pool_type='sum')
    q_ss = paddle.nn.functional.softsign(q_sum)
    q_fc = paddle.static.nn.fc(x=q_ss, size=hid_dim, weight_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01), name='__q_fc__', learning_rate=base_lr))
    pt_emb = paddle.static.nn.embedding(input=pt, is_distributed=is_distributed, size=[dict_dim, emb_dim], param_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01), name='__emb__', learning_rate=emb_lr), is_sparse=is_sparse)
    pt_emb = paddle.reshape(pt_emb, [-1, emb_dim])
    pt_sum = paddle.static.nn.sequence_lod.sequence_pool(input=pt_emb, pool_type='sum')
    pt_ss = paddle.nn.functional.softsign(pt_sum)
    pt_fc = paddle.static.nn.fc(x=pt_ss, size=hid_dim, weight_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01), name='__fc__'), bias_attr=base.ParamAttr(name='__fc_b__'))
    nt_emb = paddle.static.nn.embedding(input=nt, is_distributed=is_distributed, size=[dict_dim, emb_dim], param_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01), name='__emb__'), is_sparse=is_sparse)
    nt_emb = paddle.reshape(nt_emb, [-1, emb_dim])
    nt_sum = paddle.static.nn.sequence_lod.sequence_pool(input=nt_emb, pool_type='sum')
    nt_ss = paddle.nn.functional.softsign(nt_sum)
    nt_fc = paddle.static.nn.fc(x=nt_ss, size=hid_dim, weight_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01), name='__fc__'), bias_attr=base.ParamAttr(name='__fc_b__'))
    cos_q_pt = paddle.nn.functional.cosine_similarity(q_fc, pt_fc)
    cos_q_nt = paddle.nn.functional.cosine_similarity(q_fc, nt_fc)
    avg_cost = get_loss(cos_q_pt, cos_q_nt)
    acc = get_acc(cos_q_nt, cos_q_pt, batch_size)
    return (avg_cost, acc, cos_q_pt, reader)

class TestDistSimnetBow2x2(FleetDistRunnerBase):
    """
    For test SimnetBow model, use Fleet api
    """

    def net(self, args, batch_size=4, lr=0.01):
        if False:
            i = 10
            return i + 15
        (avg_cost, _, predict, self.reader) = train_network(batch_size=batch_size, is_distributed=False, is_sparse=True, is_self_contained_lr=False, is_pyreader=args.reader == 'pyreader')
        self.avg_cost = avg_cost
        self.predict = predict
        return avg_cost

    def check_model_right(self, dirname):
        if False:
            print('Hello World!')
        model_filename = os.path.join(dirname, '__model__')
        with open(model_filename, 'rb') as f:
            program_desc_str = f.read()
        program = base.Program.parse_from_string(program_desc_str)
        with open(os.path.join(dirname, '__model__.proto'), 'w') as wn:
            wn.write(str(program))

    def do_pyreader_training(self, fleet):
        if False:
            for i in range(10):
                print('nop')
        '\n        do training using dataset, using fetch handler to catch variable\n        Args:\n            fleet(Fleet api): the fleet object of Parameter Server, define distribute training role\n        '
        exe = base.Executor(base.CPUPlace())
        exe.run(base.default_startup_program())
        fleet.init_worker()
        batch_size = 4
        train_reader = paddle.batch(fake_simnet_reader(), batch_size=batch_size)
        self.reader.decorate_sample_list_generator(train_reader)
        for epoch_id in range(1):
            self.reader.start()
            try:
                pass_start = time.time()
                while True:
                    loss_val = exe.run(program=base.default_main_program(), fetch_list=[self.avg_cost.name])
                    loss_val = np.mean(loss_val)
                    message = f'TRAIN ---> pass: {epoch_id} loss: {loss_val}\n'
                    fleet.util.print_on_rank(message, 0)
                pass_time = time.time() - pass_start
            except base.core.EOFException:
                self.reader.reset()

    def do_dataset_training(self, fleet):
        if False:
            for i in range(10):
                print('nop')
        pass
if __name__ == '__main__':
    runtime_main(TestDistSimnetBow2x2)