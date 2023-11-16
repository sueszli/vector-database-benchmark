import paddle
from paddle import base
from paddle.distributed import fleet
base.disable_dygraph()

def get_dataset(inputs):
    if False:
        i = 10
        return i + 15
    dataset = base.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_batch_size(1)
    dataset.set_filelist([])
    dataset.set_thread(1)
    return dataset

def net(batch_size=4, lr=0.01):
    if False:
        while True:
            i = 10
    '\n    network definition\n\n    Args:\n        batch_size(int): the size of mini-batch for training\n        lr(float): learning rate of training\n    Returns:\n        avg_cost: LoDTensor of cost.\n    '
    (dnn_input_dim, lr_input_dim) = (2, 2)
    with base.device_guard('cpu'):
        dnn_data = paddle.static.data(name='dnn_data', shape=[-1, 1], dtype='int64', lod_level=1)
        lr_data = paddle.static.data(name='lr_data', shape=[-1, 1], dtype='int64', lod_level=1)
        label = paddle.static.data(name='click', shape=[-1, 1], dtype='float32', lod_level=0)
        datas = [dnn_data, lr_data, label]
        dnn_layer_dims = [2, 1]
        dnn_embedding = paddle.static.nn.embedding(is_distributed=False, input=dnn_data, size=[dnn_input_dim, dnn_layer_dims[0]], param_attr=base.ParamAttr(name='deep_embedding', initializer=paddle.nn.initializer.Constant(value=0.01)), is_sparse=True)
        dnn_pool = paddle.static.nn.sequence_lod.sequence_pool(input=dnn_embedding, pool_type='sum')
        dnn_out = dnn_pool
        lr_embedding = paddle.static.nn.embedding(is_distributed=False, input=lr_data, size=[lr_input_dim, 1], param_attr=base.ParamAttr(name='wide_embedding', initializer=paddle.nn.initializer.Constant(value=0.01)), is_sparse=True)
        lr_pool = paddle.static.nn.sequence_lod.sequence_pool(input=lr_embedding, pool_type='sum')
    with base.device_guard('gpu'):
        for (i, dim) in enumerate(dnn_layer_dims[1:]):
            fc = paddle.static.nn.fc(x=dnn_out, size=dim, activation='relu', weight_attr=base.ParamAttr(initializer=paddle.nn.initializer.Constant(value=0.01)), name='dnn-fc-%d' % i)
            dnn_out = fc
        merge_layer = paddle.concat([dnn_out, lr_pool], axis=1)
        label = paddle.cast(label, dtype='int64')
        predict = paddle.static.nn.fc(x=merge_layer, size=2, activation='softmax')
        cost = paddle.nn.functional.cross_entropy(input=predict, label=label, reduction='none', use_softmax=False)
        avg_cost = paddle.mean(x=cost)
    return (datas, avg_cost)
'\noptimizer = paddle.optimizer.Adam(learning_rate=0.01)\n\nrole = role_maker.PaddleCloudRoleMaker()\nfleet.init(role)\n\nstrategy = paddle.distributed.fleet.DistributedStrategy()\nstrategy.a_sync = True\nstrategy.a_sync_configs = {"heter_worker_device_guard": \'gpu\'}\n\nstrategy.pipeline = True\nstrategy.pipeline_configs = {"accumulate_steps": 1, "micro_batch_size": 2048}\nfeeds, avg_cost = net()\noptimizer = fleet.distributed_optimizer(optimizer, strategy)\noptimizer.minimize(avg_cost)\ndataset = get_dataset(feeds)\n'
if fleet.is_server():
    pass
elif fleet.is_heter_worker():
    pass
    fleet.stop_worker()
elif fleet.is_worker():
    pass