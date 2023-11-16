import unittest
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet import auto
_g_mesh = auto.ProcessMesh([0, 1])
PP_MESH_0 = auto.ProcessMesh([0])
PP_MESH_1 = auto.ProcessMesh([1])
image_size = 1024
class_num = 10

class MyDataset(paddle.io.Dataset):

    def __init__(self, num_samples):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        input = np.random.uniform(size=image_size).astype('float32')
        input = np.random.uniform(size=image_size).astype('float32')
        return (input, input)

    def __len__(self):
        if False:
            while True:
                i = 10
        return self.num_samples

class MLPLayer(nn.Layer):

    def __init__(self, hidden_size=1024, intermediate_size=4 * 1024, dropout_ratio=0.1, initializer_range=0.02):
        if False:
            return 10
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range))
        bias_attr = None
        self.linear0 = nn.Linear(d_model, dim_feedforward, weight_attr, bias_attr=bias_attr)
        self.linear1 = nn.Linear(dim_feedforward, d_model, weight_attr, bias_attr=bias_attr)
        self.linear2 = nn.Linear(d_model, 1, weight_attr, bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-05)
        self.dropout = nn.Dropout(dropout_ratio, mode='upscale_in_train')

    def forward(self, input):
        if False:
            return 10
        out = auto.shard_op(self.norm, PP_MESH_0)(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = auto.shard_op(self.linear1, PP_MESH_1)(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out

class GEN(nn.Layer):

    def __init__(self, mlp):
        if False:
            print('Hello World!')
        super().__init__()
        self.mlp = mlp

    def forward(self, input):
        if False:
            for i in range(10):
                print('nop')
        model_kwargs = {}
        output = self.mlp(input)
        cur_step = paddle.full([1], 0, dtype='int64')
        total_step = paddle.full([1], 10, dtype='int64')
        model_kwargs['input'] = input
        model_kwargs['output'] = output
        while cur_step < total_step:
            out = self.mlp(model_kwargs['input'])
            paddle.increment(cur_step)
            out_assign = auto.shard_op(paddle.assign, _g_mesh)(out)
        model_kwargs['output'] = paddle.assign(out_assign)
        return (model_kwargs['output'], cur_step)

def get_model():
    if False:
        return 10
    with paddle.LazyGuard():
        mlp = MLPLayer()
        gen = GEN(mlp)
    return gen

class TestGenerationPipeline(unittest.TestCase):

    def test_pp2(self):
        if False:
            i = 10
            return i + 15
        model = get_model()
        strategy = auto.Strategy()
        pipeline = strategy.pipeline
        pipeline.enable = True
        pipeline.schedule_mode = 'stream'
        pipeline.generation_batch_size = 2
        pipeline.accumulate_steps = 20
        engine = auto.Engine(model, strategy=strategy)
        engine.prepare(inputs_spec=paddle.static.InputSpec(shape=[20, 1024], name='input', dtype='float32'), labels_spec=paddle.static.InputSpec(shape=[20, 1024], name='label', dtype='float32'), mode='eval')
        train_data = MyDataset(20)
        train_dataloader = engine._prepare_dataloader_from_generator(dataset=train_data, capacity=20, iterable=False, batch_size=1, epochs=1)
        fleet_opt = engine.main_program._pipeline_opt['fleet_opt']
        assert len(fleet_opt['tasks']) == 5
        assert fleet_opt['inference_generation']
        assert fleet_opt['num_micro_batches'] == 20
        num_task_in_rank = 5
        for (idx, (task_id, rank_id)) in enumerate(fleet_opt['task_id_to_rank'].items()):
            assert task_id == rank_id * num_task_in_rank + idx % num_task_in_rank
        train_dataloader._inner_dataloader.start()
        try:
            engine._executor.run(engine.main_program, use_program_cache=False, return_numpy=False)
        except paddle.base.core.EOFException:
            print('test done')
            train_dataloader._inner_dataloader.reset()
if __name__ == '__main__':
    unittest.main()