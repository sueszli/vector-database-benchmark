import unittest
import paddle
import paddle.nn.functional as F
from paddle import nn, static, utils
from paddle.base import core
from paddle.distributed import fleet
from paddle.distributed.fleet import auto
paddle.enable_static()
_global_parallel_strategy = None
_global_process_mesh = None

class MLPLayer(nn.Layer):

    def __init__(self, hidden_size=1024, intermediate_size=4 * 1024, dropout_ratio=0.1, initializer_range=0.02):
        if False:
            print('Hello World!')
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
        out = self.norm(input)
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out

def mlp_pretrain_forward(train_program, start_program):
    if False:
        i = 10
        return i + 15
    with static.program_guard(train_program, start_program), utils.unique_name.guard():
        batch_size = 4
        hidden_size = 1024
        sequence_len = 512
        input = static.data(name='input', shape=[batch_size, sequence_len, hidden_size], dtype='float32')
        label = static.data(name='label', shape=[batch_size, sequence_len, 1], dtype='float32')
        auto.shard_tensor(input, _global_process_mesh, [None, None, None])
        mlp = MLPLayer(hidden_size=hidden_size, intermediate_size=4 * hidden_size, dropout_ratio=0.1, initializer_range=0.02)
        predict = mlp(input)
        cost = paddle.nn.functional.cross_entropy(input=predict, label=label, reduction='none', use_softmax=False)
        avg_cost = paddle.mean(x=cost)
    return (avg_cost, train_program, start_program)

class TestMLPAutoParallelizer(unittest.TestCase):

    def test_mlp_serial(self):
        if False:
            return 10
        global _global_process_mesh
        _global_process_mesh = auto.ProcessMesh(mesh=[0, 1], dim_names=['x'])
        dist_strategy = fleet.DistributedStrategy()
        dist_strategy.amp = False
        dist_strategy.pipeline = False
        dist_strategy.recompute = False
        dist_strategy.semi_auto = True
        fleet.init(is_collective=True, strategy=dist_strategy)
        train_program = static.Program()
        start_program = static.Program()
        (loss, train_program, start_program) = mlp_pretrain_forward(train_program, start_program)
        optimizer = paddle.optimizer.Adam(learning_rate=1e-05, beta1=0.9, beta2=0.999, epsilon=1e-08, grad_clip=None)
        optimizer = fleet.distributed_optimizer(optimizer)
        (_, _, distributed_startup_program, distributed_main_program) = optimizer.minimize(loss, start_program)
        suffix = core.kAutoParallelSuffix()
        for block in distributed_main_program.blocks:
            for op in block.ops:
                for attr_name in op.attr_names:
                    self.assertTrue(suffix not in attr_name)
        self.assertIsNotNone(distributed_startup_program)
        self.assertIsNotNone(distributed_main_program)
if __name__ == '__main__':
    unittest.main()