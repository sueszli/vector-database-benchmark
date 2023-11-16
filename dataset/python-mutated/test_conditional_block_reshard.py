import unittest
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet import auto
from paddle.static import InputSpec

class MLPLayer(nn.Layer):

    def __init__(self, hidden_size=64, intermediate_size=4 * 64, initializer_range=0.02):
        if False:
            return 10
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, epsilon=1e-05)
        self.linear0 = nn.Linear(hidden_size, intermediate_size, paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)), bias_attr=None)
        self.linear1 = nn.Linear(intermediate_size, hidden_size, paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)), bias_attr=None)

    def forward(self, input):
        if False:
            i = 10
            return i + 15
        out = self.norm(input)
        auto.shard_tensor(self.linear0.weight, auto.ProcessMesh([0, 1], ['x']), [None, 'x'])
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        auto.shard_tensor(self.linear1.weight, auto.ProcessMesh([0, 1], ['x']), ['x', None])
        out = self.linear1(out)
        if paddle.mean(out) < 2:
            out = self.norm(out)
            out = self.linear0(out)
            out = F.gelu(out, approximate=True)
            out = self.linear1(out)
        else:
            out = self.norm(out)
            out = self.linear0(out)
            out = self.linear1(out)
        return out

def loss_fn(predict, label):
    if False:
        for i in range(10):
            print('nop')
    error_cost = paddle.nn.functional.square_error_cost(predict, label)
    loss = paddle.mean(error_cost)
    return loss

class TestSubblock(unittest.TestCase):

    def test_subblock(self):
        if False:
            for i in range(10):
                print('nop')
        mlp = MLPLayer()
        strategy = auto.Strategy()
        strategy.auto_mode = 'semi'
        engine = auto.Engine(model=mlp, loss=loss_fn, strategy=strategy)
        input_sepc = InputSpec([4, 64], 'float32', 'input')
        label_spec = InputSpec([4, 1], 'float32', 'label')
        engine.prepare(inputs_spec=[input_sepc], labels_spec=[label_spec], mode='predict')
if __name__ == '__main__':
    unittest.main()