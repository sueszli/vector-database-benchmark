import os
import tempfile
import unittest
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet import auto
paddle.enable_static()

class MLPLayer(nn.Layer):

    def __init__(self, hidden_size=1024, intermediate_size=4 * 1024, dropout_ratio=0.1, initializer_range=0.02):
        if False:
            i = 10
            return i + 15
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

    def forward(self, input0, input1):
        if False:
            print('Hello World!')
        out = self.norm(input0)
        out = self.linear0(out)
        out = out + input1
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out

class TestDistSaver(unittest.TestCase):

    def test_dist_saver(self):
        if False:
            while True:
                i = 10
        mlp = MLPLayer()
        loss = paddle.nn.CrossEntropyLoss()
        optimizer = paddle.optimizer.Adam(learning_rate=1e-05, beta1=0.9, beta2=0.999, epsilon=1e-08, grad_clip=None)
        metric = paddle.metric.Accuracy()
        strategy = auto.Strategy()
        strategy.auto_mode = 'semi'
        engine = auto.Engine(mlp, loss, optimizer, metric, strategy=strategy)
        inputs_spec = [paddle.static.InputSpec(shape=[2, 1024], dtype='float32', name='input0'), paddle.static.InputSpec(shape=[2, 4096], dtype='float32', name='input1')]
        engine.prepare(inputs_spec, mode='predict')
        temp_dir = tempfile.TemporaryDirectory()
        model_filename = os.path.join(temp_dir.name, 'mlp')
        engine.save(model_filename, training=False)
        with open(model_filename + '_dist0.pdmodel', 'rb') as f:
            data = f.read()
        program = paddle.static.io.deserialize_program(data)
        input_vars = []
        for op in program.global_block().ops:
            if op.type == 'feed':
                input_vars.append(op.output_arg_names[0])
            else:
                break
        assert input_vars == ['input0', 'input1']
if __name__ == '__main__':
    unittest.main()