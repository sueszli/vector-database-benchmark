import random
import unittest
import numpy as np
import paddle
from paddle import nn
from paddle.base import core
from paddle.distributed.fleet import auto
from paddle.static import InputSpec
from paddle.static.amp.bf16.amp_utils import _valid_types
from paddle.static.amp.fp16_utils import find_true_prev_op
from paddle.vision.datasets import MNIST
paddle.enable_static()

def apply_pass(use_bf16=False):
    if False:
        i = 10
        return i + 15
    strategy = auto.Strategy()
    strategy.auto_mode = 'semi'
    strategy.reinit = True
    if use_bf16:
        amp = strategy.amp
        amp.enable = True
        amp.dtype = 'bfloat16'
        amp.level = 'o1'
    return strategy

class MnistDataset(MNIST):

    def __init__(self, mode, return_label=True):
        if False:
            return 10
        super().__init__(mode=mode)
        self.return_label = return_label

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        img = np.reshape(self.images[idx], [1, 28, 28])
        if self.return_label:
            return (img, np.array(self.labels[idx]).astype('int64'))
        return (img,)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.images)

def reset_prog():
    if False:
        for i in range(10):
            print('nop')
    paddle.base.framework.switch_main_program(paddle.static.Program())
    paddle.base.framework.switch_startup_program(paddle.static.Program())

class Model(nn.Layer):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 120)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(120, 10)

    def forward(self, input):
        if False:
            i = 10
            return i + 15
        input.stop_gradient = True
        x = self.flatten(input)
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x

class TestBF16Pass(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.rtol = 1e-05
        self.atol = 1e-08
        self.batch_size = 256
        self.batch_num = 10
        self.dataset = MnistDataset('train')
        self.eval_dataset = MnistDataset('test')

    def init(self, engine):
        if False:
            return 10
        paddle.seed(2021)
        np.random.seed(2021)
        random.seed(2021)
        place = paddle.base.CUDAPlace(paddle.distributed.ParallelEnv().dev_id)
        engine._executor = paddle.static.Executor(place)

    def get_engine(self, use_bf16=False):
        if False:
            return 10
        reset_prog()
        strategy = apply_pass(use_bf16)
        model = Model()
        opt = paddle.optimizer.SGD(0.001, parameters=model.parameters())
        loss = nn.CrossEntropyLoss()
        engine = auto.Engine(model, loss, opt, strategy=strategy)
        self.init(engine)
        return engine

    def check_program(self, program):
        if False:
            print('Hello World!')
        bf16_op_list = {'matmul_v2', 'elementwise_add', 'relu', 'elementwise_add_grad', 'matmul_v2_grad', 'relu_grad'}
        fp32_op_list = {'flatten_contiguous_range', 'reduce_mean', 'softmax_with_cross_entropy', 'fill_constant', 'reduce_mean_grad', 'softmax_with_cross_entropy_grad'}
        for block in program.blocks:
            for op in block.ops:
                if op not in bf16_op_list and op not in fp32_op_list:
                    continue
                for in_name in op.input_names:
                    for in_var_name in op.input(in_name):
                        var = None
                        try:
                            var = block.var(in_var_name)
                        except ValueError as e:
                            var = block._var_recursive(in_var_name)
                        if var is None or var.type not in _valid_types:
                            break
                        if op.type in bf16_op_list:
                            assert var.dtype == core.VarDesc.VarType.BF16
                            if 'cast_bf16' in in_var_name:
                                if '@GRAD' in in_var_name:
                                    tmp_in_var_name = in_var_name[:in_var_name.find('@GRAD')]
                                else:
                                    tmp_in_var_name = in_var_name
                                prev_op = find_true_prev_op(block.ops, op, tmp_in_var_name)
                                assert prev_op is not None
                                assert prev_op.type == 'cast'
                                for in_name in prev_op.input_names:
                                    for in_var_name in prev_op.input(in_name):
                                        var = block.var(in_var_name)
                                        assert var.dtype == core.VarDesc.VarType.FP32
                        elif op.type in fp32_op_list:
                            if (op.type == 'softmax_with_cross_entropy' or op.type == 'softmax_with_cross_entropy_grad') and in_var_name == 'label0':
                                continue
                            assert var.dtype == core.VarDesc.VarType.FP32
                            if 'cast_fp32' in in_var_name:
                                prev_op = find_true_prev_op(block.ops, op, tmp_in_var_name)
                                assert prev_op is not None
                                assert prev_op.type == 'cast'
                                for in_name in prev_op.input_names:
                                    for in_var_name in prev_op.input(in_name):
                                        var = block.var(in_var_name)
                                        assert var.dtype == core.VarDesc.VarType.BF16
                for out_name in op.output_names:
                    for out_var_name in op.output(out_name):
                        var = None
                        try:
                            var = block.var(out_var_name)
                        except ValueError as e:
                            var = block._var_recursive(out_var_name)
                        if var is None or var.type not in _valid_types:
                            break
                        if op.type in bf16_op_list:
                            assert var.dtype == core.VarDesc.VarType.BF16
                        elif op.type in fp32_op_list:
                            assert var.dtype == core.VarDesc.VarType.FP32

    def test_bf16_pass(self):
        if False:
            print('Hello World!')
        bf16_o1_engine = self.get_engine(True)
        inputs_spec = [InputSpec([None, 1, 28, 28], 'float32', 'input0')]
        labels_spec = [InputSpec([None, 1], 'int64', 'label0')]
        bf16_o1_engine.prepare(inputs_spec=inputs_spec, labels_spec=labels_spec, mode='train')
        self.check_program(bf16_o1_engine.main_program)
        print('BF16!check program successfully!')
if __name__ == '__main__':
    unittest.main()