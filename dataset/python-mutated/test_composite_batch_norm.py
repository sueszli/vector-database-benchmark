import unittest
import numpy as np
from prim.composite_ops.utils import SUB_TOLERANCE
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.base import core, framework
from paddle.incubate.autograd import primapi
from paddle.nn import BatchNorm
from paddle.tensor import ones
np.random.seed(2023)

def generate_data(shape, dtype='float32'):
    if False:
        print('Hello World!')
    np_data = np.random.random(shape).astype(dtype)
    return np_data

class Attr:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.dtype = 'float32'
        self.shape = [4, 6, 12, 24]
        self.training = True
        self.momentum = 0.9
        self.epsilon = 1e-05
        self.data_format = 'NCHW'
        self.use_global_stats = None

    def set_dtype(self, dtype) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.dtype = dtype

    def set_shape(self, shape) -> None:
        if False:
            return 10
        self.shape = shape

    def set_training(self, training) -> None:
        if False:
            print('Hello World!')
        self.training = training

    def set_momentum(self, momentum) -> None:
        if False:
            while True:
                i = 10
        self.momentum = momentum

    def set_epsilon(self, epsilon) -> None:
        if False:
            i = 10
            return i + 15
        self.epsilon = epsilon

    def set_data_format(self, data_format) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.data_format = data_format

    def set_use_global_stats(self, use_global_stats) -> None:
        if False:
            while True:
                i = 10
        self.use_global_stats = use_global_stats

    def get_rtol(self, flag):
        if False:
            for i in range(10):
                print('nop')
        rtol = SUB_TOLERANCE[self.dtype][flag].get('rtol')
        return rtol

    def get_atol(self, flag):
        if False:
            for i in range(10):
                print('nop')
        atol = SUB_TOLERANCE[self.dtype][flag].get('atol')
        return atol
attrs = Attr()

def fn(x, running_mean, running_variance, weight, bias, training, momentum, epsilon, data_format, use_global_stats):
    if False:
        for i in range(10):
            print('nop')
    z = F.batch_norm(x, running_mean, running_variance, weight, bias, training=training, momentum=momentum, epsilon=epsilon, data_format=data_format, use_global_stats=use_global_stats)
    return z

def expect_forward(inputs, running_mean, running_variance, weight, bias, training, momentum, epsilon, data_format, use_global_stats):
    if False:
        i = 10
        return i + 15
    return fn(inputs, running_mean, running_variance, weight, bias, training, momentum, epsilon, data_format, use_global_stats)

def cal_static(inputs, running_mean, running_variance, weight, bias, mode=None):
    if False:
        print('Hello World!')
    paddle.enable_static()
    core._set_prim_all_enabled(True)
    startup_program = paddle.static.Program()
    main_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        x1 = paddle.static.data('x1', shape=inputs.shape, dtype=str(inputs.dtype))
        x2 = paddle.static.data('x2', shape=running_mean.shape, dtype=str(running_mean.dtype))
        x3 = paddle.static.data('x3', shape=running_variance.shape, dtype=str(running_variance.dtype))
        x4 = paddle.static.data('x4', shape=weight.shape, dtype=str(weight.dtype))
        x5 = paddle.static.data('x5', shape=bias.shape, dtype=str(bias.dtype))
        if attrs.use_global_stats is None:
            attrs.use_global_stats = not attrs.training
            trainable_statistics = False
        else:
            trainable_statistics = not attrs.use_global_stats
        use_run_stat = not attrs.training and (not trainable_statistics) or attrs.use_global_stats
        y = fn(x1, x2, x3, x4, x5, attrs.training, attrs.momentum, attrs.epsilon, attrs.data_format, attrs.use_global_stats)
        blocks = main_program.blocks
        names = dict(zip(blocks[0].ops[0].output_names, blocks[0].ops[0].output_arg_names))
        if not use_run_stat:
            vars_list = [names[key] for key in ['Y', 'MeanOut', 'VarianceOut', 'SavedMean', 'SavedVariance']]
        else:
            vars_list = [names[key] for key in ['Y', 'MeanOut', 'VarianceOut']]
        fwd_ops = [op.type for op in blocks[0].ops]
        assert 'batch_norm' in fwd_ops
        if mode:
            primapi.to_prim(blocks)
            fwd_ops_new = [op.type for op in blocks[0].ops]
            assert 'batch_norm' not in fwd_ops_new and 'reduce_mean' not in fwd_ops_new
    exe = paddle.static.Executor()
    exe.run(startup_program)
    if not use_run_stat:
        (Y, MeanOut, VarianceOut, SavedMean, SavedVariance) = exe.run(main_program, feed={'x1': inputs, 'x2': running_mean, 'x3': running_variance, 'x4': weight, 'x5': bias}, fetch_list=vars_list)
    else:
        (Y, MeanOut, VarianceOut) = exe.run(main_program, feed={'x1': inputs, 'x2': running_mean, 'x3': running_variance, 'x4': weight, 'x5': bias}, fetch_list=vars_list)
    paddle.disable_static()
    core._set_prim_all_enabled(False)
    if not use_run_stat:
        return (Y, MeanOut, VarianceOut, SavedMean, SavedVariance)
    else:
        return (Y, MeanOut, VarianceOut)

class TestCompositeBatchNorm(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.dtypes = ['float32', 'float64']
        self.training = [False, True]
        self.shapes = [[8, 8, 16, 16], [2, 3, 4, 4]]
        self.momentum = [0.1, 0.9]
        self.data_formats = ['NCHW', 'NHWC']
        self.use_global_stats = [None, True, False]

    def compare_forward(self):
        if False:
            i = 10
            return i + 15
        np_data = generate_data(attrs.shape, attrs.dtype)
        tensor_data = paddle.to_tensor(np_data)
        if attrs.data_format == 'NCHW':
            C = np_data.shape[1]
        elif attrs.data_format == 'NHWC':
            C = np_data.shape[-1]
        else:
            raise TypeError
        running_mean = paddle.zeros(C, dtype=attrs.dtype)
        running_variance = paddle.ones(C, dtype=attrs.dtype)
        weight = paddle.ones(C, dtype=attrs.dtype) * 2
        bias = paddle.ones(C, dtype=attrs.dtype)
        expect = expect_forward(tensor_data, running_mean, running_variance, weight, bias, attrs.training, attrs.momentum, attrs.epsilon, attrs.data_format, attrs.use_global_stats).numpy()
        np_running_mean = np.zeros(C, dtype=attrs.dtype)
        np_running_variance = np.ones(C, dtype=attrs.dtype)
        np_weight = np.ones(C, dtype=attrs.dtype) * 2
        np_bias = np.ones(C, dtype=attrs.dtype)
        res_origin = cal_static(np_data, np_running_mean, np_running_variance, np_weight, np_bias)
        res_prim = cal_static(np_data, np_running_mean, np_running_variance, np_weight, np_bias, mode='prim')
        assert expect.dtype == res_prim[0].dtype
        np.testing.assert_allclose(expect, res_prim[0], rtol=attrs.get_rtol('forward'), atol=attrs.get_atol('forward'))
        use_global_stats = attrs.use_global_stats
        if use_global_stats is None:
            use_global_stats = not attrs.training
            trainable_statistics = False
        else:
            trainable_statistics = not use_global_stats
        test_mode = not attrs.training and (not trainable_statistics)
        global_stats = test_mode or use_global_stats
        vars_name = ['Y', 'MeanOut', 'VarianceOut', 'SavedMean', 'SavedVariance']
        assert len(res_origin) == len(res_prim)
        for idx in range(len(res_origin)):
            if global_stats and idx >= 3:
                continue
            origin_item = res_origin[idx]
            prim_item = res_prim[idx]
            assert origin_item.dtype == prim_item.dtype
            rtol = attrs.get_rtol('forward')
            atol = attrs.get_atol('forward')
            if attrs.dtype == 'float64' and idx in (1, 2, 3):
                atol = 1e-07
                rtol = 1e-07
            if not isinstance(framework._current_expected_place(), core.CPUPlace) and idx in (2, 3):
                atol = 0.005
                rtol = 0.005
            np.testing.assert_allclose(origin_item, prim_item, rtol=atol, atol=rtol, err_msg=f'Check diff failed of output: {vars_name[idx]}')

    def test_forward(self):
        if False:
            i = 10
            return i + 15
        for i in self.training:
            for j in self.dtypes:
                for k in self.use_global_stats:
                    attrs.set_training(i)
                    attrs.set_dtype(j)
                    attrs.set_use_global_stats(k)
                    self.compare_forward()
        for n in self.shapes:
            for m in self.momentum:
                for s in self.data_formats:
                    attrs.set_momentum(m)
                    attrs.set_shape(n)
                    attrs.set_data_format(s)
                    self.compare_forward()

def apply_to_static(net, use_cinn):
    if False:
        while True:
            i = 10
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(net, build_strategy=False)

class PrimeNet(paddle.nn.Layer):

    def __init__(self, data_layout='NCHW', is_test=False):
        if False:
            print('Hello World!')
        super().__init__()
        self.conv = nn.Conv2D(2, 4, (3, 3), bias_attr=False)
        self.bn = BatchNorm(4, act='relu', data_layout=data_layout, is_test=is_test)

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        y = self.conv(x)
        out = self.bn(y)
        res = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        return res

class TestPrimForwardAndBackward(unittest.TestCase):
    """
    Test PrimeNet with @to_static + prim forward + prim backward + cinn v.s Dygraph
    """

    def setUp(self):
        if False:
            return 10
        paddle.seed(2022)
        self.x = paddle.randn([4, 2, 6, 6], dtype='float32')
        self.x.stop_gradient = False

    def train(self, use_prim, data_layout='NCHW', is_test=False):
        if False:
            print('Hello World!')
        core._set_prim_all_enabled(use_prim)
        paddle.seed(2022)
        net = PrimeNet(data_layout=data_layout, is_test=is_test)
        sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=net.parameters())
        net = paddle.amp.decorate(models=net, level='O2')
        net = apply_to_static(net, False)
        with paddle.amp.auto_cast(level='O2'):
            out = net(self.x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
            return loss

    def test_amp_nchw(self):
        if False:
            while True:
                i = 10
        if not isinstance(framework._current_expected_place(), core.CPUPlace):
            expected = self.train(use_prim=False)
            actual = self.train(use_prim=True)
            np.testing.assert_allclose(expected, actual, rtol=0.001, atol=0.001)

    def test_amp_nchw_eval(self):
        if False:
            i = 10
            return i + 15
        if not isinstance(framework._current_expected_place(), core.CPUPlace):
            expected = self.train(use_prim=False, is_test=True)
            actual = self.train(use_prim=True, is_test=True)
            np.testing.assert_allclose(expected, actual, rtol=0.001, atol=0.001)

    def test_amp_nhwc(self):
        if False:
            return 10
        if not isinstance(framework._current_expected_place(), core.CPUPlace):
            expected = self.train(use_prim=False, data_layout='NHWC')
            actual = self.train(use_prim=True, data_layout='NHWC')
            np.testing.assert_allclose(expected, actual, rtol=0.001, atol=0.001)

    def test_amp_nhwc_eval(self):
        if False:
            while True:
                i = 10
        if not isinstance(framework._current_expected_place(), core.CPUPlace):
            expected = self.train(use_prim=False, data_layout='NHWC', is_test=True)
            actual = self.train(use_prim=True, data_layout='NHWC', is_test=True)
            np.testing.assert_allclose(expected, actual, rtol=0.001, atol=0.001)

class TestPrimEvalBranch(unittest.TestCase):
    """
    Test eval branch or composite rule of batch_norm.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        paddle.seed(2022)
        self.x = paddle.randn([4, 2, 6, 6], dtype='float32')
        self.x.stop_gradient = False

    def train(self, use_prim):
        if False:
            i = 10
            return i + 15
        core._set_prim_all_enabled(use_prim)
        paddle.seed(2022)
        net = BatchNorm(2, is_test=True)
        net = apply_to_static(net, False)
        out = net(self.x)
        loss = paddle.mean(out)
        return loss

    def test_eval_branch(self):
        if False:
            return 10
        expected = self.train(False)
        actual = self.train(True)
        np.testing.assert_allclose(expected, actual, rtol=1e-06, atol=1e-06)
if __name__ == '__main__':
    unittest.main()