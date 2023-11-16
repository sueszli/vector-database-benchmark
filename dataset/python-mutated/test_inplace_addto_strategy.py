import unittest
import numpy as np
import paddle
from paddle import base

class ConvBNLayer(paddle.nn.Layer):

    def __init__(self, num_channels, num_filters, filter_size, stride=1, groups=1, data_format='NCHW'):
        if False:
            return 10
        super().__init__()
        self._conv = paddle.nn.Conv2D(in_channels=num_channels, out_channels=num_filters, kernel_size=filter_size, stride=stride, padding=(filter_size - 1) // 2, groups=groups, bias_attr=False, data_format=data_format)
        self._batch_norm = paddle.nn.BatchNorm(num_filters, data_layout=data_format)

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y

def create_program(data_format='NCHW'):
    if False:
        return 10
    main = base.Program()
    startup = base.Program()
    with base.program_guard(main, startup):
        x = paddle.static.data(name='img', shape=[-1, 3, 224, 224])
        x.stop_gradient = False
        if data_format == 'NHWC':
            x = paddle.transpose(x, [0, 2, 3, 1])
        x = paddle.static.nn.prelu(x, mode='channel')
        conv = ConvBNLayer(num_channels=3, num_filters=3, filter_size=1, data_format=data_format)
        y = conv(x) + x
        loss = paddle.sum(y)
        sgd = paddle.optimizer.SGD(learning_rate=0.01)
        sgd.minimize(loss)
    return (loss, main, startup, conv._conv.weight)

class TestInplaceAddto(unittest.TestCase):

    def check_result(self, data_format='NCHW'):
        if False:
            while True:
                i = 10

        def run_program(enable_addto):
            if False:
                return 10
            np.random.seed(10)
            paddle.seed(10)
            paddle.framework.random._manual_program_seed(10)
            if base.core.is_compiled_with_cuda():
                base.set_flags({'FLAGS_cudnn_deterministic': True})
            base.set_flags({'FLAGS_max_inplace_grad_add': 2})
            (loss, main, startup, w) = create_program(data_format=data_format)
            place = base.CUDAPlace(0) if base.core.is_compiled_with_cuda() else base.CPUPlace()
            exe = base.Executor(place)
            strategy = base.BuildStrategy()
            strategy.enable_addto = enable_addto
            compiled = base.CompiledProgram(main, build_strategy=strategy)
            exe.run(startup)
            img = np.random.uniform(-128, 128, [8, 3, 224, 224]).astype(np.float32)
            for i in range(10):
                res = exe.run(compiled, feed={'img': img}, fetch_list=[loss.name, w.name])
            return res
        (res1, w1) = run_program(True)
        (res2, w2) = run_program(False)
        np.testing.assert_array_equal(res1, res2)

    def test_nchw(self):
        if False:
            i = 10
            return i + 15
        self.check_result()

    def test_nhwc(self):
        if False:
            i = 10
            return i + 15
        self.check_result('NHWC')
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()