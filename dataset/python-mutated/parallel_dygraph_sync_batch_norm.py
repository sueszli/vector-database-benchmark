import numpy as np
from legacy_test.test_dist_base import TestParallelDyGraphRunnerBase, runtime_main
import paddle
from paddle.base.dygraph.base import to_variable
from paddle.nn import Conv2D, SyncBatchNorm

class TestLayer(paddle.nn.Layer):

    def __init__(self, num_channels, num_filters, filter_size, stride=1, groups=1, act=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._conv = Conv2D(in_channels=num_channels, out_channels=num_filters, kernel_size=filter_size, stride=stride, padding=(filter_size - 1) // 2, groups=groups, bias_attr=False)
        self._sync_batch_norm = SyncBatchNorm(num_filters)
        self._conv2 = Conv2D(in_channels=num_filters, out_channels=num_filters, kernel_size=filter_size, stride=stride, padding=(filter_size - 1) // 2, groups=groups, bias_attr=False)
        self._sync_batch_norm2 = SyncBatchNorm(num_filters, weight_attr=False, bias_attr=False)

    def forward(self, inputs):
        if False:
            return 10
        y = self._conv(inputs)
        y = self._sync_batch_norm(y)
        y = self._conv2(y)
        y = self._sync_batch_norm2(y)
        return y

class TestSyncBatchNorm(TestParallelDyGraphRunnerBase):

    def get_model(self):
        if False:
            return 10
        model = TestLayer(3, 64, 7)
        train_reader = paddle.batch(paddle.dataset.flowers.test(use_xmap=False), batch_size=32, drop_last=True)
        opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
        return (model, train_reader, opt)

    def run_one_loop(self, model, opt, data):
        if False:
            while True:
                i = 10
        batch_size = len(data)
        dy_x_data = np.array([x[0].reshape(3, 224, 224) for x in data]).astype('float32')
        img = to_variable(dy_x_data)
        img.stop_gradient = False
        out = model(img)
        out = paddle.mean(out)
        return out
if __name__ == '__main__':
    runtime_main(TestSyncBatchNorm)