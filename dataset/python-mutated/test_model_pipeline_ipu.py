import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        data = np.random.uniform(size=[2, 3, 10, 10])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}

    def set_feed_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_shape = [(1, 3, 10, 10)]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            return 10
        image = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        with paddle.static.ipu_shard_guard(index=0):
            conv1 = paddle.static.nn.conv2d(image, num_filters=3, filter_size=3, bias_attr=False)
        with paddle.static.ipu_shard_guard(index=1):
            conv2 = paddle.static.nn.conv2d(conv1, num_filters=3, filter_size=3, bias_attr=False)
            loss = paddle.mean(conv2)
        self.fetch_list = [loss.name]

    def run_model(self, exec_mode):
        if False:
            return 10
        ipu_strategy = paddle.static.IpuStrategy()
        ipu_strategy.set_graph_config(num_ipus=2, is_training=False, enable_manual_shard=True)
        ipu_strategy.set_pipelining_config(enable_pipelining=True, batches_per_step=2)
        self.run_op_test(exec_mode, ipu_strategy=ipu_strategy)

    def test(self):
        if False:
            i = 10
            return i + 15
        self.build_model()
        self.run_model(IPUOpTest.ExecutionMode.IPU_FP32)
if __name__ == '__main__':
    unittest.main()