import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.nn.functional as F
import paddle.static

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        if False:
            while True:
                i = 10
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}
        self.feed_list = list(self.feed_fp32.keys())

    def set_feed_attr(self):
        if False:
            print('Hello World!')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        array = np.random.uniform(size=[1]).astype(np.float32)
        result1 = paddle.zeros(shape=[1], dtype='float32')
        weight = paddle.assign(array, result1)
        out = F.prelu(x, weight=weight, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            return 10
        ipu_strategy = paddle.static.IpuStrategy()
        ipu_strategy.set_graph_config(is_training=self.is_training)
        self.run_op_test(exec_mode, ipu_strategy=ipu_strategy)

    def test(self):
        if False:
            print('Hello World!')
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

class TestCase1(TestBase):

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        array = np.random.uniform(size=[3]).astype(np.float32)
        result1 = paddle.zeros(shape=[3], dtype='float32')
        weight = paddle.assign(array, result1)
        out = F.prelu(x, weight=weight, **self.attrs)
        self.fetch_list = [out.name]
if __name__ == '__main__':
    unittest.main()