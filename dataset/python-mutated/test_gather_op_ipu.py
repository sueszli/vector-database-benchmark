import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        if False:
            print('Hello World!')
        x = np.random.uniform(size=[10, 20])
        y = np.array([1, 3, 5])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.int32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.int32)}

    def set_feed_attr(self):
        if False:
            i = 10
            return i + 15
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        y = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='int32')
        out = paddle.gather(x, index=y, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            return 10
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            return 10
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

class TestCase1(TestBase):

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(size=[100])
        y = np.array([1, 3, 5])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.int32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.int32)}
if __name__ == '__main__':
    unittest.main()