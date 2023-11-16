import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()

    def set_data_feed(self):
        if False:
            return 10
        data_x = np.random.uniform(size=[1, 3])
        data_y = np.random.uniform(size=[2, 2, 3])
        self.feed_fp32 = {'x': data_x.astype(np.float32), 'y': data_y.astype(np.float32)}
        self.feed_fp16 = {'x': data_x.astype(np.float16), 'y': data_y.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            return 10
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            return 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        y = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='float32')
        out = paddle.expand_as(x, y)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            while True:
                i = 10
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            i = 10
            return i + 15
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

class TestCase1(TestBase):

    def set_data_feed(self):
        if False:
            print('Hello World!')
        data_x = np.random.uniform(size=[2, 3])
        data_y = np.random.uniform(size=[2, 4, 2, 3])
        self.feed_fp32 = {'x': data_x.astype(np.float32), 'y': data_y.astype(np.float32)}
        self.feed_fp16 = {'x': data_x.astype(np.float16), 'y': data_y.astype(np.float16)}

@unittest.skip('corresponding dimensions must have the same value.')
class TestCase2(TestBase):

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        data_x = np.random.uniform(size=[2, 3])
        data_y = np.random.uniform(size=[2, 4, 3, 3])
        self.feed_fp32 = {'x': data_x.astype(np.float32), 'y': data_y.astype(np.float32)}
        self.feed_fp16 = {'x': data_x.astype(np.float16), 'y': data_y.astype(np.float16)}
if __name__ == '__main__':
    unittest.main()