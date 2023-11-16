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
        data_x = np.random.uniform(size=[4, 5, 6])
        data_y = np.random.uniform(size=[4, 5, 6])
        self.feed_fp32 = {'in_0': data_x.astype(np.float32), 'in_1': data_y.astype(np.float32)}
        self.feed_fp16 = {'in_0': data_x.astype(np.float16), 'in_1': data_y.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        y = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='float32')
        out = paddle.where(x > 1, x, y)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            return 10
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            print('Hello World!')
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()
if __name__ == '__main__':
    unittest.main()