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
        self.set_op_attrs()

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        data = np.random.uniform(size=[2, 2, 4, 6])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            i = 10
            return i + 15
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {}
        self.attrs['axis'] = 1

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        if self.attrs['axis'] == 0:
            x = paddle.flatten(x, 0, -1)
            out = paddle.unsqueeze(x, 0)
        else:
            x = paddle.flatten(x, self.attrs['axis'], -1)
            out = paddle.flatten(x, 0, self.attrs['axis'] - 1)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            print('Hello World!')
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            while True:
                i = 10
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {}
        self.attrs['axis'] = 0

class TestCase2(TestBase):

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}
        self.attrs['axis'] = 2
if __name__ == '__main__':
    unittest.main()