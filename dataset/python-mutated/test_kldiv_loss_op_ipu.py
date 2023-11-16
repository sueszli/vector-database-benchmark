import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            return 10
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        if False:
            print('Hello World!')
        x = np.random.uniform(size=[3, 4, 2, 2])
        target = np.random.uniform(size=[3, 4, 2, 2])
        self.feed_fp32 = {'x': x.astype(np.float32), 'target': target.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'target': target.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {'reduction': 'mean'}

    @IPUOpTest.static_graph
    def build_model(self, on_ipu):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        target = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='float32')
        out = paddle.nn.functional.kl_div(x, target, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            while True:
                i = 10
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            print('Hello World!')
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model(self.is_ipu_mode(m))
                self.run_model(m)
        self.check()

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {'reduction': 'sum'}

class TestCase2(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'reduction': 'none'}
if __name__ == '__main__':
    unittest.main()