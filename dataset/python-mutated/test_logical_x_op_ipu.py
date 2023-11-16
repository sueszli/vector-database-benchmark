import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestLogicalAnd(IPUOpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_atol()
        self.set_training()
        self.set_test_op()

    @property
    def fp16_enabled(self):
        if False:
            print('Hello World!')
        return False

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = paddle.logical_and

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype=self.feed_dtype[0])
        y = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype=self.feed_dtype[1])
        out = self.op(x, y, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            for i in range(10):
                print('nop')
        self.run_op_test(exec_mode)

    def run_test_base(self):
        if False:
            print('Hello World!')
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

    def set_feed_attr(self):
        if False:
            i = 10
            return i + 15
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = ['bool', 'bool']

    def set_data_feed0(self):
        if False:
            print('Hello World!')
        x = np.random.choice([True, False], size=(1, 3, 5, 5))
        y = np.random.choice([True, False], size=(1, 3, 5, 5))
        self.feed_fp32 = {'x': x.astype('bool'), 'y': y.astype('bool')}
        self.set_feed_attr()

    def test_case0(self):
        if False:
            while True:
                i = 10
        self.set_data_feed0()
        self.set_op_attrs()
        self.run_test_base()

class TestLogicalOr(TestLogicalAnd):

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = paddle.logical_or
if __name__ == '__main__':
    unittest.main()