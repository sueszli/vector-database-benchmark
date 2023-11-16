import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestGreaterThan(IPUOpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_atol()
        self.set_training()
        self.set_test_op()

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = paddle.base.layers.greater_than

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            return 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        y = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='float32')
        out = self.op(x, y, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            i = 10
            return i + 15
        self.run_op_test(exec_mode)

    def run_test_base(self):
        if False:
            i = 10
            return i + 15
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

    def set_feed_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_data_feed0(self):
        if False:
            while True:
                i = 10
        x = np.random.randn(3, 4, 5)
        y = np.random.randn(3, 4, 5)
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}
        self.set_feed_attr()

    def set_data_feed1(self):
        if False:
            print('Hello World!')
        x = np.ones([1, 10])
        y = np.ones([10])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}
        self.set_feed_attr()

    def set_data_feed2(self):
        if False:
            return 10
        x = np.ones([1, 10])
        y = np.zeros([1, 10])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}
        self.set_feed_attr()

    def set_data_feed3(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.zeros([1, 10])
        y = np.ones([1, 10])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}
        self.set_feed_attr()

    def test_case0(self):
        if False:
            while True:
                i = 10
        self.set_data_feed0()
        self.set_op_attrs()
        self.run_test_base()

    def test_case1(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_data_feed1()
        self.set_op_attrs()
        self.run_test_base()

    def test_case2(self):
        if False:
            print('Hello World!')
        self.set_data_feed2()
        self.set_op_attrs()
        self.run_test_base()

    def test_case3(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_data_feed3()
        self.set_op_attrs()
        self.run_test_base()

class TestLessThan(TestGreaterThan):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = paddle.base.layers.less_than

class TestEqual(TestGreaterThan):

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = paddle.base.layers.equal

class TestGreaterEqual(TestGreaterThan):

    def set_test_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = paddle.base.layers.greater_equal

class TestLessEqual(TestGreaterThan):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = paddle.base.layers.less_equal
if __name__ == '__main__':
    unittest.main()