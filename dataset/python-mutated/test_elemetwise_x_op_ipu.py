import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestMul(IPUOpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_atol()
        self.set_training()
        self.set_test_op()

    @property
    def fp16_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        if IPUOpTest.use_ipumodel():
            return False
        else:
            return True

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = paddle.tensor.math._multiply_with_axis

    def set_feed_attr(self):
        if False:
            i = 10
            return i + 15
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        y = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='float32')
        out = self.op(x, y, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            while True:
                i = 10
        self.run_op_test(exec_mode)

    def run_test_base(self):
        if False:
            for i in range(10):
                print('nop')
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

    def test_case0(self):
        if False:
            while True:
                i = 10
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(2, 3, 4, 5))
        self.feed_fp32 = {'x': data_x.astype('float32'), 'y': data_y.astype('float32')}
        self.feed_fp16 = {'x': data_x.astype('float16'), 'y': data_y.astype('float16')}
        self.attrs = {}
        self.set_feed_attr()
        self.run_test_base()

    def test_case1(self):
        if False:
            i = 10
            return i + 15
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=(3, 4))
        self.feed_fp32 = {'x': data_x.astype('float32'), 'y': data_y.astype('float32')}
        self.feed_fp16 = {'x': data_x.astype('float16'), 'y': data_y.astype('float16')}
        self.set_feed_attr()
        self.attrs = {'axis': 1}
        self.run_test_base()

    def test_case2(self):
        if False:
            for i in range(10):
                print('nop')
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=5)
        self.feed_fp32 = {'x': data_x.astype('float32'), 'y': data_y.astype('float32')}
        self.feed_fp16 = {'x': data_x.astype('float16'), 'y': data_y.astype('float16')}
        self.set_feed_attr()
        self.attrs = {'axis': -1}
        self.run_test_base()

    def test_case3(self):
        if False:
            while True:
                i = 10
        data_x = np.random.uniform(size=(2, 3, 4, 5))
        data_y = np.random.uniform(size=2)
        self.feed_fp32 = {'x': data_x.astype('float32'), 'y': data_y.astype('float32')}
        self.feed_fp16 = {'x': data_x.astype('float16'), 'y': data_y.astype('float16')}
        self.set_feed_attr()
        self.attrs = {'axis': 0}
        self.run_test_base()

class TestAdd(TestMul):

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = paddle.add

class TestSub(TestMul):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = paddle.subtract

class TestDiv(TestMul):

    def set_test_op(self):
        if False:
            print('Hello World!')
        self.op = paddle.divide

class TestMin(TestMul):

    def set_test_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = paddle.minimum

class TestMax(TestMul):

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = paddle.maximum

class TestPow(TestMul):

    def set_test_op(self):
        if False:
            print('Hello World!')
        self.op = paddle.pow

class TestMod(TestMul):

    def set_atol(self):
        if False:
            while True:
                i = 10
        self.atol = 1e-07
        self.rtol = 1e-05
        self.atol_fp16 = 0.01
        self.rtol_fp16 = 0.001

    def set_test_op(self):
        if False:
            return 10
        self.op = paddle.remainder
if __name__ == '__main__':
    unittest.main()