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
        self.set_feed()
        self.set_op_attrs()

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {}

    def set_feed(self):
        if False:
            for i in range(10):
                print('nop')
        data = np.random.uniform(size=[32, 100])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        x = paddle.static.nn.data_norm(input=x, **self.attrs)
        self.fetch_list = [x.name]

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
                self.build_model()
                self.run_model(m)
        self.check()

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {'in_place': True}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        x = paddle.static.nn.data_norm(input=x, **self.attrs)
        x = x + 1
        self.fetch_list = [x.name]

@unittest.skip('Do not support in_place=True when test single data_norm Op')
class TestCase2(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {'in_place': True}

class TestCase3(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'data_layout': 'NHWC'}

class TestCase4(TestBase):

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'epsilon': 0.001}

class TestCase5(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'do_model_average_for_mean_and_var': True}

class TestCase6(TestBase):

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {'param_attr': {'scale_w': 0.5, 'bias': 0.1}, 'enable_scale_and_shift': True}

class TestCase7(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'param_attr': {'batch_size': 1000.0, 'batch_sum': 0.1, 'batch_square': 1000.0, 'scale_w': 0.5, 'bias': 0.1}, 'enable_scale_and_shift': True}
if __name__ == '__main__':
    unittest.main()