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

    def set_atol(self):
        if False:
            return 10
        self.atol = 1e-06
        self.rtol = 1e-05
        self.atol_fp16 = 0.01
        self.rtol_fp16 = 0.001

    def set_data_feed(self):
        if False:
            while True:
                i = 10
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {}
        self.attrs['is_test'] = False
        self.attrs['data_layout'] = 'NCHW'
        self.attrs['in_place'] = False

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            return 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        x = paddle.static.nn.conv2d(x, num_filters=3, filter_size=3, bias_attr=False)
        x = paddle.static.nn.batch_norm(x, **self.attrs)
        self.fetch_list = [x.name]

    def run_model(self, exec_mode):
        if False:
            for i in range(10):
                print('nop')
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

    def set_atol(self):
        if False:
            i = 10
            return i + 15
        self.atol = 1e-06
        self.rtol = 1e-06
        self.atol_fp16 = 0.001
        self.rtol_fp16 = 0.001

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {}
        self.attrs['is_test'] = True
        self.attrs['data_layout'] = 'NCHW'
        self.attrs['in_place'] = False

class TestCase2(TestBase):

    def set_atol(self):
        if False:
            return 10
        self.atol = 1e-06
        self.rtol = 1e-06
        self.atol_fp16 = 0.001
        self.rtol_fp16 = 0.001

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {}
        self.attrs['is_test'] = True
        self.attrs['data_layout'] = 'NCHW'
        self.attrs['in_place'] = True
if __name__ == '__main__':
    unittest.main()