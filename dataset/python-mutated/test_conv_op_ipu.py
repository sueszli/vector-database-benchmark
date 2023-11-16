import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_atol()
        self.set_training()
        self.set_feed()
        self.set_op_attrs()

    def set_atol(self):
        if False:
            print('Hello World!')
        self.atol = 1e-06
        self.rtol = 1e-06
        self.atol_fp16 = 0.001
        self.rtol_fp16 = 0.001

    def set_feed(self):
        if False:
            for i in range(10):
                print('nop')
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {}
        self.attrs['num_filters'] = 3
        self.attrs['filter_size'] = 3
        self.attrs['stride'] = 1
        self.attrs['padding'] = 0
        self.attrs['dilation'] = 1
        self.attrs['groups'] = 1
        self.attrs['data_format'] = 'NCHW'

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            return 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        x = paddle.static.nn.conv2d(x, **self.attrs)
        self.fetch_list = [x.name]

    def run_model(self, exec_mode):
        if False:
            print('Hello World!')
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        super().set_op_attrs()
        self.attrs['num_filters'] = 1

class TestCase2(TestBase):

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        super().set_op_attrs()
        self.attrs['filter_size'] = [3, 3]

class TestCase2_1(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        super().set_op_attrs()
        self.attrs['filter_size'] = [3, 2]

class TestCase3(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        super().set_op_attrs()
        self.attrs['stride'] = [2, 3]

class TestCase4(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        super().set_op_attrs()
        self.attrs['dilation'] = [2, 2]

class TestCase5(TestBase):

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        super().set_op_attrs()
        self.attrs['groups'] = 3

class TestCase6(TestBase):

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        super().set_op_attrs()
        self.attrs['padding'] = 2

class TestCase7(TestBase):

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        super().set_op_attrs()
        self.attrs['padding'] = [2, 3]

class TestCase8(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        super().set_op_attrs()
        self.attrs['padding'] = [1, 2, 2, 3]

class TestCase9(TestBase):

    def set_feed(self):
        if False:
            return 10
        data = np.random.uniform(size=[1, 3, 10, 10])
        weight = np.random.uniform(size=[3, 1, 3, 3])
        self.feed_fp32 = {'in_0': data.astype(np.float32), 'in_1': weight.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16), 'in_1': weight.astype(np.float16)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {}
        self.attrs['groups'] = 3

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        weight = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='float32')
        x = paddle.nn.functional.conv2d(x, weight, **self.attrs)
        self.fetch_list = [x.name]
if __name__ == '__main__':
    unittest.main()