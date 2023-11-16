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
        self.set_data_feed()
        self.set_feed_attr()
        self.set_attrs()

    def set_data_feed(self):
        if False:
            return 10
        data = np.random.uniform(size=[2, 3])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            return 10
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'shape': [2, 2, 3]}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        out = paddle.expand(x, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            print('Hello World!')
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

    def set_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'shape': [5, 2, 2, 3]}

class TestCase2(TestBase):

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        data = np.random.uniform(size=[2, 1, 3])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}

    def set_attrs(self):
        if False:
            return 10
        self.attrs = {'shape': [5, 2, 2, 3]}

@unittest.skip('corresponding dimensions must have the same value.')
class TestCase3(TestBase):

    def set_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'shape': [5, 2, 4, 3]}

@unittest.skip('Do not support `shape` = Tensors.')
class TestCase4(TestBase):

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        data = np.random.uniform(size=[3, 3])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        self.attrs = {'name': 'y', 'shape': [3], 'dtype': 'int32', 'value': 3}
        y = paddle.tensor.fill_constant(**self.attrs)
        out = paddle.expand(x, shape=y)
        self.fetch_list = [out.name]
if __name__ == '__main__':
    unittest.main()