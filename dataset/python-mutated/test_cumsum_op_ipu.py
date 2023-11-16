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

    @property
    def fp16_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(size=[1, 128])
        self.feed_fp32 = {'x': x.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            return 10
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
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        out = paddle.cumsum(x, **self.attrs)
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

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'exclusive': True, 'reverse': False}

class TestCase2(TestBase):

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'exclusive': False, 'reverse': True}

class TestCase3(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {'exclusive': True, 'reverse': True}

class TestCase4(TestBase):

    def set_data_feed(self):
        if False:
            while True:
                i = 10
        x = np.random.uniform(size=[1, 128])
        self.feed_fp32 = {'x': x.astype(np.int32)}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='int32')
        out = paddle.cumsum(x, **self.attrs)
        self.fetch_list = [out.name]

class TestCase5(TestBase):

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(size=[1, 128])
        self.feed_fp32 = {'x': x.astype(np.int64)}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='int64')
        out = paddle.cumsum(x, **self.attrs)
        self.fetch_list = [out.name]
if __name__ == '__main__':
    unittest.main()