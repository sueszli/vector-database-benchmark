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
            for i in range(10):
                print('nop')
        self.atol = 1e-06
        self.rtol = 1e-06
        self.atol_fp16 = 0.001
        self.rtol_fp16 = 0.001

    def set_feed(self):
        if False:
            i = 10
            return i + 15
        data = np.random.uniform(size=[5, 5])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}
        self.attrs['min'] = 0.1
        self.attrs['max'] = 3.4

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        x = paddle.clip(x, **self.attrs)
        self.fetch_list = [x.name]

    def run_model(self, exec_mode):
        if False:
            return 10
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            i = 10
            return i + 15
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

class TestNoMin(TestBase):

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}
        self.attrs['max'] = 3.4

class TestNoMax(TestBase):

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {}
        self.attrs['min'] = 0.1

class TestNoMinNoMax(TestBase):

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}

class TestMinMaxTensor(TestBase):

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            return 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        min = paddle.tensor.fill_constant(name='min', shape=[1], dtype='float32', value=0.1)
        max = paddle.tensor.fill_constant(name='max', shape=[1], dtype='float32', value=3.4)
        x = paddle.clip(x, min=min, max=max)
        self.fetch_list = [x.name]

class TestMinTensor(TestBase):

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        min = paddle.tensor.fill_constant(name='min', shape=[1], dtype='float32', value=0.1)
        x = paddle.clip(x, min=min)
        self.fetch_list = [x.name]

class TestMaxTensor(TestBase):

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            return 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        max = paddle.tensor.fill_constant(name='max', shape=[1], dtype='float32', value=3.4)
        x = paddle.clip(x, max=max)
        self.fetch_list = [x.name]

class TestCombine1(TestBase):

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        min = paddle.tensor.fill_constant(name='min', shape=[1], dtype='float32', value=0.1)
        x = paddle.clip(x, min=min, max=3.4)
        self.fetch_list = [x.name]

class TestCombine2(TestBase):

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        max = paddle.tensor.fill_constant(name='max', shape=[1], dtype='float32', value=3.4)
        x = paddle.clip(x, min=0.1, max=max)
        self.fetch_list = [x.name]

class TestIntInput(TestBase):

    def set_feed(self):
        if False:
            while True:
                i = 10
        data = np.random.uniform(size=[5, 5])
        self.feed_fp32 = {'x': data.astype(np.int32)}
        self.feed_fp16 = {'x': data.astype(np.int32)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='int32')
        x = paddle.clip(x, min=0.1, max=3.4)
        self.fetch_list = [x.name]

class TestIntMinMax(TestBase):

    def set_feed(self):
        if False:
            while True:
                i = 10
        data = np.random.uniform(size=[5, 5])
        self.feed_fp32 = {'x': data.astype(np.int32)}
        self.feed_fp16 = {'x': data.astype(np.int32)}
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='int32')
        min = paddle.tensor.fill_constant(name='min', shape=[1], dtype='int32', value=1)
        max = paddle.tensor.fill_constant(name='max', shape=[1], dtype='int32', value=3)
        x = paddle.clip(x, min=min, max=max)
        self.fetch_list = [x.name]
if __name__ == '__main__':
    unittest.main()