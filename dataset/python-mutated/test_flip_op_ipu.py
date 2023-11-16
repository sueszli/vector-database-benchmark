import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_atol()
        self.set_training()
        self.set_feed()
        self.set_feed_attr()
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
            while True:
                i = 10
        data = np.random.uniform(size=[3, 2, 2])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            i = 10
            return i + 15
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {}
        self.attrs['axis'] = [0, 1]

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype=self.feed_dtype[0])
        x = paddle.flip(x, **self.attrs)
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

    def set_feed(self):
        if False:
            while True:
                i = 10
        data = np.random.randint(0, 10, size=[3, 2, 2])
        self.feed_fp32 = {'x': data.astype(np.int32)}
        self.feed_fp16 = {'x': data.astype(np.int32)}

class TestCase2(TestBase):

    def set_feed(self):
        if False:
            return 10
        data = np.random.randint(0, 2, size=[4, 3, 2, 2])
        self.feed_fp32 = {'x': data.astype(np.bool_)}
        self.feed_fp16 = {'x': data.astype(np.bool_)}
if __name__ == '__main__':
    unittest.main()