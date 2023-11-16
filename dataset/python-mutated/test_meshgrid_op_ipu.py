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
            for i in range(10):
                print('nop')
        self.atol = 1e-06
        self.rtol = 1e-06
        self.atol_fp16 = 0.001
        self.rtol_fp16 = 0.001

    def set_feed(self):
        if False:
            print('Hello World!')
        data1 = np.random.uniform(size=[10])
        data2 = np.random.uniform(size=[20])
        self.feed_fp32 = {'x': data1.astype(np.float32), 'y': data2.astype(np.float32)}
        self.feed_fp16 = {'x': data1.astype(np.float16), 'y': data2.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            return 10
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}
        self.attrs['axis'] = [0, 1]

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype=self.feed_dtype[0])
        y = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype=self.feed_dtype[1])
        (r1, r2) = paddle.meshgrid(x, y)
        self.fetch_list = [r1.name, r2.name]

    def run_model(self, exec_mode):
        if False:
            return 10
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        for (k, v) in self.output_dict.items():
            self.output_dict[k] = np.concatenate([vv.flatten() for vv in v])
        self.check()

class TestCase1(TestBase):

    def set_feed(self):
        if False:
            for i in range(10):
                print('nop')
        data1 = np.random.uniform(size=[10])
        data2 = np.random.uniform(size=[20])
        data3 = np.random.uniform(size=[30])
        self.feed_fp32 = {'x': data1.astype(np.float32), 'y': data2.astype(np.float32), 'z': data3.astype(np.float32)}
        self.feed_fp16 = {'x': data1.astype(np.float16), 'y': data2.astype(np.float16), 'z': data3.astype(np.float16)}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            return 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype=self.feed_dtype[0])
        y = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype=self.feed_dtype[1])
        z = paddle.static.data(name=self.feed_list[2], shape=self.feed_shape[2], dtype=self.feed_dtype[2])
        (r1, r2, r3) = paddle.meshgrid(x, y, z)
        self.fetch_list = [r1.name, r2.name, r3.name]

class TestCase2(TestBase):

    def set_feed(self):
        if False:
            print('Hello World!')
        data1 = np.random.uniform(size=[100])
        data2 = np.random.uniform(size=[200])
        self.feed_fp32 = {'x': data1.astype(np.int32), 'y': data2.astype(np.int32)}
        self.feed_fp16 = {'x': data1.astype(np.int32), 'y': data2.astype(np.int32)}
if __name__ == '__main__':
    unittest.main()