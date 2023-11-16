import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        data = np.random.uniform(size=[1, 2, 3, 3]).astype(np.float16)
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            while True:
                i = 10
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'axis': -1, 'descending': False}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        (out, _) = paddle.argsort(x, **self.attrs)
        self.fetch_list = [out.name]

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
        for (k, v) in self.output_dict.items():
            self.output_dict[k] = v.astype(np.int32)
        self.check()

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {'axis': 0, 'descending': False}

class TestCase2(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'axis': 1, 'descending': True}
if __name__ == '__main__':
    unittest.main()