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
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        if False:
            while True:
                i = 10
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            return 10
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {'p': 0.5, 'training': False, 'mode': 'downgrade_in_infer'}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        x = paddle.nn.functional.dropout(x, **self.attrs)
        out = paddle.add(x, x)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            i = 10
            return i + 15
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

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'p': 0.5, 'training': False, 'mode': 'upscale_in_train'}

class TestCase2(TestBase):

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'p': 0.0, 'training': True, 'mode': 'upscale_in_train'}
if __name__ == '__main__':
    unittest.main()