import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            return 10
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_op_attrs()

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        data = np.random.uniform(size=[2, 3, 1])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            while True:
                i = 10
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'fill_value': 0.3, 'dtype': 'float32'}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        x_fill = paddle.full_like(x, **self.attrs)
        out = paddle.add(x_fill, x_fill)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            i = 10
            return i + 15
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            return 10
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'fill_value': 3, 'dtype': 'int32'}

class TestError(TestBase):

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data('x', [-1, 3, 13], 'float32')
        x_fill = paddle.full_like(x, **self.attrs)
        out = paddle.add(x_fill, x_fill)
        self.fetch_list = [out.name]

    def test(self):
        if False:
            while True:
                i = 10
        self.build_model()

        def test_error():
            if False:
                for i in range(10):
                    print('nop')
            self.run_op_test(IPUOpTest.ExecutionMode.IPU_FP32)
        self.assertRaisesRegex(Exception, 'Please check tensor shape setting', test_error)
if __name__ == '__main__':
    unittest.main()