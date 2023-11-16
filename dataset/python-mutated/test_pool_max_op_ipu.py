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

    def set_data_feed(self):
        if False:
            print('Hello World!')
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            print('Hello World!')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'kernel_size': 3, 'stride': 1, 'padding': 0, 'ceil_mode': False, 'data_format': 'NCHW'}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        out = paddle.nn.functional.max_pool2d(x, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            print('Hello World!')
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
        super().set_op_attrs()
        self.attrs['kernel_size'] = 3

class TestCase1_2(TestBase):

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        super().set_op_attrs()
        self.attrs['kernel_size'] = [3, 1]

class TestCase2(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        super().set_op_attrs()
        self.attrs['stride'] = 2

class TestCase2_2(TestBase):

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        super().set_op_attrs()
        self.attrs['stride'] = [2, 1]

class TestCase3(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        super().set_op_attrs()
        self.attrs['padding'] = [1, 1]

class TestCase3_2(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        super().set_op_attrs()
        self.attrs['padding'] = [1, 1, 2, 2]

@unittest.skip('auto_pad is not currently supported')
class TestCase3_3(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        super().set_op_attrs()
        self.attrs['padding'] = 'VALID'

@unittest.skip('auto_pad is not currently supported')
class TestCase3_4(TestBase):

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        super().set_op_attrs()
        self.attrs['padding'] = 'SAME'

class TestCase5(TestBase):

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        super().set_op_attrs()
        self.attrs['ceil_mode'] = True
if __name__ == '__main__':
    unittest.main()