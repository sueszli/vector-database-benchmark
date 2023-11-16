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
        self.set_op_attrs()

    def set_data_feed(self):
        if False:
            return 10
        x = np.random.uniform(size=[2, 3])
        y = np.random.uniform(size=[3, 2])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            return 10
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'transpose_x': False, 'transpose_y': False}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        y = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='float32')
        out = paddle.matmul(x, y, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            while True:
                i = 10
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

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'transpose_x': True, 'transpose_y': True}

class TestCase3(TestBase):

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(size=[5, 4, 2, 3])
        y = np.random.uniform(size=[5, 4, 3, 2])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}

class TestCase4(TestBase):

    def set_data_feed(self):
        if False:
            return 10
        x = np.random.uniform(size=[4, 2, 3])
        y = np.random.uniform(size=[4, 3, 2])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}

class TestCase5(TestBase):

    def set_data_feed(self):
        if False:
            print('Hello World!')
        x = np.random.uniform(size=[4, 2, 3])
        y = np.random.uniform(size=[3, 2])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}

class TestCase6(TestBase):

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(size=[3])
        y = np.random.uniform(size=[3])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}

@unittest.skip('not supported')
class TestCase6_2(TestCase6):

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(size=[3])
        y = np.random.uniform(size=[3])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {'transpose_x': True, 'transpose_y': True}

class TestCase7(TestBase):

    def set_data_feed(self):
        if False:
            print('Hello World!')
        x = np.random.uniform(size=[3, 1])
        y = np.random.uniform(size=[1, 2])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}

@unittest.skip('dim > 4 is not supported')
class TestCase8(TestBase):

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed = {'x': np.random.uniform(size=[6, 5, 4, 2, 3]).astype('float32'), 'y': np.random.uniform(size=[6, 5, 4, 3, 2]).astype('float32')}

class TestCase9(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'transpose_y': True}

    def set_data_feed(self):
        if False:
            print('Hello World!')
        x = np.random.uniform(size=[4, 2, 3])
        y = np.random.uniform(size=[2, 3])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}

class TestCase10(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {'transpose_x': True}

    def set_data_feed(self):
        if False:
            print('Hello World!')
        x = np.random.uniform(size=[4, 3, 2])
        y = np.random.uniform(size=[3, 2])
        self.feed_fp32 = {'x': x.astype(np.float32), 'y': y.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'y': y.astype(np.float16)}
if __name__ == '__main__':
    unittest.main()