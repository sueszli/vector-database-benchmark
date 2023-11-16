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
            while True:
                i = 10
        return False

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        data = np.random.uniform(size=[1, 3, 3, 3])
        self.feed_fp32 = {'x': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            i = 10
            return i + 15
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}
        self.attrs['dtype'] = 'float32'

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype=self.feed_dtype[0])
        out = paddle.cast(x, **self.attrs)
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

class TestEnableFp16(TestBase):

    @property
    def fp16_enabled(self):
        if False:
            while True:
                i = 10
        return True

    def run_model(self, exec_mode):
        if False:
            while True:
                i = 10
        self.run_op_test(exec_mode)

    def set_data_feed(self):
        if False:
            print('Hello World!')
        data = np.random.uniform(size=[1, 3, 3, 3])
        self.feed_fp32 = {'x': data.astype(np.float32)}
        self.feed_fp16 = {'x': data.astype(np.float16)}

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {}
        self.attrs['dtype'] = 'float32'

class TestCase2(TestBase):

    def set_atol(self):
        if False:
            return 10
        super().set_atol()
        self.atol = 0.001
        self.rtol = 0.001

    def set_data_feed(self):
        if False:
            print('Hello World!')
        self.feed_fp32 = {'x': np.random.uniform(size=[1, 3, 3, 3]).astype('float32')}

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {}
        self.attrs['dtype'] = 'float16'

class TestCase3(TestBase):

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_fp32 = {'x': np.random.uniform(size=[1, 3, 3, 3]).astype('float32')}

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}
        self.attrs['dtype'] = 'int32'

class TestCase4(TestBase):

    def set_data_feed(self):
        if False:
            print('Hello World!')
        self.feed_fp32 = {'x': np.random.uniform(size=[1, 3, 3, 3]).astype('int32')}

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}
        self.attrs['dtype'] = 'float32'

class TestCase5(TestBase):

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_fp32 = {'x': np.random.uniform(size=[1, 3, 3, 3]).astype('float16')}

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {}
        self.attrs['dtype'] = 'int32'

class TestCase6(TestBase):

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_fp32 = {'x': np.random.uniform(size=[1, 3, 3, 3]).astype('int32')}

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {}
        self.attrs['dtype'] = 'float16'

@unittest.skip('float64 is not supported')
class TestCase7(TestBase):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {}
        self.attrs['dtype'] = 'float64'

@unittest.skip('skip float16 to float32')
class TestCase8(TestBase):

    def set_data_feed(self):
        if False:
            while True:
                i = 10
        self.feed_fp32 = {'x': np.random.uniform(size=[1, 3, 3, 3]).astype('float16')}

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {}
        self.attrs['dtype'] = 'float32'

@unittest.skip('int32 to int8 is not supported')
class TestCase9(TestBase):

    def set_atol(self):
        if False:
            for i in range(10):
                print('nop')
        super().set_atol()
        self.atol = 1

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        self.feed_fp32 = {'x': np.random.randint(low=1, high=100, size=[1, 3, 3, 3]).astype('int32')}

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {}
        self.attrs['dtype'] = 'int8'
if __name__ == '__main__':
    unittest.main()