import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestMean(IPUOpTest):

    def setUp(self):
        if False:
            return 10
        self.set_atol()
        self.set_training()
        self.set_test_op()

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = paddle.mean

    def set_feed_attr(self):
        if False:
            print('Hello World!')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        out = self.op(x, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            for i in range(10):
                print('nop')
        self.run_op_test(exec_mode)

    def run_test_base(self):
        if False:
            return 10
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

    def set_data_feed0(self):
        if False:
            while True:
                i = 10
        data = np.random.uniform(size=[2, 4])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}
        self.set_feed_attr()

    def set_data_feed1(self):
        if False:
            return 10
        data = np.random.uniform(size=[2, 2, 2])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}
        self.set_feed_attr()

    def set_op_attr0(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {}
        self.attrs['dim'] = None
        self.attrs['keep_dim'] = False

    def test_case0(self):
        if False:
            print('Hello World!')
        self.set_data_feed0()
        self.set_op_attr0()
        self.run_test_base()

    def test_case1(self):
        if False:
            while True:
                i = 10
        self.set_data_feed0()
        self.set_op_attr0()
        self.attrs['dim'] = 0
        self.run_test_base()

    def test_case2(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_data_feed0()
        self.set_op_attr0()
        self.attrs['dim'] = -1
        self.run_test_base()

    def test_case3(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_data_feed0()
        self.set_op_attr0()
        self.attrs['dim'] = 1
        self.run_test_base()

    def test_case4(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_data_feed0()
        self.attrs = {}
        self.attrs['dim'] = 1
        self.attrs['keep_dim'] = True
        self.run_test_base()

    def test_case5(self):
        if False:
            while True:
                i = 10
        self.set_data_feed1()
        self.attrs = {}
        self.attrs['dim'] = [1, 2]
        self.attrs['keep_dim'] = False
        self.run_test_base()

    def test_case6(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_data_feed1()
        self.attrs = {}
        self.attrs['dim'] = [0, 1]
        self.attrs['keep_dim'] = False
        self.run_test_base()

    def test_case7(self):
        if False:
            i = 10
            return i + 15
        self.set_data_feed1()
        self.attrs = {}
        self.attrs['dim'] = [0, 1]
        self.attrs['keep_dim'] = True
        self.run_test_base()

class TestMax(TestMean):

    def set_test_op(self):
        if False:
            return 10
        self.op = paddle.max

class TestMin(TestMean):

    def set_test_op(self):
        if False:
            print('Hello World!')
        self.op = paddle.min

class TestSum(TestMean):

    def set_test_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = paddle.sum

class TestLogsumexp(TestMean):

    def set_test_op(self):
        if False:
            return 10
        self.op = paddle.logsumexp

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        if 'dim' in self.attrs:
            self.attrs['axis'] = self.attrs['dim']
            del self.attrs['dim']
        if 'keep_dim' in self.attrs:
            self.attrs['keepdim'] = self.attrs['keep_dim']
            del self.attrs['keep_dim']
        out = self.op(x, **self.attrs)
        self.fetch_list = [out.name]

class TestAll(TestMean):

    @property
    def fp16_enabled(self):
        if False:
            i = 10
            return i + 15
        return False

    def set_data_feed0(self):
        if False:
            while True:
                i = 10
        data = np.random.choice(a=[False, True], size=(2, 4))
        self.feed_fp32 = {'in_0': data.astype(bool)}
        self.set_feed_attr()

    def set_data_feed1(self):
        if False:
            i = 10
            return i + 15
        data = np.random.choice(a=[False, True], size=(2, 2, 2))
        self.feed_fp32 = {'in_0': data.astype(bool)}
        self.set_feed_attr()

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='bool')
        out = self.op(x, **self.attrs)
        self.fetch_list = [out.name]

    def set_test_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = paddle.all

class TestAny(TestAll):

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = paddle.any
if __name__ == '__main__':
    unittest.main()