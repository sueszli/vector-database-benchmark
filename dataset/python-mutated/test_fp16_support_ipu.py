import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.nn.functional as F
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

    def set_atol(self):
        if False:
            for i in range(10):
                print('nop')
        self.atol = 5e-06
        self.rtol = 1e-05
        self.atol_fp16 = 0.01
        self.rtol_fp16 = 0.001

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        np_data = np.random.uniform(low=-1, high=1, size=[1, 3, 100, 100])
        self.feed_fp32 = {'x': np_data.astype('float32')}
        self.feed_fp16 = {'x': np_data.astype('float16')}

    def set_feed_attr(self):
        if False:
            while True:
                i = 10
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        conv1 = paddle.static.nn.conv2d(x, num_filters=3, filter_size=3, bias_attr=False)
        conv2 = paddle.static.nn.conv2d(x, num_filters=3, filter_size=3, bias_attr=False)
        add1 = conv1 + conv2
        conv3 = paddle.static.nn.conv2d(add1, num_filters=8, filter_size=8, bias_attr=False)
        out = F.relu(conv3, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            print('Hello World!')
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            return 10
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

class TestIntInput(TestBase):

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        embedding = np.random.uniform(size=[10, 20])
        indice = np.array([1, 3, 5]).astype(np.int32)
        self.feed_fp32 = {'embedding': embedding.astype(np.float32), 'indice': indice}
        self.feed_fp16 = {'embedding': embedding.astype(np.float16), 'indice': indice}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        y = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='int32')
        out = paddle.gather(x, index=y)
        self.fetch_list = [out.name]
if __name__ == '__main__':
    unittest.main()