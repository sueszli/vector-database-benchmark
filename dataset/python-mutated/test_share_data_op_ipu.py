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

    def set_data_feed(self):
        if False:
            while True:
                i = 10
        data = np.random.uniform(size=[2, 3, 1])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            return 10
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        y = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        self.main_prog.global_block().append_op(type='share_data', inputs={'X': x}, outputs={'Out': y})
        out = paddle.add(y, y)
        self.fetch_list = [out.name]

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

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        data = np.random.uniform(size=[2, 3, 1])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}
        data = np.random.uniform(size=[2, 3, 1])
        self.assign_fp32 = data.astype(np.float32)

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        y = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        self.main_prog.global_block().append_op(type='share_data', inputs={'X': x}, outputs={'Out': y})
        out = paddle.add(x, y)
        self.fetch_list = [out.name]
if __name__ == '__main__':
    unittest.main()