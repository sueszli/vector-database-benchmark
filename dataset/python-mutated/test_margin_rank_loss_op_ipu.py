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
            print('Hello World!')
        label = np.random.uniform(size=[3, 1])
        left = np.random.uniform(size=[3, 1])
        right = np.random.uniform(size=[3, 1])
        self.feed_fp32 = {'label': label.astype(np.float32), 'left': left.astype(np.float32), 'right': right.astype(np.float32)}
        self.feed_fp16 = {'label': label.astype(np.float16), 'left': left.astype(np.float16), 'right': right.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            print('Hello World!')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'margin': 0.1}

    @IPUOpTest.static_graph
    def build_model(self, on_ipu):
        if False:
            return 10
        label = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        left = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='float32')
        right = paddle.static.data(name=self.feed_list[2], shape=self.feed_shape[2], dtype='float32')
        out = paddle.nn.functional.margin_ranking_loss(left, right, label)
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
                self.build_model(self.is_ipu_mode(m))
                self.run_model(m)
        self.check()

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'margin': 0.5}
if __name__ == '__main__':
    unittest.main()