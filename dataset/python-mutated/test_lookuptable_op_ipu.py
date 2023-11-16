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
        data = np.array([[[1], [3]], [[2], [4]], [[4], [127]]])
        self.feed_fp32 = {'x': data.astype(np.int64)}
        self.feed_fp16 = {'x': data.astype(np.int32)}

    def set_feed_attr(self):
        if False:
            return 10
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'size': [128, 16], 'is_sparse': False, 'is_distributed': False, 'padding_idx': -1, 'dtype': 'float32'}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            return 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='int64')
        out = paddle.static.nn.embedding(x, **self.attrs)
        if self.is_training:
            loss = paddle.mean(out)
            adam = paddle.optimizer.Adam(learning_rate=0.01)
            adam.minimize(loss)
            self.fetch_list = [loss.name]
        else:
            self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            for i in range(10):
                print('nop')
        if self.is_ipu_mode(exec_mode):
            self.feed_fp32['x'] = self.feed_fp32['x'].astype(np.int32)
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

class TestTrainCase1(TestBase):

    def set_atol(self):
        if False:
            while True:
                i = 10
        self.atol = 1e-07
        self.rtol = 1e-06
        self.atol_fp16 = 0.001
        self.rtol_fp16 = 0.001

    def set_training(self):
        if False:
            i = 10
            return i + 15
        self.is_training = True
        self.epoch = 10
if __name__ == '__main__':
    unittest.main()