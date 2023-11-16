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
            print('Hello World!')
        data1 = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'x': data1.astype(np.float32)}
        self.feed_fp16 = {'x': data1.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            i = 10
            return i + 15
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {'num_or_sections': [1, 1, 1], 'axis': 1}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        out = paddle.split(x, **self.attrs)
        self.fetch_list = [fetch.name for fetch in out]

    def run_model(self, exec_mode):
        if False:
            for i in range(10):
                print('nop')
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            print('Hello World!')
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        for (k, v) in self.output_dict.items():
            self.output_dict[k] = np.concatenate([vv.flatten() for vv in v])
        self.check()

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'num_or_sections': [2, 8], 'axis': 2}
if __name__ == '__main__':
    unittest.main()