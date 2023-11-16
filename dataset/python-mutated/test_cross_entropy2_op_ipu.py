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
        x = np.random.uniform(size=[3, 7])
        label = np.arange(3).reshape([3, 1])
        self.feed_fp32 = {'x': x.astype(np.float32), 'label': label.astype(np.int64)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'label': label.astype(np.int32)}

    def set_feed_attr(self):
        if False:
            print('Hello World!')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {'soft_label': False}

    @IPUOpTest.static_graph
    def build_model(self, on_ipu):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        if on_ipu:
            label = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='int32')
        else:
            label = paddle.static.data(name=self.feed_list[1], shape=self.feed_shape[1], dtype='int64')
        out = paddle.nn.functional.cross_entropy(input=x, label=label, reduction='none', use_softmax=False, **self.attrs)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            for i in range(10):
                print('nop')
        if self.is_ipu_mode(exec_mode):
            self.feed_fp32['label'] = self.feed_fp32['label'].astype(np.int32)
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            return 10
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model(self.is_ipu_mode(m))
                self.run_model(m)
        self.check()

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {'soft_label': False, 'ignore_index': 1}

class TestCase2(TestBase):

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(size=[30, 70])
        label = np.arange(30).reshape([30, 1])
        self.feed_fp32 = {'x': x.astype(np.float32), 'label': label.astype(np.int64)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'label': label.astype(np.int32)}

@unittest.skip('soft_label=True is not supported')
class TestCase3(TestBase):

    def set_op_attrs(self):
        if False:
            return 10
        self.attrs = {'soft_label': True}

class TestCase4(TestBase):

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        x = np.random.uniform(size=[3, 5, 7])
        label = np.random.randint(0, 7, [3, 5, 1], dtype='int64')
        self.feed_fp32 = {'x': x.astype(np.float32), 'label': label.astype(np.int64)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'label': label.astype(np.int32)}

class TestCase5(TestBase):

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.random.uniform(size=[3, 5, 6, 7])
        label = np.random.randint(0, 7, [3, 5, 6], dtype='int64')
        self.feed_fp32 = {'x': x.astype(np.float32), 'label': label.astype(np.int64)}
        self.feed_fp16 = {'x': x.astype(np.float16), 'label': label.astype(np.int32)}
if __name__ == '__main__':
    unittest.main()