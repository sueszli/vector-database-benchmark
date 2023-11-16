import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

class TestTopKOp(IPUOpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_atol()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_test_op()
        self.set_op_attrs()

    def set_test_op(self):
        if False:
            i = 10
            return i + 15
        self.op = paddle.topk

    def set_data_feed(self):
        if False:
            for i in range(10):
                print('nop')
        data = np.random.uniform(size=[3, 5])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            i = 10
            return i + 15
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.use_k_as_const_variable = False
        self.attrs = {}
        if not self.use_k_as_const_variable:
            self.attrs['k'] = 3

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            print('Hello World!')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        if not self.use_k_as_const_variable:
            (topk_values, topk_indices) = self.op(x, **self.attrs)
        else:
            K_t = paddle.tensor.fill_constant(shape=[1], dtype='int32', value=self.k, name='in_2')
            (topk_values, topk_indices) = self.op(x, K_t, **self.attrs)
        self.fetch_list = [topk_values.name, topk_indices.name]

    def run_model(self, exec_mode):
        if False:
            return 10
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            i = 10
            return i + 15
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        value_dict = {}
        index_dict = {}
        for (k, v) in self.output_dict.items():
            value_dict[k] = v[0]
            index_dict[k] = v[1]
        self.check(output_dict=value_dict)
        self.check(output_dict=index_dict)

class TestCase2(TestTopKOp):

    def set_test_op(self):
        if False:
            while True:
                i = 10
        self.op = paddle.topk

@unittest.skip('Trying to get data as int64 but it is of type int32')
class TestCase3(TestTopKOp):

    def set_op_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.use_k_as_const_variable = True
        self.attrs = {}
        self.k = 2

@unittest.skip('Trying to get data as int64 but it is of type int32')
class TestCase4(TestCase3):

    def set_test_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = paddle.topk
if __name__ == '__main__':
    unittest.main()