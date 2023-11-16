import unittest
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
            print('Hello World!')
        self.feed_fp32 = {}
        self.feed_fp16 = {}

    def set_feed_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'name': 'x', 'shape': [1, 3, 3, 3], 'dtype': 'float32', 'value': 0.3}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.tensor.fill_constant(**self.attrs)
        out = paddle.add(x, x)
        self.fetch_list = [out.name]

    def run_model(self, exec_mode):
        if False:
            while True:
                i = 10
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            return 10
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'name': 'x', 'shape': [1, 3, 3, 3], 'dtype': 'int32', 'value': 3.0}
if __name__ == '__main__':
    unittest.main()