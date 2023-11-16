import os
import sys
import unittest
import numpy as np
import paddle
import paddle.static
from paddle.utils.cpp_extension import load
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from op_test_ipu import IPUOpTest

def load_custom_ops():
    if False:
        while True:
            i = 10
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    custom_ops = load(name='checkpointoutput', sources=[f'{cur_dir}/custom_checkpointoutput.cc'], extra_cxx_cflags=['-DONNX_NAMESPACE=onnx'])
    return custom_ops

class TestCheckpointoutput(IPUOpTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.load_custom_ops()
        self.set_atol()
        self.set_test_op()
        self.set_training()
        self.set_data_feed()
        self.set_feed_attr()

    @property
    def fp16_enabled(self):
        if False:
            print('Hello World!')
        return False

    def load_custom_ops(self):
        if False:
            return 10
        self.custom_ops = load_custom_ops()

    def set_test_op(self):
        if False:
            for i in range(10):
                print('nop')
        self.op = self.custom_ops.checkpointoutput
        self.op_attrs = {}

    def set_data_feed(self):
        if False:
            return 10
        data = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}

    def set_feed_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        x = paddle.add(x, x)
        x = self.op(x, **self.op_attrs)
        x = paddle.mean(x)
        self.fetch_list = [x.name]

    def run_model(self, exec_mode):
        if False:
            for i in range(10):
                print('nop')
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            i = 10
            return i + 15
        self.build_model()
        self.run_model(IPUOpTest.ExecutionMode.IPU_FP32)
        print(self.output_dict)
if __name__ == '__main__':
    unittest.main()