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
        self.set_op_attrs()

    def set_atol(self):
        if False:
            while True:
                i = 10
        self.atol = 1e-06
        self.rtol = 1e-06
        self.atol_fp16 = 0.01
        self.rtol_fp16 = 0.01

    def set_data_feed(self):
        if False:
            return 10
        data = np.random.uniform(size=[1, 255, 13, 13])
        self.feed_fp32 = {'in_0': data.astype(np.float32)}
        self.feed_fp16 = {'in_0': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            i = 10
            return i + 15
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_op_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'class_num': 80, 'anchors': [10, 13, 16, 30, 33, 23], 'conf_thresh': 0.01, 'downsample_ratio': 32}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            while True:
                i = 10
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        attrs = {'name': 'img_size', 'shape': [1, 2], 'dtype': 'int32', 'value': 6}
        img_size = paddle.tensor.fill_constant(**attrs)
        out = paddle.vision.ops.yolo_box(x=x, img_size=img_size, **self.attrs)
        self.fetch_list = [x.name for x in out]

    def run_model(self, exec_mode):
        if False:
            while True:
                i = 10
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()
if __name__ == '__main__':
    unittest.main()