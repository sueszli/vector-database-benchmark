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

    def set_atol(self):
        if False:
            for i in range(10):
                print('nop')
        self.atol = 1e-06
        self.rtol = 1e-05
        self.atol_fp16 = 0.01
        self.rtol_fp16 = 0.001

    def set_data_feed(self):
        if False:
            while True:
                i = 10
        x = np.random.uniform(size=[1, 3, 10, 10])
        self.feed_fp32 = {'x': x.astype(np.float32)}
        self.feed_fp16 = {'x': x.astype(np.float16)}

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
        self.attrs = {'scale': True, 'shift': True, 'begin_norm_axis': 1, 'epsilon': 1e-05}
        self.optimizer = None

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            i = 10
            return i + 15
        x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        if self.is_training:
            ch = self.feed_shape[0][1]
            conv1 = paddle.static.nn.conv2d(x, num_filters=ch, filter_size=3, bias_attr=False)
            scale = paddle.ParamAttr(trainable=True)
            bias = paddle.ParamAttr(trainable=True)
            out = paddle.static.nn.layer_norm(conv1, param_attr=scale, bias_attr=bias, **self.attrs)
            loss = paddle.mean(out)
            self.fetch_list = [loss.name]
        else:
            scale = self.attrs['scale']
            bias = self.attrs['shift']
            out = paddle.static.nn.layer_norm(x, param_attr=scale, bias_attr=bias, **self.attrs)
            self.fetch_list = [out.name]
        if self.is_training:
            optimizer = None
            if self.optimizer == 'sgd':
                optimizer = paddle.optimizer.SGD(learning_rate=0.01)
            elif self.optimizer == 'adam':
                optimizer = paddle.optimizer.Adam(learning_rate=0.01)
            elif self.optimizer == 'lamb':
                optimizer = paddle.optimizer.Lamb(learning_rate=0.01, lamb_weight_decay=0.0)
            if optimizer is not None:
                optimizer.minimize(loss)

    def run_model(self, exec_mode):
        if False:
            return 10
        self.run_op_test(exec_mode)

    def test(self):
        if False:
            return 10
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

@unittest.skip('raise error')
class TestCase1(TestBase):

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {'scale': False, 'shift': True, 'begin_norm_axis': 1, 'epsilon': 1e-05}

@unittest.skip('raise error')
class TestCase2(TestBase):

    def set_op_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {'scale': True, 'shift': False, 'begin_norm_axis': 1, 'epsilon': 1e-05}

class TestCase3(TestBase):

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'scale': True, 'shift': True, 'begin_norm_axis': 2, 'epsilon': 1e-05}
        self.optimizer = None

class TestTrainCase1(TestBase):

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'scale': True, 'shift': True, 'begin_norm_axis': 1, 'epsilon': 1e-05}
        self.optimizer = 'sgd'

    def set_atol(self):
        if False:
            for i in range(10):
                print('nop')
        super().set_atol()
        self.atol = 1e-06

    def set_training(self):
        if False:
            for i in range(10):
                print('nop')
        self.is_training = True
        self.epoch = 20

class TestTrainCase3(TestBase):

    def set_atol(self):
        if False:
            for i in range(10):
                print('nop')
        super().set_atol()
        self.atol = 0.005

    def set_op_attrs(self):
        if False:
            i = 10
            return i + 15
        self.attrs = {'scale': True, 'shift': True, 'begin_norm_axis': 2, 'epsilon': 1e-05}
        self.optimizer = 'lamb'

    def set_training(self):
        if False:
            i = 10
            return i + 15
        self.is_training = True
        self.epoch = 20
if __name__ == '__main__':
    unittest.main()