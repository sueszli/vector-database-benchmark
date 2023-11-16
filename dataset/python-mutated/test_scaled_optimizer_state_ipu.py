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
        self.set_attrs()

    @property
    def fp16_enabled(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def set_training(self):
        if False:
            for i in range(10):
                print('nop')
        self.is_training = True
        self.epoch = 100

    def set_data_feed(self):
        if False:
            while True:
                i = 10
        data = np.random.uniform(size=[1, 3, 10, 10]).astype('float32')
        self.feed_fp32 = {'image': data.astype(np.float32)}
        self.feed_fp16 = {'image': data.astype(np.float16)}

    def set_feed_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_shape = [x.shape for x in self.feed_fp32.values()]
        self.feed_list = list(self.feed_fp32.keys())
        self.feed_dtype = [x.dtype for x in self.feed_fp32.values()]

    def set_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'optimizer': 'lamb', 'weight_decay': 0.0, 'scaled_optimizer_state': True}

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            return 10
        image = paddle.static.data(name='image', shape=[1, 3, 10, 10], dtype='float32')
        conv1 = paddle.static.nn.conv2d(image, num_filters=3, filter_size=3, bias_attr=False)
        loss = paddle.mean(conv1)
        weight_decay = self.attrs['weight_decay']
        opt = paddle.optimizer.Adam(learning_rate=0.1, weight_decay=weight_decay)
        if self.attrs['optimizer'] == 'lamb':
            opt = paddle.optimizer.Lamb(learning_rate=0.1, lamb_weight_decay=weight_decay)
        opt.minimize(loss)
        self.feed_list = [image.name]
        self.fetch_list = [loss.name]

    def run_model(self, exec_mode):
        if False:
            for i in range(10):
                print('nop')
        ipu_strategy = paddle.static.IpuStrategy()
        ipu_strategy.set_graph_config(is_training=self.is_training)
        if self.is_ipu_mode(exec_mode):
            if 'use_no_bias_optimizer' in self.attrs.keys():
                ipu_strategy.set_options({'use_no_bias_optimizer': self.attrs['use_no_bias_optimizer']})
            if 'scaled_optimizer_state' in self.attrs.keys():
                ipu_strategy.set_options({'scaled_optimizer_state': self.attrs['scaled_optimizer_state']})
        self.run_op_test(exec_mode, ipu_strategy=ipu_strategy)

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        for m in IPUOpTest.ExecutionMode:
            if not self.skip_mode(m):
                self.build_model()
                self.run_model(m)
        self.check()

class TestScaledAdam(TestBase):

    def set_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.attrs = {'optimizer': 'adam', 'weight_decay': 0.0, 'scaled_optimizer_state': True}

    def set_atol(self):
        if False:
            return 10
        super().set_atol()
        self.atol = 1e-05
        self.rtol = 1e-05

@unittest.skip('cpu do not support AdamNoBias')
class TestScaledAdamNoBias(TestBase):

    def set_attrs(self):
        if False:
            return 10
        self.attrs = {'optimizer': 'adam', 'weight_decay': 0.0, 'use_no_bias_optimizer': True, 'scaled_optimizer_state': True}

@unittest.skip('cpu do not support LambNoBias')
class TestScaledLambNoBias(TestBase):

    def set_attrs(self):
        if False:
            print('Hello World!')
        self.attrs = {'optimizer': 'lamb', 'weight_decay': 0.0, 'use_no_bias_optimizer': True, 'scaled_optimizer_state': True}
if __name__ == '__main__':
    unittest.main()