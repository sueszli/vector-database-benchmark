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
        self.set_data_feed()
        self.set_feed_attr()
        self.set_attrs()
        self.set_training()

    @property
    def fp16_enabled(self):
        if False:
            i = 10
            return i + 15
        return False

    def set_atol(self):
        if False:
            print('Hello World!')
        super().set_atol()
        self.atol = 1e-06
        self.rtol = 1e-05

    def set_data_feed(self):
        if False:
            i = 10
            return i + 15
        self.feed_fp32 = {'image': np.random.uniform(size=[1, 3, 10, 10]).astype('float32')}

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
        self.attrs = {'optimizer': 'sgd', 'weight_decay': 0.0}

    def set_training(self):
        if False:
            for i in range(10):
                print('nop')
        self.is_training = True
        self.epoch = 100

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            i = 10
            return i + 15
        image = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype='float32')
        conv1 = paddle.static.nn.conv2d(image, num_filters=3, filter_size=3, bias_attr=False)
        loss = paddle.mean(conv1)
        self.fetch_list = [loss.name]
        weight_decay = self.attrs['weight_decay']
        clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0)
        if self.attrs['optimizer'] == 'sgd':
            opt = paddle.optimizer.SGD(learning_rate=0.1, weight_decay=weight_decay, grad_clip=clip)
        elif self.attrs['optimizer'] == 'adam':
            opt = paddle.optimizer.Adam(learning_rate=0.1, weight_decay=weight_decay, grad_clip=clip)
        elif self.attrs['optimizer'] == 'lamb':
            opt = paddle.optimizer.Lamb(learning_rate=0.1, lamb_weight_decay=weight_decay, grad_clip=clip)
        else:
            raise ValueError(f"Not supported optimizer {self.attrs['optimizer']} for test")
        opt.minimize(loss)

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

class TestAdam(TestBase):

    def set_attrs(self):
        if False:
            return 10
        self.attrs = {'optimizer': 'adam', 'weight_decay': 0.0}

class TestLamb(TestBase):

    def set_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {'optimizer': 'lamb', 'weight_decay': 0.1}
if __name__ == '__main__':
    unittest.main()