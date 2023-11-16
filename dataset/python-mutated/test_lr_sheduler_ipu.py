import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static
from paddle.optimizer.lr import LRScheduler

class LR_New(LRScheduler):

    def __init__(self, learning_rate=1e-05, last_epoch=-1, verbose=False):
        if False:
            print('Hello World!')
        super().__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if False:
            i = 10
            return i + 15
        self.base_lr = self.base_lr + 0.0001
        self.last_epoch = self.last_epoch + 1
        return self.base_lr

class TestConvNet(IPUOpTest):

    @IPUOpTest.static_graph
    def build_model(self):
        if False:
            return 10
        image = paddle.static.data(name='image', shape=[1, 3, 10, 10], dtype='float32')
        conv1 = paddle.static.nn.conv2d(image, num_filters=3, filter_size=3, bias_attr=False)
        loss = paddle.mean(conv1)
        opt = paddle.optimizer.Lamb(learning_rate=LR_New())
        opt.minimize(loss)
        self.feed_list = [image.name]
        self.fetch_list = [loss.name]

    def run_model(self, run_ipu=True):
        if False:
            i = 10
            return i + 15
        self.build_model()
        if run_ipu:
            place = paddle.IPUPlace()
        else:
            place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(self.startup_prog)
        if run_ipu:
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(is_training=True)
            program = paddle.static.IpuCompiledProgram(self.main_prog, ipu_strategy=ipu_strategy).compile(self.feed_list, self.fetch_list)
        else:
            program = self.main_prog
        result = []
        for _ in range(100):
            if hasattr(program, 'lr_scheduler'):
                program.lr_scheduler.step()
            loss_res = exe.run(program, feed=self.feed, fetch_list=self.fetch_list)
            result.append(loss_res)
        return np.array(result)

    def test_training(self):
        if False:
            while True:
                i = 10
        data = np.random.rand(1, 3, 10, 10).astype(np.float32)
        self.feed = {'image': data}
        ipu_loss = self.run_model(True).flatten()
        cpu_loss = self.run_model(False).flatten()
        np.testing.assert_allclose(ipu_loss, cpu_loss, rtol=1e-05, atol=1e-10)
if __name__ == '__main__':
    unittest.main()