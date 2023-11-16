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

    def set_atol(self):
        if False:
            i = 10
            return i + 15
        self.atol = 0.0001

    def set_data_feed(self):
        if False:
            while True:
                i = 10
        self.feed = {'image': np.random.uniform(size=[1, 3, 10, 10]).astype('float32')}

    def set_feed_attr(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [x.dtype for x in self.feed.values()]

    def set_attrs(self):
        if False:
            while True:
                i = 10
        self.attrs = {'optimizer': 'lamb', 'weight_decay': 2.0}

    def _test_optimizer(self, run_ipu=True):
        if False:
            return 10
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED
        np.random.seed(self.SEED)
        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                image = paddle.static.data(name='image', shape=[1, 3, 10, 10], dtype='float32')
                conv1 = paddle.static.nn.conv2d(image, num_filters=3, filter_size=3, bias_attr=False)
                loss = paddle.mean(conv1)
                weight_decay = self.attrs['weight_decay']
                opt = paddle.optimizer.SGD(learning_rate=0.1, weight_decay=weight_decay)
                if self.attrs['optimizer'] == 'adam':
                    opt = paddle.optimizer.Adam(learning_rate=0.1, weight_decay=weight_decay)
                elif self.attrs['optimizer'] == 'lamb':
                    opt = paddle.optimizer.Lamb(learning_rate=0.1, lamb_weight_decay=weight_decay)
                opt.minimize(loss)
            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)
            if run_ipu:
                feed_list = [image.name]
                fetch_list = [loss.name]
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(is_training=True)
                ipu_strategy.set_options({'runtime_options.enable_eval': True})
                program = paddle.static.IpuCompiledProgram(main_prog, ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog
            result = []
            if run_ipu:
                for epoch in range(200):
                    if epoch == 100:
                        ipu_strategy.set_options({'runtime_options.enable_eval': False})
                    loss_res = exe.run(program, feed=self.feed, fetch_list=[loss])
                    result.append(loss_res)
            else:
                for epoch in range(100):
                    loss_res = exe.run(program, feed=self.feed, fetch_list=[loss])
                    result.append(loss_res)
            return np.array(result)

    def test(self):
        if False:
            print('Hello World!')
        ipu_loss = self._test_optimizer(True).flatten()
        cpu_loss = self._test_optimizer(False).flatten()
        self.assertTrue(ipu_loss[0] == ipu_loss[99])
        np.testing.assert_allclose(ipu_loss[100:], cpu_loss, rtol=1e-05, atol=self.atol)
if __name__ == '__main__':
    unittest.main()