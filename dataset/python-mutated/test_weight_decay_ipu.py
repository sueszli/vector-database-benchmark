import os
import tempfile
import unittest
import numpy as np
from op_test_ipu import IPUOpTest
import paddle
import paddle.static

@unittest.skipIf(IPUOpTest.use_ipumodel(), 'skip for ipumodel')
class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.set_atol()
        self.set_data_feed()
        self.set_feed_attr()
        self.set_attrs()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, 'weight_decay')

    def tearDown(self):
        if False:
            return 10
        super().tearDown()
        self.temp_dir.cleanup()

    def set_atol(self):
        if False:
            i = 10
            return i + 15
        self.atol = 1e-06

    def set_data_feed(self):
        if False:
            print('Hello World!')
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
            print('Hello World!')
        self.attrs = {'weight_decay': 4.0, 'loss_scaling': 1.0}

    def _test_optimizer(self, run_ipu=True):
        if False:
            for i in range(10):
                print('nop')

        def exclude_fn(param):
            if False:
                i = 10
                return i + 15
            return param.name.endswith('.w_0')
        scope = paddle.static.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        main_prog.random_seed = self.SEED
        startup_prog.random_seed = self.SEED
        np.random.seed(self.SEED)
        with paddle.static.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                image = paddle.static.data(name='image', shape=[1, 3, 10, 10], dtype='float32')
                bias = paddle.create_parameter(shape=[1, 3, 10, 10], is_bias=True, dtype='float32')
                add1 = image + bias
                conv1 = paddle.static.nn.conv2d(add1, num_filters=3, filter_size=3, bias_attr=False)
                loss = paddle.mean(conv1)
                opt = paddle.optimizer.Lamb(learning_rate=0.001, lamb_weight_decay=self.attrs['weight_decay'], exclude_from_weight_decay_fn=exclude_fn)
                opt.minimize(loss)
            if run_ipu:
                place = paddle.IPUPlace()
            else:
                place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)
            paddle.static.save(main_prog, self.model_path)
            if run_ipu:
                feed_list = [image.name]
                fetch_list = [loss.name]
                ipu_strategy = paddle.static.IpuStrategy()
                ipu_strategy.set_graph_config(is_training=True)
                ipu_strategy.set_options({'loss_scaling': self.attrs['loss_scaling']})
                program = paddle.static.IpuCompiledProgram(main_prog, ipu_strategy=ipu_strategy).compile(feed_list, fetch_list)
            else:
                program = main_prog
            result = []
            for epoch in range(100):
                loss_res = exe.run(program, feed=self.feed, fetch_list=[loss])
                result.append(loss_res)
            return np.array(result)

    def test(self):
        if False:
            print('Hello World!')
        ipu_loss = self._test_optimizer(True).flatten()
        cpu_loss = self._test_optimizer(False).flatten()
        np.testing.assert_allclose(ipu_loss, cpu_loss, rtol=1e-05, atol=self.atol)
if __name__ == '__main__':
    unittest.main()