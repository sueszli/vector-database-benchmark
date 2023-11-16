import unittest
import numpy as np
from op_test_ipu import IPUOpTest, np_dtype_to_base_str
import paddle
import paddle.optimizer
import paddle.static
from paddle import base
from paddle.base import compiler
paddle.enable_static()

class TestBase(IPUOpTest):

    def setUp(self):
        if False:
            return 10
        self.set_atol()
        self.set_training()
        self.set_feed()
        self.set_feed_attr()
        self.set_op()

    def set_op(self):
        if False:
            return 10
        self.op = paddle.incubate.identity_loss

    def set_feed(self):
        if False:
            i = 10
            return i + 15
        self.feed = {'x': np.random.uniform(low=-2, high=2, size=[3, 5]).astype('float32')}

    def set_feed_attr(self):
        if False:
            i = 10
            return i + 15
        self.feed_shape = [x.shape for x in self.feed.values()]
        self.feed_list = list(self.feed.keys())
        self.feed_dtype = [np_dtype_to_base_str(x.dtype) for x in self.feed.values()]

    def _test_base(self, reduction):
        if False:
            print('Hello World!')
        scope = base.core.Scope()
        main_prog = paddle.static.Program()
        startup_prog = paddle.static.Program()
        SEED = 0
        main_prog.random_seed = SEED
        startup_prog.random_seed = SEED
        with base.scope_guard(scope):
            with paddle.static.program_guard(main_prog, startup_prog):
                x = paddle.static.data(name=self.feed_list[0], shape=self.feed_shape[0], dtype=self.feed_dtype[0])
                out = self.op(x, reduction)
                fetch_list = [out.name]
            place = paddle.IPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(startup_prog)
            feed_list = self.feed_list
            ipu_strategy = paddle.static.IpuStrategy()
            ipu_strategy.set_graph_config(num_ipus=1, is_training=False)
            ipu_compiler = compiler.IpuCompiledProgram(main_prog, ipu_strategy=ipu_strategy)
            program = ipu_compiler.compile(feed_list, fetch_list)
            ipu_res = exe.run(program, self.feed, fetch_list)
            if reduction == 0:
                cpu_res = self.feed['x'].sum()
            elif reduction == 1:
                cpu_res = self.feed['x'].mean()
            else:
                cpu_res = self.feed['x']
            np.testing.assert_allclose(ipu_res[0], cpu_res, rtol=1e-05, atol=self.atol)

    def test_base(self):
        if False:
            print('Hello World!')
        for reduction in [0, 1, 2]:
            self._test_base(reduction)
if __name__ == '__main__':
    unittest.main()