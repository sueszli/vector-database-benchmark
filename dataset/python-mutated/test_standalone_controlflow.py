import unittest
import numpy as np
import paddle
from paddle.base import core
from paddle.base.framework import Program, program_guard
paddle.enable_static()

class TestCompatibility(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda() else paddle.CPUPlace()
        self.iter_run = 4

    def _get_feed(self):
        if False:
            for i in range(10):
                print('nop')
        'return the feeds'
        return None

    def build_program(self):
        if False:
            return 10

        def true_func():
            if False:
                while True:
                    i = 10
            return (paddle.tensor.fill_constant(shape=[1, 2], dtype='int32', value=1), paddle.tensor.fill_constant(shape=[2, 3], dtype='bool', value=True))

        def false_func():
            if False:
                return 10
            return (paddle.tensor.fill_constant(shape=[3, 4], dtype='float32', value=3), paddle.tensor.fill_constant(shape=[4, 5], dtype='int64', value=2))
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program, startup_program):
            x = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=0.1)
            y = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=0.23)
            pred = paddle.less_than(x, y)
            out = paddle.static.nn.cond(pred, true_func, false_func)
            return (main_program, startup_program, out)

    def _run(self, feed):
        if False:
            return 10
        paddle.seed(2020)
        (main_program, startup_program, fetch_vars) = self.build_program()
        exe = paddle.static.Executor(self.place)
        exe.run(startup_program)
        ret = []
        for i in range(self.iter_run):
            ret.append(exe.run(main_program, feed=feed, fetch_list=fetch_vars))
        return ret

    def run_dygraph_once(self, feed):
        if False:
            for i in range(10):
                print('nop')
        x = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=0.1)
        y = paddle.tensor.fill_constant(shape=[1], dtype='float32', value=0.23)
        if x < y:
            out = [paddle.tensor.fill_constant(shape=[1, 2], dtype='int32', value=1).numpy(), paddle.tensor.fill_constant(shape=[2, 3], dtype='bool', value=True).numpy()]
        else:
            out = [paddle.tensor.fill_constant(shape=[3, 4], dtype='float32', value=3).numpy(), paddle.tensor.fill_constant(shape=[4, 5], dtype='int64', value=2).numpy()]
        return out

    def run_dygraph(self, feed):
        if False:
            while True:
                i = 10
        ret = []
        for _ in range(self.iter_run):
            ret.append(self.run_dygraph_once(feed))
        return ret

    def run_new_executor(self, feed):
        if False:
            print('Hello World!')
        out = self._run(feed)
        return out

    def test_with_feed(self):
        if False:
            for i in range(10):
                print('nop')
        feed = self._get_feed()
        paddle.enable_static()
        res = self.run_new_executor(feed)
        paddle.disable_static()
        gt = self.run_dygraph(feed)
        for (x, y) in zip(gt, res):
            if isinstance(x, list):
                for (tx, ty) in zip(x, y):
                    np.testing.assert_array_equal(tx, ty)
            elif isinstance(x, np.ndarray):
                np.testing.assert_array_equal(x, y)
            else:
                raise Exception('Not Implement!')

class TestWhile(TestCompatibility):

    def _get_feed(self):
        if False:
            while True:
                i = 10
        'return the feeds'
        return None

    def build_program(self):
        if False:
            i = 10
            return i + 15

        def cond(i, ten):
            if False:
                while True:
                    i = 10
            return i < ten

        def body(i, ten):
            if False:
                return 10
            i = i + 1
            return [i, ten]
        main_program = paddle.static.default_main_program()
        startup_program = paddle.static.default_startup_program()
        with paddle.static.program_guard(main_program, startup_program):
            i = paddle.full(shape=[1], fill_value=0, dtype='int64')
            ten = paddle.full(shape=[1], fill_value=10, dtype='int64')
            (i, ten) = paddle.static.nn.while_loop(cond, body, [i, ten])
            exe = paddle.static.Executor(paddle.CPUPlace())
        return (main_program, startup_program, i)

    def run_dygraph_once(self, feed):
        if False:
            for i in range(10):
                print('nop')
        i = 1
        while i < 10:
            i = i + 1
        return [i]
if __name__ == '__main__':
    unittest.main()