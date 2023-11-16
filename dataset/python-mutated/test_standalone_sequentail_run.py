import os
import unittest
import numpy as np
import paddle

class TestStandaloneExecutor(unittest.TestCase):

    def build_program(self):
        if False:
            print('Hello World!')
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            a = paddle.static.data(name='data', shape=[2, 2], dtype='float32')
            b = paddle.ones([2, 2]) * 2
            t = paddle.static.nn.fc(a, 2)
            c = t + b
        return (main_program, startup_program, [c])

    def run_program(self, sequential_run=False):
        if False:
            return 10
        seed = 100
        paddle.seed(seed)
        np.random.seed(seed)
        (main, startup, outs) = self.build_program()
        build_strategy = paddle.static.BuildStrategy()
        build_strategy.sequential_run = sequential_run
        print(build_strategy)
        compiled_program = paddle.static.CompiledProgram(main, build_strategy=build_strategy)
        exe = paddle.static.Executor()
        scope = paddle.static.Scope()
        with paddle.static.scope_guard(scope):
            exe.run(startup)
            data = np.ones([2, 2], dtype='float32')
            ret = exe.run(compiled_program, feed={'data': data}, fetch_list=[v.name for v in outs])
            return ret

    def test_result(self):
        if False:
            return 10
        paddle.enable_static()
        ret1 = self.run_program(True)
        ret2 = self.run_program(False)
        np.testing.assert_array_equal(ret1, ret2)

    def test_str_flag(self):
        if False:
            i = 10
            return i + 15
        paddle.enable_static()
        os.environ['FLAGS_new_executor_sequential_run'] = 'true'
        ret1 = self.run_program(True)
        assert os.environ['FLAGS_new_executor_sequential_run'] == 'true'
if __name__ == '__main__':
    unittest.main()