import unittest
from test_standalone_executor import build_program
import paddle
from paddle.base import core
paddle.enable_static()

class TestCustomStream(unittest.TestCase):
    """
    fill_constant(cpu)     gaussian_random
      |     |      |              |
      |     | matmul_v2(s1) fill_constant
      |     |      |              |    |
      |     |     elementwise_add(s1)  |
      |     |           |              |
      |  elementwise_sub(cpu)          |
      |     |           |              |
      |  tanh(cpu)     elementwise_add(s2)
      |     |                  |
    elementwise_sub(s1)      tanh(s2)
                 |             |
                elementwise_add(s2)
                        |
                  reduce_mean(s2)
    """

    def setUp(self):
        if False:
            return 10
        self.steps = 3

    def set_custom_stream(self, prog):
        if False:
            i = 10
            return i + 15
        op_index_for_stream1 = [2, 4, 9]
        op_index_for_stream2 = [7, 8, 10, 11]
        ops = prog.global_block().ops
        for op_index in op_index_for_stream1:
            ops[op_index].dist_attr.execution_stream = 's1'
            ops[op_index].dist_attr.stream_priority = 0
        for op_index in op_index_for_stream2:
            ops[op_index].dist_attr.execution_stream = 's2'
            ops[op_index].dist_attr.stream_priority = -1

    def run_program(self, apply_custom_stream=False):
        if False:
            for i in range(10):
                print('nop')
        paddle.seed(2022)
        (main_program, startup_program, fetch_list) = build_program()
        self.assertEqual(len(startup_program.global_block().ops), 0)
        if apply_custom_stream:
            self.set_custom_stream(main_program)
        with paddle.static.program_guard(main_program, startup_program):
            exe = paddle.static.Executor(paddle.CUDAPlace(0))
            scope = core.Scope()
            outs = []
            for i in range(self.steps):
                outs.append(exe.run(main_program, scope=scope, fetch_list=fetch_list))
        return outs

    def test_result(self):
        if False:
            i = 10
            return i + 15
        if not core.is_compiled_with_cuda():
            return
        baselines = self.run_program()
        outs = self.run_program(apply_custom_stream=True)
        for (bl, out) in zip(baselines, outs):
            self.assertEqual(bl[0], out[0])
if __name__ == '__main__':
    unittest.main()