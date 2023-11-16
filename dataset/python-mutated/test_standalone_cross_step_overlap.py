import unittest
import numpy as np
import paddle
from paddle import static
paddle.enable_static()

class TestCrossStepOverlap(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.shape = [16, 513, 513, 19]
        self.x_value = 2
        self.y_value = 3
        self.overlap_op_num = 1500
        self.step_num = 3

    def test_cross_step_overlap(self):
        if False:
            while True:
                i = 10
        if not paddle.base.core.is_compiled_with_cuda():
            return
        program = static.Program()
        with static.program_guard(program):
            x = paddle.full(self.shape, fill_value=self.x_value, dtype='float64')
            y = paddle.full(self.shape, fill_value=self.y_value, dtype='float64')
            z = paddle.add(x, y)
            block = program.global_block()
            block.var(x.name).desc.set_persistable(True)
            block.var(y.name).desc.set_persistable(True)
            for i in range(self.overlap_op_num):
                block.append_op(type='reduce_min', inputs={'X': x.name}, outputs={'Out': y.name}, attrs={'axis': 0, 'keepdim': True})
                block.ops[-1].dist_attr.execution_stream = 'custom'
            exe = static.Executor()
            results = []
            for i in range(self.step_num):
                result = exe.run(program, fetch_list=[z])
                results.append(result)
            for result in results:
                self.assertAlmostEqual(np.sum(result), (self.x_value + self.y_value) * np.prod(self.shape))
if __name__ == '__main__':
    unittest.main()