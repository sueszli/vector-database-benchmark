import unittest
import paddle
from paddle import static
paddle.enable_static()

class TestOpPriority(unittest.TestCase):

    def test_op_priority(self):
        if False:
            print('Hello World!')
        program = static.Program()
        with static.program_guard(program):
            x = paddle.zeros(shape=[1], dtype='int32')
            block = program.global_block()
            y = block.create_var(dtype='int32')
            block.append_op(type='share_data', inputs={'X': x.name}, outputs={'Out': y.name})
            paddle.increment(x)
            block.ops[-1].dist_attr.scheduling_priority = 1
            paddle.increment(y)
            block.ops[-1].dist_attr.scheduling_priority = -1
            paddle.framework.set_flags({'FLAGS_new_executor_serial_run': 1})
            exe = static.Executor()
            result = exe.run(program, fetch_list=[y])
            result = exe.run(program, fetch_list=[y])
            self.assertEqual(result[0], 1)
if __name__ == '__main__':
    unittest.main()