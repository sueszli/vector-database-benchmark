import unittest
import paddle
from paddle import base

class TestAvoidTwiceInitialization(unittest.TestCase):

    def test_avoid_twice_initialization(self):
        if False:
            print('Hello World!')
        cur_program = base.Program()
        cur_block = cur_program.current_block()
        var = cur_block.create_parameter(initializer=paddle.nn.initializer.Constant(value=0.01), shape=[2, 2], dtype='float32', name='var_a')
        cur_block.append_op(type='c_broadcast', inputs={'X': [var]}, outputs={'Out': [var]}, attrs={'root': 0, 'ring_id': 0, 'use_calc_stream': False})
        cur_block.append_op(type='c_sync_comm_stream', inputs={'X': [var]}, outputs={'Out': [var]}, attrs={'ring_id': 0})
        var2 = cur_block.create_parameter(initializer=paddle.nn.initializer.Constant(value=0.01), shape=[2, 2], dtype='float32', name='var_a')
if __name__ == '__main__':
    unittest.main()