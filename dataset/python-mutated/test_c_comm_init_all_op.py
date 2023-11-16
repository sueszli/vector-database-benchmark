import unittest
import paddle
from paddle import base

class TestCCommInitAllOp(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.place = base.CUDAPlace(0)
        self.exe = base.Executor(self.place)

    def test_default_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        program = base.Program()
        block = program.global_block()
        block.append_op(type='c_comm_init_all', attrs={'ring_id': 0})
        self.exe.run(program)

    def test_init_with_same_ring_id(self):
        if False:
            while True:
                i = 10
        program = base.Program()
        block = program.global_block()
        block.append_op(type='c_comm_init_all', attrs={'ring_id': 0})
        with self.assertRaises(ValueError):
            self.exe.run(program)

    def test_specifying_devices(self):
        if False:
            for i in range(10):
                print('nop')
        program = base.Program()
        block = program.global_block()
        block.append_op(type='c_comm_init_all', attrs={'devices': [0], 'ring_id': 1})
        self.exe.run(program)
if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()