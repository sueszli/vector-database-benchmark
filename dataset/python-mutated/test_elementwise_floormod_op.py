import unittest
import paddle
from paddle import base

class TestFloorModOp(unittest.TestCase):

    def test_dygraph(self):
        if False:
            while True:
                i = 10
        with base.dygraph.guard(base.CPUPlace()):
            x = paddle.to_tensor([59], dtype='int32')
            y = paddle.to_tensor([0], dtype='int32')
            try:
                paddle.floor_mod(x, y)
            except Exception as e:
                print('Error: Mod by zero encounter in floor_mod\n')
if __name__ == '__main__':
    unittest.main()