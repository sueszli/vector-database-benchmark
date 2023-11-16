import unittest
from paddle.base import core

class TestOpSupportGPU(unittest.TestCase):

    def test_case(self):
        if False:
            return 10
        self.assertEqual(core.is_compiled_with_cuda(), core.op_support_gpu('sum'))
if __name__ == '__main__':
    unittest.main()