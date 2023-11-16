import unittest
from paddle.base import core

class TestGetAllRegisteredOpKernels(unittest.TestCase):

    def test_phi_kernels(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(core._get_all_register_op_kernels('phi')['sign'])
        with self.assertRaises(KeyError):
            core._get_all_register_op_kernels('phi')['reshape']

    def test_base_kernels(self):
        if False:
            return 10
        self.assertTrue(core._get_all_register_op_kernels('fluid')['reshape'])
        with self.assertRaises(KeyError):
            core._get_all_register_op_kernels('fluid')['sign']

    def test_all_kernels(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(core._get_all_register_op_kernels('all')['reshape'])
        self.assertTrue(core._get_all_register_op_kernels('all')['sign'])
        self.assertTrue(core._get_all_register_op_kernels()['reshape'])
        self.assertTrue(core._get_all_register_op_kernels()['sign'])

class TestGetAllOpNames(unittest.TestCase):

    def test_get_all_op_names(self):
        if False:
            while True:
                i = 10
        all_op_names = core.get_all_op_names()
        all_op_with_phi_kernels = core.get_all_op_names('phi')
        all_op_with_fluid_kernels = core.get_all_op_names('fluid')
        self.assertTrue(len(all_op_names) > len(set(all_op_with_phi_kernels) | set(all_op_with_fluid_kernels)))
        self.assertTrue('scale' in all_op_with_phi_kernels)
if __name__ == '__main__':
    unittest.main()