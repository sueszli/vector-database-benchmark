"""Tests for querying registered kernels."""
from tensorflow.python.framework import kernels
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

class GetAllRegisteredKernelsTest(test_util.TensorFlowTestCase):

    def testFindsAtLeastOneKernel(self):
        if False:
            return 10
        kernel_list = kernels.get_all_registered_kernels()
        self.assertGreater(len(kernel_list.kernel), 0)

class GetRegisteredKernelsForOp(test_util.TensorFlowTestCase):

    def testFindsAtLeastOneKernel(self):
        if False:
            for i in range(10):
                print('nop')
        kernel_list = kernels.get_registered_kernels_for_op('KernelLabel')
        self.assertGreater(len(kernel_list.kernel), 0)
        self.assertEqual(kernel_list.kernel[0].op, 'KernelLabel')
if __name__ == '__main__':
    googletest.main()