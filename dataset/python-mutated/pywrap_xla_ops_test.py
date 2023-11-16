from tensorflow.python.platform import googletest
from tensorflow.python.util import pywrap_xla_ops

class XlaOpsetUtilsTest(googletest.TestCase):

    def testGetGpuCompilableKernelNames(self):
        if False:
            print('Hello World!')
        'Tests retrieving compilable op names for GPU.'
        op_names = pywrap_xla_ops.get_gpu_kernel_names()
        self.assertGreater(op_names.__len__(), 0)
        self.assertEqual(op_names.count('Max'), 1)
        self.assertEqual(op_names.count('Min'), 1)
        self.assertEqual(op_names.count('MatMul'), 1)

    def testGetCpuCompilableKernelNames(self):
        if False:
            while True:
                i = 10
        'Tests retrieving compilable op names for CPU.'
        op_names = pywrap_xla_ops.get_cpu_kernel_names()
        self.assertGreater(op_names.__len__(), 0)
        self.assertEqual(op_names.count('Max'), 1)
        self.assertEqual(op_names.count('Min'), 1)
        self.assertEqual(op_names.count('MatMul'), 1)
if __name__ == '__main__':
    googletest.main()