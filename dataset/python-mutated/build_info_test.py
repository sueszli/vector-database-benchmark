"""Test for the generated build_info script."""
import platform
from tensorflow.python.platform import build_info
from tensorflow.python.platform import test

class BuildInfoTest(test.TestCase):

    def testBuildInfo(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(build_info.build_info['is_rocm_build'], test.is_built_with_rocm())
        self.assertEqual(build_info.build_info['is_cuda_build'], test.is_built_with_cuda())
        if platform.system() != 'Windows':
            from tensorflow.compiler.tf2tensorrt._pywrap_py_utils import is_tensorrt_enabled
            self.assertEqual(build_info.build_info['is_tensorrt_build'], is_tensorrt_enabled())

    def testDeterministicOrder(self):
        if False:
            while True:
                i = 10
        self.assertContainsSubsequence(build_info.build_info.keys(), ('is_cuda_build', 'is_rocm_build', 'is_tensorrt_build'))
if __name__ == '__main__':
    test.main()