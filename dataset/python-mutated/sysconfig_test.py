import re
from tensorflow.python.platform import googletest
from tensorflow.python.platform import sysconfig as sysconfig_lib
from tensorflow.python.platform import test

class SysconfigTest(googletest.TestCase):

    def test_get_build_info_works(self):
        if False:
            return 10
        build_info = sysconfig_lib.get_build_info()
        self.assertIsInstance(build_info, dict)

    def test_rocm_cuda_info_matches(self):
        if False:
            for i in range(10):
                print('nop')
        build_info = sysconfig_lib.get_build_info()
        self.assertEqual(build_info['is_rocm_build'], test.is_built_with_rocm())
        self.assertEqual(build_info['is_cuda_build'], test.is_built_with_cuda())

    def test_compile_flags(self):
        if False:
            while True:
                i = 10
        compile_flags = sysconfig_lib.get_compile_flags()

        def list_contains(items, regex_str):
            if False:
                print('Hello World!')
            regex = re.compile(regex_str)
            return any((regex.match(item) for item in items))
        self.assertTrue(list_contains(compile_flags, '.*/include'))
        self.assertTrue(list_contains(compile_flags, '.*_GLIBCXX_USE_CXX11_ABI.*'))
        self.assertTrue(list_contains(compile_flags, '.*EIGEN_MAX_ALIGN_BYTES.*'))
        self.assertTrue(list_contains(compile_flags, '.*std.*'))
if __name__ == '__main__':
    googletest.main()