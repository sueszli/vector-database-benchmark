import os
import unittest
import paddle

class SysConfigTest(unittest.TestCase):

    def test_include(self):
        if False:
            for i in range(10):
                print('nop')
        inc_dir = paddle.sysconfig.get_include()
        inc_dirs = inc_dir.split(os.sep)
        self.assertEqual(inc_dirs[-1], 'include')
        self.assertEqual(inc_dirs[-2], 'paddle')

    def test_libs(self):
        if False:
            print('Hello World!')
        lib_dir = paddle.sysconfig.get_lib()
        lib_dirs = lib_dir.split(os.sep)
        self.assertEqual(lib_dirs[-1], 'libs')
        self.assertEqual(lib_dirs[-2], 'paddle')
if __name__ == '__main__':
    unittest.main()