from numba import njit, cfunc
from numba.tests.support import TestCase, unittest
from numba.core import cgutils
unicode_name1 = u'\ndef unicode_name1(ಠ_ರೃ, ಠਊಠ):\n    return (ಠ_ರೃ) + (ಠਊಠ)\n'
unicode_name2 = u'\ndef Ծ_Ծ(ಠ_ರೃ, ಠਊಠ):\n    return (ಠ_ರೃ) + (ಠਊಠ)\n'

class TestUnicodeNames(TestCase):

    def make_testcase(self, src, fname):
        if False:
            return 10
        glb = {}
        exec(src, glb)
        fn = glb[fname]
        return fn

    def test_unicode_name1(self):
        if False:
            for i in range(10):
                print('nop')
        fn = self.make_testcase(unicode_name1, 'unicode_name1')
        cfn = njit(fn)
        self.assertEqual(cfn(1, 2), 3)

    def test_unicode_name2(self):
        if False:
            print('Hello World!')
        fn = self.make_testcase(unicode_name2, 'Ծ_Ծ')
        cfn = njit(fn)
        self.assertEqual(cfn(1, 2), 3)

    def test_cfunc(self):
        if False:
            for i in range(10):
                print('nop')
        fn = self.make_testcase(unicode_name2, 'Ծ_Ծ')
        cfn = cfunc('int32(int32, int32)')(fn)
        self.assertEqual(cfn.ctypes(1, 2), 3)

class TestUnicodeUtils(TestCase):

    def test_normalize_ir_text(self):
        if False:
            i = 10
            return i + 15
        out = cgutils.normalize_ir_text('abc')
        self.assertIsInstance(out, str)
        out.encode('latin1')

    def test_normalize_ir_text_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        out = cgutils.normalize_ir_text(unicode_name2)
        self.assertIsInstance(out, str)
        out.encode('latin1')
if __name__ == '__main__':
    unittest.main()