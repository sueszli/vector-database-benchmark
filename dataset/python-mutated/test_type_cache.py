""" Tests for the internal type cache in CPython. """
import unittest
from test import support
from test.support import import_helper
try:
    from sys import _clear_type_cache
except ImportError:
    _clear_type_cache = None
type_get_version = import_helper.import_module('_testcapi').type_get_version
type_assign_version = import_helper.import_module('_testcapi').type_assign_version

@support.cpython_only
@unittest.skipIf(_clear_type_cache is None, 'requires sys._clear_type_cache')
class TypeCacheTests(unittest.TestCase):

    def test_tp_version_tag_unique(self):
        if False:
            for i in range(10):
                print('nop')
        'tp_version_tag should be unique assuming no overflow, even after\n        clearing type cache.\n        '
        Y = type('Y', (), {})
        Y.x = 1
        Y.x
        y_ver = type_get_version(Y)
        if y_ver == 0 or y_ver > 4294963200:
            self.skipTest('Out of type version tags')
        all_version_tags = []
        append_result = all_version_tags.append
        assertNotEqual = self.assertNotEqual
        for _ in range(30):
            _clear_type_cache()
            X = type('Y', (), {})
            X.x = 1
            X.x
            tp_version_tag_after = type_get_version(X)
            assertNotEqual(tp_version_tag_after, 0, msg='Version overflowed')
            append_result(tp_version_tag_after)
        self.assertEqual(len(set(all_version_tags)), 30, msg=f'{all_version_tags} contains non-unique versions')

    def test_type_assign_version(self):
        if False:
            i = 10
            return i + 15

        class C:
            x = 5
        self.assertEqual(type_assign_version(C), 1)
        c_ver = type_get_version(C)
        C.x = 6
        self.assertEqual(type_get_version(C), 0)
        self.assertEqual(type_assign_version(C), 1)
        self.assertNotEqual(type_get_version(C), 0)
        self.assertNotEqual(type_get_version(C), c_ver)
if __name__ == '__main__':
    support.run_unittest(TypeCacheTests)