"""Tests for pytype.pytd.parse.builtins."""
from pytype import file_utils
from pytype.imports import builtin_stubs
from pytype.platform_utils import path_utils
from pytype.pyi import parser
from pytype.pytd import pytd
from pytype.pytd import pytd_utils
from pytype.pytd import visitors
from pytype.tests import test_base
import unittest

class TestDataFiles(test_base.UnitTest):
    """Test GetPredefinedFile()."""
    BUILTINS = 'builtins'

    def test_get_predefined_file_basic(self):
        if False:
            for i in range(10):
                print('nop')
        (path, src) = builtin_stubs.GetPredefinedFile(self.BUILTINS, 'builtins')
        self.assertEqual(path, file_utils.replace_separator('stubs/builtins/builtins.pytd'))
        self.assertIsInstance(src, str)

    def test_get_predefined_file_throws(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(IOError, 'File not found|Resource not found|No such file or directory'):
            builtin_stubs.GetPredefinedFile(self.BUILTINS, '-does-not-exist')

    def test_pytd_builtin3(self):
        if False:
            for i in range(10):
                print('nop')
        "Verify 'import sys' for python3."
        subdir = 'builtins'
        (_, import_contents) = builtin_stubs.GetPredefinedFile(subdir, 'builtins')
        with open(path_utils.join(path_utils.dirname(file_utils.__file__), 'stubs', subdir, 'builtins.pytd')) as fi:
            file_contents = fi.read()
        self.assertMultiLineEqual(import_contents, file_contents)

    def test_pytd_builtin_is_package(self):
        if False:
            print('Hello World!')
        subdir = 'builtins'
        (path, _) = builtin_stubs.GetPredefinedFile(subdir, 'attr', as_package=True)
        self.assertEqual(path, file_utils.replace_separator('stubs/builtins/attr/__init__.pytd'))

class UtilsTest(test_base.UnitTest):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super().setUpClass()
        cls.builtins = pytd_utils.Concat(*builtin_stubs.GetBuiltinsAndTyping(parser.PyiOptions(python_version=cls.python_version)))

    def test_get_builtins_pytd(self):
        if False:
            return 10
        self.assertIsNotNone(self.builtins)
        self.builtins.Visit(visitors.VerifyLookup())

    def test_has_mutable_parameters(self):
        if False:
            print('Hello World!')
        append = self.builtins.Lookup('builtins.list').Lookup('append')
        self.assertIsNotNone(append.signatures[0].params[0].mutated_type)

    def test_has_correct_self(self):
        if False:
            print('Hello World!')
        update = self.builtins.Lookup('builtins.dict').Lookup('update')
        t = update.signatures[0].params[0].type
        self.assertIsInstance(t, pytd.GenericType)
        self.assertEqual(t.base_type, pytd.ClassType('builtins.dict'))

    def test_has_object_superclass(self):
        if False:
            for i in range(10):
                print('nop')
        cls = self.builtins.Lookup('builtins.slice')
        self.assertEqual(cls.bases, (pytd.ClassType('builtins.object'),))
        cls = self.builtins.Lookup('builtins.object')
        self.assertEqual(cls.bases, ())
if __name__ == '__main__':
    unittest.main()