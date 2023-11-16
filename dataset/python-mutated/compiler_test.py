"""Tests for compiler.py."""
import os
from pytype.pyc import compiler
import unittest

class PythonExeTest(unittest.TestCase):
    """Test python exe utilities."""

    def test_parse_interpreter_version(self):
        if False:
            for i in range(10):
                print('nop')
        test_cases = (('Python 3.8.3', (3, 8)), ('Python 3.8.4 :: Something custom (64-bit)', (3, 8)), ('[OS-Y 64-bit] Python 3.9.1', (3, 9)))
        for (version_str, expected) in test_cases:
            self.assertEqual(expected, compiler._parse_exe_version_string(version_str))

    def test_get_python_exe_version(self):
        if False:
            while True:
                i = 10
        version = compiler._get_python_exe_version(['python'])
        self.assertIsInstance(version, tuple)
        self.assertEqual(len(version), 2)

    def test_custom_python_exe(self):
        if False:
            i = 10
            return i + 15
        temp = compiler._CUSTOM_PYTHON_EXES
        compiler._CUSTOM_PYTHON_EXES = {(3, 10): 'utils.py'}
        ((exe,),) = compiler._get_python_exes((3, 10))
        self.assertEqual(os.path.basename(exe), 'utils.py')
        compiler._CUSTOM_PYTHON_EXES = temp
if __name__ == '__main__':
    unittest.main()