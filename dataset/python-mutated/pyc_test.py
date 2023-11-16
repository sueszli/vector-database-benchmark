"""Tests for pyc.py."""
from pytype.pyc import compiler
from pytype.pyc import opcodes
from pytype.pyc import pyc
from pytype.tests import test_base
import unittest

class TestCompileError(unittest.TestCase):

    def test_error_matches_re(self):
        if False:
            return 10
        e = pyc.CompileError('some error (foo.py, line 123)')
        self.assertEqual('foo.py', e.filename)
        self.assertEqual(123, e.lineno)
        self.assertEqual('some error', e.error)

    def test_error_does_not_match_re(self):
        if False:
            while True:
                i = 10
        e = pyc.CompileError('some error in foo.py at line 123')
        self.assertIsNone(e.filename)
        self.assertEqual(1, e.lineno)
        self.assertEqual('some error in foo.py at line 123', e.error)

class TestPyc(test_base.UnitTest):
    """Tests for pyc.py."""

    def _compile(self, src, mode='exec'):
        if False:
            for i in range(10):
                print('nop')
        exe = (['python' + '.'.join(map(str, self.python_version))], [])
        pyc_data = compiler.compile_src_string_to_pyc_string(src, filename='test_input.py', python_version=self.python_version, python_exe=exe, mode=mode)
        return pyc.parse_pyc_string(pyc_data)

    def test_compile(self):
        if False:
            print('Hello World!')
        code = self._compile('foobar = 3')
        self.assertIn('foobar', code.co_names)
        self.assertEqual(self.python_version, code.python_version)

    def test_compile_utf8(self):
        if False:
            return 10
        src = 'foobar = "abc□def"'
        code = self._compile(src)
        self.assertIn('foobar', code.co_names)
        self.assertEqual(self.python_version, code.python_version)

    def test_erroneous_file(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(pyc.CompileError) as ctx:
            self._compile('\nfoo ==== bar--')
        self.assertEqual('test_input.py', ctx.exception.filename)
        self.assertEqual(2, ctx.exception.lineno)
        self.assertEqual('invalid syntax', ctx.exception.error)

    def test_lineno(self):
        if False:
            while True:
                i = 10
        code = self._compile('a = 1\n\na = a + 1\n')
        self.assertIn('a', code.co_names)
        op_and_line = [(op.name, op.line) for op in opcodes.dis(code)]
        expected = [('LOAD_CONST', 1), ('STORE_NAME', 1), ('LOAD_NAME', 3), ('LOAD_CONST', 3), ('BINARY_ADD', 3), ('STORE_NAME', 3), ('LOAD_CONST', 3), ('RETURN_VALUE', 3)]
        if self.python_version >= (3, 11):
            expected = [('RESUME', 0)] + expected
            expected[5] = ('BINARY_OP', 3)
        self.assertEqual(expected, op_and_line)

    def test_mode(self):
        if False:
            while True:
                i = 10
        code = self._compile('foo', mode='eval')
        self.assertIn('foo', code.co_names)
        ops = [op.name for op in opcodes.dis(code)]
        expected = ['LOAD_NAME', 'RETURN_VALUE']
        if self.python_version >= (3, 11):
            expected = ['RESUME'] + expected
        self.assertEqual(expected, ops)

    def test_singlelineno(self):
        if False:
            for i in range(10):
                print('nop')
        code = self._compile('a = 1\n')
        self.assertIn('a', code.co_names)
        op_and_line = [(op.name, op.line) for op in opcodes.dis(code)]
        expected = [('LOAD_CONST', 1), ('STORE_NAME', 1), ('LOAD_CONST', 1), ('RETURN_VALUE', 1)]
        if self.python_version >= (3, 11):
            expected = [('RESUME', 0)] + expected
        self.assertEqual(expected, op_and_line)

    def test_singlelinenowithspace(self):
        if False:
            print('Hello World!')
        code = self._compile('\n\na = 1\n')
        self.assertIn('a', code.co_names)
        op_and_line = [(op.name, op.line) for op in opcodes.dis(code)]
        expected = [('LOAD_CONST', 3), ('STORE_NAME', 3), ('LOAD_CONST', 3), ('RETURN_VALUE', 3)]
        if self.python_version >= (3, 11):
            expected = [('RESUME', 0)] + expected
        self.assertEqual(expected, op_and_line)
if __name__ == '__main__':
    unittest.main()