import opcode
import re
import sys
import textwrap
import unittest
from test.support import os_helper, verbose
from test.support.script_helper import assert_python_ok
Py_DEBUG = hasattr(sys, 'gettotalrefcount')

@unittest.skipUnless(Py_DEBUG, 'lltrace requires Py_DEBUG')
class TestLLTrace(unittest.TestCase):

    def test_lltrace_does_not_crash_on_subscript_operator(self):
        if False:
            for i in range(10):
                print('nop')
        with open(os_helper.TESTFN, 'w', encoding='utf-8') as fd:
            self.addCleanup(os_helper.unlink, os_helper.TESTFN)
            fd.write(textwrap.dedent("            import code\n\n            console = code.InteractiveConsole()\n            console.push('__ltrace__ = 1')\n            console.push('a = [1, 2, 3]')\n            console.push('a[0] = 1')\n            print('unreachable if bug exists')\n            "))
            assert_python_ok(os_helper.TESTFN)

    def run_code(self, code):
        if False:
            print('Hello World!')
        code = textwrap.dedent(code).strip()
        with open(os_helper.TESTFN, 'w', encoding='utf-8') as fd:
            self.addCleanup(os_helper.unlink, os_helper.TESTFN)
            fd.write(code)
        (status, stdout, stderr) = assert_python_ok(os_helper.TESTFN)
        self.assertEqual(stderr, b'')
        self.assertEqual(status, 0)
        result = stdout.decode('utf-8')
        if verbose:
            print('\n\n--- code ---')
            print(code)
            print('\n--- stdout ---')
            print(result)
            print()
        return result

    def check_op(self, op, stdout, present):
        if False:
            while True:
                i = 10
        op = opcode.opmap[op]
        regex = re.compile(f': {op}($|, )', re.MULTILINE)
        if present:
            self.assertTrue(regex.search(stdout), f'": {op}" not found in: {stdout}')
        else:
            self.assertFalse(regex.search(stdout), f'": {op}" found in: {stdout}')

    def check_op_in(self, op, stdout):
        if False:
            print('Hello World!')
        self.check_op(op, stdout, True)

    def check_op_not_in(self, op, stdout):
        if False:
            i = 10
            return i + 15
        self.check_op(op, stdout, False)

    def test_lltrace(self):
        if False:
            while True:
                i = 10
        stdout = self.run_code('\n            def dont_trace_1():\n                a = "a"\n                a = 10 * a\n            def trace_me():\n                for i in range(3):\n                    +i\n            def dont_trace_2():\n                x = 42\n                y = -x\n            dont_trace_1()\n            __ltrace__ = 1\n            trace_me()\n            del __ltrace__\n            dont_trace_2()\n        ')
        self.check_op_in('GET_ITER', stdout)
        self.check_op_in('FOR_ITER', stdout)
        self.check_op_in('UNARY_POSITIVE', stdout)
        self.check_op_in('POP_TOP', stdout)
        self.check_op_not_in('BINARY_MULTIPLY', stdout)
        self.check_op_not_in('UNARY_NEGATIVE', stdout)
if __name__ == '__main__':
    unittest.main()