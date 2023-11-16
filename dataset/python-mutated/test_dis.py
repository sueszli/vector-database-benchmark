from test.support import captured_stdout
from test.support.bytecode_helper import BytecodeTestCase
from textwrap import dedent
from opcode import shadowop
import unittest
import sys
import dis
import io
import os
import re
import types
import contextlib

def get_tb():
    if False:
        print('Hello World!')

    def _error():
        if False:
            return 10
        try:
            1 / 0
        except Exception as e:
            tb = e.__traceback__
        return tb
    tb = _error()
    while tb.tb_next:
        tb = tb.tb_next
    return tb
TRACEBACK_CODE = get_tb().tb_frame.f_code

class _C:

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.x = x == 1

    @staticmethod
    def sm(x):
        if False:
            return 10
        x = x == 1

    @classmethod
    def cm(cls, x):
        if False:
            return 10
        cls.x = x == 1
dis_c_instance_method = '%3d           0 LOAD_FAST                1 (x)\n              2 LOAD_CONST               1 (1)\n              4 COMPARE_OP               2 (==)\n              6 LOAD_FAST                0 (self)\n              8 STORE_ATTR               0 (x)\n             10 LOAD_CONST               0 (None)\n             12 RETURN_VALUE\n' % (_C.__init__.__code__.co_firstlineno + 1,)
dis_c_instance_method_bytes = '          0 LOAD_FAST                1 (1)\n          2 LOAD_CONST               1 (1)\n          4 COMPARE_OP               2 (==)\n          6 LOAD_FAST                0 (0)\n          8 STORE_ATTR               0 (0)\n         10 LOAD_CONST               0 (0)\n         12 RETURN_VALUE\n'
dis_c_class_method = '%3d           0 LOAD_FAST                1 (x)\n              2 LOAD_CONST               1 (1)\n              4 COMPARE_OP               2 (==)\n              6 LOAD_FAST                0 (cls)\n              8 STORE_ATTR               0 (x)\n             10 LOAD_CONST               0 (None)\n             12 RETURN_VALUE\n' % (_C.cm.__code__.co_firstlineno + 2,)
dis_c_static_method = '%3d           0 LOAD_FAST                0 (x)\n              2 LOAD_CONST               1 (1)\n              4 COMPARE_OP               2 (==)\n              6 STORE_FAST               0 (x)\n              8 LOAD_CONST               0 (None)\n             10 RETURN_VALUE\n' % (_C.sm.__code__.co_firstlineno + 2,)
dis_c = 'Disassembly of %s:\n%s\nDisassembly of %s:\n%s\nDisassembly of %s:\n%s\n' % (_C.__init__.__name__, dis_c_instance_method, _C.cm.__name__, dis_c_class_method, _C.sm.__name__, dis_c_static_method)

def _f(a):
    if False:
        i = 10
        return i + 15
    print(a)
    return 1
dis_f = '%3d           0 LOAD_GLOBAL              0 (print)\n              2 LOAD_FAST                0 (a)\n              4 CALL_FUNCTION            1\n              6 POP_TOP\n\n%3d           8 LOAD_CONST               1 (1)\n             10 RETURN_VALUE\n' % (_f.__code__.co_firstlineno + 1, _f.__code__.co_firstlineno + 2)
dis_f_co_code = '          0 LOAD_GLOBAL              0 (0)\n          2 LOAD_FAST                0 (0)\n          4 CALL_FUNCTION            1\n          6 POP_TOP\n          8 LOAD_CONST               1 (1)\n         10 RETURN_VALUE\n'

def bug708901():
    if False:
        i = 10
        return i + 15
    for res in range(1, 10):
        pass
dis_bug708901 = '%3d           0 LOAD_GLOBAL              0 (range)\n              2 LOAD_CONST               1 (1)\n\n%3d           4 LOAD_CONST               2 (10)\n\n%3d           6 CALL_FUNCTION            2\n              8 GET_ITER\n        >>   10 FOR_ITER                 2 (to 16)\n             12 STORE_FAST               0 (res)\n\n%3d          14 JUMP_ABSOLUTE            5 (to 10)\n\n%3d     >>   16 LOAD_CONST               0 (None)\n             18 RETURN_VALUE\n' % (bug708901.__code__.co_firstlineno + 1, bug708901.__code__.co_firstlineno + 2, bug708901.__code__.co_firstlineno + 1, bug708901.__code__.co_firstlineno + 3, bug708901.__code__.co_firstlineno + 1)

def compile_and_get(code_str, funcname):
    if False:
        return 10
    _tmp_globals = {}
    code = compile(code_str, __file__, 'exec')
    exec(code, _tmp_globals)
    return _tmp_globals[funcname]
bug1333982_str = '\ndef bug1333982(x=[]):\n    assert 0, ([s for s in x] +\n              1)\n    pass\n'
bug1333982 = compile_and_get(bug1333982_str, 'bug1333982')
dis_bug1333982 = '%3d           0 LOAD_ASSERTION_ERROR\n              2 LOAD_CONST               2 (<code object <listcomp> at 0x..., file "%s", line %d>)\n              4 LOAD_CONST               3 (\'bug1333982.<locals>.<listcomp>\')\n              6 MAKE_FUNCTION            0\n              8 LOAD_FAST                0 (x)\n             10 GET_ITER\n             12 CALL_FUNCTION            1\n\n%3d          14 LOAD_CONST               4 (1)\n\n%3d          16 BINARY_ADD\n             18 CALL_FUNCTION            1\n             20 RAISE_VARARGS            1\n' % (bug1333982.__code__.co_firstlineno + 1, __file__, bug1333982.__code__.co_firstlineno + 1, bug1333982.__code__.co_firstlineno + 2, bug1333982.__code__.co_firstlineno + 1)
dis_bug1333982_with_inline_comprehensions = '%3d           0 LOAD_ASSERTION_ERROR\n              2 BUILD_LIST               0\n              4 LOAD_FAST                0 (x)\n              6 GET_ITER\n        >>    8 FOR_ITER                 4 (to 18)\n             10 STORE_FAST               1 (s)\n             12 LOAD_FAST                1 (s)\n             14 LIST_APPEND              2\n             16 JUMP_ABSOLUTE            4 (to 8)\n        >>   18 DELETE_FAST              1 (s)\n\n%3d          20 LOAD_CONST               2 (1)\n\n%3d          22 BINARY_ADD\n             24 CALL_FUNCTION            1\n             26 RAISE_VARARGS            1\n' % (bug1333982.__code__.co_firstlineno + 1, bug1333982.__code__.co_firstlineno + 2, bug1333982.__code__.co_firstlineno + 1)

def bug42562():
    if False:
        print('Hello World!')
    pass
bug42562.__code__ = bug42562.__code__.replace(co_linetable=b'\x04\x80')
dis_bug42562 = '          0 LOAD_CONST               0 (None)\n          2 RETURN_VALUE\n'
code_bug_45757 = bytes([144, 1, 9, 255, 144, 1, 100, 41, 83, 0])
dis_bug_45757 = '          0 EXTENDED_ARG             1\n          2 NOP\n          4 EXTENDED_ARG             1\n          6 LOAD_CONST             297 (297)\n          8 RETURN_VALUE\n'
_BIG_LINENO_FORMAT = '%3d           0 LOAD_GLOBAL              0 (spam)\n              2 POP_TOP\n              4 LOAD_CONST               0 (None)\n              6 RETURN_VALUE\n'
_BIG_LINENO_FORMAT2 = '%4d           0 LOAD_GLOBAL              0 (spam)\n               2 POP_TOP\n               4 LOAD_CONST               0 (None)\n               6 RETURN_VALUE\n'
dis_module_expected_results = 'Disassembly of f:\n  4           0 LOAD_CONST               0 (None)\n              2 RETURN_VALUE\n\nDisassembly of g:\n  5           0 LOAD_CONST               0 (None)\n              2 RETURN_VALUE\n\n'
expr_str = 'x + 1'
dis_expr_str = '  1           0 LOAD_NAME                0 (x)\n              2 LOAD_CONST               0 (1)\n              4 BINARY_ADD\n              6 RETURN_VALUE\n'
simple_stmt_str = 'x = x + 1'
dis_simple_stmt_str = '  1           0 LOAD_NAME                0 (x)\n              2 LOAD_CONST               0 (1)\n              4 BINARY_ADD\n              6 STORE_NAME               0 (x)\n              8 LOAD_CONST               1 (None)\n             10 RETURN_VALUE\n'
annot_stmt_str = '\nx: int = 1\ny: fun(1)\nlst[fun(0)]: int = 1\n'
dis_annot_stmt_str = "  2           0 SETUP_ANNOTATIONS\n              2 LOAD_CONST               0 (1)\n              4 STORE_NAME               0 (x)\n              6 LOAD_NAME                1 (int)\n              8 LOAD_NAME                2 (__annotations__)\n             10 LOAD_CONST               1 ('x')\n             12 STORE_SUBSCR\n\n  3          14 LOAD_NAME                3 (fun)\n             16 LOAD_CONST               0 (1)\n             18 CALL_FUNCTION            1\n             20 LOAD_NAME                2 (__annotations__)\n             22 LOAD_CONST               2 ('y')\n             24 STORE_SUBSCR\n\n  4          26 LOAD_CONST               0 (1)\n             28 LOAD_NAME                4 (lst)\n             30 LOAD_NAME                3 (fun)\n             32 LOAD_CONST               3 (0)\n             34 CALL_FUNCTION            1\n             36 STORE_SUBSCR\n             38 LOAD_NAME                1 (int)\n             40 POP_TOP\n             42 LOAD_CONST               4 (None)\n             44 RETURN_VALUE\n"
compound_stmt_str = 'x = 0\nwhile 1:\n    x += 1'
dis_compound_stmt_str = '  1           0 LOAD_CONST               0 (0)\n              2 STORE_NAME               0 (x)\n\n  2           4 NOP\n\n  3     >>    6 LOAD_NAME                0 (x)\n              8 LOAD_CONST               1 (1)\n             10 INPLACE_ADD\n             12 STORE_NAME               0 (x)\n\n  2          14 JUMP_ABSOLUTE            3 (to 6)\n'
dis_traceback = '%3d           0 SETUP_FINALLY            7 (to 16)\n\n%3d           2 LOAD_CONST               1 (1)\n              4 LOAD_CONST               2 (0)\n    -->       6 BINARY_TRUE_DIVIDE\n              8 POP_TOP\n             10 POP_BLOCK\n\n%3d          12 LOAD_FAST                1 (tb)\n             14 RETURN_VALUE\n\n%3d     >>   16 DUP_TOP\n             18 LOAD_GLOBAL              0 (Exception)\n             20 JUMP_IF_NOT_EXC_MATCH    29 (to 58)\n             22 POP_TOP\n             24 STORE_FAST               0 (e)\n             26 POP_TOP\n             28 SETUP_FINALLY           10 (to 50)\n\n%3d          30 LOAD_FAST                0 (e)\n             32 LOAD_ATTR                1 (__traceback__)\n             34 STORE_FAST               1 (tb)\n             36 POP_BLOCK\n             38 POP_EXCEPT\n             40 LOAD_CONST               0 (None)\n             42 STORE_FAST               0 (e)\n             44 DELETE_FAST              0 (e)\n\n%3d          46 LOAD_FAST                1 (tb)\n             48 RETURN_VALUE\n        >>   50 LOAD_CONST               0 (None)\n             52 STORE_FAST               0 (e)\n             54 DELETE_FAST              0 (e)\n             56 RERAISE                  1\n\n%3d     >>   58 RERAISE                  0\n' % (TRACEBACK_CODE.co_firstlineno + 1, TRACEBACK_CODE.co_firstlineno + 2, TRACEBACK_CODE.co_firstlineno + 5, TRACEBACK_CODE.co_firstlineno + 3, TRACEBACK_CODE.co_firstlineno + 4, TRACEBACK_CODE.co_firstlineno + 5, TRACEBACK_CODE.co_firstlineno + 3)

def _fstring(a, b, c, d):
    if False:
        return 10
    return f'{a} {b:4} {c!r} {d!r:4}'
dis_fstring = "%3d           0 LOAD_FAST                0 (a)\n              2 FORMAT_VALUE             0\n              4 LOAD_CONST               1 (' ')\n              6 LOAD_FAST                1 (b)\n              8 LOAD_CONST               2 ('4')\n             10 FORMAT_VALUE             4 (with format)\n             12 LOAD_CONST               1 (' ')\n             14 LOAD_FAST                2 (c)\n             16 FORMAT_VALUE             2 (repr)\n             18 LOAD_CONST               1 (' ')\n             20 LOAD_FAST                3 (d)\n             22 LOAD_CONST               2 ('4')\n             24 FORMAT_VALUE             6 (repr, with format)\n             26 BUILD_STRING             7\n             28 RETURN_VALUE\n" % (_fstring.__code__.co_firstlineno + 1,)

def _tryfinally(a, b):
    if False:
        for i in range(10):
            print('nop')
    try:
        return a
    finally:
        b()

def _tryfinallyconst(b):
    if False:
        i = 10
        return i + 15
    try:
        return 1
    finally:
        b()
dis_tryfinally = '%3d           0 SETUP_FINALLY            6 (to 14)\n\n%3d           2 LOAD_FAST                0 (a)\n              4 POP_BLOCK\n\n%3d           6 LOAD_FAST                1 (b)\n              8 CALL_FUNCTION            0\n             10 POP_TOP\n             12 RETURN_VALUE\n        >>   14 LOAD_FAST                1 (b)\n             16 CALL_FUNCTION            0\n             18 POP_TOP\n             20 RERAISE                  0\n' % (_tryfinally.__code__.co_firstlineno + 1, _tryfinally.__code__.co_firstlineno + 2, _tryfinally.__code__.co_firstlineno + 4)
dis_tryfinallyconst = '%3d           0 SETUP_FINALLY            6 (to 14)\n\n%3d           2 POP_BLOCK\n\n%3d           4 LOAD_FAST                0 (b)\n              6 CALL_FUNCTION            0\n              8 POP_TOP\n             10 LOAD_CONST               1 (1)\n             12 RETURN_VALUE\n        >>   14 LOAD_FAST                0 (b)\n             16 CALL_FUNCTION            0\n             18 POP_TOP\n             20 RERAISE                  0\n' % (_tryfinallyconst.__code__.co_firstlineno + 1, _tryfinallyconst.__code__.co_firstlineno + 2, _tryfinallyconst.__code__.co_firstlineno + 4)

def _g(x):
    if False:
        i = 10
        return i + 15
    yield x

async def _ag(x):
    yield x

async def _co(x):
    async for item in _ag(x):
        pass
_h_str = "\ndef _h(y):\n    def foo(x):\n        '''funcdoc'''\n        return [x + z for z in y]\n    return foo\n"
_h = compile_and_get(_h_str, '_h')
dis_nested_0 = '%3d           0 LOAD_CLOSURE             0 (y)\n              2 BUILD_TUPLE              1\n              4 LOAD_CONST               1 (<code object foo at 0x..., file "%s", line %d>)\n              6 LOAD_CONST               2 (\'_h.<locals>.foo\')\n              8 MAKE_FUNCTION            8 (closure)\n             10 STORE_FAST               1 (foo)\n\n%3d          12 LOAD_FAST                1 (foo)\n             14 RETURN_VALUE\n' % (_h.__code__.co_firstlineno + 1, __file__, _h.__code__.co_firstlineno + 1, _h.__code__.co_firstlineno + 4)
dis_nested_1 = '%s\nDisassembly of <code object foo at 0x..., file "%s", line %d>:\n%3d           0 LOAD_CLOSURE             0 (x)\n              2 BUILD_TUPLE              1\n              4 LOAD_CONST               1 (<code object <listcomp> at 0x..., file "%s", line %d>)\n              6 LOAD_CONST               2 (\'_h.<locals>.foo.<locals>.<listcomp>\')\n              8 MAKE_FUNCTION            8 (closure)\n             10 LOAD_DEREF               1 (y)\n             12 GET_ITER\n             14 CALL_FUNCTION            1\n             16 RETURN_VALUE\n' % (dis_nested_0, __file__, _h.__code__.co_firstlineno + 1, _h.__code__.co_firstlineno + 3, __file__, _h.__code__.co_firstlineno + 3)
dis_nested_1_with_inline_comprehensions = '%s\nDisassembly of <code object foo at 0x..., file "%s", line %d>:\n%3d           0 BUILD_LIST               0\n              2 LOAD_DEREF               0 (y)\n              4 GET_ITER\n        >>    6 FOR_ITER                 6 (to 20)\n              8 STORE_FAST               1 (z)\n             10 LOAD_FAST                0 (x)\n             12 LOAD_FAST                1 (z)\n             14 BINARY_ADD\n             16 LIST_APPEND              2\n             18 JUMP_ABSOLUTE            3 (to 6)\n        >>   20 DELETE_FAST              1 (z)\n             22 RETURN_VALUE\n' % (dis_nested_0, __file__, _h.__code__.co_firstlineno + 1, _h.__code__.co_firstlineno + 3)

class DisTests(unittest.TestCase):
    _inline_comprehensions = os.getenv('PYTHONINLINECOMPREHENSIONS')
    maxDiff = None

    def get_disassembly(self, func, lasti=-1, wrapper=True, **kwargs):
        if False:
            while True:
                i = 10
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            if wrapper:
                dis.dis(func, **kwargs)
            else:
                dis.disassemble(func, lasti, **kwargs)
        return output.getvalue()

    def get_disassemble_as_string(self, func, lasti=-1):
        if False:
            return 10
        return self.get_disassembly(func, lasti, False)

    def strip_addresses(self, text):
        if False:
            for i in range(10):
                print('nop')
        return re.sub('\\b0x[0-9A-Fa-f]+\\b', '0x...', text)

    def do_disassembly_test(self, func, expected):
        if False:
            print('Hello World!')
        got = self.get_disassembly(func, depth=0)
        if got != expected:
            got = self.strip_addresses(got)
        self.assertEqual(got, expected)

    def test_opmap(self):
        if False:
            print('Hello World!')
        self.assertEqual(dis.opmap['NOP'], 9)
        self.assertIn(dis.opmap['LOAD_CONST'], dis.hasconst)
        self.assertIn(dis.opmap['STORE_NAME'], dis.hasname)

    def test_opname(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(dis.opname[dis.opmap['LOAD_FAST']], 'LOAD_FAST')

    def test_boundaries(self):
        if False:
            print('Hello World!')
        self.assertEqual(dis.opmap['EXTENDED_ARG'], dis.EXTENDED_ARG)
        self.assertEqual(dis.opmap['STORE_NAME'], dis.HAVE_ARGUMENT)

    def test_widths(self):
        if False:
            while True:
                i = 10
        for (opcode, opname) in enumerate(dis.opname):
            if opname in ('BUILD_MAP_UNPACK_WITH_CALL', 'BUILD_TUPLE_UNPACK_WITH_CALL', 'JUMP_IF_NONZERO_OR_POP', 'JUMP_IF_NOT_EXC_MATCH') or opcode in shadowop:
                continue
            with self.subTest(opname=opname):
                width = dis._OPNAME_WIDTH
                if opcode < dis.HAVE_ARGUMENT:
                    width += 1 + dis._OPARG_WIDTH
                self.assertLessEqual(len(opname), width)

    def test_dis(self):
        if False:
            print('Hello World!')
        self.do_disassembly_test(_f, dis_f)

    def test_bug_708901(self):
        if False:
            for i in range(10):
                print('nop')
        self.do_disassembly_test(bug708901, dis_bug708901)

    def test_bug_1333982(self):
        if False:
            for i in range(10):
                print('nop')
        if not __debug__:
            self.skipTest('need asserts, run without -O')
        if self._inline_comprehensions:
            self.do_disassembly_test(bug1333982, dis_bug1333982_with_inline_comprehensions)
        else:
            self.do_disassembly_test(bug1333982, dis_bug1333982)

    def test_bug_42562(self):
        if False:
            while True:
                i = 10
        self.do_disassembly_test(bug42562, dis_bug42562)

    def test_bug_45757(self):
        if False:
            for i in range(10):
                print('nop')
        self.do_disassembly_test(code_bug_45757, dis_bug_45757)

    def test_big_linenos(self):
        if False:
            i = 10
            return i + 15

        def func(count):
            if False:
                return 10
            namespace = {}
            func = 'def foo():\n ' + ''.join(['\n '] * count + ['spam\n'])
            exec(func, namespace)
            return namespace['foo']
        for i in range(1, 300):
            expected = _BIG_LINENO_FORMAT % (i + 2)
            self.do_disassembly_test(func(i), expected)
        for i in range(300, 1000, 10):
            expected = _BIG_LINENO_FORMAT % (i + 2)
            self.do_disassembly_test(func(i), expected)
        for i in range(1000, 5000, 10):
            expected = _BIG_LINENO_FORMAT2 % (i + 2)
            self.do_disassembly_test(func(i), expected)
        from test import dis_module
        self.do_disassembly_test(dis_module, dis_module_expected_results)

    def test_big_offsets(self):
        if False:
            return 10

        def func(count):
            if False:
                for i in range(10):
                    print('nop')
            namespace = {}
            func = 'def foo(x):\n ' + ';'.join(['x = x + 1'] * count) + '\n return x'
            exec(func, namespace)
            return namespace['foo']

        def expected(count, w):
            if False:
                i = 10
                return i + 15
            s = ['           %*d LOAD_FAST                0 (x)\n           %*d LOAD_CONST               1 (1)\n           %*d BINARY_ADD\n           %*d STORE_FAST               0 (x)\n' % (w, 8 * i, w, 8 * i + 2, w, 8 * i + 4, w, 8 * i + 6) for i in range(count)]
            s += ['\n  3        %*d LOAD_FAST                0 (x)\n           %*d RETURN_VALUE\n' % (w, 8 * count, w, 8 * count + 2)]
            s[0] = '  2' + s[0][3:]
            return ''.join(s)
        for i in range(1, 5):
            self.do_disassembly_test(func(i), expected(i, 4))
        self.do_disassembly_test(func(1249), expected(1249, 4))
        self.do_disassembly_test(func(1250), expected(1250, 5))

    def test_disassemble_str(self):
        if False:
            while True:
                i = 10
        self.do_disassembly_test(expr_str, dis_expr_str)
        self.do_disassembly_test(simple_stmt_str, dis_simple_stmt_str)
        self.do_disassembly_test(annot_stmt_str, dis_annot_stmt_str)
        self.do_disassembly_test(compound_stmt_str, dis_compound_stmt_str)

    def test_disassemble_bytes(self):
        if False:
            return 10
        self.do_disassembly_test(_f.__code__.co_code, dis_f_co_code)

    def test_disassemble_class(self):
        if False:
            i = 10
            return i + 15
        self.do_disassembly_test(_C, dis_c)

    def test_disassemble_instance_method(self):
        if False:
            return 10
        self.do_disassembly_test(_C(1).__init__, dis_c_instance_method)

    def test_disassemble_instance_method_bytes(self):
        if False:
            while True:
                i = 10
        method_bytecode = _C(1).__init__.__code__.co_code
        self.do_disassembly_test(method_bytecode, dis_c_instance_method_bytes)

    def test_disassemble_static_method(self):
        if False:
            i = 10
            return i + 15
        self.do_disassembly_test(_C.sm, dis_c_static_method)

    def test_disassemble_class_method(self):
        if False:
            i = 10
            return i + 15
        self.do_disassembly_test(_C.cm, dis_c_class_method)

    def test_disassemble_generator(self):
        if False:
            i = 10
            return i + 15
        gen_func_disas = self.get_disassembly(_g)
        gen_disas = self.get_disassembly(_g(1))
        self.assertEqual(gen_disas, gen_func_disas)

    def test_disassemble_async_generator(self):
        if False:
            print('Hello World!')
        agen_func_disas = self.get_disassembly(_ag)
        agen_disas = self.get_disassembly(_ag(1))
        self.assertEqual(agen_disas, agen_func_disas)

    def test_disassemble_coroutine(self):
        if False:
            print('Hello World!')
        coro_func_disas = self.get_disassembly(_co)
        coro = _co(1)
        coro.close()
        coro_disas = self.get_disassembly(coro)
        self.assertEqual(coro_disas, coro_func_disas)

    def test_disassemble_fstring(self):
        if False:
            i = 10
            return i + 15
        self.do_disassembly_test(_fstring, dis_fstring)

    def test_disassemble_try_finally(self):
        if False:
            print('Hello World!')
        self.do_disassembly_test(_tryfinally, dis_tryfinally)
        self.do_disassembly_test(_tryfinallyconst, dis_tryfinallyconst)

    def test_dis_none(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            del sys.last_traceback
        except AttributeError:
            pass
        self.assertRaises(RuntimeError, dis.dis, None)

    def test_dis_traceback(self):
        if False:
            while True:
                i = 10
        try:
            del sys.last_traceback
        except AttributeError:
            pass
        try:
            1 / 0
        except Exception as e:
            tb = e.__traceback__
            sys.last_traceback = tb
        tb_dis = self.get_disassemble_as_string(tb.tb_frame.f_code, tb.tb_lasti)
        self.do_disassembly_test(None, tb_dis)

    def test_dis_object(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, dis.dis, object())

    def test_disassemble_recursive(self):
        if False:
            i = 10
            return i + 15

        def check(expected, **kwargs):
            if False:
                i = 10
                return i + 15
            dis = self.get_disassembly(_h, **kwargs)
            dis = self.strip_addresses(dis)
            self.assertEqual(dis, expected)
        check(dis_nested_0, depth=0)
        if self._inline_comprehensions:
            check(dis_nested_1_with_inline_comprehensions, depth=1)
        else:
            check(dis_nested_1, depth=1)

class DisWithFileTests(DisTests):

    def get_disassembly(self, func, lasti=-1, wrapper=True, **kwargs):
        if False:
            while True:
                i = 10
        output = io.StringIO()
        if wrapper:
            dis.dis(func, file=output, **kwargs)
        else:
            dis.disassemble(func, lasti, file=output, **kwargs)
        return output.getvalue()
if sys.flags.optimize:
    code_info_consts = '0: None'
else:
    code_info_consts = "0: 'Formatted details of methods, functions, or code.'"
code_info_code_info = f'Name:              code_info\nFilename:          (.*)\nArgument count:    1\nPositional-only arguments: 0\nKw-only arguments: 0\nNumber of locals:  1\nStack size:        3\nFlags:             OPTIMIZED, NEWLOCALS, NOFREE\nConstants:\n   {code_info_consts}\nNames:\n   0: _format_code_info\n   1: _get_code_object\nVariable names:\n   0: x'

@staticmethod
def tricky(a, b, /, x, y, z=True, *args, c, d, e=[], **kwds):
    if False:
        print('Hello World!')

    def f(c=c):
        if False:
            return 10
        print(a, b, x, y, z, c, d, e, f)
    yield (a, b, x, y, z, c, d, e, f)
code_info_tricky = 'Name:              tricky\nFilename:          (.*)\nArgument count:    5\nPositional-only arguments: 2\nKw-only arguments: 3\nNumber of locals:  10\nStack size:        9\nFlags:             OPTIMIZED, NEWLOCALS, VARARGS, VARKEYWORDS, GENERATOR\nConstants:\n   0: None\n   1: <code object f at (.*), file "(.*)", line (.*)>\n   2: \'tricky.<locals>.f\'\nVariable names:\n   0: a\n   1: b\n   2: x\n   3: y\n   4: z\n   5: c\n   6: d\n   7: e\n   8: args\n   9: kwds\nCell variables:\n   0: [abedfxyz]\n   1: [abedfxyz]\n   2: [abedfxyz]\n   3: [abedfxyz]\n   4: [abedfxyz]\n   5: [abedfxyz]'
co_tricky_nested_f = tricky.__func__.__code__.co_consts[1]
code_info_tricky_nested_f = 'Filename:          (.*)\nArgument count:    1\nPositional-only arguments: 0\nKw-only arguments: 0\nNumber of locals:  1\nStack size:        10\nFlags:             OPTIMIZED, NEWLOCALS, NESTED\nConstants:\n   0: None\nNames:\n   0: print\nVariable names:\n   0: c\nFree variables:\n   0: [abedfxyz]\n   1: [abedfxyz]\n   2: [abedfxyz]\n   3: [abedfxyz]\n   4: [abedfxyz]\n   5: [abedfxyz]'
code_info_expr_str = 'Name:              <module>\nFilename:          <disassembly>\nArgument count:    0\nPositional-only arguments: 0\nKw-only arguments: 0\nNumber of locals:  0\nStack size:        2\nFlags:             NOFREE\nConstants:\n   0: 1\nNames:\n   0: x'
code_info_simple_stmt_str = 'Name:              <module>\nFilename:          <disassembly>\nArgument count:    0\nPositional-only arguments: 0\nKw-only arguments: 0\nNumber of locals:  0\nStack size:        2\nFlags:             NOFREE\nConstants:\n   0: 1\n   1: None\nNames:\n   0: x'
code_info_compound_stmt_str = 'Name:              <module>\nFilename:          <disassembly>\nArgument count:    0\nPositional-only arguments: 0\nKw-only arguments: 0\nNumber of locals:  0\nStack size:        2\nFlags:             NOFREE\nConstants:\n   0: 0\n   1: 1\nNames:\n   0: x'

async def async_def():
    await 1
    async for a in b:
        pass
    async with c as d:
        pass
code_info_async_def = 'Name:              async_def\nFilename:          (.*)\nArgument count:    0\nPositional-only arguments: 0\nKw-only arguments: 0\nNumber of locals:  2\nStack size:        9\nFlags:             OPTIMIZED, NEWLOCALS, NOFREE, COROUTINE\nConstants:\n   0: None\n   1: 1\nNames:\n   0: b\n   1: c\nVariable names:\n   0: a\n   1: d'

class CodeInfoTests(unittest.TestCase):
    test_pairs = [(dis.code_info, code_info_code_info), (tricky, code_info_tricky), (co_tricky_nested_f, code_info_tricky_nested_f), (expr_str, code_info_expr_str), (simple_stmt_str, code_info_simple_stmt_str), (compound_stmt_str, code_info_compound_stmt_str), (async_def, code_info_async_def)]

    def test_code_info(self):
        if False:
            while True:
                i = 10
        self.maxDiff = 1000
        for (x, expected) in self.test_pairs:
            self.assertRegex(dis.code_info(x), expected)

    def test_show_code(self):
        if False:
            for i in range(10):
                print('nop')
        self.maxDiff = 1000
        for (x, expected) in self.test_pairs:
            with captured_stdout() as output:
                dis.show_code(x)
            self.assertRegex(output.getvalue(), expected + '\n')
            output = io.StringIO()
            dis.show_code(x, file=output)
            self.assertRegex(output.getvalue(), expected)

    def test_code_info_object(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, dis.code_info, object())

    def test_pretty_flags_no_flags(self):
        if False:
            while True:
                i = 10
        self.assertEqual(dis.pretty_flags(0), '0x0')

def outer(a=1, b=2):
    if False:
        return 10

    def f(c=3, d=4):
        if False:
            return 10

        def inner(e=5, f=6):
            if False:
                while True:
                    i = 10
            print(a, b, c, d, e, f)
        print(a, b, c, d)
        return inner
    print(a, b, '', 1, [], {}, 'Hello world!')
    return f

def jumpy():
    if False:
        for i in range(10):
            print('nop')
    for i in range(10):
        print(i)
        if i < 4:
            continue
        if i > 6:
            break
    else:
        print('I can haz else clause?')
    while i:
        print(i)
        i -= 1
        if i > 6:
            continue
        if i < 4:
            break
    else:
        print('Who let lolcatz into this test suite?')
    try:
        1 / 0
    except ZeroDivisionError:
        print('Here we go, here we go, here we go...')
    else:
        with i as dodgy:
            print('Never reach this')
    finally:
        print("OK, now we're done")
expected_outer_line = 1
_line_offset = outer.__code__.co_firstlineno - 1
code_object_f = outer.__code__.co_consts[3]
expected_f_line = code_object_f.co_firstlineno - _line_offset
code_object_inner = code_object_f.co_consts[3]
expected_inner_line = code_object_inner.co_firstlineno - _line_offset
expected_jumpy_line = 1
Instruction = dis.Instruction
expected_opinfo_outer = [Instruction(opname='LOAD_CONST', opcode=100, arg=8, argval=(3, 4), argrepr='(3, 4)', offset=0, starts_line=2, is_jump_target=False), Instruction(opname='LOAD_CLOSURE', opcode=135, arg=0, argval='a', argrepr='a', offset=2, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CLOSURE', opcode=135, arg=1, argval='b', argrepr='b', offset=4, starts_line=None, is_jump_target=False), Instruction(opname='BUILD_TUPLE', opcode=102, arg=2, argval=2, argrepr='', offset=6, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=3, argval=code_object_f, argrepr=repr(code_object_f), offset=8, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=4, argval='outer.<locals>.f', argrepr="'outer.<locals>.f'", offset=10, starts_line=None, is_jump_target=False), Instruction(opname='MAKE_FUNCTION', opcode=132, arg=9, argval=9, argrepr='defaults, closure', offset=12, starts_line=None, is_jump_target=False), Instruction(opname='STORE_FAST', opcode=125, arg=2, argval='f', argrepr='f', offset=14, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=0, argval='print', argrepr='print', offset=16, starts_line=7, is_jump_target=False), Instruction(opname='LOAD_DEREF', opcode=136, arg=0, argval='a', argrepr='a', offset=18, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_DEREF', opcode=136, arg=1, argval='b', argrepr='b', offset=20, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=5, argval='', argrepr="''", offset=22, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=6, argval=1, argrepr='1', offset=24, starts_line=None, is_jump_target=False), Instruction(opname='BUILD_LIST', opcode=103, arg=0, argval=0, argrepr='', offset=26, starts_line=None, is_jump_target=False), Instruction(opname='BUILD_MAP', opcode=105, arg=0, argval=0, argrepr='', offset=28, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=7, argval='Hello world!', argrepr="'Hello world!'", offset=30, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=7, argval=7, argrepr='', offset=32, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=34, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_FAST', opcode=124, arg=2, argval='f', argrepr='f', offset=36, starts_line=8, is_jump_target=False), Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=38, starts_line=None, is_jump_target=False)]
expected_opinfo_f = [Instruction(opname='LOAD_CONST', opcode=100, arg=5, argval=(5, 6), argrepr='(5, 6)', offset=0, starts_line=3, is_jump_target=False), Instruction(opname='LOAD_CLOSURE', opcode=135, arg=2, argval='a', argrepr='a', offset=2, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CLOSURE', opcode=135, arg=3, argval='b', argrepr='b', offset=4, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CLOSURE', opcode=135, arg=0, argval='c', argrepr='c', offset=6, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CLOSURE', opcode=135, arg=1, argval='d', argrepr='d', offset=8, starts_line=None, is_jump_target=False), Instruction(opname='BUILD_TUPLE', opcode=102, arg=4, argval=4, argrepr='', offset=10, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=3, argval=code_object_inner, argrepr=repr(code_object_inner), offset=12, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=4, argval='outer.<locals>.f.<locals>.inner', argrepr="'outer.<locals>.f.<locals>.inner'", offset=14, starts_line=None, is_jump_target=False), Instruction(opname='MAKE_FUNCTION', opcode=132, arg=9, argval=9, argrepr='defaults, closure', offset=16, starts_line=None, is_jump_target=False), Instruction(opname='STORE_FAST', opcode=125, arg=2, argval='inner', argrepr='inner', offset=18, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=0, argval='print', argrepr='print', offset=20, starts_line=5, is_jump_target=False), Instruction(opname='LOAD_DEREF', opcode=136, arg=2, argval='a', argrepr='a', offset=22, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_DEREF', opcode=136, arg=3, argval='b', argrepr='b', offset=24, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_DEREF', opcode=136, arg=0, argval='c', argrepr='c', offset=26, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_DEREF', opcode=136, arg=1, argval='d', argrepr='d', offset=28, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=4, argval=4, argrepr='', offset=30, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=32, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_FAST', opcode=124, arg=2, argval='inner', argrepr='inner', offset=34, starts_line=6, is_jump_target=False), Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=36, starts_line=None, is_jump_target=False)]
expected_opinfo_inner = [Instruction(opname='LOAD_GLOBAL', opcode=116, arg=0, argval='print', argrepr='print', offset=0, starts_line=4, is_jump_target=False), Instruction(opname='LOAD_DEREF', opcode=136, arg=0, argval='a', argrepr='a', offset=2, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_DEREF', opcode=136, arg=1, argval='b', argrepr='b', offset=4, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_DEREF', opcode=136, arg=2, argval='c', argrepr='c', offset=6, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_DEREF', opcode=136, arg=3, argval='d', argrepr='d', offset=8, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='e', argrepr='e', offset=10, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_FAST', opcode=124, arg=1, argval='f', argrepr='f', offset=12, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=6, argval=6, argrepr='', offset=14, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=16, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=0, argval=None, argrepr='None', offset=18, starts_line=None, is_jump_target=False), Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=20, starts_line=None, is_jump_target=False)]
expected_opinfo_jumpy = [Instruction(opname='LOAD_GLOBAL', opcode=116, arg=0, argval='range', argrepr='range', offset=0, starts_line=3, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=1, argval=10, argrepr='10', offset=2, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='', offset=4, starts_line=None, is_jump_target=False), Instruction(opname='GET_ITER', opcode=68, arg=None, argval=None, argrepr='', offset=6, starts_line=None, is_jump_target=False), Instruction(opname='FOR_ITER', opcode=93, arg=17, argval=44, argrepr='to 44', offset=8, starts_line=None, is_jump_target=True), Instruction(opname='STORE_FAST', opcode=125, arg=0, argval='i', argrepr='i', offset=10, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=12, starts_line=4, is_jump_target=False), Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=14, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='', offset=16, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=18, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=20, starts_line=5, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=2, argval=4, argrepr='4', offset=22, starts_line=None, is_jump_target=False), Instruction(opname='COMPARE_OP', opcode=107, arg=0, argval='<', argrepr='<', offset=24, starts_line=None, is_jump_target=False), Instruction(opname='POP_JUMP_IF_FALSE', opcode=114, arg=15, argval=30, argrepr='to 30', offset=26, starts_line=None, is_jump_target=False), Instruction(opname='JUMP_ABSOLUTE', opcode=113, arg=4, argval=8, argrepr='to 8', offset=28, starts_line=6, is_jump_target=False), Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=30, starts_line=7, is_jump_target=True), Instruction(opname='LOAD_CONST', opcode=100, arg=3, argval=6, argrepr='6', offset=32, starts_line=None, is_jump_target=False), Instruction(opname='COMPARE_OP', opcode=107, arg=4, argval='>', argrepr='>', offset=34, starts_line=None, is_jump_target=False), Instruction(opname='POP_JUMP_IF_FALSE', opcode=114, arg=21, argval=42, argrepr='to 42', offset=36, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=38, starts_line=8, is_jump_target=False), Instruction(opname='JUMP_FORWARD', opcode=110, arg=5, argval=52, argrepr='to 52', offset=40, starts_line=None, is_jump_target=False), Instruction(opname='JUMP_ABSOLUTE', opcode=113, arg=4, argval=8, argrepr='to 8', offset=42, starts_line=7, is_jump_target=True), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=44, starts_line=10, is_jump_target=True), Instruction(opname='LOAD_CONST', opcode=100, arg=4, argval='I can haz else clause?', argrepr="'I can haz else clause?'", offset=46, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='', offset=48, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=50, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=52, starts_line=11, is_jump_target=True), Instruction(opname='POP_JUMP_IF_FALSE', opcode=114, arg=48, argval=96, argrepr='to 96', offset=54, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=56, starts_line=12, is_jump_target=True), Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=58, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='', offset=60, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=62, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=64, starts_line=13, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=5, argval=1, argrepr='1', offset=66, starts_line=None, is_jump_target=False), Instruction(opname='INPLACE_SUBTRACT', opcode=56, arg=None, argval=None, argrepr='', offset=68, starts_line=None, is_jump_target=False), Instruction(opname='STORE_FAST', opcode=125, arg=0, argval='i', argrepr='i', offset=70, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=72, starts_line=14, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=3, argval=6, argrepr='6', offset=74, starts_line=None, is_jump_target=False), Instruction(opname='COMPARE_OP', opcode=107, arg=4, argval='>', argrepr='>', offset=76, starts_line=None, is_jump_target=False), Instruction(opname='POP_JUMP_IF_FALSE', opcode=114, arg=41, argval=82, argrepr='to 82', offset=78, starts_line=None, is_jump_target=False), Instruction(opname='JUMP_ABSOLUTE', opcode=113, arg=26, argval=52, argrepr='to 52', offset=80, starts_line=15, is_jump_target=False), Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=82, starts_line=16, is_jump_target=True), Instruction(opname='LOAD_CONST', opcode=100, arg=2, argval=4, argrepr='4', offset=84, starts_line=None, is_jump_target=False), Instruction(opname='COMPARE_OP', opcode=107, arg=0, argval='<', argrepr='<', offset=86, starts_line=None, is_jump_target=False), Instruction(opname='POP_JUMP_IF_FALSE', opcode=114, arg=46, argval=92, argrepr='to 92', offset=88, starts_line=None, is_jump_target=False), Instruction(opname='JUMP_FORWARD', opcode=110, arg=6, argval=104, argrepr='to 104', offset=90, starts_line=17, is_jump_target=False), Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=92, starts_line=11, is_jump_target=True), Instruction(opname='POP_JUMP_IF_TRUE', opcode=115, arg=28, argval=56, argrepr='to 56', offset=94, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=96, starts_line=19, is_jump_target=True), Instruction(opname='LOAD_CONST', opcode=100, arg=6, argval='Who let lolcatz into this test suite?', argrepr="'Who let lolcatz into this test suite?'", offset=98, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='', offset=100, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=102, starts_line=None, is_jump_target=False), Instruction(opname='SETUP_FINALLY', opcode=122, arg=63, argval=232, argrepr='to 232', offset=104, starts_line=20, is_jump_target=True), Instruction(opname='SETUP_FINALLY', opcode=122, arg=6, argval=120, argrepr='to 120', offset=106, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=5, argval=1, argrepr='1', offset=108, starts_line=21, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=7, argval=0, argrepr='0', offset=110, starts_line=None, is_jump_target=False), Instruction(opname='BINARY_TRUE_DIVIDE', opcode=27, arg=None, argval=None, argrepr='', offset=112, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=114, starts_line=None, is_jump_target=False), Instruction(opname='POP_BLOCK', opcode=87, arg=None, argval=None, argrepr='', offset=116, starts_line=None, is_jump_target=False), Instruction(opname='JUMP_FORWARD', opcode=110, arg=13, argval=146, argrepr='to 146', offset=118, starts_line=None, is_jump_target=False), Instruction(opname='DUP_TOP', opcode=4, arg=None, argval=None, argrepr='', offset=120, starts_line=22, is_jump_target=True), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=2, argval='ZeroDivisionError', argrepr='ZeroDivisionError', offset=122, starts_line=None, is_jump_target=False), Instruction(opname='JUMP_IF_NOT_EXC_MATCH', opcode=121, arg=72, argval=144, argrepr='to 144', offset=124, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=126, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=128, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=130, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=132, starts_line=23, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=8, argval='Here we go, here we go, here we go...', argrepr="'Here we go, here we go, here we go...'", offset=134, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='', offset=136, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=138, starts_line=None, is_jump_target=False), Instruction(opname='POP_EXCEPT', opcode=89, arg=None, argval=None, argrepr='', offset=140, starts_line=None, is_jump_target=False), Instruction(opname='JUMP_FORWARD', opcode=110, arg=30, argval=204, argrepr='to 204', offset=142, starts_line=None, is_jump_target=False), Instruction(opname='RERAISE', opcode=119, arg=0, argval=0, argrepr='', offset=144, starts_line=22, is_jump_target=True), Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='i', argrepr='i', offset=146, starts_line=25, is_jump_target=True), Instruction(opname='SETUP_WITH', opcode=143, arg=12, argval=174, argrepr='to 174', offset=148, starts_line=None, is_jump_target=False), Instruction(opname='STORE_FAST', opcode=125, arg=1, argval='dodgy', argrepr='dodgy', offset=150, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=152, starts_line=26, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=9, argval='Never reach this', argrepr="'Never reach this'", offset=154, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='', offset=156, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=158, starts_line=None, is_jump_target=False), Instruction(opname='POP_BLOCK', opcode=87, arg=None, argval=None, argrepr='', offset=160, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=0, argval=None, argrepr='None', offset=162, starts_line=25, is_jump_target=False), Instruction(opname='DUP_TOP', opcode=4, arg=None, argval=None, argrepr='', offset=164, starts_line=None, is_jump_target=False), Instruction(opname='DUP_TOP', opcode=4, arg=None, argval=None, argrepr='', offset=166, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=3, argval=3, argrepr='', offset=168, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=170, starts_line=None, is_jump_target=False), Instruction(opname='JUMP_FORWARD', opcode=110, arg=22, argval=218, argrepr='to 218', offset=172, starts_line=None, is_jump_target=False), Instruction(opname='WITH_EXCEPT_START', opcode=49, arg=None, argval=None, argrepr='', offset=174, starts_line=None, is_jump_target=True), Instruction(opname='POP_JUMP_IF_TRUE', opcode=115, arg=90, argval=180, argrepr='to 180', offset=176, starts_line=None, is_jump_target=False), Instruction(opname='RERAISE', opcode=119, arg=1, argval=1, argrepr='', offset=178, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=180, starts_line=None, is_jump_target=True), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=182, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=184, starts_line=None, is_jump_target=False), Instruction(opname='POP_EXCEPT', opcode=89, arg=None, argval=None, argrepr='', offset=186, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=188, starts_line=None, is_jump_target=False), Instruction(opname='POP_BLOCK', opcode=87, arg=None, argval=None, argrepr='', offset=190, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=192, starts_line=28, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=10, argval="OK, now we're done", argrepr='"OK, now we\'re done"', offset=194, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='', offset=196, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=198, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=0, argval=None, argrepr='None', offset=200, starts_line=None, is_jump_target=False), Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=202, starts_line=None, is_jump_target=False), Instruction(opname='POP_BLOCK', opcode=87, arg=None, argval=None, argrepr='', offset=204, starts_line=23, is_jump_target=True), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=206, starts_line=28, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=10, argval="OK, now we're done", argrepr='"OK, now we\'re done"', offset=208, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='', offset=210, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=212, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=0, argval=None, argrepr='None', offset=214, starts_line=None, is_jump_target=False), Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=216, starts_line=None, is_jump_target=False), Instruction(opname='POP_BLOCK', opcode=87, arg=None, argval=None, argrepr='', offset=218, starts_line=25, is_jump_target=True), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=220, starts_line=28, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=10, argval="OK, now we're done", argrepr='"OK, now we\'re done"', offset=222, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='', offset=224, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=226, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_CONST', opcode=100, arg=0, argval=None, argrepr='None', offset=228, starts_line=None, is_jump_target=False), Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=230, starts_line=None, is_jump_target=False), Instruction(opname='LOAD_GLOBAL', opcode=116, arg=1, argval='print', argrepr='print', offset=232, starts_line=None, is_jump_target=True), Instruction(opname='LOAD_CONST', opcode=100, arg=10, argval="OK, now we're done", argrepr='"OK, now we\'re done"', offset=234, starts_line=None, is_jump_target=False), Instruction(opname='CALL_FUNCTION', opcode=131, arg=1, argval=1, argrepr='', offset=236, starts_line=None, is_jump_target=False), Instruction(opname='POP_TOP', opcode=1, arg=None, argval=None, argrepr='', offset=238, starts_line=None, is_jump_target=False), Instruction(opname='RERAISE', opcode=119, arg=0, argval=0, argrepr='', offset=240, starts_line=None, is_jump_target=False)]

def simple():
    if False:
        print('Hello World!')
    pass
expected_opinfo_simple = [Instruction(opname='LOAD_CONST', opcode=100, arg=0, argval=None, argrepr='None', offset=0, starts_line=simple.__code__.co_firstlineno, is_jump_target=False), Instruction(opname='RETURN_VALUE', opcode=83, arg=None, argval=None, argrepr='', offset=2, starts_line=None, is_jump_target=False)]

class InstructionTests(BytecodeTestCase):

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        super().__init__(*args)
        self.maxDiff = None

    def test_default_first_line(self):
        if False:
            return 10
        actual = dis.get_instructions(simple)
        self.assertEqual(list(actual), expected_opinfo_simple)

    def test_first_line_set_to_None(self):
        if False:
            i = 10
            return i + 15
        actual = dis.get_instructions(simple, first_line=None)
        self.assertEqual(list(actual), expected_opinfo_simple)

    def test_outer(self):
        if False:
            while True:
                i = 10
        actual = dis.get_instructions(outer, first_line=expected_outer_line)
        self.assertEqual(list(actual), expected_opinfo_outer)

    def test_nested(self):
        if False:
            return 10
        with captured_stdout():
            f = outer()
        actual = dis.get_instructions(f, first_line=expected_f_line)
        self.assertEqual(list(actual), expected_opinfo_f)

    def test_doubly_nested(self):
        if False:
            while True:
                i = 10
        with captured_stdout():
            inner = outer()()
        actual = dis.get_instructions(inner, first_line=expected_inner_line)
        self.assertEqual(list(actual), expected_opinfo_inner)

    def test_jumpy(self):
        if False:
            return 10
        actual = dis.get_instructions(jumpy, first_line=expected_jumpy_line)
        self.assertEqual(list(actual), expected_opinfo_jumpy)

class BytecodeTests(unittest.TestCase):

    def test_instantiation(self):
        if False:
            i = 10
            return i + 15
        for obj in [_f, _C(1).__init__, 'a=1', _f.__code__]:
            with self.subTest(obj=obj):
                b = dis.Bytecode(obj)
                self.assertIsInstance(b.codeobj, types.CodeType)
        self.assertRaises(TypeError, dis.Bytecode, object())

    def test_iteration(self):
        if False:
            return 10
        for obj in [_f, _C(1).__init__, 'a=1', _f.__code__]:
            with self.subTest(obj=obj):
                via_object = list(dis.Bytecode(obj))
                via_generator = list(dis.get_instructions(obj))
                self.assertEqual(via_object, via_generator)

    def test_explicit_first_line(self):
        if False:
            i = 10
            return i + 15
        actual = dis.Bytecode(outer, first_line=expected_outer_line)
        self.assertEqual(list(actual), expected_opinfo_outer)

    def test_source_line_in_disassembly(self):
        if False:
            print('Hello World!')
        actual = dis.Bytecode(simple).dis()
        actual = actual.strip().partition(' ')[0]
        expected = str(simple.__code__.co_firstlineno)
        self.assertEqual(actual, expected)
        actual = dis.Bytecode(simple, first_line=350).dis()
        actual = actual.strip().partition(' ')[0]
        self.assertEqual(actual, '350')

    def test_info(self):
        if False:
            while True:
                i = 10
        self.maxDiff = 1000
        for (x, expected) in CodeInfoTests.test_pairs:
            b = dis.Bytecode(x)
            self.assertRegex(b.info(), expected)

    def test_disassembled(self):
        if False:
            return 10
        actual = dis.Bytecode(_f).dis()
        self.assertEqual(actual, dis_f)

    def test_from_traceback(self):
        if False:
            i = 10
            return i + 15
        tb = get_tb()
        b = dis.Bytecode.from_traceback(tb)
        while tb.tb_next:
            tb = tb.tb_next
        self.assertEqual(b.current_offset, tb.tb_lasti)

    def test_from_traceback_dis(self):
        if False:
            while True:
                i = 10
        tb = get_tb()
        b = dis.Bytecode.from_traceback(tb)
        self.assertEqual(b.dis(), dis_traceback)

class TestBytecodeTestCase(BytecodeTestCase):

    def test_assert_not_in_with_op_not_in_bytecode(self):
        if False:
            for i in range(10):
                print('nop')
        code = compile('a = 1', '<string>', 'exec')
        self.assertInBytecode(code, 'LOAD_CONST', 1)
        self.assertNotInBytecode(code, 'LOAD_NAME')
        self.assertNotInBytecode(code, 'LOAD_NAME', 'a')

    def test_assert_not_in_with_arg_not_in_bytecode(self):
        if False:
            for i in range(10):
                print('nop')
        code = compile('a = 1', '<string>', 'exec')
        self.assertInBytecode(code, 'LOAD_CONST')
        self.assertInBytecode(code, 'LOAD_CONST', 1)
        self.assertNotInBytecode(code, 'LOAD_CONST', 2)

    def test_assert_not_in_with_arg_in_bytecode(self):
        if False:
            print('Hello World!')
        code = compile('a = 1', '<string>', 'exec')
        with self.assertRaises(AssertionError):
            self.assertNotInBytecode(code, 'LOAD_CONST', 1)

class TestDisTraceback(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            del sys.last_traceback
        except AttributeError:
            pass
        return super().setUp()

    def get_disassembly(self, tb):
        if False:
            i = 10
            return i + 15
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            dis.distb(tb)
        return output.getvalue()

    def test_distb_empty(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(RuntimeError):
            dis.distb()

    def test_distb_last_traceback(self):
        if False:
            while True:
                i = 10
        tb = get_tb()
        sys.last_traceback = tb
        self.assertEqual(self.get_disassembly(None), dis_traceback)

    def test_distb_explicit_arg(self):
        if False:
            return 10
        tb = get_tb()
        self.assertEqual(self.get_disassembly(tb), dis_traceback)

class TestDisTracebackWithFile(TestDisTraceback):

    def get_disassembly(self, tb):
        if False:
            i = 10
            return i + 15
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            dis.distb(tb, file=output)
        return output.getvalue()
if __name__ == '__main__':
    unittest.main()