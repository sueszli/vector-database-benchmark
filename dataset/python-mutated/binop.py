from compiler.errors import TypedSyntaxError
from compiler.static.types import PRIM_OP_ADD_INT, PRIM_OP_DIV_INT, TYPED_INT16, TYPED_INT8
from unittest import skip
from .common import StaticTestBase
try:
    import cinderjit
except ImportError:
    cinderjit = None

class BinopTests(StaticTestBase):

    def test_pow_of_int64s_returns_double(self):
        if False:
            return 10
        codestr = '\n        from __static__ import int64\n        def foo():\n            x: int64 = 0\n            y: int64 = 1\n            z: int64 = x ** y\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'double cannot be assigned to int64'):
            self.compile(codestr, modname='foo')

    def test_int_binop(self):
        if False:
            print('Hello World!')
        tests = [('int8', 1, 2, '/', 0), ('int8', 4, 2, '/', 2), ('int8', 4, -2, '/', -2), ('uint8', 255, 127, '/', 2), ('int16', 4, -2, '/', -2), ('uint16', 255, 127, '/', 2), ('uint32', 65535, 32767, '/', 2), ('int32', 4, -2, '/', -2), ('uint32', 255, 127, '/', 2), ('uint32', 4294967295, 2147483647, '/', 2), ('int64', 4, -2, '/', -2), ('uint64', 255, 127, '/', 2), ('uint64', 18446744073709551615, 9223372036854775807, '/', 2), ('int8', 1, -2, '-', 3), ('int8', 1, 2, '-', -1), ('int16', 1, -2, '-', 3), ('int16', 1, 2, '-', -1), ('int32', 1, -2, '-', 3), ('int32', 1, 2, '-', -1), ('int64', 1, -2, '-', 3), ('int64', 1, 2, '-', -1), ('int8', 1, -2, '*', -2), ('int8', 1, 2, '*', 2), ('int16', 1, -2, '*', -2), ('int16', 1, 2, '*', 2), ('int32', 1, -2, '*', -2), ('int32', 1, 2, '*', 2), ('int64', 1, -2, '*', -2), ('int64', 1, 2, '*', 2), ('int8', 1, -2, '&', 0), ('int8', 1, 3, '&', 1), ('int16', 1, 3, '&', 1), ('int16', 1, 3, '&', 1), ('int32', 1, 3, '&', 1), ('int32', 1, 3, '&', 1), ('int64', 1, 3, '&', 1), ('int64', 1, 3, '&', 1), ('int8', 1, 2, '|', 3), ('uint8', 1, 2, '|', 3), ('int16', 1, 2, '|', 3), ('uint16', 1, 2, '|', 3), ('int32', 1, 2, '|', 3), ('uint32', 1, 2, '|', 3), ('int64', 1, 2, '|', 3), ('uint64', 1, 2, '|', 3), ('int8', 1, 3, '^', 2), ('uint8', 1, 3, '^', 2), ('int16', 1, 3, '^', 2), ('uint16', 1, 3, '^', 2), ('int32', 1, 3, '^', 2), ('uint32', 1, 3, '^', 2), ('int64', 1, 3, '^', 2), ('uint64', 1, 3, '^', 2), ('int8', 1, 3, '%', 1), ('uint8', 1, 3, '%', 1), ('int16', 1, 3, '%', 1), ('uint16', 1, 3, '%', 1), ('int32', 1, 3, '%', 1), ('uint32', 1, 3, '%', 1), ('int64', 1, 3, '%', 1), ('uint64', 1, 3, '%', 1), ('int8', 1, -3, '%', 1), ('uint8', 1, 255, '%', 1), ('int16', 1, -3, '%', 1), ('uint16', 1, 65535, '%', 1), ('int32', 1, -3, '%', 1), ('uint32', 1, 4294967295, '%', 1), ('int64', 1, -3, '%', 1), ('uint64', 1, 18446744073709551615, '%', 1), ('int8', 1, 2, '<<', 4), ('uint8', 1, 2, '<<', 4), ('int16', 1, 2, '<<', 4), ('uint16', 1, 2, '<<', 4), ('int32', 1, 2, '<<', 4), ('uint32', 1, 2, '<<', 4), ('int64', 1, 2, '<<', 4), ('uint64', 1, 2, '<<', 4), ('int8', 4, 1, '>>', 2), ('int8', -1, 1, '>>', -1), ('uint8', 255, 1, '>>', 127), ('int16', 4, 1, '>>', 2), ('int16', -1, 1, '>>', -1), ('uint16', 65535, 1, '>>', 32767), ('int32', 4, 1, '>>', 2), ('int32', -1, 1, '>>', -1), ('uint32', 4294967295, 1, '>>', 2147483647), ('int64', 4, 1, '>>', 2), ('int64', -1, 1, '>>', -1), ('uint64', 18446744073709551615, 1, '>>', 9223372036854775807), ('int64', 2, 2, '**', 4.0, 'double'), ('int16', -1, 1, '**', -1, 'double'), ('int32', -1, 1, '**', -1, 'double'), ('int64', -1, 1, '**', -1, 'double'), ('int64', -2, -3, '**', -0.125, 'double'), ('uint8', 255, 2, '**', float(255 * 255), 'double'), ('uint16', 65535, 2, '**', float(65535 * 65535), 'double'), ('uint32', 4294967295, 2, '**', float(4294967295 * 4294967295), 'double'), ('uint64', 18446744073709551615, 1, '**', float(18446744073709551615), 'double')]
        for (type, x, y, op, res, *output_type_option) in tests:
            if len(output_type_option) == 0:
                output_type = type
            else:
                output_type = output_type_option[0]
            codestr = f'\n            from __static__ import {type}, box\n            from __static__ import {output_type}\n            def testfunc(tst):\n                x: {type} = {x}\n                y: {type} = {y}\n                if tst:\n                    x = x + 1\n                    y = y + 2\n\n                z: {output_type} = x {op} y\n                return box(z), box(x {op} y)\n            '
            with self.subTest(type=type, x=x, y=y, op=op, res=res):
                with self.in_module(codestr) as mod:
                    f = mod.testfunc
                    self.assertEqual(f(False), (res, res), f'{type} {x} {op} {y} {res} {output_type}')

    def test_primitive_arithmetic(self):
        if False:
            for i in range(10):
                print('nop')
        cases = [('int8', 127, '*', 1, 127), ('int8', -64, '*', 2, -128), ('int8', 0, '*', 4, 0), ('uint8', 51, '*', 5, 255), ('uint8', 5, '*', 0, 0), ('int16', 3123, '*', -10, -31230), ('int16', -32767, '*', -1, 32767), ('int16', -32768, '*', 1, -32768), ('int16', 3, '*', 0, 0), ('uint16', 65535, '*', 1, 65535), ('uint16', 0, '*', 4, 0), ('int32', (1 << 31) - 1, '*', 1, (1 << 31) - 1), ('int32', -(1 << 30), '*', 2, -(1 << 31)), ('int32', 0, '*', 1, 0), ('uint32', (1 << 32) - 1, '*', 1, (1 << 32) - 1), ('uint32', 0, '*', 4, 0), ('int64', (1 << 63) - 1, '*', 1, (1 << 63) - 1), ('int64', -(1 << 62), '*', 2, -(1 << 63)), ('int64', 0, '*', 1, 0), ('uint64', (1 << 64) - 1, '*', 1, (1 << 64) - 1), ('uint64', 0, '*', 4, 0), ('int8', 127, '//', 4, 31), ('int8', -128, '//', 4, -32), ('int8', 0, '//', 4, 0), ('uint8', 255, '//', 5, 51), ('uint8', 0, '//', 5, 0), ('int16', 32767, '//', -1000, -32), ('int16', -32768, '//', -1000, 32), ('int16', 0, '//', 4, 0), ('uint16', 65535, '//', 5, 13107), ('uint16', 0, '//', 4, 0), ('int32', (1 << 31) - 1, '//', (1 << 31) - 1, 1), ('int32', -(1 << 31), '//', 1, -(1 << 31)), ('int32', 0, '//', 1, 0), ('uint32', (1 << 32) - 1, '//', 500, 8589934), ('uint32', 0, '//', 4, 0), ('int64', (1 << 63) - 1, '//', 2, (1 << 62) - 1), ('int64', -(1 << 63), '//', 2, -(1 << 62)), ('int64', 0, '//', 1, 0), ('uint64', (1 << 64) - 1, '//', (1 << 64) - 1, 1), ('uint64', 0, '//', 4, 0), ('int8', 127, '%', 4, 3), ('int8', -128, '%', 4, 0), ('int8', 0, '%', 4, 0), ('uint8', 255, '%', 6, 3), ('uint8', 0, '%', 5, 0), ('int16', 32767, '%', -1000, 767), ('int16', -32768, '%', -1000, -768), ('int16', 0, '%', 4, 0), ('uint16', 65535, '%', 7, 1), ('uint16', 0, '%', 4, 0), ('int32', (1 << 31) - 1, '%', (1 << 31) - 1, 0), ('int32', -(1 << 31), '%', 1, 0), ('int32', 0, '%', 1, 0), ('uint32', (1 << 32) - 1, '%', 500, 295), ('uint32', 0, '%', 4, 0), ('int64', (1 << 63) - 1, '%', 2, 1), ('int64', -(1 << 63), '%', 2, 0), ('int64', 0, '%', 1, 0), ('uint64', (1 << 64) - 1, '%', (1 << 64) - 1, 0), ('uint64', 0, '%', 4, 0)]
        for (typ, a, op, b, res) in cases:
            for const in ['noconst', 'constfirst', 'constsecond']:
                if const == 'noconst':
                    codestr = f'\n                        from __static__ import {typ}\n\n                        def f(a: {typ}, b: {typ}) -> {typ}:\n                            return a {op} b\n                    '
                elif const == 'constfirst':
                    codestr = f'\n                        from __static__ import {typ}\n\n                        def f(b: {typ}) -> {typ}:\n                            return {a} {op} b\n                    '
                elif const == 'constsecond':
                    codestr = f'\n                        from __static__ import {typ}\n\n                        def f(a: {typ}) -> {typ}:\n                            return a {op} {b}\n                    '
                with self.subTest(typ=typ, a=a, op=op, b=b, res=res, const=const):
                    with self.in_module(codestr) as mod:
                        f = mod.f
                        act = None
                        if const == 'noconst':
                            act = f(a, b)
                        elif const == 'constfirst':
                            act = f(b)
                        elif const == 'constsecond':
                            act = f(a)
                        self.assertEqual(act, res)

    def test_int_binop_type_context(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = f'\n            from __static__ import box, int8, int16\n\n            def f(x: int8, y: int8) -> int:\n                z: int16 = x * y\n                return box(z)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertInBytecode(f, 'CONVERT_PRIMITIVE', TYPED_INT8 | TYPED_INT16 << 4)
            self.assertEqual(f(120, 120), 14400)

    def test_mixed_binop(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(TypedSyntaxError, 'cannot add int64 and Literal\\[1\\]'):
            self.bind_module('\n                from __static__ import ssize_t\n\n                def f():\n                    x: ssize_t = 1\n                    y = 1\n                    x + y\n            ')
        with self.assertRaisesRegex(TypedSyntaxError, 'cannot add Literal\\[1\\] and int64'):
            self.bind_module('\n                from __static__ import ssize_t\n\n                def f():\n                    x: ssize_t = 1\n                    y = 1\n                    y + x\n            ')

    def test_mixed_binop_okay(self):
        if False:
            return 10
        codestr = '\n            from __static__ import ssize_t, box\n\n            def f():\n                x: ssize_t = 1\n                y = x + 1\n                return box(y)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(), 2)

    def test_mixed_binop_okay_1(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__ import ssize_t, box\n\n            def f():\n                x: ssize_t = 1\n                y = 1 + x\n                return box(y)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(), 2)

    def test_inferred_primitive_type(self):
        if False:
            while True:
                i = 10
        codestr = '\n        from __static__ import ssize_t, box\n\n        def f():\n            x: ssize_t = 1\n            y = x\n            return box(y)\n        '
        with self.in_module(codestr) as mod:
            f = mod.f
            self.assertEqual(f(), 1)

    def test_mixed_binop_sign(self):
        if False:
            while True:
                i = 10
        'mixed signed/unsigned ops should be promoted to signed'
        codestr = '\n            from __static__ import int8, uint8, box\n            def testfunc():\n                x: uint8 = 42\n                y: int8 = 2\n                return box(x / y)\n        '
        code = self.compile(codestr)
        f = self.find_code(code)
        self.assertInBytecode(f, 'PRIMITIVE_BINARY_OP', PRIM_OP_DIV_INT)
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            self.assertEqual(f(), 21)
        codestr = '\n            from __static__ import int8, uint8, box\n            def testfunc():\n                x: int8 = 42\n                y: uint8 = 2\n                return box(x / y)\n        '
        code = self.compile(codestr)
        f = self.find_code(code)
        self.assertInBytecode(f, 'PRIMITIVE_BINARY_OP', PRIM_OP_DIV_INT)
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            self.assertEqual(f(), 21)
        codestr = '\n            from __static__ import uint32, box\n            def testfunc():\n                x: uint32 = 2\n                a = box(x / -2)\n                return box(x ** -2)\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            self.assertEqual(f(), 0.25)
        codestr = '\n            from __static__ import int32, box\n            def testfunc():\n                x: int32 = 2\n                return box(x ** -2)\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            self.assertEqual(f(), 0.25)
        codestr = '\n            from __static__ import uint32, box\n            def testfunc():\n                x: uint32 = 2\n                return box(x ** -2)\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            self.assertEqual(f(), 0.25)
        codestr = '\n            from __static__ import int8, uint8, box\n            def testfunc():\n                x: int8 = 4\n                y: uint8 = 2\n                return box(x ** y)\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'cannot pow int8 and uint8'):
            self.compile(codestr)
        codestr = '\n            from __static__ import int8, uint8, box\n            def testfunc():\n                x: uint8 = 2\n                y: int8 = -3\n                return box(x ** y)\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'cannot pow uint8 and int8'):
            self.compile(codestr)
        codestr = '\n            from __static__ import uint8, box, double\n            def testfunc():\n                x: uint8 = 2\n                y: double = -3.0\n                return box(x ** y)\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'cannot pow uint8 and double'):
            self.compile(codestr)

    def test_double_binop(self):
        if False:
            for i in range(10):
                print('nop')
        tests = [(1.732, 2.0, '+', 3.732), (1.732, 2.0, '-', -0.268), (1.732, 2.0, '/', 0.866), (1.732, 2.0, '*', 3.464), (1.732, 2, '+', 3.732), (2.5, 2, '**', 6.25), (2.5, 2.5, '**', 9.882117688026186)]
        if cinderjit is not None:
            tests.append((1.732, 0.0, '/', float('inf')))
        for (x, y, op, res) in tests:
            codestr = f'\n            from __static__ import double, box\n            def testfunc(tst):\n                x: double = {x}\n                y: double = {y}\n\n                z: double = x {op} y\n                return box(z)\n            '
            with self.subTest(type=type, x=x, y=y, op=op, res=res):
                with self.in_module(codestr) as mod:
                    f = mod.testfunc
                    self.assertEqual(f(False), res, f'{type} {x} {op} {y} {res}')

    def test_double_sub_with_reg_pressure(self):
        if False:
            while True:
                i = 10
        "\n        Test the behavior of double subtraction under register pressure:\n        we had one bug where a rewrite rule inserted an invalid instruction,\n        and another where the register allocator didn't keep all inputs to the\n        Fsub instruction alive long enough.\n        "
        codestr = f'\n        from __static__ import box, double\n\n        def testfunc(f0: double, f1: double) -> double:\n            f2 = f0 + f1\n            f3 = f1 + f2\n            f4 = f2 + f3\n            f5 = f3 + f4\n            f6 = f4 + f5\n            f7 = f5 + f6\n            f8 = f6 + f7\n            f9 = f7 + f8\n            f10 = f8 + f9\n            f11 = f9 + f10\n            f12 = f10 + f11\n            f13 = f11 + f12\n            f14 = f12 + f13\n            f15 = f13 + f14\n            f16 = f1 - f0\n            return (\n                f1\n                + f2\n                + f3\n                + f4\n                + f5\n                + f6\n                + f7\n                + f8\n                + f9\n                + f10\n                + f11\n                + f12\n                + f13\n                + f14\n                + f15\n                + f16\n            )\n        '
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            self.assertEqual(f(1.0, 2.0), 4179.0)

    def test_double_binop_with_literal(self):
        if False:
            return 10
        codestr = f'\n            from __static__ import double, unbox\n\n            def f():\n                y: double = 1.2\n                y + 1.0\n        '
        f = self.run_code(codestr)['f']
        f()

    def test_subclass_binop(self):
        if False:
            while True:
                i = 10
        codestr = '\n            class C: pass\n            class D(C): pass\n\n            def f(x: C, y: D):\n                return x + y\n        '
        code = self.compile(codestr, modname='foo')
        f = self.find_code(code, 'f')
        self.assertInBytecode(f, 'BINARY_ADD')

    def test_mixed_add_reversed(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            from __static__ import int8, uint8, int64, box, int16\n            def testfunc(tst=False):\n                x: int8 = 42\n                y: int16 = 2\n                if tst:\n                    x += 1\n                    y += 1\n\n                return box(y + x)\n        '
        code = self.compile(codestr)
        f = self.find_code(code)
        self.assertInBytecode(f, 'PRIMITIVE_BINARY_OP', PRIM_OP_ADD_INT)
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            self.assertEqual(f(), 44)

    def test_mixed_tri_add(self):
        if False:
            while True:
                i = 10
        codestr = '\n            from __static__ import int8, uint8, int64, box\n            def testfunc(tst=False):\n                x: uint8 = 42\n                y: int8 = 2\n                z: int64 = 3\n                if tst:\n                    x += 1\n                    y += 1\n\n                return box(x + y + z)\n        '
        code = self.compile(codestr)
        f = self.find_code(code)
        self.assertInBytecode(f, 'PRIMITIVE_BINARY_OP', PRIM_OP_ADD_INT)
        with self.in_module(codestr) as mod:
            f = mod.testfunc
            self.assertEqual(f(), 47)

    def test_mixed_tri_add_unsigned(self):
        if False:
            i = 10
            return i + 15
        "promote int/uint to int, can't add to uint64"
        codestr = '\n            from __static__ import int8, uint8, uint64, box\n            def testfunc(tst=False):\n                x: uint8 = 42\n                y: int8 = 2\n                z: uint64 = 3\n\n                return box(x + y + z)\n        '
        with self.assertRaisesRegex(TypedSyntaxError, 'cannot add int16 and uint64'):
            self.compile(codestr)

    def test_literal_int_binop_inferred_type(self):
        if False:
            return 10
        "primitive literal doesn't wrongly carry through arithmetic"
        for rev in [False, True]:
            with self.subTest(rev=rev):
                op = '1 + x' if rev else 'x + 1'
                codestr = f'\n                    from __static__ import int64\n\n                    def f(x: int64):\n                        reveal_type({op})\n                '
                self.type_error(codestr, "'int64'", f'reveal_type({op})')

    def test_error_type_ctx_left_operand_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = f'\n            from __static__ import int64\n\n            def f(k: int64):\n                l = [1, 2, 3]\n                # slices cannot be primitives, so this is invalid\n                l[:k + 1] = [0]\n                return l\n        '
        self.type_error(codestr, 'int64 cannot be assigned to dynamic', f'k + 1')