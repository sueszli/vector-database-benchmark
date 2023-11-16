import unittest
import cython
from Cython.Compiler.Main import CompileError
from Cython.Build.Inline import cython_inline
import re
import sys
if cython.compiled:
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO

    class StdErrHider:

        def __enter__(self):
            if False:
                print('Hello World!')
            self.old_stderr = sys.stderr
            self.new_stderr = StringIO()
            sys.stderr = self.new_stderr
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if False:
                print('Hello World!')
            sys.stderr = self.old_stderr

        @property
        def stderr_contents(self):
            if False:
                i = 10
                return i + 15
            return self.new_stderr.getvalue()

    def exec(code, globals_=None, locals_=None):
        if False:
            while True:
                i = 10
        if locals_ and globals_ and (locals_ is not globals_):
            code = 'class Cls:\n' + '\n'.join(('    ' + line for line in code.split('\n')))
        code += '\nreturn globals(), locals()'
        try:
            with StdErrHider() as stderr_handler:
                try:
                    (g, l) = cython_inline(code, globals=globals_, locals=locals_)
                finally:
                    err_messages = stderr_handler.stderr_contents
            if globals_ is not None:
                globals_.update(l)
                globals_.update(g)
        except CompileError as exc:
            raised_message = str(exc)
            if raised_message.endswith('.pyx'):
                raised_message = []
                for line in err_messages.split('\n'):
                    match = re.match('(.+?):\\d+:\\d+:(.*)', line)
                    if match and match.group(1).endswith('.pyx'):
                        raised_message.append(match.group(2))
                raised_message = '; '.join(raised_message)
            raise SyntaxError(raised_message) from None

class NamedExpressionInvalidTest(unittest.TestCase):

    def test_named_expression_invalid_01(self):
        if False:
            print('Hello World!')
        code = 'x := 0'
        with self.assertRaisesRegex(SyntaxError, 'invalid syntax'):
            exec(code, {}, {})

    def test_named_expression_invalid_02(self):
        if False:
            for i in range(10):
                print('nop')
        code = 'x = y := 0'
        with self.assertRaisesRegex(SyntaxError, 'invalid syntax'):
            exec(code, {}, {})

    def test_named_expression_invalid_03(self):
        if False:
            i = 10
            return i + 15
        code = 'y := f(x)'
        with self.assertRaisesRegex(SyntaxError, 'invalid syntax'):
            exec(code, {}, {})

    def test_named_expression_invalid_04(self):
        if False:
            while True:
                i = 10
        code = 'y0 = y1 := f(x)'
        with self.assertRaisesRegex(SyntaxError, 'invalid syntax'):
            exec(code, {}, {})

    def test_named_expression_invalid_06(self):
        if False:
            i = 10
            return i + 15
        code = '((a, b) := (1, 2))'
        with self.assertRaisesRegex(SyntaxError, ''):
            exec(code, {}, {})

    def test_named_expression_invalid_07(self):
        if False:
            print('Hello World!')
        code = 'def spam(a = b := 42): pass'
        with self.assertRaisesRegex(SyntaxError, 'invalid syntax'):
            exec(code, {}, {})

    def test_named_expression_invalid_08(self):
        if False:
            print('Hello World!')
        code = 'def spam(a: b := 42 = 5): pass'
        with self.assertRaisesRegex(SyntaxError, 'invalid syntax'):
            exec(code, {}, {})

    def test_named_expression_invalid_09(self):
        if False:
            for i in range(10):
                print('nop')
        code = "spam(a=b := 'c')"
        with self.assertRaisesRegex(SyntaxError, 'invalid syntax'):
            exec(code, {}, {})

    def test_named_expression_invalid_10(self):
        if False:
            return 10
        code = 'spam(x = y := f(x))'
        with self.assertRaisesRegex(SyntaxError, 'invalid syntax'):
            exec(code, {}, {})

    def test_named_expression_invalid_11(self):
        if False:
            i = 10
            return i + 15
        code = 'spam(a=1, b := 2)'
        with self.assertRaisesRegex(SyntaxError, 'follow.* keyword arg'):
            exec(code, {}, {})

    def test_named_expression_invalid_12(self):
        if False:
            i = 10
            return i + 15
        code = 'spam(a=1, (b := 2))'
        with self.assertRaisesRegex(SyntaxError, 'follow.* keyword arg'):
            exec(code, {}, {})

    def test_named_expression_invalid_13(self):
        if False:
            while True:
                i = 10
        code = 'spam(a=1, (b := 2))'
        with self.assertRaisesRegex(SyntaxError, 'follow.* keyword arg'):
            exec(code, {}, {})

    def test_named_expression_invalid_14(self):
        if False:
            while True:
                i = 10
        code = '(x := lambda: y := 1)'
        with self.assertRaisesRegex(SyntaxError, 'invalid syntax'):
            exec(code, {}, {})

    def test_named_expression_invalid_15(self):
        if False:
            return 10
        code = '(lambda: x := 1)'
        with self.assertRaisesRegex(SyntaxError, ''):
            exec(code, {}, {})

    def test_named_expression_invalid_16(self):
        if False:
            for i in range(10):
                print('nop')
        code = '[i + 1 for i in i := [1,2]]'
        with self.assertRaisesRegex(SyntaxError, ''):
            exec(code, {}, {})

    def test_named_expression_invalid_17(self):
        if False:
            return 10
        code = '[i := 0, j := 1 for i, j in [(1, 2), (3, 4)]]'
        with self.assertRaisesRegex(SyntaxError, ''):
            exec(code, {}, {})

    def test_named_expression_invalid_in_class_body(self):
        if False:
            while True:
                i = 10
        code = 'class Foo():\n            [(42, 1 + ((( j := i )))) for i in range(5)]\n        '
        with self.assertRaisesRegex(SyntaxError, 'assignment expression within a comprehension cannot be used in a class body'):
            exec(code, {}, {})

    def test_named_expression_invalid_rebinding_comprehension_iteration_variable(self):
        if False:
            for i in range(10):
                print('nop')
        cases = [('Local reuse', 'i', '[i := 0 for i in range(5)]'), ('Nested reuse', 'j', '[[(j := 0) for i in range(5)] for j in range(5)]'), ('Reuse inner loop target', 'j', '[(j := 0) for i in range(5) for j in range(5)]'), ('Unpacking reuse', 'i', '[i := 0 for i, j in [(0, 1)]]'), ('Reuse in loop condition', 'i', '[i+1 for i in range(5) if (i := 0)]'), ('Unreachable reuse', 'i', '[False or (i:=0) for i in range(5)]'), ('Unreachable nested reuse', 'i', '[(i, j) for i in range(5) for j in range(5) if True or (i:=10)]')]
        for (case, target, code) in cases:
            msg = f"assignment expression cannot rebind comprehension iteration variable '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {})

    def test_named_expression_invalid_rebinding_comprehension_inner_loop(self):
        if False:
            while True:
                i = 10
        cases = [('Inner reuse', 'j', '[i for i in range(5) if (j := 0) for j in range(5)]'), ('Inner unpacking reuse', 'j', '[i for i in range(5) if (j := 0) for j, k in [(0, 1)]]')]
        for (case, target, code) in cases:
            msg = f"comprehension inner loop cannot rebind assignment expression target '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {})
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {})
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f'lambda: {code}', {})

    def test_named_expression_invalid_comprehension_iterable_expression(self):
        if False:
            for i in range(10):
                print('nop')
        cases = [('Top level', '[i for i in (i := range(5))]'), ('Inside tuple', '[i for i in (2, 3, i := range(5))]'), ('Inside list', '[i for i in [2, 3, i := range(5)]]'), ('Different name', '[i for i in (j := range(5))]'), ('Lambda expression', '[i for i in (lambda:(j := range(5)))()]'), ('Inner loop', '[i for i in range(5) for j in (i := range(5))]'), ('Nested comprehension', '[i for i in [j for j in (k := range(5))]]'), ('Nested comprehension condition', '[i for i in [j for j in range(5) if (j := True)]]'), ('Nested comprehension body', '[i for i in [(j := True) for j in range(5)]]')]
        msg = 'assignment expression cannot be used in a comprehension iterable expression'
        for (case, code) in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {})
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {})
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f'lambda: {code}', {})

class NamedExpressionAssignmentTest(unittest.TestCase):

    def test_named_expression_assignment_01(self):
        if False:
            for i in range(10):
                print('nop')
        (a := 10)
        self.assertEqual(a, 10)

    def test_named_expression_assignment_02(self):
        if False:
            print('Hello World!')
        a = 20
        (a := a)
        self.assertEqual(a, 20)

    def test_named_expression_assignment_03(self):
        if False:
            return 10
        (total := (1 + 2))
        self.assertEqual(total, 3)

    def test_named_expression_assignment_04(self):
        if False:
            print('Hello World!')
        (info := (1, 2, 3))
        self.assertEqual(info, (1, 2, 3))

    def test_named_expression_assignment_05(self):
        if False:
            while True:
                i = 10
        ((x := 1), 2)
        self.assertEqual(x, 1)

    def test_named_expression_assignment_06(self):
        if False:
            print('Hello World!')
        (z := (y := (x := 0)))
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_named_expression_assignment_07(self):
        if False:
            for i in range(10):
                print('nop')
        (loc := (1, 2))
        self.assertEqual(loc, (1, 2))

    def test_named_expression_assignment_08(self):
        if False:
            return 10
        if (spam := 'eggs'):
            self.assertEqual(spam, 'eggs')
        else:
            self.fail('variable was not assigned using named expression')

    def test_named_expression_assignment_09(self):
        if False:
            print('Hello World!')
        if True and (spam := True):
            self.assertTrue(spam)
        else:
            self.fail('variable was not assigned using named expression')

    def test_named_expression_assignment_10(self):
        if False:
            while True:
                i = 10
        if (match := 10) == 10:
            pass
        else:
            self.fail('variable was not assigned using named expression')

    def test_named_expression_assignment_11(self):
        if False:
            for i in range(10):
                print('nop')

        def spam(a):
            if False:
                i = 10
                return i + 15
            return a
        input_data = [1, 2, 3]
        res = [(x, y, x / y) for x in input_data if (y := spam(x)) > 0]
        self.assertEqual(res, [(1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)])

    def test_named_expression_assignment_12(self):
        if False:
            i = 10
            return i + 15

        def spam(a):
            if False:
                print('Hello World!')
            return a
        res = [[(y := spam(x)), x / y] for x in range(1, 5)]
        self.assertEqual(res, [[1, 1.0], [2, 1.0], [3, 1.0], [4, 1.0]])

    def test_named_expression_assignment_13(self):
        if False:
            return 10
        length = len((lines := [1, 2]))
        self.assertEqual(length, 2)
        self.assertEqual(lines, [1, 2])

    def test_named_expression_assignment_14(self):
        if False:
            return 10
        "\n        Where all variables are positive integers, and a is at least as large\n        as the n'th root of x, this algorithm returns the floor of the n'th\n        root of x (and roughly doubling the number of accurate bits per\n        iteration):\n        "
        a = 9
        n = 2
        x = 3
        while a > (d := (x // a ** (n - 1))):
            a = ((n - 1) * a + d) // n
        self.assertEqual(a, 1)

    def test_named_expression_assignment_15(self):
        if False:
            while True:
                i = 10
        while (a := False):
            pass
        self.assertEqual(a, False)

    def test_named_expression_assignment_16(self):
        if False:
            i = 10
            return i + 15
        (a, b) = (1, 2)
        fib = {(c := a): (a := b) + (b := (a + c)) - b for __ in range(6)}
        self.assertEqual(fib, {1: 2, 2: 3, 3: 5, 5: 8, 8: 13, 13: 21})

class NamedExpressionScopeTest(unittest.TestCase):

    def test_named_expression_scope_01(self):
        if False:
            i = 10
            return i + 15
        code = 'def spam():\n    (a := 5)\nprint(a)'
        with self.assertRaisesRegex(SyntaxError if cython.compiled else NameError, ''):
            exec(code, {}, {})

    def test_named_expression_scope_02(self):
        if False:
            i = 10
            return i + 15
        total = 0
        partial_sums = [(total := (total + v)) for v in range(5)]
        self.assertEqual(partial_sums, [0, 1, 3, 6, 10])
        self.assertEqual(total, 10)

    def test_named_expression_scope_03(self):
        if False:
            while True:
                i = 10
        containsOne = any(((lastNum := num) == 1 for num in [1, 2, 3]))
        self.assertTrue(containsOne)
        self.assertEqual(lastNum, 1)

    def test_named_expression_scope_04(self):
        if False:
            while True:
                i = 10

        def spam(a):
            if False:
                i = 10
                return i + 15
            return a
        res = [[(y := spam(x)), x / y] for x in range(1, 5)]
        self.assertEqual(y, 4)

    def test_named_expression_scope_05(self):
        if False:
            for i in range(10):
                print('nop')

        def spam(a):
            if False:
                print('Hello World!')
            return a
        input_data = [1, 2, 3]
        res = [(x, y, x / y) for x in input_data if (y := spam(x)) > 0]
        self.assertEqual(res, [(1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)])
        self.assertEqual(y, 3)

    def test_named_expression_scope_06(self):
        if False:
            for i in range(10):
                print('nop')
        res = [[(spam := i) for i in range(3)] for j in range(2)]
        self.assertEqual(res, [[0, 1, 2], [0, 1, 2]])
        self.assertEqual(spam, 2)

    def test_named_expression_scope_07(self):
        if False:
            print('Hello World!')
        len((lines := [1, 2]))
        self.assertEqual(lines, [1, 2])

    def test_named_expression_scope_08(self):
        if False:
            while True:
                i = 10

        def spam(a):
            if False:
                i = 10
                return i + 15
            return a

        def eggs(b):
            if False:
                while True:
                    i = 10
            return b * 2
        res = [spam((a := eggs((b := h)))) for h in range(2)]
        self.assertEqual(res, [0, 2])
        self.assertEqual(a, 2)
        self.assertEqual(b, 1)

    def test_named_expression_scope_09(self):
        if False:
            print('Hello World!')

        def spam(a):
            if False:
                for i in range(10):
                    print('nop')
            return a

        def eggs(b):
            if False:
                for i in range(10):
                    print('nop')
            return b * 2
        res = [spam((a := eggs((a := h)))) for h in range(2)]
        self.assertEqual(res, [0, 2])
        self.assertEqual(a, 2)

    def test_named_expression_scope_10(self):
        if False:
            while True:
                i = 10
        res = [(b := [(a := 1) for i in range(2)]) for j in range(2)]
        self.assertEqual(res, [[1, 1], [1, 1]])
        self.assertEqual(a, 1)
        self.assertEqual(b, [1, 1])

    def test_named_expression_scope_11(self):
        if False:
            i = 10
            return i + 15
        res = [(j := i) for i in range(5)]
        self.assertEqual(res, [0, 1, 2, 3, 4])
        self.assertEqual(j, 4)

    def test_named_expression_scope_17(self):
        if False:
            i = 10
            return i + 15
        b = 0
        res = [(b := (i + b)) for i in range(5)]
        self.assertEqual(res, [0, 1, 3, 6, 10])
        self.assertEqual(b, 10)

    def test_named_expression_scope_18(self):
        if False:
            print('Hello World!')

        def spam(a):
            if False:
                while True:
                    i = 10
            return a
        res = spam((b := 2))
        self.assertEqual(res, 2)
        self.assertEqual(b, 2)

    def test_named_expression_scope_19(self):
        if False:
            print('Hello World!')

        def spam(a):
            if False:
                return 10
            return a
        res = spam((b := 2))
        self.assertEqual(res, 2)
        self.assertEqual(b, 2)

    def test_named_expression_scope_20(self):
        if False:
            while True:
                i = 10

        def spam(a):
            if False:
                while True:
                    i = 10
            return a
        res = spam(a=(b := 2))
        self.assertEqual(res, 2)
        self.assertEqual(b, 2)

    def test_named_expression_scope_21(self):
        if False:
            for i in range(10):
                print('nop')

        def spam(a, b):
            if False:
                return 10
            return a + b
        res = spam((c := 2), b=1)
        self.assertEqual(res, 3)
        self.assertEqual(c, 2)

    def test_named_expression_scope_22(self):
        if False:
            while True:
                i = 10

        def spam(a, b):
            if False:
                print('Hello World!')
            return a + b
        res = spam((c := 2), b=1)
        self.assertEqual(res, 3)
        self.assertEqual(c, 2)

    def test_named_expression_scope_23(self):
        if False:
            return 10

        def spam(a, b):
            if False:
                return 10
            return a + b
        res = spam(b=(c := 2), a=1)
        self.assertEqual(res, 3)
        self.assertEqual(c, 2)

    def test_named_expression_scope_24(self):
        if False:
            while True:
                i = 10
        a = 10

        def spam():
            if False:
                i = 10
                return i + 15
            nonlocal a
            (a := 20)
        spam()
        self.assertEqual(a, 20)

    def test_named_expression_scope_25(self):
        if False:
            for i in range(10):
                print('nop')
        ns = {}
        code = 'a = 10\ndef spam():\n    global a\n    (a := 20)\nspam()'
        exec(code, ns, {})
        self.assertEqual(ns['a'], 20)

    def test_named_expression_variable_reuse_in_comprehensions(self):
        if False:
            return 10
        rebinding = '[x := i for i in range(3) if (x := i) or not x]'
        filter_ref = '[x := i for i in range(3) if x or not x]'
        body_ref = '[x for i in range(3) if (x := i) or not x]'
        nested_ref = '[j for i in range(3) if x or not x for j in range(3) if (x := i)][:-3]'
        cases = [('Rebind global', f'x = 1; result = {rebinding}'), ('Rebind nonlocal', f'result, x = (lambda x=1: ({rebinding}, x))()'), ('Filter global', f'x = 1; result = {filter_ref}'), ('Filter nonlocal', f'result, x = (lambda x=1: ({filter_ref}, x))()'), ('Body global', f'x = 1; result = {body_ref}'), ('Body nonlocal', f'result, x = (lambda x=1: ({body_ref}, x))()'), ('Nested global', f'x = 1; result = {nested_ref}'), ('Nested nonlocal', f'result, x = (lambda x=1: ({nested_ref}, x))()')]
        for (case, code) in cases:
            with self.subTest(case=case):
                ns = {}
                exec(code, ns)
                self.assertEqual(ns['x'], 2)
                self.assertEqual(ns['result'], [0, 1, 2])
if __name__ == '__main__':
    unittest.main()