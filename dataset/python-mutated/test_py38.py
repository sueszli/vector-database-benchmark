import io
from compiler.dis_stable import Disassembler
from textwrap import dedent
from .common import CompilerTest

def dump_code(code):
    if False:
        while True:
            i = 10
    f = io.StringIO()
    Disassembler().dump_code(code, file=f)
    text = f.getvalue()
    return text

class Python38Tests(CompilerTest):
    maxDiff = None

    def _check(self, src, optimize=-1):
        if False:
            i = 10
            return i + 15
        src = dedent(src).strip()
        actual = dump_code(self.compile(src, optimize=optimize))
        expected = dump_code(compile(src, '', mode='exec', optimize=optimize))
        self.assertEqual(actual, expected)

    def test_sanity(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'basic test that the compiler can generate a function'
        code = self.compile('f()')
        self.assertInBytecode(code, 'CALL_FUNCTION')
        self.assertEqual(code.co_posonlyargcount, 0)

    def test_walrus_if(self):
        if False:
            return 10
        code = self.compile('if x:= y: pass')
        self.assertInBytecode(code, 'STORE_NAME', 'x')

    def test_walrus_call(self):
        if False:
            for i in range(10):
                print('nop')
        code = self.compile('f(x:= y)')
        self.assertInBytecode(code, 'STORE_NAME', 'x')

    def test_while_codegen(self) -> None:
        if False:
            print('Hello World!')
        source = '\n            def f(l):\n                while x:\n                    if y: pass\n        '
        self._check(source)

    def test_for_codegen(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        source = '\n            def f(l):\n                for i, j in l: pass\n        '
        self._check(source)

    def test_async_for_codegen(self) -> None:
        if False:
            return 10
        source = '\n            async def f(l):\n                async for i, j in l: pass\n        '
        self._check(source)

    def test_continue(self) -> None:
        if False:
            print('Hello World!')
        source = '\n            while x:\n                if y: continue\n                print(1)\n        '
        self._check(source)
        source = '\n            for y in l:\n                if y: continue\n                print(1)\n        '
        self._check(source)

    def test_break(self) -> None:
        if False:
            print('Hello World!')
        source = '\n            while x:\n                if y: break\n                print(1)\n        '
        self._check(source)
        source = '\n            for y in l:\n                if y: break\n                print(1)\n        '
        self._check(source)

    def test_try_finally(self):
        if False:
            i = 10
            return i + 15
        source = '\n            def g():\n                for i in l:\n                    try:\n                        if i > 0:\n                            continue\n                        if i == 3:\n                            break\n                        if i == 5:\n                            return\n                    except:\n                        pass\n                    finally: pass\n            '
        self._check(source)
        source = '\n            def g():\n                for i in l:\n                    try:\n                        if i == 5:\n                            return\n                    except BaseException as e:\n                        print(1)\n                    except:\n                        pass\n                    finally:\n                        continue\n            '
        self._check(source)

    def test_async_comprehension(self) -> None:
        if False:
            i = 10
            return i + 15
        source = '\n            async def f():\n                async for z in (x async for x in y if x > 10):\n                    yield z\n        '
        self._check(source)
        source = '\n            async def f():\n                x = 1\n                return [x async for x in y if x > 10]\n        '
        self._check(source)
        source = '\n            async def f():\n                x = 1\n                return {x async for x in y if x > 10}\n        '
        self._check(source)
        source = '\n            async def f():\n                x = 1\n                return {x:str(x) async for x in y if x > 10}\n        '
        self._check(source)

    def test_try_except(self):
        if False:
            return 10
        source = '\n            try:\n                f()\n            except:\n                g()\n        '
        self._check(source)
        source = '\n            try:\n                f()\n            except BaseException:\n                g(e)\n        '
        self._check(source)
        source = '\n            try:\n                f()\n            except BaseException as e:\n                g(e)\n        '
        self._check(source)

    def test_with(self):
        if False:
            while True:
                i = 10
        source = '\n            with foo():\n                pass\n        '
        self._check(source)
        source = '\n            with foo() as f:\n                pass\n        '
        self._check(source)
        source = '\n            with foo() as f, bar() as b, baz():\n                pass\n        '
        self._check(source)

    def test_async_with(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\n            async def f():\n                async with foo():\n                    pass\n        '
        self._check(source)
        source = '\n            async def g():\n                async with foo() as f:\n                    pass\n        '
        self._check(source)
        source = '\n            async def g():\n                async with foo() as f, bar() as b, baz():\n                    pass\n        '
        self._check(source)

    def test_constants(self):
        if False:
            while True:
                i = 10
        source = '\n            # formerly ast.Num\n            i = 1\n            f = 1.1\n            c = 1j\n            # formerly ast.Str\n            s = "foo"\n            b = b"foo"\n            # formerly ast.Ellipsis\n            e = ...\n            # formerly ast.NameConstant\n            t = True\n            f = False\n            n = None\n        '
        self._check(source)

    def test_key_value_order(self):
        if False:
            for i in range(10):
                print('nop')
        source = "\norder = []\ndef f(place, val):\n    order.append((place, val))\n\n{f('key', k): f('val', v) for k, v in zip('abc', [1, 2, 3])}\n        "
        self._check(source)

    def test_return(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\ndef f():\n    return 1\n        '
        self._check(source)
        source = '\ndef f():\n    return\n        '
        self._check(source)
        source = '\ndef f():\n    return x\n        '
        self._check(source)
        source = '\ndef f():\n    try:\n        return 1\n    finally:\n        print(1)\n        '
        self._check(source)

    def test_break_continue_in_finally(self):
        if False:
            return 10
        source = "\n            def test_break_in_finally_after_return(self):\n                # See issue #37830\n                def g1(x):\n                    for count in [0, 1]:\n                        count2 = 0\n                        while count2 < 20:\n                            count2 += 10\n                            try:\n                                return count + count2\n                            finally:\n                                if x:\n                                    break\n                    return 'end', count, count2\n                self.assertEqual(g1(False), 10)\n                self.assertEqual(g1(True), ('end', 1, 10))\n\n                def g2(x):\n                    for count in [0, 1]:\n                        for count2 in [10, 20]:\n                            try:\n                                return count + count2\n                            finally:\n                                if x:\n                                    break\n                    return 'end', count, count2\n\n            def test_continue_in_finally_after_return(self):\n                # See issue #37830\n                def g1(x):\n                    count = 0\n                    while count < 100:\n                        count += 1\n                        try:\n                            return count\n                        finally:\n                            if x:\n                                continue\n                    return 'end', count\n\n                def g2(x):\n                    for count in [0, 1]:\n                        try:\n                            return count\n                        finally:\n                            if x:\n                                continue\n                    return 'end', count\n        "
        self._check(source)

    def test_continue_in_finally(self):
        if False:
            i = 10
            return i + 15
        source = '\n            def test_continue_in_finally(self):\n                count = 0\n                while count < 2:\n                    count += 1\n                    try:\n                        pass\n                    finally:\n                        continue\n                    break\n\n                count = 0\n                while count < 2:\n                    count += 1\n                    try:\n                        break\n                    finally:\n                        continue\n\n                count = 0\n                while count < 2:\n                    count += 1\n                    try:\n                        1/0\n                    finally:\n                        continue\n                    break\n\n                for count in [0, 1]:\n                    try:\n                        pass\n                    finally:\n                        continue\n                    break\n\n                for count in [0, 1]:\n                    try:\n                        break\n                    finally:\n                        continue\n\n                for count in [0, 1]:\n                    try:\n                        1/0\n                    finally:\n                        continue\n                    break\n        '
        self._check(source)

    def test_asyncgen(self):
        if False:
            for i in range(10):
                print('nop')
        source = '\n            async def f(it):\n                for i in it:\n                    yield i\n\n            async def run_list():\n                i  = 1\n                return [i + 10 async for i in f(range(5)) if 0 < i < 4]\n\n            async def run_set():\n                i  = 1\n                return {i + 10 async for i in f(range(5)) if 0 < i < 4}\n\n            async def run_dict():\n                i  = 1\n                return {i + 10: i + 100 async for i in f(range(5)) if 0 < i < 4}\n\n            async def run_gen():\n                g = 1\n                gen = (i + 10 async for i in f(range(5)) if 0 < i < 4)\n                return [g + 100 async for g in gen]\n        '
        self._check(source)

    def test_posonly_args(self):
        if False:
            print('Hello World!')
        code = self.compile('def f(a, /, b): pass')
        f = self.find_code(code)
        self.assertEqual(f.co_posonlyargcount, 1)
        self.assertEqual(f.co_argcount, 2)
        self.assertEqual(f.co_varnames, ('a', 'b'))

    def test_multiline_expr_line_nos(self):
        if False:
            i = 10
            return i + 15
        codestr = '\n            import traceback\n\n            def some_inner(k, v):\n                a = 1\n                b = 2\n                return traceback.StackSummary.extract(\n                    traceback.walk_stack(None), capture_locals=True, limit=1)\n        '
        self._check(codestr)

    def test_decorator_line_nos(self):
        if False:
            i = 10
            return i + 15
        dec_func = '\n            @a\n            @b\n            @c\n            def x():\n                pass\n        '
        self._check(dec_func)
        dec_class = '\n            @a\n            @b\n            @c\n            class C():\n                pass\n        '
        self._check(dec_class)
        dec_async_func = '\n            @a\n            @b\n            @c\n            async def x():\n                pass\n        '
        self._check(dec_async_func)

    def test_yield_outside_function_dead_code(self):
        if False:
            while True:
                i = 10
        'Yield syntax errors are still reported in dead code: bpo-37500.'
        cases = ['if 0: yield', 'class C:\n    if 0: yield', 'if 0: yield\nelse:  x=1', 'if 1: pass\nelse: yield', 'while 0: yield', 'while 0: yield\nelse:  x=1', 'class C:\n  if 0: yield', 'class C:\n  if 1: pass\n  else: yield', 'class C:\n  while 0: yield', 'class C:\n  while 0: yield\n  else:  x = 1']
        for case in cases:
            with self.subTest(case):
                with self.assertRaisesRegex(SyntaxError, 'outside function'):
                    self.compile(case)

    def test_return_outside_function_dead_code(self):
        if False:
            for i in range(10):
                print('nop')
        'Return syntax errors are still reported in dead code: bpo-37500.'
        cases = ['if 0: return', 'class C:\n    if 0: return', 'if 0: return\nelse:  x=1', 'if 1: pass\nelse: return', 'while 0: return', 'class C:\n  if 0: return', 'class C:\n  while 0: return', 'class C:\n  while 0: return\n  else:  x=1', 'class C:\n  if 0: return\n  else: x= 1', 'class C:\n  if 1: pass\n  else: return']
        for case in cases:
            with self.subTest(case):
                with self.assertRaisesRegex(SyntaxError, 'outside function'):
                    self.compile(case)

    def test_break_outside_loop_dead_code(self):
        if False:
            return 10
        'Break syntax errors are still reported in dead code: bpo-37500.'
        cases = ['if 0: break', 'if 0: break\nelse:  x=1', 'if 1: pass\nelse: break', 'class C:\n  if 0: break', 'class C:\n  if 1: pass\n  else: break']
        for case in cases:
            with self.subTest(case):
                with self.assertRaisesRegex(SyntaxError, 'outside loop'):
                    self.compile(case)

    def test_continue_outside_loop_dead_code(self):
        if False:
            while True:
                i = 10
        'Continue syntax errors are still reported in dead code: bpo-37500.'
        cases = ['if 0: continue', 'if 0: continue\nelse:  x=1', 'if 1: pass\nelse: continue', 'class C:\n  if 0: continue', 'class C:\n  if 1: pass\n  else: continue']
        for case in cases:
            with self.subTest(case):
                with self.assertRaisesRegex(SyntaxError, 'not properly in loop'):
                    self.compile(case)

    def test_jump_offsets(self):
        if False:
            while True:
                i = 10
        codestr = '\n        def f(a):\n            return g(i for i in x if i not in j)\n        '
        self._check(codestr)

    def test_jump_forward(self):
        if False:
            return 10
        codestr = '\n        def f():\n            if yes:\n                for c in a:\n                    print(c)\n            elif no:\n                for c in a.d():\n                    print(c)\n        '
        self._check(codestr)

    def test_with_setup(self):
        if False:
            return 10
        codestr = '\n        def _read_output(commandstring):\n            import contextlib\n            try:\n                import tempfile\n                fp = tempfile.NamedTemporaryFile()\n            except ImportError:\n                fp = open("/tmp/_osx_support.%s"%(\n                    os.getpid(),), "w+b")\n\n            with contextlib.closing(fp) as fp:\n                cmd = "%s 2>/dev/null >\'%s\'" % (commandstring, fp.name)\n                return fp.read().decode(\'utf-8\').strip() if not os.system(cmd) else None\n        '
        self._check(codestr)

    def test_try_finally_return(self):
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        def f():\n            try:\n                a\n            finally:\n                return 42\n        '
        self._check(codestr)

    def test_break_in_false_loop(self):
        if False:
            print('Hello World!')
        codestr = '\n        def break_in_while():\n            while False:\n                break\n        '
        self._check(codestr)

    def test_true_loop_lineno(self):
        if False:
            print('Hello World!')
        codestr = '\n            while True:\n                b\n        '
        self._check(codestr)

    def test_syntax_error_rebind_comp_iter_nonlocal(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(SyntaxError, "comprehension inner loop cannot rebind assignment expression target 'j'"):
            self.compile('[i for i in range(5) if (j := 0) for j in range(5)]')

    def test_syntax_error_rebind_comp_iter(self):
        if False:
            return 10
        with self.assertRaisesRegex(SyntaxError, "assignment expression cannot rebind comprehension iteration variable 'x'"):
            self.compile("[x:=42 for x in 'abc']")

    def test_syntax_error_assignment_expr_in_comp_iterable(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(SyntaxError, 'assignment expression cannot be used in a comprehension iterable expression'):
            self.compile("[x for x in (x:='abc')]")

    def test_syntax_error_assignment_expr_in_class_comp(self):
        if False:
            return 10
        code = "\n        class C:\n            [y:=42 for x in 'abc']\n        "
        with self.assertRaisesRegex(SyntaxError, 'assignment expression within a comprehension cannot be used in a class body'):
            self.compile(code)

    def test_future_annotated_assign_validation(self):
        if False:
            i = 10
            return i + 15
        code = '\n        from __future__ import annotations\n        def f(x):\n            self.y: int # this should be invalid\n            self.y = x\n        '
        self._check(code)

    def test_assert_with_opt_0(self):
        if False:
            print('Hello World!')
        code = '\n        def f(x):\n            if x > 1:\n                if x > 2:\n                    pass\n                else:\n                    assert x > 3\n            else:\n                x = 5\n        '
        self._check(code)

    def test_assert_with_opt_1(self):
        if False:
            return 10
        code = '\n        def f(x):\n            if x > 1:\n                if x > 2:\n                    pass\n                else:\n                    assert x > 3\n            else:\n                x = 5\n        '
        self._check(code, optimize=1)

    def test_unary_op_jump_folding(self):
        if False:
            for i in range(10):
                print('nop')
        code = '\n        def f(x):\n            return (not f(x) and x > 3) or x < 4\n        '
        self._check(code)

    def test_dunder_class_cellvar_in_nested(self):
        if False:
            i = 10
            return i + 15
        '\n        __class__ should not be cell since the functions are\n        not defined in a class\n        '
        code = '\n        def f(x):\n            def g(y):\n                def __new__(cls):\n                    return super(x, cls).__new__(cls)\n                y.__new__ = __new__\n            return g(x)\n        '
        self._check(code)

    def test_class_dunder_class_as_local(self):
        if False:
            i = 10
            return i + 15
        code = '\n        class C:\n            def f(__class__):\n                return lambda: __class__\n        '
        self._check(code)

    def test_class_dunder_class_declared(self):
        if False:
            for i in range(10):
                print('nop')
        code = '\n        def f():\n            class C:\n                def g():\n                    return __class__\n                __class__\n        '
        self._check(code)