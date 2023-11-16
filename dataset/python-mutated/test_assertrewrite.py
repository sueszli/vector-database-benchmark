import ast
import errno
import glob
import importlib
import marshal
import os
import py_compile
import stat
import sys
import textwrap
import zipfile
from functools import partial
from pathlib import Path
from typing import cast
from typing import Dict
from typing import Generator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Set
from unittest import mock
import _pytest._code
import pytest
from _pytest._io.saferepr import DEFAULT_REPR_MAX_SIZE
from _pytest.assertion import util
from _pytest.assertion.rewrite import _get_assertion_exprs
from _pytest.assertion.rewrite import _get_maxsize_for_saferepr
from _pytest.assertion.rewrite import AssertionRewritingHook
from _pytest.assertion.rewrite import get_cache_dir
from _pytest.assertion.rewrite import PYC_TAIL
from _pytest.assertion.rewrite import PYTEST_TAG
from _pytest.assertion.rewrite import rewrite_asserts
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.pathlib import make_numbered_dir
from _pytest.pytester import Pytester

def rewrite(src: str) -> ast.Module:
    if False:
        for i in range(10):
            print('nop')
    tree = ast.parse(src)
    rewrite_asserts(tree, src.encode())
    return tree

def getmsg(f, extra_ns: Optional[Mapping[str, object]]=None, *, must_pass: bool=False) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    'Rewrite the assertions in f, run it, and get the failure message.'
    src = '\n'.join(_pytest._code.Code.from_function(f).source().lines)
    mod = rewrite(src)
    code = compile(mod, '<test>', 'exec')
    ns: Dict[str, object] = {}
    if extra_ns is not None:
        ns.update(extra_ns)
    exec(code, ns)
    func = ns[f.__name__]
    try:
        func()
    except AssertionError:
        if must_pass:
            pytest.fail("shouldn't have raised")
        s = str(sys.exc_info()[1])
        if not s.startswith('assert'):
            return 'AssertionError: ' + s
        return s
    else:
        if not must_pass:
            pytest.fail("function didn't raise at all")
        return None

class TestAssertionRewrite:

    def test_place_initial_imports(self) -> None:
        if False:
            i = 10
            return i + 15
        s = "'Doc string'\nother = stuff"
        m = rewrite(s)
        assert isinstance(m.body[0], ast.Expr)
        for imp in m.body[1:3]:
            assert isinstance(imp, ast.Import)
            assert imp.lineno == 2
            assert imp.col_offset == 0
        assert isinstance(m.body[3], ast.Assign)
        s = 'from __future__ import division\nother_stuff'
        m = rewrite(s)
        assert isinstance(m.body[0], ast.ImportFrom)
        for imp in m.body[1:3]:
            assert isinstance(imp, ast.Import)
            assert imp.lineno == 2
            assert imp.col_offset == 0
        assert isinstance(m.body[3], ast.Expr)
        s = "'doc string'\nfrom __future__ import division"
        m = rewrite(s)
        assert isinstance(m.body[0], ast.Expr)
        assert isinstance(m.body[1], ast.ImportFrom)
        for imp in m.body[2:4]:
            assert isinstance(imp, ast.Import)
            assert imp.lineno == 2
            assert imp.col_offset == 0
        s = "'doc string'\nfrom __future__ import division\nother"
        m = rewrite(s)
        assert isinstance(m.body[0], ast.Expr)
        assert isinstance(m.body[1], ast.ImportFrom)
        for imp in m.body[2:4]:
            assert isinstance(imp, ast.Import)
            assert imp.lineno == 3
            assert imp.col_offset == 0
        assert isinstance(m.body[4], ast.Expr)
        s = 'from . import relative\nother_stuff'
        m = rewrite(s)
        for imp in m.body[:2]:
            assert isinstance(imp, ast.Import)
            assert imp.lineno == 1
            assert imp.col_offset == 0
        assert isinstance(m.body[3], ast.Expr)

    def test_location_is_set(self) -> None:
        if False:
            i = 10
            return i + 15
        s = textwrap.dedent('\n\n        assert False, (\n\n            "Ouch"\n          )\n\n        ')
        m = rewrite(s)
        for node in m.body:
            if isinstance(node, ast.Import):
                continue
            for n in [node, *ast.iter_child_nodes(node)]:
                assert n.lineno == 3
                assert n.col_offset == 0
                assert n.end_lineno == 6
                assert n.end_col_offset == 3

    def test_dont_rewrite(self) -> None:
        if False:
            return 10
        s = "'PYTEST_DONT_REWRITE'\nassert 14"
        m = rewrite(s)
        assert len(m.body) == 2
        assert isinstance(m.body[1], ast.Assert)
        assert m.body[1].msg is None

    def test_dont_rewrite_plugin(self, pytester: Pytester) -> None:
        if False:
            return 10
        contents = {'conftest.py': "pytest_plugins = 'plugin'; import plugin", 'plugin.py': "'PYTEST_DONT_REWRITE'", 'test_foo.py': 'def test_foo(): pass'}
        pytester.makepyfile(**contents)
        result = pytester.runpytest_subprocess()
        assert 'warning' not in ''.join(result.outlines)

    def test_rewrites_plugin_as_a_package(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pkgdir = pytester.mkpydir('plugin')
        pkgdir.joinpath('__init__.py').write_text('import pytest\n@pytest.fixture\ndef special_asserter():\n    def special_assert(x, y):\n        assert x == y\n    return special_assert\n', encoding='utf-8')
        pytester.makeconftest('pytest_plugins = ["plugin"]')
        pytester.makepyfile('def test(special_asserter): special_asserter(1, 2)\n')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*assert 1 == 2*'])

    def test_honors_pep_235(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile(test_y='x = 1')
        xdir = pytester.mkdir('x')
        pytester.mkpydir(str(xdir.joinpath('test_Y')))
        xdir.joinpath('test_Y').joinpath('__init__.py').write_text('x = 2', encoding='utf-8')
        pytester.makepyfile('import test_y\nimport test_Y\ndef test():\n    assert test_y.x == 1\n    assert test_Y.x == 2\n')
        monkeypatch.syspath_prepend(str(xdir))
        pytester.runpytest().assert_outcomes(passed=1)

    def test_name(self, request) -> None:
        if False:
            for i in range(10):
                print('nop')

        def f1() -> None:
            if False:
                while True:
                    i = 10
            assert False
        assert getmsg(f1) == 'assert False'

        def f2() -> None:
            if False:
                i = 10
                return i + 15
            f = False
            assert f
        assert getmsg(f2) == 'assert False'

        def f3() -> None:
            if False:
                return 10
            assert a_global
        assert getmsg(f3, {'a_global': False}) == 'assert False'

        def f4() -> None:
            if False:
                while True:
                    i = 10
            assert sys == 42
        msg = getmsg(f4, {'sys': sys})
        assert msg == 'assert sys == 42'

        def f5() -> None:
            if False:
                for i in range(10):
                    print('nop')
            assert cls == 42

        class X:
            pass
        msg = getmsg(f5, {'cls': X})
        assert msg is not None
        lines = msg.splitlines()
        assert lines == ['assert cls == 42']

    def test_assertrepr_compare_same_width(self, request) -> None:
        if False:
            return 10
        'Should use same width/truncation with same initial width.'

        def f() -> None:
            if False:
                print('Hello World!')
            assert '1234567890' * 5 + 'A' == '1234567890' * 5 + 'B'
        msg = getmsg(f)
        assert msg is not None
        line = msg.splitlines()[0]
        if request.config.getoption('verbose') > 1:
            assert line == "assert '12345678901234567890123456789012345678901234567890A' == '12345678901234567890123456789012345678901234567890B'"
        else:
            assert line == "assert '123456789012...901234567890A' == '123456789012...901234567890B'"

    def test_dont_rewrite_if_hasattr_fails(self, request) -> None:
        if False:
            for i in range(10):
                print('nop')

        class Y:
            """A class whose getattr fails, but not with `AttributeError`."""

            def __getattr__(self, attribute_name):
                if False:
                    i = 10
                    return i + 15
                raise KeyError()

            def __repr__(self) -> str:
                if False:
                    while True:
                        i = 10
                return 'Y'

            def __init__(self) -> None:
                if False:
                    print('Hello World!')
                self.foo = 3

        def f() -> None:
            if False:
                i = 10
                return i + 15
            assert cls().foo == 2
        msg = getmsg(f, {'cls': Y})
        assert msg is not None
        lines = msg.splitlines()
        assert lines == ['assert 3 == 2', ' +  where 3 = Y.foo', ' +    where Y = cls()']

    def test_assert_already_has_message(self) -> None:
        if False:
            return 10

        def f():
            if False:
                for i in range(10):
                    print('nop')
            assert False, 'something bad!'
        assert getmsg(f) == 'AssertionError: something bad!\nassert False'

    def test_assertion_message(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            def test_foo():\n                assert 1 == 2, "The failure message"\n        ')
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(['*AssertionError*The failure message*', '*assert 1 == 2*'])

    def test_assertion_message_multiline(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            def test_foo():\n                assert 1 == 2, "A multiline\\nfailure message"\n        ')
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(['*AssertionError*A multiline*', '*failure message*', '*assert 1 == 2*'])

    def test_assertion_message_tuple(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            def test_foo():\n                assert 1 == 2, (1, 2)\n        ')
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(['*AssertionError*%s*' % repr((1, 2)), '*assert 1 == 2*'])

    def test_assertion_message_expr(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            def test_foo():\n                assert 1 == 2, 1 + 2\n        ')
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(['*AssertionError*3*', '*assert 1 == 2*'])

    def test_assertion_message_escape(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile("\n            def test_foo():\n                assert 1 == 2, 'To be escaped: %'\n        ")
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(['*AssertionError: To be escaped: %', '*assert 1 == 2'])

    def test_assertion_messages_bytes(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile("def test_bytes_assertion():\n    assert False, b'ohai!'\n")
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(["*AssertionError: b'ohai!'", '*assert False'])

    def test_boolop(self) -> None:
        if False:
            while True:
                i = 10

        def f1() -> None:
            if False:
                print('Hello World!')
            f = g = False
            assert f and g
        assert getmsg(f1) == 'assert (False)'

        def f2() -> None:
            if False:
                i = 10
                return i + 15
            f = True
            g = False
            assert f and g
        assert getmsg(f2) == 'assert (True and False)'

        def f3() -> None:
            if False:
                i = 10
                return i + 15
            f = False
            g = True
            assert f and g
        assert getmsg(f3) == 'assert (False)'

        def f4() -> None:
            if False:
                while True:
                    i = 10
            f = g = False
            assert f or g
        assert getmsg(f4) == 'assert (False or False)'

        def f5() -> None:
            if False:
                while True:
                    i = 10
            f = g = False
            assert not f and (not g)
        getmsg(f5, must_pass=True)

        def x() -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return False

        def f6() -> None:
            if False:
                for i in range(10):
                    print('nop')
            assert x() and x()
        assert getmsg(f6, {'x': x}) == 'assert (False)\n +  where False = x()'

        def f7() -> None:
            if False:
                print('Hello World!')
            assert False or x()
        assert getmsg(f7, {'x': x}) == 'assert (False or False)\n +  where False = x()'

        def f8() -> None:
            if False:
                i = 10
                return i + 15
            assert 1 in {} and 2 in {}
        assert getmsg(f8) == 'assert (1 in {})'

        def f9() -> None:
            if False:
                while True:
                    i = 10
            x = 1
            y = 2
            assert x in {1: None} and y in {}
        assert getmsg(f9) == 'assert (1 in {1: None} and 2 in {})'

        def f10() -> None:
            if False:
                for i in range(10):
                    print('nop')
            f = True
            g = False
            assert f or g
        getmsg(f10, must_pass=True)

        def f11() -> None:
            if False:
                return 10
            f = g = h = lambda : True
            assert f() and g() and h()
        getmsg(f11, must_pass=True)

    def test_short_circuit_evaluation(self) -> None:
        if False:
            i = 10
            return i + 15

        def f1() -> None:
            if False:
                print('Hello World!')
            assert True or explode
        getmsg(f1, must_pass=True)

        def f2() -> None:
            if False:
                return 10
            x = 1
            assert x == 1 or x == 2
        getmsg(f2, must_pass=True)

    def test_unary_op(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def f1() -> None:
            if False:
                i = 10
                return i + 15
            x = True
            assert not x
        assert getmsg(f1) == 'assert not True'

        def f2() -> None:
            if False:
                while True:
                    i = 10
            x = 0
            assert ~x + 1
        assert getmsg(f2) == 'assert (~0 + 1)'

        def f3() -> None:
            if False:
                return 10
            x = 3
            assert -x + x
        assert getmsg(f3) == 'assert (-3 + 3)'

        def f4() -> None:
            if False:
                return 10
            x = 0
            assert +x + x
        assert getmsg(f4) == 'assert (+0 + 0)'

    def test_binary_op(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def f1() -> None:
            if False:
                return 10
            x = 1
            y = -1
            assert x + y
        assert getmsg(f1) == 'assert (1 + -1)'

        def f2() -> None:
            if False:
                for i in range(10):
                    print('nop')
            assert not 5 % 4
        assert getmsg(f2) == 'assert not (5 % 4)'

    def test_boolop_percent(self) -> None:
        if False:
            return 10

        def f1() -> None:
            if False:
                return 10
            assert 3 % 2 and False
        assert getmsg(f1) == 'assert ((3 % 2) and False)'

        def f2() -> None:
            if False:
                while True:
                    i = 10
            assert False or 4 % 2
        assert getmsg(f2) == 'assert (False or (4 % 2))'

    def test_at_operator_issue1290(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            class Matrix(object):\n                def __init__(self, num):\n                    self.num = num\n                def __matmul__(self, other):\n                    return self.num * other.num\n\n            def test_multmat_operator():\n                assert Matrix(2) @ Matrix(3) == 6')
        pytester.runpytest().assert_outcomes(passed=1)

    def test_starred_with_side_effect(self, pytester: Pytester) -> None:
        if False:
            return 10
        'See #4412'
        pytester.makepyfile('            def test():\n                f = lambda x: x\n                x = iter([1, 2, 3])\n                assert 2 * next(x) == f(*[next(x)])\n            ')
        pytester.runpytest().assert_outcomes(passed=1)

    def test_call(self) -> None:
        if False:
            print('Hello World!')

        def g(a=42, *args, **kwargs) -> bool:
            if False:
                while True:
                    i = 10
            return False
        ns = {'g': g}

        def f1() -> None:
            if False:
                return 10
            assert g()
        assert getmsg(f1, ns) == 'assert False\n +  where False = g()'

        def f2() -> None:
            if False:
                i = 10
                return i + 15
            assert g(1)
        assert getmsg(f2, ns) == 'assert False\n +  where False = g(1)'

        def f3() -> None:
            if False:
                return 10
            assert g(1, 2)
        assert getmsg(f3, ns) == 'assert False\n +  where False = g(1, 2)'

        def f4() -> None:
            if False:
                print('Hello World!')
            assert g(1, g=42)
        assert getmsg(f4, ns) == 'assert False\n +  where False = g(1, g=42)'

        def f5() -> None:
            if False:
                while True:
                    i = 10
            assert g(1, 3, g=23)
        assert getmsg(f5, ns) == 'assert False\n +  where False = g(1, 3, g=23)'

        def f6() -> None:
            if False:
                print('Hello World!')
            seq = [1, 2, 3]
            assert g(*seq)
        assert getmsg(f6, ns) == 'assert False\n +  where False = g(*[1, 2, 3])'

        def f7() -> None:
            if False:
                for i in range(10):
                    print('nop')
            x = 'a'
            assert g(**{x: 2})
        assert getmsg(f7, ns) == "assert False\n +  where False = g(**{'a': 2})"

    def test_attribute(self) -> None:
        if False:
            while True:
                i = 10

        class X:
            g = 3
        ns = {'x': X}

        def f1() -> None:
            if False:
                print('Hello World!')
            assert not x.g
        assert getmsg(f1, ns) == 'assert not 3\n +  where 3 = x.g'

        def f2() -> None:
            if False:
                i = 10
                return i + 15
            x.a = False
            assert x.a
        assert getmsg(f2, ns) == 'assert False\n +  where False = x.a'

    def test_comparisons(self) -> None:
        if False:
            while True:
                i = 10

        def f1() -> None:
            if False:
                print('Hello World!')
            (a, b) = range(2)
            assert b < a
        assert getmsg(f1) == 'assert 1 < 0'

        def f2() -> None:
            if False:
                print('Hello World!')
            (a, b, c) = range(3)
            assert a > b > c
        assert getmsg(f2) == 'assert 0 > 1'

        def f3() -> None:
            if False:
                for i in range(10):
                    print('nop')
            (a, b, c) = range(3)
            assert a < b > c
        assert getmsg(f3) == 'assert 1 > 2'

        def f4() -> None:
            if False:
                i = 10
                return i + 15
            (a, b, c) = range(3)
            assert a < b <= c
        getmsg(f4, must_pass=True)

        def f5() -> None:
            if False:
                return 10
            (a, b, c) = range(3)
            assert a < b
            assert b < c
        getmsg(f5, must_pass=True)

    def test_len(self, request) -> None:
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                while True:
                    i = 10
            values = list(range(10))
            assert len(values) == 11
        msg = getmsg(f)
        assert msg == 'assert 10 == 11\n +  where 10 = len([0, 1, 2, 3, 4, 5, ...])'

    def test_custom_reprcompare(self, monkeypatch) -> None:
        if False:
            print('Hello World!')

        def my_reprcompare1(op, left, right) -> str:
            if False:
                while True:
                    i = 10
            return '42'
        monkeypatch.setattr(util, '_reprcompare', my_reprcompare1)

        def f1() -> None:
            if False:
                print('Hello World!')
            assert 42 < 3
        assert getmsg(f1) == 'assert 42'

        def my_reprcompare2(op, left, right) -> str:
            if False:
                print('Hello World!')
            return f'{left} {op} {right}'
        monkeypatch.setattr(util, '_reprcompare', my_reprcompare2)

        def f2() -> None:
            if False:
                while True:
                    i = 10
            assert 1 < 3 < 5 <= 4 < 7
        assert getmsg(f2) == 'assert 5 <= 4'

    def test_assert_raising__bool__in_comparison(self) -> None:
        if False:
            print('Hello World!')

        def f() -> None:
            if False:
                for i in range(10):
                    print('nop')

            class A:

                def __bool__(self):
                    if False:
                        print('Hello World!')
                    raise ValueError(42)

                def __lt__(self, other):
                    if False:
                        i = 10
                        return i + 15
                    return A()

                def __repr__(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    return '<MY42 object>'

            def myany(x) -> bool:
                if False:
                    print('Hello World!')
                return False
            assert myany(A() < 0)
        msg = getmsg(f)
        assert msg is not None
        assert '<MY42 object> < 0' in msg

    def test_assert_handling_raise_in__iter__(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('            class A:\n                def __iter__(self):\n                    raise ValueError()\n\n                def __eq__(self, o: object) -> bool:\n                    return self is o\n\n                def __repr__(self):\n                    return "<A object>"\n\n            assert A() == A()\n            ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*E*assert <A object> == <A object>'])

    def test_formatchar(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def f() -> None:
            if False:
                while True:
                    i = 10
            assert '%test' == 'test'
        msg = getmsg(f)
        assert msg is not None
        assert msg.startswith("assert '%test' == 'test'")

    def test_custom_repr(self, request) -> None:
        if False:
            i = 10
            return i + 15

        def f() -> None:
            if False:
                i = 10
                return i + 15

            class Foo:
                a = 1

                def __repr__(self):
                    if False:
                        return 10
                    return '\n{ \n~ \n}'
            f = Foo()
            assert 0 == f.a
        msg = getmsg(f)
        assert msg is not None
        lines = util._format_lines([msg])
        assert lines == ['assert 0 == 1\n +  where 1 = \\n{ \\n~ \\n}.a']

    def test_custom_repr_non_ascii(self) -> None:
        if False:
            return 10

        def f() -> None:
            if False:
                for i in range(10):
                    print('nop')

            class A:
                name = 'ä'

                def __repr__(self):
                    if False:
                        print('Hello World!')
                    return self.name.encode('UTF-8')
            a = A()
            assert not a.name
        msg = getmsg(f)
        assert msg is not None
        assert 'UnicodeDecodeError' not in msg
        assert 'UnicodeEncodeError' not in msg

class TestRewriteOnImport:

    def test_pycache_is_a_file(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.path.joinpath('__pycache__').write_text('Hello', encoding='utf-8')
        pytester.makepyfile('\n            def test_rewritten():\n                assert "@py_builtins" in globals()')
        assert pytester.runpytest().ret == 0

    def test_pycache_is_readonly(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        cache = pytester.mkdir('__pycache__')
        old_mode = cache.stat().st_mode
        cache.chmod(old_mode ^ stat.S_IWRITE)
        pytester.makepyfile('\n            def test_rewritten():\n                assert "@py_builtins" in globals()')
        try:
            assert pytester.runpytest().ret == 0
        finally:
            cache.chmod(old_mode)

    def test_zipfile(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        z = pytester.path.joinpath('myzip.zip')
        z_fn = str(z)
        f = zipfile.ZipFile(z_fn, 'w')
        try:
            f.writestr('test_gum/__init__.py', '')
            f.writestr('test_gum/test_lizard.py', '')
        finally:
            f.close()
        z.chmod(256)
        pytester.makepyfile('\n            import sys\n            sys.path.append(%r)\n            import test_gum.test_lizard' % (z_fn,))
        assert pytester.runpytest().ret == ExitCode.NO_TESTS_COLLECTED

    @pytest.mark.skipif(sys.version_info < (3, 9), reason='importlib.resources.files was introduced in 3.9')
    def test_load_resource_via_files_with_rewrite(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        example = pytester.path.joinpath('demo') / 'example'
        init = pytester.path.joinpath('demo') / '__init__.py'
        pytester.makepyfile(**{'demo/__init__.py': '\n                from importlib.resources import files\n\n                def load():\n                    return files(__name__)\n                ', 'test_load': f'\n                pytest_plugins = ["demo"]\n\n                def test_load():\n                    from demo import load\n                    found = {{str(i) for i in load().iterdir() if i.name != "__pycache__"}}\n                    assert found == {{{str(example)!r}, {str(init)!r}}}\n                '})
        example.mkdir()
        assert pytester.runpytest('-vv').ret == ExitCode.OK

    def test_readonly(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        sub = pytester.mkdir('testing')
        sub.joinpath('test_readonly.py').write_bytes(b'\ndef test_rewritten():\n    assert "@py_builtins" in globals()\n            ')
        old_mode = sub.stat().st_mode
        sub.chmod(320)
        try:
            assert pytester.runpytest().ret == 0
        finally:
            sub.chmod(old_mode)

    def test_dont_write_bytecode(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            return 10
        monkeypatch.delenv('PYTHONPYCACHEPREFIX', raising=False)
        pytester.makepyfile('\n            import os\n            def test_no_bytecode():\n                assert "__pycache__" in __cached__\n                assert not os.path.exists(__cached__)\n                assert not os.path.exists(os.path.dirname(__cached__))')
        monkeypatch.setenv('PYTHONDONTWRITEBYTECODE', '1')
        assert pytester.runpytest_subprocess().ret == 0

    def test_orphaned_pyc_file(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            return 10
        monkeypatch.delenv('PYTHONPYCACHEPREFIX', raising=False)
        monkeypatch.setattr(sys, 'pycache_prefix', None, raising=False)
        pytester.makepyfile('\n            import orphan\n            def test_it():\n                assert orphan.value == 17\n            ')
        pytester.makepyfile(orphan='\n            value = 17\n            ')
        py_compile.compile('orphan.py')
        os.remove('orphan.py')
        if not os.path.exists('orphan.pyc'):
            pycs = glob.glob('__pycache__/orphan.*.pyc')
            assert len(pycs) == 1
            os.rename(pycs[0], 'orphan.pyc')
        assert pytester.runpytest().ret == 0

    def test_cached_pyc_includes_pytest_version(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            i = 10
            return i + 15
        'Avoid stale caches (#1671)'
        monkeypatch.delenv('PYTHONDONTWRITEBYTECODE', raising=False)
        monkeypatch.delenv('PYTHONPYCACHEPREFIX', raising=False)
        pytester.makepyfile(test_foo='\n            def test_foo():\n                assert True\n            ')
        result = pytester.runpytest_subprocess()
        assert result.ret == 0
        found_names = glob.glob(f'__pycache__/*-pytest-{pytest.__version__}.pyc')
        assert found_names, 'pyc with expected tag not found in names: {}'.format(glob.glob('__pycache__/*.pyc'))

    @pytest.mark.skipif('"__pypy__" in sys.modules')
    def test_pyc_vs_pyo(self, pytester: Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            import pytest\n            def test_optimized():\n                "hello"\n                assert test_optimized.__doc__ is None')
        p = make_numbered_dir(root=Path(pytester.path), prefix='runpytest-')
        tmp = '--basetemp=%s' % p
        with monkeypatch.context() as mp:
            mp.setenv('PYTHONOPTIMIZE', '2')
            mp.delenv('PYTHONDONTWRITEBYTECODE', raising=False)
            mp.delenv('PYTHONPYCACHEPREFIX', raising=False)
            assert pytester.runpytest_subprocess(tmp).ret == 0
            tagged = 'test_pyc_vs_pyo.' + PYTEST_TAG
            assert tagged + '.pyo' in os.listdir('__pycache__')
        monkeypatch.delenv('PYTHONDONTWRITEBYTECODE', raising=False)
        monkeypatch.delenv('PYTHONPYCACHEPREFIX', raising=False)
        assert pytester.runpytest_subprocess(tmp).ret == 1
        assert tagged + '.pyc' in os.listdir('__pycache__')

    def test_package(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pkg = pytester.path.joinpath('pkg')
        pkg.mkdir()
        pkg.joinpath('__init__.py')
        pkg.joinpath('test_blah.py').write_text('\ndef test_rewritten():\n    assert "@py_builtins" in globals()', encoding='utf-8')
        assert pytester.runpytest().ret == 0

    def test_translate_newlines(self, pytester: Pytester) -> None:
        if False:
            return 10
        content = "def test_rewritten():\r\n assert '@py_builtins' in globals()"
        b = content.encode('utf-8')
        pytester.path.joinpath('test_newlines.py').write_bytes(b)
        assert pytester.runpytest().ret == 0

    def test_package_without__init__py(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pkg = pytester.mkdir('a_package_without_init_py')
        pkg.joinpath('module.py').touch()
        pytester.makepyfile('import a_package_without_init_py.module')
        assert pytester.runpytest().ret == ExitCode.NO_TESTS_COLLECTED

    def test_rewrite_warning(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makeconftest('\n            import pytest\n            pytest.register_assert_rewrite("_pytest")\n        ')
        result = pytester.runpytest_subprocess()
        result.stdout.fnmatch_lines(['*Module already imported*: _pytest'])

    def test_rewrite_module_imported_from_conftest(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makeconftest('\n            import test_rewrite_module_imported\n        ')
        pytester.makepyfile(test_rewrite_module_imported='\n            def test_rewritten():\n                assert "@py_builtins" in globals()\n        ')
        assert pytester.runpytest_subprocess().ret == 0

    def test_remember_rewritten_modules(self, pytestconfig, pytester: Pytester, monkeypatch) -> None:
        if False:
            while True:
                i = 10
        "`AssertionRewriteHook` should remember rewritten modules so it\n        doesn't give false positives (#2005)."
        monkeypatch.syspath_prepend(pytester.path)
        pytester.makepyfile(test_remember_rewritten_modules='')
        warnings = []
        hook = AssertionRewritingHook(pytestconfig)
        monkeypatch.setattr(hook, '_warn_already_imported', lambda code, msg: warnings.append(msg))
        spec = hook.find_spec('test_remember_rewritten_modules')
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        hook.exec_module(module)
        hook.mark_rewrite('test_remember_rewritten_modules')
        hook.mark_rewrite('test_remember_rewritten_modules')
        assert warnings == []

    def test_rewrite_warning_using_pytest_plugins(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile(**{'conftest.py': "pytest_plugins = ['core', 'gui', 'sci']", 'core.py': '', 'gui.py': "pytest_plugins = ['core', 'sci']", 'sci.py': "pytest_plugins = ['core']", 'test_rewrite_warning_pytest_plugins.py': 'def test(): pass'})
        pytester.chdir()
        result = pytester.runpytest_subprocess()
        result.stdout.fnmatch_lines(['*= 1 passed in *=*'])
        result.stdout.no_fnmatch_line('*pytest-warning summary*')

    def test_rewrite_warning_using_pytest_plugins_env_var(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            return 10
        monkeypatch.setenv('PYTEST_PLUGINS', 'plugin')
        pytester.makepyfile(**{'plugin.py': '', 'test_rewrite_warning_using_pytest_plugins_env_var.py': "\n                import plugin\n                pytest_plugins = ['plugin']\n                def test():\n                    pass\n            "})
        pytester.chdir()
        result = pytester.runpytest_subprocess()
        result.stdout.fnmatch_lines(['*= 1 passed in *=*'])
        result.stdout.no_fnmatch_line('*pytest-warning summary*')

class TestAssertionRewriteHookDetails:

    def test_sys_meta_path_munged(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            def test_meta_path():\n                import sys; sys.meta_path = []')
        assert pytester.runpytest().ret == 0

    def test_write_pyc(self, pytester: Pytester, tmp_path) -> None:
        if False:
            print('Hello World!')
        from _pytest.assertion.rewrite import _write_pyc
        from _pytest.assertion import AssertionState
        config = pytester.parseconfig()
        state = AssertionState(config, 'rewrite')
        tmp_path.joinpath('source.py').touch()
        source_path = str(tmp_path)
        pycpath = tmp_path.joinpath('pyc')
        co = compile('1', 'f.py', 'single')
        assert _write_pyc(state, co, os.stat(source_path), pycpath)
        with mock.patch.object(os, 'replace', side_effect=OSError):
            assert not _write_pyc(state, co, os.stat(source_path), pycpath)

    def test_resources_provider_for_loader(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        '\n        Attempts to load resources from a package should succeed normally,\n        even when the AssertionRewriteHook is used to load the modules.\n\n        See #366 for details.\n        '
        pytest.importorskip('pkg_resources')
        pytester.mkpydir('testpkg')
        contents = {'testpkg/test_pkg': "\n                import pkg_resources\n\n                import pytest\n                from _pytest.assertion.rewrite import AssertionRewritingHook\n\n                def test_load_resource():\n                    assert isinstance(__loader__, AssertionRewritingHook)\n                    res = pkg_resources.resource_string(__name__, 'resource.txt')\n                    res = res.decode('ascii')\n                    assert res == 'Load me please.'\n                "}
        pytester.makepyfile(**contents)
        pytester.maketxtfile(**{'testpkg/resource': 'Load me please.'})
        result = pytester.runpytest_subprocess()
        result.assert_outcomes(passed=1)

    def test_read_pyc(self, tmp_path: Path) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Ensure that the `_read_pyc` can properly deal with corrupted pyc files.\n        In those circumstances it should just give up instead of generating\n        an exception that is propagated to the caller.\n        '
        import py_compile
        from _pytest.assertion.rewrite import _read_pyc
        source = tmp_path / 'source.py'
        pyc = Path(str(source) + 'c')
        source.write_text('def test(): pass', encoding='utf-8')
        py_compile.compile(str(source), str(pyc))
        contents = pyc.read_bytes()
        strip_bytes = 20
        assert len(contents) > strip_bytes
        pyc.write_bytes(contents[:strip_bytes])
        assert _read_pyc(source, pyc) is None

    def test_read_pyc_success(self, tmp_path: Path, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        '\n        Ensure that the _rewrite_test() -> _write_pyc() produces a pyc file\n        that can be properly read with _read_pyc()\n        '
        from _pytest.assertion import AssertionState
        from _pytest.assertion.rewrite import _read_pyc
        from _pytest.assertion.rewrite import _rewrite_test
        from _pytest.assertion.rewrite import _write_pyc
        config = pytester.parseconfig()
        state = AssertionState(config, 'rewrite')
        fn = tmp_path / 'source.py'
        pyc = Path(str(fn) + 'c')
        fn.write_text('def test(): assert True', encoding='utf-8')
        (source_stat, co) = _rewrite_test(fn, config)
        _write_pyc(state, co, source_stat, pyc)
        assert _read_pyc(fn, pyc, state.trace) is not None

    def test_read_pyc_more_invalid(self, tmp_path: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        from _pytest.assertion.rewrite import _read_pyc
        source = tmp_path / 'source.py'
        pyc = tmp_path / 'source.pyc'
        source_bytes = b'def test(): pass\n'
        source.write_bytes(source_bytes)
        magic = importlib.util.MAGIC_NUMBER
        flags = b'\x00\x00\x00\x00'
        mtime = b'X<\xb0_'
        mtime_int = int.from_bytes(mtime, 'little')
        os.utime(source, (mtime_int, mtime_int))
        size = len(source_bytes).to_bytes(4, 'little')
        code = marshal.dumps(compile(source_bytes, str(source), 'exec'))
        pyc.write_bytes(magic + flags + mtime + size + code)
        assert _read_pyc(source, pyc, print) is not None
        pyc.write_bytes(magic + flags + mtime)
        assert _read_pyc(source, pyc, print) is None
        pyc.write_bytes(b'\x124Vx' + flags + mtime + size + code)
        assert _read_pyc(source, pyc, print) is None
        pyc.write_bytes(magic + b'\x00\xff\x00\x00' + mtime + size + code)
        assert _read_pyc(source, pyc, print) is None
        pyc.write_bytes(magic + flags + b'X=\xb0_' + size + code)
        assert _read_pyc(source, pyc, print) is None
        pyc.write_bytes(magic + flags + mtime + b'\x99\x00\x00\x00' + code)
        assert _read_pyc(source, pyc, print) is None

    def test_reload_is_same_and_reloads(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        'Reloading a (collected) module after change picks up the change.'
        pytester.makeini('\n            [pytest]\n            python_files = *.py\n            ')
        pytester.makepyfile(file="\n            def reloaded():\n                return False\n\n            def rewrite_self():\n                with open(__file__, 'w', encoding='utf-8') as self:\n                    self.write('def reloaded(): return True')\n            ", test_fun='\n            import sys\n            from importlib import reload\n\n            def test_loader():\n                import file\n                assert not file.reloaded()\n                file.rewrite_self()\n                assert sys.modules["file"] is reload(file)\n                assert file.reloaded()\n            ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 1 passed*'])

    def test_get_data_support(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        'Implement optional PEP302 api (#808).'
        path = pytester.mkpydir('foo')
        path.joinpath('test_foo.py').write_text(textwrap.dedent("                class Test(object):\n                    def test_foo(self):\n                        import pkgutil\n                        data = pkgutil.get_data('foo.test_foo', 'data.txt')\n                        assert data == b'Hey'\n                "), encoding='utf-8')
        path.joinpath('data.txt').write_text('Hey', encoding='utf-8')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*1 passed*'])

def test_issue731(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile("\n    class LongReprWithBraces(object):\n        def __repr__(self):\n           return 'LongReprWithBraces({' + ('a' * 80) + '}' + ('a' * 120) + ')'\n\n        def some_method(self):\n            return False\n\n    def test_long_repr():\n        obj = LongReprWithBraces()\n        assert obj.some_method()\n    ")
    result = pytester.runpytest()
    result.stdout.no_fnmatch_line('*unbalanced braces*')

class TestIssue925:

    def test_simple_case(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n        def test_ternary_display():\n            assert (False == False) == False\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*E*assert (False == False) == False'])

    def test_long_case(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n        def test_ternary_display():\n             assert False == (False == True) == True\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*E*assert (False == True) == True'])

    def test_many_brackets(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            def test_ternary_display():\n                 assert True == ((False == True) == True)\n            ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*E*assert True == ((False == True) == True)'])

class TestIssue2121:

    def test_rewrite_python_files_contain_subdirs(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile(**{'tests/file.py': '\n                def test_simple_failure():\n                    assert 1 + 1 == 3\n                '})
        pytester.makeini('\n                [pytest]\n                python_files = tests/**.py\n            ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*E*assert (1 + 1) == 3'])

class TestIssue10743:

    def test_assertion_walrus_operator(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            def my_func(before, after):\n                return before == after\n\n            def change_value(value):\n                return value.lower()\n\n            def test_walrus_conversion():\n                a = "Hello"\n                assert not my_func(a, a := change_value(a))\n                assert a == "hello"\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

    def test_assertion_walrus_operator_dont_rewrite(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            \'PYTEST_DONT_REWRITE\'\n            def my_func(before, after):\n                return before == after\n\n            def change_value(value):\n                return value.lower()\n\n            def test_walrus_conversion_dont_rewrite():\n                a = "Hello"\n                assert not my_func(a, a := change_value(a))\n                assert a == "hello"\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

    def test_assertion_inline_walrus_operator(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            def my_func(before, after):\n                return before == after\n\n            def test_walrus_conversion_inline():\n                a = "Hello"\n                assert not my_func(a, a := a.lower())\n                assert a == "hello"\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

    def test_assertion_inline_walrus_operator_reverse(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            def my_func(before, after):\n                return before == after\n\n            def test_walrus_conversion_reverse():\n                a = "Hello"\n                assert my_func(a := a.lower(), a)\n                assert a == \'hello\'\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

    def test_assertion_walrus_no_variable_name_conflict(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            def test_walrus_conversion_no_conflict():\n                a = "Hello"\n                assert a == (b := a.lower())\n        ')
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(["*AssertionError: assert 'Hello' == 'hello'"])

    def test_assertion_walrus_operator_true_assertion_and_changes_variable_value(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            def test_walrus_conversion_succeed():\n                a = "Hello"\n                assert a != (a := a.lower())\n                assert a == \'hello\'\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

    def test_assertion_walrus_operator_fail_assertion(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            def test_walrus_conversion_fails():\n                a = "Hello"\n                assert a == (a := a.lower())\n        ')
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(["*AssertionError: assert 'Hello' == 'hello'"])

    def test_assertion_walrus_operator_boolean_composite(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            def test_walrus_operator_change_boolean_value():\n                a = True\n                assert a and True and ((a := False) is False) and (a is False) and ((a := None) is None)\n                assert a is None\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

    def test_assertion_walrus_operator_compare_boolean_fails(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            def test_walrus_operator_change_boolean_value():\n                a = True\n                assert not (a and ((a := False) is False))\n        ')
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(['*assert not (True and False is False)'])

    def test_assertion_walrus_operator_boolean_none_fails(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            def test_walrus_operator_change_boolean_value():\n                a = True\n                assert not (a and ((a := None) is None))\n        ')
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(['*assert not (True and None is None)'])

    def test_assertion_walrus_operator_value_changes_cleared_after_each_test(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            def test_walrus_operator_change_value():\n                a = True\n                assert (a := None) is None\n\n            def test_walrus_operator_not_override_value():\n                a = True\n                assert a is True\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

class TestIssue11028:

    def test_assertion_walrus_operator_in_operand(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            def test_in_string():\n              assert (obj := "foo") in obj\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

    def test_assertion_walrus_operator_in_operand_json_dumps(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            import json\n\n            def test_json_encoder():\n                assert (obj := "foo") in json.dumps(obj)\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

    def test_assertion_walrus_operator_equals_operand_function(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            def f(a):\n                return a\n\n            def test_call_other_function_arg():\n              assert (obj := "foo") == f(obj)\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

    def test_assertion_walrus_operator_equals_operand_function_keyword_arg(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            def f(a=\'test\'):\n                return a\n\n            def test_call_other_function_k_arg():\n              assert (obj := "foo") == f(a=obj)\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

    def test_assertion_walrus_operator_equals_operand_function_arg_as_function(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            def f(a=\'test\'):\n                return a\n\n            def test_function_of_function():\n              assert (obj := "foo") == f(f(obj))\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

    def test_assertion_walrus_operator_gt_operand_function(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            def add_one(a):\n                return a + 1\n\n            def test_gt():\n              assert (obj := 4) > add_one(obj)\n        ')
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(['*assert 4 > 5', '*where 5 = add_one(4)'])

class TestIssue11239:

    def test_assertion_walrus_different_test_cases(self, pytester: Pytester) -> None:
        if False:
            return 10
        'Regression for (#11239)\n\n        Walrus operator rewriting would leak to separate test cases if they used the same variables.\n        '
        pytester.makepyfile('\n            def test_1():\n                state = {"x": 2}.get("x")\n                assert state is not None\n\n            def test_2():\n                db = {"x": 2}\n                assert (state := db.get("x")) is not None\n        ')
        result = pytester.runpytest()
        assert result.ret == 0

@pytest.mark.skipif(sys.maxsize <= 2 ** 31 - 1, reason='Causes OverflowError on 32bit systems')
@pytest.mark.parametrize('offset', [-1, +1])
def test_source_mtime_long_long(pytester: Pytester, offset) -> None:
    if False:
        return 10
    'Support modification dates after 2038 in rewritten files (#4903).\n\n    pytest would crash with:\n\n            fp.write(struct.pack("<ll", mtime, size))\n        E   struct.error: argument out of range\n    '
    p = pytester.makepyfile('\n        def test(): pass\n    ')
    timestamp = 2 ** 32 + offset
    os.utime(str(p), (timestamp, timestamp))
    result = pytester.runpytest()
    assert result.ret == 0

def test_rewrite_infinite_recursion(pytester: Pytester, pytestconfig, monkeypatch) -> None:
    if False:
        i = 10
        return i + 15
    'Fix infinite recursion when writing pyc files: if an import happens to be triggered when writing the pyc\n    file, this would cause another call to the hook, which would trigger another pyc writing, which could\n    trigger another import, and so on. (#3506)'
    from _pytest.assertion import rewrite as rewritemod
    pytester.syspathinsert()
    pytester.makepyfile(test_foo='def test_foo(): pass')
    pytester.makepyfile(test_bar='def test_bar(): pass')
    original_write_pyc = rewritemod._write_pyc
    write_pyc_called = []

    def spy_write_pyc(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        write_pyc_called.append(True)
        assert hook.find_spec('test_bar') is None
        return original_write_pyc(*args, **kwargs)
    monkeypatch.setattr(rewritemod, '_write_pyc', spy_write_pyc)
    monkeypatch.setattr(sys, 'dont_write_bytecode', False)
    hook = AssertionRewritingHook(pytestconfig)
    spec = hook.find_spec('test_foo')
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    hook.exec_module(module)
    assert len(write_pyc_called) == 1

class TestEarlyRewriteBailout:

    @pytest.fixture
    def hook(self, pytestconfig, monkeypatch, pytester: Pytester) -> Generator[AssertionRewritingHook, None, None]:
        if False:
            print('Hello World!')
        'Returns a patched AssertionRewritingHook instance so we can configure its initial paths and track\n        if PathFinder.find_spec has been called.\n        '
        import importlib.machinery
        self.find_spec_calls: List[str] = []
        self.initial_paths: Set[Path] = set()

        class StubSession:
            _initialpaths = self.initial_paths

            def isinitpath(self, p):
                if False:
                    return 10
                return p in self._initialpaths

        def spy_find_spec(name, path):
            if False:
                for i in range(10):
                    print('nop')
            self.find_spec_calls.append(name)
            return importlib.machinery.PathFinder.find_spec(name, path)
        hook = AssertionRewritingHook(pytestconfig)
        with mock.patch.object(hook, 'fnpats', ['test_*.py', '*_test.py']):
            monkeypatch.setattr(hook, '_find_spec', spy_find_spec)
            hook.set_session(StubSession())
            pytester.syspathinsert()
            yield hook

    def test_basic(self, pytester: Pytester, hook: AssertionRewritingHook) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Ensure we avoid calling PathFinder.find_spec when we know for sure a certain\n        module will not be rewritten to optimize assertion rewriting (#3918).\n        '
        pytester.makeconftest('\n            import pytest\n            @pytest.fixture\n            def fix(): return 1\n        ')
        pytester.makepyfile(test_foo='def test_foo(): pass')
        pytester.makepyfile(bar='def bar(): pass')
        foobar_path = pytester.makepyfile(foobar='def foobar(): pass')
        self.initial_paths.add(foobar_path)
        assert hook.find_spec('conftest') is not None
        assert self.find_spec_calls == ['conftest']
        assert hook.find_spec('test_foo') is not None
        assert self.find_spec_calls == ['conftest', 'test_foo']
        assert hook.find_spec('bar') is None
        assert self.find_spec_calls == ['conftest', 'test_foo']
        assert hook.find_spec('foobar') is not None
        assert self.find_spec_calls == ['conftest', 'test_foo', 'foobar']

    def test_pattern_contains_subdirectories(self, pytester: Pytester, hook: AssertionRewritingHook) -> None:
        if False:
            while True:
                i = 10
        'If one of the python_files patterns contain subdirectories ("tests/**.py") we can\'t bailout early\n        because we need to match with the full path, which can only be found by calling PathFinder.find_spec\n        '
        pytester.makepyfile(**{'tests/file.py': '                    def test_simple_failure():\n                        assert 1 + 1 == 3\n                '})
        pytester.syspathinsert('tests')
        with mock.patch.object(hook, 'fnpats', ['tests/**.py']):
            assert hook.find_spec('file') is not None
            assert self.find_spec_calls == ['file']

    @pytest.mark.skipif(sys.platform.startswith('win32'), reason='cannot remove cwd on Windows')
    @pytest.mark.skipif(sys.platform.startswith('sunos5'), reason='cannot remove cwd on Solaris')
    def test_cwd_changed(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            i = 10
            return i + 15
        monkeypatch.syspath_prepend('')
        monkeypatch.delitem(sys.modules, 'pathlib', raising=False)
        pytester.makepyfile(**{'test_setup_nonexisting_cwd.py': '                    import os\n                    import tempfile\n\n                    with tempfile.TemporaryDirectory() as d:\n                        os.chdir(d)\n                ', 'test_test.py': '                    def test():\n                        pass\n                '})
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['* 1 passed in *'])

class TestAssertionPass:

    def test_option_default(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        config = pytester.parseconfig()
        assert config.getini('enable_assertion_pass_hook') is False

    @pytest.fixture
    def flag_on(self, pytester: Pytester):
        if False:
            return 10
        pytester.makeini('[pytest]\nenable_assertion_pass_hook = True\n')

    @pytest.fixture
    def hook_on(self, pytester: Pytester):
        if False:
            return 10
        pytester.makeconftest('            def pytest_assertion_pass(item, lineno, orig, expl):\n                raise Exception("Assertion Passed: {} {} at line {}".format(orig, expl, lineno))\n            ')

    def test_hook_call(self, pytester: Pytester, flag_on, hook_on) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('            def test_simple():\n                a=1\n                b=2\n                c=3\n                d=0\n\n                assert a+b == c+d\n\n            # cover failing assertions with a message\n            def test_fails():\n                assert False, "assert with message"\n            ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines('*Assertion Passed: a+b == c+d (1 + 2) == (3 + 0) at line 7*')

    def test_hook_call_with_parens(self, pytester: Pytester, flag_on, hook_on) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('            def f(): return 1\n            def test():\n                assert f()\n            ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines('*Assertion Passed: f() 1')

    def test_hook_not_called_without_hookimpl(self, pytester: Pytester, monkeypatch, flag_on) -> None:
        if False:
            print('Hello World!')
        'Assertion pass should not be called (and hence formatting should\n        not occur) if there is no hook declared for pytest_assertion_pass'

        def raise_on_assertionpass(*_, **__):
            if False:
                return 10
            raise Exception("Assertion passed called when it shouldn't!")
        monkeypatch.setattr(_pytest.assertion.rewrite, '_call_assertion_pass', raise_on_assertionpass)
        pytester.makepyfile('            def test_simple():\n                a=1\n                b=2\n                c=3\n                d=0\n\n                assert a+b == c+d\n            ')
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)

    def test_hook_not_called_without_cmd_option(self, pytester: Pytester, monkeypatch) -> None:
        if False:
            i = 10
            return i + 15
        'Assertion pass should not be called (and hence formatting should\n        not occur) if there is no hook declared for pytest_assertion_pass'

        def raise_on_assertionpass(*_, **__):
            if False:
                print('Hello World!')
            raise Exception("Assertion passed called when it shouldn't!")
        monkeypatch.setattr(_pytest.assertion.rewrite, '_call_assertion_pass', raise_on_assertionpass)
        pytester.makeconftest('            def pytest_assertion_pass(item, lineno, orig, expl):\n                raise Exception("Assertion Passed: {} {} at line {}".format(orig, expl, lineno))\n            ')
        pytester.makepyfile('            def test_simple():\n                a=1\n                b=2\n                c=3\n                d=0\n\n                assert a+b == c+d\n            ')
        result = pytester.runpytest()
        result.assert_outcomes(passed=1)

@pytest.mark.parametrize(('src', 'expected'), (pytest.param(b'', {}, id='trivial'), pytest.param(b'def x(): assert 1\n', {1: '1'}, id='assert statement not on own line'), pytest.param(b'def x():\n    assert 1\n    assert 1+2\n', {2: '1', 3: '1+2'}, id='multiple assertions'), pytest.param('# -*- coding: latin1\ndef ÀÀÀÀÀ(): assert 1\n'.encode('latin1'), {2: '1'}, id='latin1 encoded on first line\n'), pytest.param('def ÀÀÀÀÀ(): assert 1\n'.encode(), {1: '1'}, id='utf-8 encoded on first line'), pytest.param(b'def x():\n    assert (\n        1 + 2  # comment\n    )\n', {2: '(\n        1 + 2  # comment\n    )'}, id='multi-line assertion'), pytest.param(b'def x():\n    assert y == [\n        1, 2, 3\n    ]\n', {2: 'y == [\n        1, 2, 3\n    ]'}, id='multi line assert with list continuation'), pytest.param(b'def x():\n    assert 1 + \\\n        2\n', {2: '1 + \\\n        2'}, id='backslash continuation'), pytest.param(b'def x():\n    assert x, y\n', {2: 'x'}, id='assertion with message'), pytest.param(b"def x():\n    assert (\n        f(1, 2, 3)\n    ),  'f did not work!'\n", {2: '(\n        f(1, 2, 3)\n    )'}, id='assertion with message, test spanning multiple lines'), pytest.param(b"def x():\n    assert \\\n        x\\\n        , 'failure message'\n", {2: 'x'}, id='escaped newlines plus message'), pytest.param(b'def x(): assert 5', {1: '5'}, id='no newline at end of file')))
def test_get_assertion_exprs(src, expected) -> None:
    if False:
        return 10
    assert _get_assertion_exprs(src) == expected

def test_try_makedirs(monkeypatch, tmp_path: Path) -> None:
    if False:
        i = 10
        return i + 15
    from _pytest.assertion.rewrite import try_makedirs
    p = tmp_path / 'foo'
    assert try_makedirs(p)
    assert p.is_dir()
    assert try_makedirs(p)

    def fake_mkdir(p, exist_ok=False, *, exc):
        if False:
            return 10
        assert isinstance(p, Path)
        raise exc
    monkeypatch.setattr(os, 'makedirs', partial(fake_mkdir, exc=FileNotFoundError()))
    assert not try_makedirs(p)
    monkeypatch.setattr(os, 'makedirs', partial(fake_mkdir, exc=NotADirectoryError()))
    assert not try_makedirs(p)
    monkeypatch.setattr(os, 'makedirs', partial(fake_mkdir, exc=PermissionError()))
    assert not try_makedirs(p)
    err = OSError()
    err.errno = errno.EROFS
    monkeypatch.setattr(os, 'makedirs', partial(fake_mkdir, exc=err))
    assert not try_makedirs(p)
    err = OSError()
    err.errno = errno.ECHILD
    monkeypatch.setattr(os, 'makedirs', partial(fake_mkdir, exc=err))
    with pytest.raises(OSError) as exc_info:
        try_makedirs(p)
    assert exc_info.value.errno == errno.ECHILD

class TestPyCacheDir:

    @pytest.mark.parametrize('prefix, source, expected', [('c:/tmp/pycs', 'd:/projects/src/foo.py', 'c:/tmp/pycs/projects/src'), (None, 'd:/projects/src/foo.py', 'd:/projects/src/__pycache__'), ('/tmp/pycs', '/home/projects/src/foo.py', '/tmp/pycs/home/projects/src'), (None, '/home/projects/src/foo.py', '/home/projects/src/__pycache__')])
    def test_get_cache_dir(self, monkeypatch, prefix, source, expected) -> None:
        if False:
            i = 10
            return i + 15
        monkeypatch.delenv('PYTHONPYCACHEPREFIX', raising=False)
        monkeypatch.setattr(sys, 'pycache_prefix', prefix, raising=False)
        assert get_cache_dir(Path(source)) == Path(expected)

    @pytest.mark.skipif(sys.version_info[:2] == (3, 9) and sys.platform.startswith('win'), reason='#9298')
    def test_sys_pycache_prefix_integration(self, tmp_path, monkeypatch, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Integration test for sys.pycache_prefix (#4730).'
        pycache_prefix = tmp_path / 'my/pycs'
        monkeypatch.setattr(sys, 'pycache_prefix', str(pycache_prefix))
        monkeypatch.setattr(sys, 'dont_write_bytecode', False)
        pytester.makepyfile(**{'src/test_foo.py': '\n                import bar\n                def test_foo():\n                    pass\n            ', 'src/bar/__init__.py': ''})
        result = pytester.runpytest()
        assert result.ret == 0
        test_foo = pytester.path.joinpath('src/test_foo.py')
        bar_init = pytester.path.joinpath('src/bar/__init__.py')
        assert test_foo.is_file()
        assert bar_init.is_file()
        test_foo_pyc = get_cache_dir(test_foo) / ('test_foo' + PYC_TAIL)
        assert test_foo_pyc.is_file()
        bar_init_pyc = get_cache_dir(bar_init) / '__init__.{cache_tag}.pyc'.format(cache_tag=sys.implementation.cache_tag)
        assert bar_init_pyc.is_file()

class TestReprSizeVerbosity:
    """
    Check that verbosity also controls the string length threshold to shorten it using
    ellipsis.
    """

    @pytest.mark.parametrize('verbose, expected_size', [(0, DEFAULT_REPR_MAX_SIZE), (1, DEFAULT_REPR_MAX_SIZE * 10), (2, None), (3, None)])
    def test_get_maxsize_for_saferepr(self, verbose: int, expected_size) -> None:
        if False:
            print('Hello World!')

        class FakeConfig:

            def getoption(self, name: str) -> int:
                if False:
                    for i in range(10):
                        print('nop')
                assert name == 'verbose'
                return verbose
        config = FakeConfig()
        assert _get_maxsize_for_saferepr(cast(Config, config)) == expected_size

    def create_test_file(self, pytester: Pytester, size: int) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile(f'\n            def test_very_long_string():\n                text = "x" * {size}\n                assert "hello world" in text\n            ')

    def test_default_verbosity(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        self.create_test_file(pytester, DEFAULT_REPR_MAX_SIZE)
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*xxx...xxx*'])

    def test_increased_verbosity(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        self.create_test_file(pytester, DEFAULT_REPR_MAX_SIZE)
        result = pytester.runpytest('-v')
        result.stdout.no_fnmatch_line('*xxx...xxx*')

    def test_max_increased_verbosity(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        self.create_test_file(pytester, DEFAULT_REPR_MAX_SIZE * 10)
        result = pytester.runpytest('-vv')
        result.stdout.no_fnmatch_line('*xxx...xxx*')

class TestIssue11140:

    def test_constant_not_picked_as_module_docstring(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('            0\n\n            def test_foo():\n                pass\n            ')
        result = pytester.runpytest()
        assert result.ret == 0