from __static__ import TYPED_DOUBLE
import re
from compiler.errors import TypedSyntaxError
from .common import StaticTestBase

class UnknownNameTests(StaticTestBase):

    def test_unknown_name_toplevel(self) -> None:
        if False:
            return 10
        codestr = '\n        b = a + 1\n        '
        self.type_error(codestr, 'Name `a` is not defined.')

    def test_unknown_name_class_toplevel(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n        class C:\n            b: int = a + 1\n        '
        self.type_error(codestr, 'Name `a` is not defined.')

    def test_unknown_name_method(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n        class C:\n            def foo(self) -> int:\n                b = a + 1\n                return 0\n        '
        self.type_error(codestr, 'Name `a` is not defined.')

    def test_unknown_name_function(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n        def foo() -> int:\n            return a\n        '
        self.type_error(codestr, 'Name `a` is not defined.')

    def test_builtins_ok(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n        def foo() -> None:\n            a = open("sourcefile.hs")\n        '
        self.compile(codestr)

    def test_no_unknown_name_error_assignments(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n        def foo() -> None:\n            a: int = 1\n            b = 2\n        '
        self.compile(codestr)

    def test_unknown_name_error_augassign(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n        def foo() -> None:\n            a += 1\n        '
        self.type_error(codestr, 'Name `a` is not defined.')

    def test_with_optional_vars_are_known(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        def foo(x) -> None:\n            with x() as y:\n               pass\n        '
        self.compile(codestr)

    def test_inline_import_supported(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n        def f():\n            import math\n            return math.isnan\n        '
        self.compile(codestr)

    def test_inline_import_as_supported(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        def f():\n            import os.path as road # Modernization.\n            return road.exists\n        '
        self.compile(codestr)

    def test_inline_from_import_names_supported(self) -> None:
        if False:
            return 10
        acode = '\n        x: int = 42\n        '
        bcode = '\n            def f():\n                from a import x\n                return x\n        '
        bcomp = self.compiler(a=acode, b=bcode).compile_module('b')

    def test_inline_from_import_names_supported_alias(self) -> None:
        if False:
            while True:
                i = 10
        acode = '\n        x: int = 42\n        '
        bcode = '\n            def f():\n                from a import x as y\n                return y\n        '
        bcomp = self.compiler(a=acode, b=bcode).compile_module('b')

    def test_unknown_decorated_functions_declared(self) -> None:
        if False:
            return 10
        codestr = '\n            def foo(x):\n                return x\n            def bar():\n                baz()\n            @foo\n            def baz():\n                pass\n        '
        self.compile(codestr)

    def test_cellvars_known(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            def use(x):\n                return x\n\n            def foo(x):\n                use(x)\n                def nested():\n                    return x\n                return nested\n        '
        self.compile(codestr)

    def test_name_defined_in_except_and_else_known(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n            def foo(self):\n                try:\n                    pass\n                except Exception:\n                    a = None\n                else:\n                    a = None\n                return a\n        '
        self.compile(codestr)

    def test_name_defined_only_in_else_unknown(self) -> None:
        if False:
            while True:
                i = 10
        codestr = '\n            def foo(self):\n                try:\n                    pass\n                except Exception:\n                    pass\n                else:\n                    a = None\n                return a\n        '
        self.type_error(codestr, 'Name `a` is not defined.')

    def test_name_defined_only_in_if_unknown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            def foo(self, p):\n                if p:\n                    a = None\n                return a\n        '
        self.type_error(codestr, 'Name `a` is not defined.')

    def test_name_defined_only_in_else_unknown(self) -> None:
        if False:
            print('Hello World!')
        codestr = '\n            def foo(self, p):\n                if p:\n                    pass\n                else:\n                    a = None\n                return a\n        '
        self.type_error(codestr, 'Name `a` is not defined.')

    def test_name_defined_terminal_except_raises(self) -> None:
        if False:
            i = 10
            return i + 15
        codestr = '\n            def foo(self):\n                try:\n                    a = None\n                except:\n                    raise Exception\n                return a\n        '
        self.compile(codestr)

    def test_name_defined_terminal_except_returns(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n            def foo(self):\n                try:\n                    a = None\n                except:\n                    return None\n                return a\n        '
        self.compile(codestr)