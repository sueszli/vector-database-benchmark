import ast
import re
from compiler.static import StaticCodeGenerator
from compiler.static.compiler import Compiler
from compiler.static.module_table import ModuleTable, ModuleTableException
from compiler.static.types import Class, TypeName
from textwrap import dedent
from .common import bad_ret_type, StaticTestBase

class DeclarationVisitorTests(StaticTestBase):

    def test_cross_module(self) -> None:
        if False:
            return 10
        acode = '\n            class C:\n                def f(self):\n                    return 42\n        '
        bcode = '\n            from a import C\n\n            def f():\n                x = C()\n                return x.f()\n        '
        bcomp = self.compiler(a=acode, b=bcode).compile_module('b')
        x = self.find_code(bcomp, 'f')
        self.assertInBytecode(x, 'INVOKE_FUNCTION', (('a', 'C', 'f'), 1))

    def test_cross_module_nested(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        for (parent, close) in [('if FOO:', ''), ('for x in []:', ''), ('while True:', ''), ('with foo:', ''), ('try:', 'except: pass')]:
            with self.subTest(parent=parent, close=close):
                acode = f'\n                    {parent}\n                        class C:\n                            def f(self):\n                                return 42\n                    {close}\n                '
                bcode = '\n                    from a import C\n\n                    def f():\n                        x = C()\n                        return x.f()\n                '
                bcomp = self.compiler(a=acode, b=bcode).compile_module('b')
                x = self.find_code(bcomp, 'f')
                self.assertNotInBytecode(x, 'INVOKE_METHOD', (('a', 'C', 'f'), 0))

    def test_cross_module_inst_decl_visit_only(self) -> None:
        if False:
            i = 10
            return i + 15
        acode = '\n            class C:\n                def f(self):\n                    return 42\n\n            x: C = C()\n        '
        bcode = '\n            from a import x\n\n            def f():\n                return x.f()\n        '
        bcomp = self.compiler(a=acode, b=bcode).compile_module('b')
        x = self.find_code(bcomp, 'f')
        self.assertInBytecode(x, 'INVOKE_METHOD', (('a', 'C', 'f'), 0))

    def test_cross_module_inst_decl_final_dynamic_is_invoked(self) -> None:
        if False:
            while True:
                i = 10
        acode = '\n            from typing import Final, Protocol\n            def foo(x: int) -> int:\n                    return x + 42\n\n            class CallableProtocol(Protocol):\n                def __call__(self, x: int) -> int:\n                    pass\n\n            f: Final[CallableProtocol] = foo\n        '
        bcode = '\n            from a import f\n\n            def g():\n                return f(1)\n        '
        bcomp = self.compiler(a=acode, b=bcode).compile_module('b')
        x = self.find_code(bcomp, 'g')
        self.assertInBytecode(x, 'INVOKE_FUNCTION')

    def test_cross_module_inst_decl_alias_is_not_invoked(self) -> None:
        if False:
            return 10
        acode = '\n            from typing import Final, Protocol\n            def foo(x: int) -> int:\n                    return x + 42\n            f = foo\n        '
        bcode = '\n            from a import f\n\n            def g():\n                return f(1)\n        '
        bcomp = self.compiler(a=acode, b=bcode).compile_module('b')
        x = self.find_code(bcomp, 'g')
        self.assertNotInBytecode(x, 'INVOKE_FUNCTION')

    def test_cross_module_decl_visit_type_check_methods(self) -> None:
        if False:
            i = 10
            return i + 15
        acode = '\n            class C:\n                def f(self, x: int = 42) -> int:\n                    return x\n        '
        bcode = "\n            from a import C\n\n            def f():\n                return C().f('abc')\n        "
        self.compiler(a=acode, b=bcode).type_error('b', re.escape("type mismatch: str received for positional arg 'x', expected int"), at="'abc'")
        bcode = '\n            from a import C\n\n            def f() -> str:\n                return C().f(42)\n        '
        self.compiler(a=acode, b=bcode).type_error('b', bad_ret_type('int', 'str'), at='return')

    def test_cross_module_decl_visit_type_check_fields(self) -> None:
        if False:
            i = 10
            return i + 15
        acode = '\n            class C:\n                def __init__(self):\n                    self.x: int = 42\n        '
        bcode = "\n            from a import C\n\n            def f():\n                C().x = 'abc'\n        "
        self.compiler(a=acode, b=bcode).type_error('b', re.escape('type mismatch: str cannot be assigned to int'), at='C().x')
        bcode = '\n            from a import C\n\n            def f() -> str:\n                return C().x\n        '
        self.compiler(a=acode, b=bcode).type_error('b', bad_ret_type('int', 'str'), at='return')

    def test_cross_module_import_time_resolution(self) -> None:
        if False:
            return 10
        acode = '\n            class C:\n                def f(self):\n                    return 42\n        '
        bcode = '\n            from a import C\n\n            def f():\n                x = C()\n                return x.f()\n        '
        bcomp = self.compiler(a=acode, b=bcode).compile_module('b')
        x = self.find_code(bcomp, 'f')
        self.assertInBytecode(x, 'INVOKE_FUNCTION', (('a', 'C', 'f'), 1))

    def test_cross_module_type_checking(self) -> None:
        if False:
            i = 10
            return i + 15
        acode = '\n            class C:\n                def f(self):\n                    return 42\n        '
        bcode = '\n            from typing import TYPE_CHECKING\n\n            if TYPE_CHECKING:\n                from a import C\n\n            def f(x: C):\n                return x.f()\n        '
        bcomp = self.compiler(a=acode, b=bcode).compile_module('b')
        x = self.find_code(bcomp, 'f')
        self.assertInBytecode(x, 'INVOKE_METHOD', (('a', 'C', 'f'), 0))

    def test_cross_module_rewrite(self) -> None:
        if False:
            while True:
                i = 10
        acode = '\n            from b import B\n            class C(B):\n                def f(self):\n                    return self.g()\n        '
        bcode = '\n            class B:\n                def g(self):\n                    return 1 + 2\n        '
        testcase = self

        class CustomCompiler(Compiler):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__(StaticCodeGenerator)

            def import_module(self, name: str, optimize: int) -> ModuleTable:
                if False:
                    while True:
                        i = 10
                if name == 'b':
                    btree = ast.parse(dedent(bcode))
                    self.btree = self.add_module('b', 'b.py', btree, optimize=optimize)
                    testcase.assertFalse(self.btree is btree)
        compiler = CustomCompiler()
        acomp = compiler.compile('a', 'a.py', ast.parse(dedent(acode)), optimize=1)
        bcomp = compiler.compile('b', 'b.py', compiler.btree, optimize=1)
        x = self.find_code(self.find_code(acomp, 'C'), 'f')
        self.assertInBytecode(x, 'INVOKE_METHOD', (('b', 'B', 'g'), 0))

    def test_declaring_toplevel_local_after_decl_visit_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        codestr = '\n        class C:\n            pass\n        '

        class CustomCodeGenerator(StaticCodeGenerator):

            def visitClassDef(self, node):
                if False:
                    return 10
                super().visitClassDef(node)
                self.cur_mod.declare_class(node, Class(TypeName('mod', 'C'), self.compiler.type_env))

        class CustomCompiler(Compiler):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__(CustomCodeGenerator)

            def import_module(self, name: str, optimize: int) -> ModuleTable:
                if False:
                    while True:
                        i = 10
                if name == 'b':
                    btree = ast.parse(dedent(bcode))
        compiler = CustomCompiler()
        with self.assertRaisesRegex(ModuleTableException, 'Attempted to declare a class after the declaration visit'):
            compiler.compile('a', 'a.py', ast.parse(dedent(codestr)), optimize=1)