from __future__ import annotations
import ast
import symtable
import sys
import unittest
from compiler.strict import strict_compile
from compiler.strict.common import FIXED_MODULES
from compiler.strict.loader import StrictModule
from compiler.strict.rewriter import rewrite
from textwrap import dedent
from types import CoroutineType, FunctionType, ModuleType
from typing import Any, Dict, final, List, Optional, Set, Type, TypeVar
from weakref import ref
from .common import StrictTestWithCheckerBase

class RewriterTestCase(StrictTestWithCheckerBase):

    def compile_to_strict(self, code: str, builtins: Dict[str, Any]=__builtins__, modules: Optional[Dict[str, Dict[str, Any]]]=None, globals: Optional[Dict[str, Any]]=None) -> StrictModule:
        if False:
            return 10
        code = dedent(code)
        root = ast.parse(code)
        name = 'foo'
        filename = 'foo.py'
        symbols = symtable.symtable(code, filename, 'exec')
        root = rewrite(root, symbols, filename, name, builtins=builtins)
        c = strict_compile(name, filename, root)

        def freeze_type(freeze: Type[object]) -> None:
            if False:
                for i in range(10):
                    print('nop')
            pass

        def loose_slots(freeze: Type[object]) -> None:
            if False:
                for i in range(10):
                    print('nop')
            pass

        def strict_slots(typ: Type[object]) -> Type[object]:
            if False:
                for i in range(10):
                    print('nop')
            return typ
        fixed_modules = modules or dict(FIXED_MODULES)
        fixed_modules.update(__strict__={'freeze_type': freeze_type, 'loose_slots': loose_slots, 'strict_slots': strict_slots})
        additional_dicts = globals or {}
        additional_dicts.update({'<fixed-modules>': fixed_modules, '<builtins>': builtins})
        (d, m) = self._exec_strict_code(c, name, additional_dicts=additional_dicts)
        return m

@final
class ImmutableModuleTestCase(RewriterTestCase):

    def test_simple(self) -> None:
        if False:
            i = 10
            return i + 15
        code = '\nx = 1\ndef f():\n    return x\n'
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.x, 1)
        self.assertEqual(type(mod.f), FunctionType)
        self.assertEqual(mod.f(), 1)
        self.assertEqual(mod.f.__name__, 'f')

    def test_decorators(self) -> None:
        if False:
            print('Hello World!')
        code = '\nfrom __strict__ import strict_slots\ndef dec(x):\n    return x\n\n@dec\n@strict_slots\ndef f():\n    return 1\n'
        mod = self.compile_to_strict(code)
        self.assertEqual(type(mod.f), FunctionType)
        self.assertEqual(type(mod.dec), FunctionType)
        self.assertEqual(type(mod.f()), int)
        code = '\nfrom __strict__ import strict_slots\ndef dec(x):\n    return x\n\n@dec\n@strict_slots\nclass C:\n    x = 1\n'
        mod = self.compile_to_strict(code)
        self.assertEqual(type(mod.C), type)
        self.assertEqual(type(mod.dec), FunctionType)
        self.assertEqual(type(mod.C.x), int)

    def test_visit_method_global(self) -> None:
        if False:
            i = 10
            return i + 15
        'test visiting an explicit global decl inside of a nested scope'
        code = '\nfrom __strict__ import strict_slots\nX = 1\n@strict_slots\nclass C:\n    def f(self):\n        global X\n        X = 2\n        return X\n'
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.C().f(), 2)

    def test_class_def(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        code = '\nfrom __strict__ import strict_slots\nx = 42\n@strict_slots\nclass C:\n    def f(self):\n        return x\n    '
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.C.__name__, 'C')
        self.assertEqual(mod.C().f(), 42)

    def test_nested_class_def(self) -> None:
        if False:
            return 10
        code = '\nfrom __strict__ import strict_slots\nx = 42\n@strict_slots\nclass C:\n    def f(self):\n        return x\n    @strict_slots\n    class D:\n        def g(self):\n            return x\n    '
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.C.__name__, 'C')
        self.assertEqual(mod.C.__qualname__, 'C')
        self.assertEqual(mod.C.D.__name__, 'D')
        self.assertEqual(mod.C.D.__qualname__, 'C.D')
        self.assertEqual(mod.C.f.__name__, 'f')
        self.assertEqual(mod.C.f.__qualname__, 'C.f')
        self.assertEqual(mod.C.D.g.__name__, 'g')
        self.assertEqual(mod.C.D.g.__qualname__, 'C.D.g')
        self.assertEqual(mod.C().f(), 42)
        self.assertEqual(mod.C.D().g(), 42)

@final
class LazyLoadingTestCases(RewriterTestCase):
    """test cases which verify the behavior of lazy loading is the same as
    non-lazy"""

    def test_lazy_load_exception(self) -> None:
        if False:
            i = 10
            return i + 15
        'lazy code raising an exception should run'
        code = "\nraise Exception('no way')\n    "
        with self.assertRaises(Exception) as e:
            self.compile_to_strict(code)
        self.assertEqual(e.exception.args[0], 'no way')

    def test_lazy_load_exception_2(self) -> None:
        if False:
            i = 10
            return i + 15
        code = "\nfrom __strict__ import strict_slots\n@strict_slots\nclass MyException(Exception):\n    pass\n\nraise MyException('no way')\n    "
        with self.assertRaises(Exception) as e:
            self.compile_to_strict(code)
        self.assertEqual(type(e.exception).__name__, 'MyException')

    def test_lazy_load_exception_3(self) -> None:
        if False:
            i = 10
            return i + 15
        code = "\nfrom pickle import PicklingError\n\nraise PicklingError('no way')\n"
        with self.assertRaises(Exception) as e:
            self.compile_to_strict(code)
        self.assertEqual(type(e.exception).__name__, 'PicklingError')

    def test_lazy_load_exception_4(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        code = '\nraise ShouldBeANameError()\n'
        with self.assertRaises(NameError):
            self.compile_to_strict(code)

    def test_lazy_load_no_reinit(self) -> None:
        if False:
            while True:
                i = 10
        'only run earlier initialization once'
        code = '\ntry:\n    y.append(0)\nexcept:\n    y = []\ntry:\n    y.append(1)\n    raise Exception()\nexcept:\n    pass\nz = y\n'
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.z, [1])

    def test_finish_initialization(self) -> None:
        if False:
            return 10
        'values need to be fully initialized upon their first access'
        code = '\nx = 1\ny = x\nx = 2\n'
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.y, 1)
        self.assertEqual(mod.x, 2)

    def test_full_initialization(self) -> None:
        if False:
            i = 10
            return i + 15
        'values need to be fully initialized upon their first access'
        code = '\nx = 1\ny = x\nx = 2\n'
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.x, 2)
        self.assertEqual(mod.y, 1)

    def test_transitive_closure(self) -> None:
        if False:
            i = 10
            return i + 15
        'we run the transitive closure of things required to be initialized'
        code = '\nx = 1\ny = x\nz = y\n'
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.z, 1)

    def test_annotations(self) -> None:
        if False:
            while True:
                i = 10
        'annotations are properly initialized'
        code = '\nx: int = 1\n    '
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.__annotations__, {'x': int})
        self.assertEqual(mod.x, 1)

    def test_annotations_no_value(self) -> None:
        if False:
            while True:
                i = 10
        'annotations are properly initialized w/o values'
        code = '\nx: int\n    '
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.__annotations__, {'x': int})
        with self.assertRaises(AttributeError):
            mod.x

    def test_annotations_del(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'values deleted after use are deleted, when accessed after initial var'
        code = '\nx = 1\ny = x\ndel x\n    '
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.y, 1)
        with self.assertRaises(AttributeError):
            mod.x

    def test_annotations_del_2(self) -> None:
        if False:
            while True:
                i = 10
        'deleted values are deleted when accessed initially, previous values are okay'
        code = '\nx = 1\ny = x\ndel x\n    '
        mod = self.compile_to_strict(code)
        with self.assertRaises(AttributeError):
            mod.x
        self.assertEqual(mod.y, 1)

    def test_forward_dep(self) -> None:
        if False:
            while True:
                i = 10
        'forward dependencies cause all values to be initialized'
        code = '\nfrom __strict__ import strict_slots\n@strict_slots\nclass C:\n    pass\nC.x = 42\n    '
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.C.x, 42)

    def test_not_init(self) -> None:
        if False:
            i = 10
            return i + 15
        "unassigned values don't show up (definite assignment would disallow this)"
        code = '\nx = 1\nif x != 1:\n    y = 2\n    '
        mod = self.compile_to_strict(code)
        with self.assertRaises(AttributeError):
            mod.y

    def test_try_except_shadowed_handler_no_body_changes(self) -> None:
        if False:
            print('Hello World!')
        "the try body doesn't get rewritten, but the except handler does"
        code = '\ntry:\n    x = 2\nexcept Exception as min:\n    pass\n    '
        mod = self.compile_to_strict(code)
        self.assertEqual(mod.x, 2)
        self.assertFalse(hasattr(mod, 'min'))