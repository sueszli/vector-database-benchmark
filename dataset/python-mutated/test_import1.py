"""Tests for import."""
from pytype import file_utils
from pytype import imports_map_loader
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils
DEFAULT_PYI = '\nfrom typing import Any\ndef __getattr__(name) -> Any: ...\n'

class FakeOptions:
    """Fake options."""

    def __init__(self):
        if False:
            return 10
        self.open_function = open

class ImportTest(test_base.BaseTest):
    """Tests for import."""

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()
        cls.builder = imports_map_loader.ImportsMapBuilder(FakeOptions())

    def build_imports_map(self, path):
        if False:
            print('Hello World!')
        return self.builder.build_from_file(path)

    def test_basic_import(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import sys\n      ')
        self.assertTypesMatchPytd(ty, '\n       import sys\n    ')

    def test_basic_import2(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      import bad_import  # doesn't exist\n      ", report_errors=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      bad_import = ...  # type: Any\n    ')

    def test_from_import_smoke(self):
        if False:
            return 10
        self.assertNoCrash(self.Check, '\n      from sys import exit\n      from path.to.module import bar, baz\n      ')

    def test_long_from(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('path/to/my_module.pyi'), 'def foo() -> str: ...')
            ty = self.Infer('\n      from path.to import my_module\n      def foo():\n        return my_module.foo()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from path.to import my_module\n        def foo() -> str: ...\n      ')

    def test_star_import_smoke(self):
        if False:
            print('Hello World!')
        self.Check('\n      from sys import *\n      ')

    def test_star_import_unknown_smoke(self):
        if False:
            i = 10
            return i + 15
        self.assertNoCrash(self.Check, '\n      from unknown_module import *\n      ')

    def test_star_import(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('my_module.pyi', '\n        def f() -> str: ...\n        class A:\n          pass\n        a = ...  # type: A\n      ')
            ty = self.Infer('\n      from my_module import *\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Type\n        A = ...  # type: Type[my_module.A]\n        a = ...  # type: my_module.A\n        def f() -> str: ...\n      ')

    def test_star_import_any(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', DEFAULT_PYI)
            ty = self.Infer('\n        from a import *\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        def __getattr__(name) -> Any: ...\n      ')

    def test_star_import_in_pyi(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        class X: ...\n      ')
            d.create_file('b.pyi', '\n        from a import *\n        class Y(X): ...\n      ')
            ty = self.Infer('\n      from b import *\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        import b\n        from typing import Type\n        X = ...  # type: Type[a.X]\n        Y = ...  # type: Type[b.Y]\n      ')

    def test_bad_star_import(self):
        if False:
            print('Hello World!')
        (ty, _) = self.InferWithErrors('\n      from nonsense import *  # import-error\n      from other_nonsense import *  # import-error\n      x = foo.bar()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def __getattr__(name) -> Any: ...\n      x = ...  # type: Any\n    ')

    def test_path_import(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('path/to/my_module.pyi'), 'def qqsv() -> str: ...')
            d.create_file(file_utils.replace_separator('path/to/__init__.pyi'), '')
            d.create_file(file_utils.replace_separator('path/__init__.pyi'), '')
            ty = self.Infer('\n      import path.to.my_module\n      def foo():\n        return path.to.my_module.qqsv()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import path\n        def foo() -> str: ...\n      ')

    def test_path_import2(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('path/to/my_module.pyi'), 'def qqsv() -> str: ...')
            d.create_file(file_utils.replace_separator('path/to/__init__.pyi'), '')
            d.create_file(file_utils.replace_separator('path/__init__.pyi'), '')
            ty = self.Infer("\n      import nonexistant_path.to.my_module  # doesn't exist\n      def foo():\n        return path.to.my_module.qqsv()\n      ", deep=True, report_errors=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        nonexistant_path = ...  # type: Any\n        def foo() -> Any: ...\n      ')

    def test_import_all(self):
        if False:
            return 10
        self.assertNoCrash(self.Check, '\n      from module import *\n      from path.to.module import *\n      ')

    def test_assign_member(self):
        if False:
            print('Hello World!')
        self.Check('\n      import sys\n      sys.path = []\n      ')

    def test_return_module(self):
        if False:
            return 10
        ty = self.Infer('\n        import sys\n\n        def f():\n          return sys\n    ')
        self.assertTypesMatchPytd(ty, '\n      import sys\n      def f() -> module: ...\n    ')

    def test_match_module(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import sys\n      def f():\n        if getattr(sys, "foobar"):\n          return list({sys: sys}.keys())[0]\n        else:\n          return sys\n    ')
        self.assertTypesMatchPytd(ty, '\n      import sys\n      def f() -> module: ...\n    ')

    def test_sys(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import sys\n      def f():\n        return sys.path\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      import sys\n      def f() -> List[str]: ...\n    ')

    def test_from_sys_import(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from sys import path\n      def f():\n        return path\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      path = ...  # type: List[str]\n      def f() -> List[str]: ...\n    ')

    def test_stdlib(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import datetime\n      def f():\n        return datetime.timedelta().total_seconds()\n    ')
        self.assertTypesMatchPytd(ty, '\n      import datetime\n      def f() -> float: ...\n    ')

    def test_import_pytd(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('other_file.pyi', '\n        def f() -> int: ...\n      ')
            d.create_file('main.py', '\n        from other_file import f\n      ')
            ty = self.InferFromFile(filename=d['main.py'], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        def f() -> int: ...\n      ')

    def test_import_pytd2(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('other_file.pyi', '\n        def f() -> int: ...\n      ')
            d.create_file('main.py', '\n        from other_file import f\n        def g():\n          return f()\n      ')
            ty = self.InferFromFile(filename=d['main.py'], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        def f() -> int: ...\n        def g() -> int: ...\n      ')

    def test_import_directory(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('sub/other_file.pyi'), 'def f() -> int: ...')
            d.create_file(file_utils.replace_separator('sub/bar/baz.pyi'), 'def g() -> float: ...')
            d.create_file(file_utils.replace_separator('sub/__init__.pyi'), '')
            d.create_file(file_utils.replace_separator('sub/bar/__init__.pyi'), '')
            d.create_file('main.py', '\n        from sub import other_file\n        import sub.bar.baz\n        from sub.bar.baz import g\n        def h():\n          return other_file.f()\n        def i():\n          return g()\n        def j():\n          return sub.bar.baz.g()\n      ')
            ty = self.InferFromFile(filename=d['main.py'], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, "\n        import sub  # from 'import sub.bar.baz'\n        from sub import other_file\n        def g() -> float: ...\n        def h() -> int: ...\n        def i() -> float: ...\n        def j() -> float: ...\n      ")

    def test_import_init(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('sub/__init__.pyi'), '\n        def f() -> int: ...\n      ')
            d.create_file('main.py', '\n        from sub import f\n        def g():\n          return f()\n      ')
            ty = self.InferFromFile(filename=d['main.py'], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        def f() -> int: ...\n        def g() -> int: ...\n      ')

    def test_import_name(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class A:\n          pass\n        def f() -> A: ...\n      ')
            d.create_file('main.py', '\n        from foo import f\n        def g():\n          return f()\n      ')
            ty = self.InferFromFile(filename=d['main.py'], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        def f() -> foo.A: ...\n        def g() -> foo.A: ...\n    ')

    def test_deep_dependency(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', 'x = ...  # type: bar.Bar')
            d.create_file('bar.pyi', '\n          class Bar:\n            def bar(self) -> int: ...\n      ')
            d.create_file('main.py', '\n        from foo import x\n        def f():\n          return x.bar()\n      ')
            ty = self.InferFromFile(filename=d['main.py'], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        x = ...  # type: bar.Bar\n        def f() -> int: ...\n    ')

    def test_relative_import(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/baz.pyi'), 'x = ...  # type: int')
            d.create_file(file_utils.replace_separator('foo/bar.py'), '\n        from . import baz\n        def f():\n          return baz.x\n      ')
            d.create_file(file_utils.replace_separator('foo/__init__.pyi'), '')
            ty = self.InferFromFile(filename=d[file_utils.replace_separator('foo/bar.py')], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from foo import baz\n        def f() -> int: ...\n    ')

    def test_dot_package(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('up1/foo.py'), '\n        from .bar import x\n      ')
            d.create_file(file_utils.replace_separator('up1/bar.pyi'), 'x = ...  # type: int')
            d.create_file(file_utils.replace_separator('up1/__init__.pyi'), '')
            d.create_file('__init__.pyi', '')
            ty = self.InferFromFile(filename=d[file_utils.replace_separator('up1/foo.py')], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        x = ...  # type: int\n    ')

    def test_dot_dot_package(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('up2/baz/foo.py'), '\n        from ..bar import x\n      ')
            d.create_file(file_utils.replace_separator('up2/bar.pyi'), 'x = ...  # type: int')
            d.create_file('__init__.pyi', '')
            d.create_file(file_utils.replace_separator('up2/__init__.pyi'), '')
            d.create_file(file_utils.replace_separator('up2/baz/__init__.pyi'), '')
            ty = self.InferFromFile(filename=d[file_utils.replace_separator('up2/baz/foo.py')], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        x = ...  # type: int\n      ')

    def test_dot_package_no_init(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.py', '\n        from .bar import x\n      ')
            d.create_file('bar.pyi', 'x = ...  # type: int')
            ty = self.InferFromFile(filename=d['foo.py'], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        x = ...  # type: int\n      ')

    def test_dot_dot_packag_no_init(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('baz/foo.py'), '\n        from ..bar import x\n      ')
            d.create_file('bar.pyi', 'x = ...  # type: int')
            ty = self.InferFromFile(filename=d[file_utils.replace_separator('baz/foo.py')], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        x = ...  # type: int\n      ')

    def test_dot_dot(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/baz.pyi'), 'x = ...  # type: int')
            d.create_file(file_utils.replace_separator('foo/deep/bar.py'), '\n        from .. import baz\n        def f():\n          return baz.x\n      ')
            d.create_file(file_utils.replace_separator('foo/__init__.pyi'), '')
            d.create_file(file_utils.replace_separator('foo/deep/__init__.pyi'), '')
            ty = self.InferFromFile(filename=d[file_utils.replace_separator('foo/deep/bar.py')], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from foo import baz\n        def f() -> int: ...\n    ')

    def test_dot_dot_package_in_pyi(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('up2/baz/foo.pyi'), '\n        from ..bar import X\n      ')
            d.create_file(file_utils.replace_separator('up2/bar.pyi'), 'class X: ...')
            d.create_file('top.py', '\n                    from up2.baz.foo import X\n                    x = X()\n                    ')
            ty = self.InferFromFile(filename=d['top.py'], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Type\n        import up2.bar\n        X = ...  # type: Type[up2.bar.X]\n        x = ...  # type: up2.bar.X\n      ')

    def test_dot_dot_in_pyi(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/baz.pyi'), 'x: int')
            d.create_file(file_utils.replace_separator('foo/deep/bar.py'), '\n        from .. import baz\n        a = baz.x\n      ')
            ty = self.InferFromFile(filename=d[file_utils.replace_separator('foo/deep/bar.py')], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from foo import baz\n        a: int\n      ')

    def test_too_many_dots_in_package_in_pyi(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('up/foo.pyi'), 'from ..bar import X')
            d.create_file(file_utils.replace_separator('up/bar.pyi'), 'class X: ...')
            (_, err) = self.InferWithErrors('from up.foo import X  # pyi-error[e]', pythonpath=[d.path])
            self.assertErrorRegexes(err, {'e': 'Cannot resolve relative import \\.\\.bar'})

    def test_from_dot_in_pyi(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/a.pyi'), 'class X: ...')
            d.create_file(file_utils.replace_separator('foo/b.pyi'), '\n        from . import a\n        Y = a.X')
            d.create_file('top.py', '\n        import foo.b\n        x = foo.b.Y() ')
            ty = self.InferFromFile(filename=d['top.py'], pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Type\n        import foo\n        x = ...  # type: foo.a.X\n      ')

    def test_unused_from_dot_in_pyi(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/a.pyi'), 'class X: ...')
            d.create_file(file_utils.replace_separator('foo/b.pyi'), 'from . import a')
            self.Check('import foo.b', pythonpath=[d.path])

    def test_file_import1(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('path/to/some/module.pyi'), 'def foo(x:int) -> str: ...')
            d.create_file(file_utils.replace_separator('path/to/some/__init__.pyi'), '')
            d.create_file(file_utils.replace_separator('path/to/__init__.pyi'), '')
            d.create_file(file_utils.replace_separator('path/__init__.pyi'), '')
            ty = self.Infer('\n        import path.to.some.module\n        def my_foo(x):\n          return path.to.some.module.foo(x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import path\n        def my_foo(x) -> str: ...\n      ')

    def test_file_import2(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('path/to/some/module.pyi'), 'def foo(x:int) -> str: ...')
            d.create_file(file_utils.replace_separator('path/to/some/__init__.pyi'), '')
            d.create_file(file_utils.replace_separator('path/to/__init__.pyi'), '')
            d.create_file(file_utils.replace_separator('path/__init__.pyi'), '')
            ty = self.Infer('\n        from path.to.some import module\n        def my_foo(x):\n          return module.foo(x)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from path.to.some import module\n        def my_foo(x) -> str: ...\n      ')

    @test_base.skip('flaky')
    def test_solve_for_imported(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import StringIO\n      def my_foo(x):\n        return x.read()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Union\n      StringIO = ...  # type: module\n      def my_foo(x: Union[StringIO.StringIO[object], typing.IO[object],\n                          typing.BinaryIO, typing.TextIO]) -> Any\n    ')

    def test_import_builtins(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import builtins as __builtin__\n\n      def f():\n        return __builtin__.int()\n    ')
        self.assertTypesMatchPytd(ty, '\n      import builtins as __builtin__\n\n      def f() -> int: ...\n    ')

    def test_imported_method_as_class_attribute(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import os\n      class Foo:\n        kill = os.kill\n    ')
        self.assertTypesMatchPytd(ty, '\n      import os\n      class Foo:\n        def kill(__pid: int, __signal: int) -> None: ...\n    ')

    def test_match_against_imported(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Foo:\n          pass\n        class Bar:\n          def f1(self, x: Foo) -> Baz: ...\n        class Baz:\n          pass\n      ')
            ty = self.Infer('\n        import foo\n        def f(x, y):\n          return x.f1(y)\n        def g(x):\n          return x.f1(foo.Foo())\n        class FooSub(foo.Foo):\n          pass\n        def h(x):\n          return x.f1(FooSub())\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        import foo\n        def f(x, y) -> Any: ...\n        def g(x) -> Any: ...\n        def h(x) -> Any: ...\n\n        class FooSub(foo.Foo):\n          pass\n      ')

    def test_imported_constants(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('module.pyi', '\n        x = ...  # type: int\n        class Foo:\n          x = ...  # type: float\n      ')
            ty = self.Infer('\n        import module\n        def f():\n          return module.x\n        def g():\n          return module.Foo().x\n        def h():\n          return module.Foo.x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import module\n        def f() -> int: ...\n        def g() -> float: ...\n        def h() -> float: ...\n      ')

    def test_circular(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('x.pyi', '\n          class X:\n            pass\n          y = ...  # type: y.Y\n          z = ...  # type: z.Z\n      ')
            d.create_file('y.pyi', '\n          class Y:\n            pass\n          x = ...  # type: x.X\n      ')
            d.create_file('z.pyi', '\n          class Z:\n            pass\n          x = ...  # type: x.X\n      ')
            ty = self.Infer('\n        import x\n        xx = x.X()\n        yy = x.y\n        zz = x.z\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import x\n        xx = ...  # type: x.X\n        yy = ...  # type: y.Y\n        zz = ...  # type: z.Z\n      ')

    def test_reimport(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n          from collections import OrderedDict as MyOrderedDict\n      ')
            ty = self.Infer('\n        import foo\n        d = foo.MyOrderedDict()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import OrderedDict\n        d = ...  # type: OrderedDict[nothing, nothing]\n      ')

    def test_import_function(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import SupportsFloat\n        def pow(__x: SupportsFloat, __y: SupportsFloat) -> float: ...\n      ')
            d.create_file('bar.pyi', '\n          from foo import pow as mypow\n      ')
            ty = self.Infer('\n        import bar\n        d = bar.mypow\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import SupportsFloat\n        import bar\n        def d(__x: SupportsFloat, __y: SupportsFloat) -> float: ...\n      ')

    def test_import_constant(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('mymath.pyi', '\n          from math import pi as half_tau\n      ')
            ty = self.Infer('\n        import mymath\n        from mymath import half_tau as x\n        y = mymath.half_tau\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import mymath\n        x = ...  # type: float\n        y = ...  # type: float\n      ')

    def test_import_map(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            foo_filename = d.create_file('foo.pyi', '\n          bar = ...  # type: int\n      ')
            imports_map_filename = d.create_file('imports_map.txt', '\n          foo %s\n      ' % foo_filename)
            imports_map = self.build_imports_map(imports_map_filename)
            ty = self.Infer('\n        from foo import bar\n      ', deep=False, imports_map=imports_map, pythonpath=[''])
            self.assertTypesMatchPytd(ty, '\n        bar = ...  # type: int\n      ')

    def test_import_resolve_on_dummy(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', DEFAULT_PYI)
            d.create_file('b.pyi', '\n          from a import Foo\n          def f(x: Foo) -> Foo: ...\n      ')
            ty = self.Infer('\n        import b\n        foo = b.Foo()\n        bar = b.f(foo)\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import b\n        from typing import Any\n        foo = ...  # type: Any\n        bar = ...  # type: Any\n      ')

    def test_two_level(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        +++ /&* unparsable *&/ +++\n      ')
            d.create_file('b.pyi', '\n        import a\n        class B(a.A):\n          pass\n      ')
            (_, errors) = self.InferWithErrors('\n        import b  # pyi-error[e]\n        x = b.B()\n      ', pythonpath=[d.path])
        self.assertErrorRegexes(errors, {'e': 'a\\.pyi'})

    def test_subdir_and_module_with_same_name_as_package(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('pkg/__init__.pyi'), '\n          from pkg.pkg.pkg import *\n          from pkg.bar import *')
            d.create_file(file_utils.replace_separator('pkg/pkg/pkg.pyi'), '\n          class X: pass')
            d.create_file(file_utils.replace_separator('pkg/bar.pyi'), '\n          class Y: pass')
            ty = self.Infer('\n        import pkg\n        a = pkg.X()\n        b = pkg.Y()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import pkg\n        a = ...  # type: pkg.pkg.pkg.X\n        b = ...  # type: pkg.bar.Y\n      ')

    def test_redefined_builtin(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        object = ...  # type: Any\n        def f(x) -> Any: ...\n      ')
            ty = self.Infer('\n        import foo\n        x = foo.f("")\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any\n        import foo\n        x = ...  # type: Any\n      ')

    def test_redefined_builtin2(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class object:\n          def foo(self) -> None: ...\n        def f(x: object) -> object: ...\n      ')
            (ty, _) = self.InferWithErrors('\n        import foo\n        x = foo.f(foo.object())\n        y = foo.f(foo.object())\n        foo.f(object())  # wrong-arg-types\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        x = ...  # type: foo.object\n        y = ...  # type: foo.object\n      ')

    def test_no_fail_on_bad_symbol_lookup(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: FooBar) -> FooBar: ...\n      ')
            self.assertNoCrash(self.Check, '\n        import foo\n      ', pythonpath=[d.path])

    @test_base.skip("instantiating 'type' should use 'Type[Any]', not 'Any'")
    def test_import_type_factory(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        def factory() -> type: ...\n      ')
            ty = self.Infer('\n        import a\n        A = a.factory()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        A = ...  # type: type\n      ')

    def test_get_bad_submodule_as_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/__init__.pyi'), '')
            d.create_file(file_utils.replace_separator('foo/bar.pyi'), 'nonsense')
            self.assertNoCrash(self.Check, '\n        import foo\n        x = foo.bar\n      ', pythonpath=[d.path])

    def test_ignored_import(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import sys  # type: ignore\n      import foobar  # type: ignore\n      from os import path  # type: ignore\n      a = sys.rumplestiltskin\n      b = sys.stderr\n      c = foobar.rumplestiltskin\n      d = path.curdir\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      sys = ...  # type: Any\n      foobar = ...  # type: Any\n      path = ...  # type: Any\n      a = ...  # type: Any\n      b = ...  # type: Any\n      c = ...  # type: Any\n      d = ...  # type: Any\n    ')

    def test_attribute_on_module(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        foo = ...  # type: int\n      ')
            (_, errors) = self.InferWithErrors('\n        from a import foo, bar  # import-error[e1]\n        import a\n        a.baz  # module-attr[e2]\n      ', pythonpath=[d.path])
        self.assertErrorRegexes(errors, {'e1': 'bar', 'e2': 'baz'})

    def test_from_import(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/b.pyi'), '\n        from foo import c\n        class bar(c.X): ...\n      ')
            d.create_file(file_utils.replace_separator('foo/c.pyi'), '\n        class X: ...\n      ')
            self.Check('\n        from foo import b\n        class Foo(b.bar):\n          pass\n      ', pythonpath=[d.path])

    def test_submodule_lookup(self):
        if False:
            return 10
        init_py = '\n      from mod import submod%s\n      X = submod.X\n    '
        submod_py = '\n      class X:\n        pass\n    '
        (init_pyi_1, _) = self.InferWithErrors(init_py % '  # import-error', module_name='mod.__init__')
        (submod_pyi_1, _) = self.InferWithErrors(submod_py, module_name='mod.submod')
        with test_utils.Tempdir() as d:
            init_path = d.create_file(file_utils.replace_separator('mod/__init__.pyi'), pytd_utils.Print(init_pyi_1))
            submod_path = d.create_file(file_utils.replace_separator('mod/submod.pyi'), pytd_utils.Print(submod_pyi_1))
            imports_info = d.create_file('imports_info', f"\n        {file_utils.replace_separator('mod/__init__')} {init_path}\n        {file_utils.replace_separator('mod/submod')} {submod_path}\n      ")
            imports_map = self.build_imports_map(imports_info)
            init_pyi = self.Infer(init_py % '', imports_map=imports_map, module_name='mod.__init__')
        self.assertTypesMatchPytd(init_pyi, '\n      from mod import submod\n      from typing import Type\n      X: Type[mod.submod.X]\n    ')

    def test_circular_dep(self):
        if False:
            for i in range(10):
                print('nop')
        submod_py = '\n      from mod import Y%s\n      class X:\n        pass\n    '
        init_py = "\n      import typing\n      if typing.TYPE_CHECKING:\n        from mod.submod import X%s\n      class Y:\n        def __init__(self, x):\n          # type: ('X') -> None\n          pass\n    "
        (submod_pyi_1, _) = self.InferWithErrors(submod_py % '  # import-error', module_name='mod.submod')
        (init_pyi_1, _) = self.InferWithErrors(init_py % '  # import-error', module_name='mod.__init__')
        with test_utils.Tempdir() as d:
            submod_path = d.create_file(file_utils.replace_separator('mod/submod.pyi'), pytd_utils.Print(submod_pyi_1))
            init_path = d.create_file(file_utils.replace_separator('mod/__init__.pyi'), pytd_utils.Print(init_pyi_1))
            imports_info = d.create_file('imports_info', f"\n        {file_utils.replace_separator('mod/submod')} {submod_path}\n        {file_utils.replace_separator('mod/__init__')} {init_path}\n      ")
            imports_map = self.build_imports_map(imports_info)
            submod_pyi = self.Infer(submod_py % '', imports_map=imports_map, module_name='mod.submod')
            with open(submod_path, 'w') as f:
                f.write(pytd_utils.Print(submod_pyi))
            init_pyi = self.Infer(init_py % '', imports_map=imports_map, module_name='mod.__init__')
        self.assertTypesMatchPytd(init_pyi, '\n      import mod.submod\n      import typing\n      from typing import Type\n      X: Type[mod.submod.X]\n      class Y:\n        def __init__(self, x: X) -> None: ...\n    ')

    def test_mutual_imports(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('pkg/a.pyi'), "\n        from typing import TypeVar, Generic, List\n        from .b import Foo\n        T = TypeVar('T')\n        class Bar(Foo, List[T], Generic[T]): ...\n        class Baz(List[T], Generic[T]): ...\n      ")
            d.create_file(file_utils.replace_separator('pkg/b.pyi'), "\n        from typing import TypeVar, Generic\n        from .a import Baz\n        T = TypeVar('T')\n        class Foo(): ...\n        class Quux(Baz[T], Generic[T]): ...\n      ")
            ty = self.Infer('from pkg.a import *', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, "\n        import pkg.a\n        import pkg.b\n        from typing import Type, TypeVar\n        Bar = ...  # type: Type[pkg.a.Bar]\n        Baz = ...  # type: Type[pkg.a.Baz]\n        Foo = ...  # type: Type[pkg.b.Foo]\n        T = TypeVar('T')\n      ")

    def test_module_reexports_and_aliases(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('pkg/a.pyi'), '\n        from pkg import b as c\n        from pkg.b import e as f\n        import pkg.d as x\n        import pkg.g  # should not cause unused import errors\n      ')
            d.create_file(file_utils.replace_separator('pkg/b.pyi'), '\n        class X: ...\n        class e: ...\n      ')
            d.create_file(file_utils.replace_separator('pkg/d.pyi'), '\n        class Y: ...\n      ')
            d.create_file(file_utils.replace_separator('pkg/g.pyi'), '\n        class Z: ...\n      ')
            ty = self.Infer('\n        import pkg.a\n        s = pkg.a.c.X()\n        t = pkg.a.f()\n        u = pkg.a.x\n        v = u.Y()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import pkg\n        from pkg import d as u\n        s = ...  # type: pkg.b.X\n        t = ...  # type: pkg.b.e\n        v = ...  # type: u.Y\n      ')

    def test_import_package_as_alias(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', 'class A: ...')
            d.create_file('b.pyi', '\n        import a as _a\n        f: _a.A\n      ')
            self.Check('\n        import b\n        c = b.f\n      ', pythonpath=[d.path])

    def test_import_package_alias_name_conflict(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', 'A: str')
            d.create_file('b.pyi', '\n        import a as _a\n        class a:\n          A: int\n        x = _a.A\n        y = a.A\n      ')
            ty = self.Infer('\n        import b\n        x = b.x\n        y = b.y\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import b\n        x: str\n        y: int\n      ')

    def test_import_package_alias_name_conflict2(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', 'A: str')
            d.create_file('b.pyi', 'A: int')
            d.create_file('c.pyi', '\n        import a as _a\n        import b as a\n        x = _a.A\n        y = a.A\n      ')
            ty = self.Infer('\n        import c\n        x = c.x\n        y = c.y\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import c\n        x: str\n        y: int\n      ')

    def test_import_package_alias_name_conflict3(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', 'A: str')
            d.create_file('b.pyi', 'A: int')
            d.create_file('c.pyi', '\n        import b as a\n        import a as _a\n        x = _a.A\n        y = a.A\n      ')
            ty = self.Infer('\n        import c\n        x = c.x\n        y = c.y\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import c\n        x: str\n        y: int\n      ')

    def test_module_class_conflict(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/bar.pyi'), DEFAULT_PYI)
            ty = self.Infer('\n        from foo import bar\n        class foo:\n          def __new__(cls):\n            return object.__new__(cls)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from foo import bar\n        from typing import Type, TypeVar\n        _Tfoo = TypeVar("_Tfoo", bound=foo)\n        class foo:\n          def __new__(cls: Type[_Tfoo]) -> _Tfoo: ...\n      ')

    def test_class_alias(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/bar.pyi'), DEFAULT_PYI)
            ty = self.Infer('\n        from foo import bar\n        class foo:\n          pass\n        baz = foo\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from foo import bar\n        from typing import Type\n        class foo: ...\n        baz: Type[foo]\n      ')

    def test_relative_star_import(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/bar.pyi'), 'from .baz.qux import *')
            d.create_file(file_utils.replace_separator('foo/baz/qux.pyi'), 'v = ...  # type: int')
            ty = self.Infer('\n        from foo.bar import *\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        v = ...  # type: int\n      ')

    def test_relative_star_import2(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/bar/baz.pyi'), 'from ..bar.qux import *')
            d.create_file(file_utils.replace_separator('foo/bar/qux.pyi'), 'v = ...  # type: int')
            ty = self.Infer('\n        from foo.bar.baz import *\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        v = ...  # type: int\n      ')

    def test_unimported_submodule_failure(self):
        if False:
            while True:
                i = 10
        "Fail when accessing a submodule we haven't imported."
        self.options.tweak(strict_import=True)
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('sub/bar/baz.pyi'), 'class A: ...')
            d.create_file(file_utils.replace_separator('sub/bar/quux.pyi'), 'class B: ...')
            d.create_file(file_utils.replace_separator('sub/__init__.pyi'), '')
            d.create_file(file_utils.replace_separator('sub/bar/__init__.pyi'), '')
            (_, errors) = self.InferWithErrors('\n        import sub.bar.baz\n        x = sub.bar.baz.A()\n        y = sub.bar.quux.B()  # module-attr[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'quux.*sub\\.bar'})

    def test_submodule_attribute_error(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('package/__init__.pyi'), 'submodule: module')
            d.create_file(file_utils.replace_separator('package/submodule.pyi'), '')
            self.CheckWithErrors('\n        from package import submodule\n        submodule.asd  # module-attr\n      ', pythonpath=[d.path])

    def test_init_only_submodule(self):
        if False:
            print('Hello World!')
        'Test a submodule without its own stub file.'
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('package/__init__.pyi'), 'submodule: module')
            self.Check('\n        from package import submodule\n        submodule.asd\n      ', pythonpath=[d.path])

    def test_import_alias(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/__init__.pyi'), '')
            d.create_file(file_utils.replace_separator('foo/bar.pyi'), '\n        from foo import baz as qux\n        X = qux.X\n      ')
            d.create_file(file_utils.replace_separator('foo/baz.pyi'), 'X = str')
            self.Check('from foo import bar', pythonpath=[d.path])

    def test_subpackage(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/__init__.pyi'), 'from .bar import baz as baz')
            d.create_file(file_utils.replace_separator('foo/bar/baz.pyi'), 'v: str')
            ty = self.Infer('\n        import foo\n        v = foo.baz.v\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        v: str\n      ')

    def test_attr_and_module(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/__init__.pyi'), 'class X: ...')
            d.create_file(file_utils.replace_separator('foo/bar.pyi'), 'v: str')
            d.create_file('other.pyi', '\n        from foo import X as X\n        from foo import bar as bar\n      ')
            ty = self.Infer('\n        import other\n        X = other.X\n        v = other.bar.v\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Type\n        import foo\n        import other\n        X: Type[foo.X]\n        v: str\n      ')

    def test_submodule_imports_info(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            empty = d.create_file('empty.pyi')
            imports_info = d.create_file('imports_info', f'email/_header_value_parser {empty}')
            imports_map = self.build_imports_map(imports_info)
            self.Check('\n        from email import message_from_bytes\n      ', imports_map=imports_map)

    def test_directory_module_clash(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            foo = d.create_file('foo.pyi', 'x: int')
            foo_bar = d.create_file(file_utils.replace_separator('foo/bar.pyi'), 'y: str')
            imports_info = d.create_file('imports_info', f"\n        foo {foo}\n        {file_utils.replace_separator('foo/bar')} {foo_bar}\n      ")
            imports_map = self.build_imports_map(imports_info)
            self.CheckWithErrors('\n        import foo\n        x = foo.x  # module-attr\n      ', imports_map=imports_map)

    def test_missing_submodule(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            foo = d.create_file(file_utils.replace_separator('foo/__init__.pyi'), 'import bar.baz as baz')
            foo_bar = d.create_file(file_utils.replace_separator('foo/bar.pyi'), 'y: str')
            imports_info = d.create_file(file_utils.replace_separator('imports_info'), f"\n        foo {foo}\n        {file_utils.replace_separator('foo/bar')} {foo_bar}\n      ")
            imports_map = self.build_imports_map(imports_info)
            self.CheckWithErrors('\n        from foo import baz  # import-error\n      ', imports_map=imports_map)

    def test_module_prefix_alias(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            foo_bar = d.create_file(file_utils.replace_separator('foo/bar.pyi'), '\n            import foo as _foo\n            x: _foo.baz.X\n          ')
            foo_baz = d.create_file(file_utils.replace_separator('foo/baz.pyi'), 'class X: ...')
            imports_info = d.create_file(file_utils.replace_separator('imports_info'), f"\n            {file_utils.replace_separator('foo/bar')} {foo_bar}\n            {file_utils.replace_separator('foo/baz')} {foo_baz}\n          ")
            imports_map = self.build_imports_map(imports_info)
            self.Check('\n        from foo import bar\n      ', imports_map=imports_map)
if __name__ == '__main__':
    test_base.main()