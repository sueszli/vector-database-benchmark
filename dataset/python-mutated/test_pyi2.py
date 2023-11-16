"""Tests for handling PYI code."""
from pytype.tests import test_base
from pytype.tests import test_utils

class PYITest(test_base.BaseTest):
    """Tests for PYI."""

    def test_unneccessary_any_import(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n        import typing\n        def foo(**kwargs: typing.Any) -> int: return 1\n        def bar(*args: typing.Any) -> int: return 2\n        ')
        self.assertTypesMatchPytd(ty, '\n        import typing\n        def foo(**kwargs) -> int: ...\n        def bar(*args) -> int: ...\n        ')

    def test_static_method_from_pyi_as_callable(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class A:\n          @staticmethod\n          def callback(msg: str) -> None: ...\n      ')
            self.Check("\n        from typing import Any, Callable\n        import foo\n        def func(c: Callable[[Any], None], arg: Any) -> None:\n          c(arg)\n        func(foo.A.callback, 'hello, world')\n      ", pythonpath=[d.path])

    def test_class_method_from_pyi_as_callable(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class A:\n          @classmethod\n          def callback(cls, msg: str) -> None: ...\n      ')
            self.Check("\n        from typing import Any, Callable\n        import foo\n        def func(c: Callable[[Any], None], arg: Any) -> None:\n          c(arg)\n        func(foo.A.callback, 'hello, world')\n      ", pythonpath=[d.path])

    def test_ellipsis(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(x: Ellipsis) -> None: ...\n      ')
            self.CheckWithErrors('\n        import foo\n        x = foo.f(...)\n        y = foo.f(1)  # wrong-arg-types\n      ', pythonpath=[d.path])

    def test_resolve_nested_type(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('meta.pyi', '\n        class Meta(type): ...\n      ')
            d.create_file('foo.pyi', '\n        import meta\n        class Foo:\n          class Bar(int, metaclass=meta.Meta): ...\n          CONST: Foo.Bar\n      ')
            self.Check('\n        import foo\n        print(foo.Foo.CONST)\n      ', pythonpath=[d.path])

    def test_partial_forward_reference(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        from typing import Generic, TypeVar\n        X1 = list['Y']\n        X2 = list['Z[str]']\n        X3 = int | 'Z[int]'\n        Y = int\n        T = TypeVar('T')\n        class Z(Generic[T]): ...\n      ")
            self.Check('\n        import foo\n        assert_type(foo.X1, "Type[List[int]]")\n        assert_type(foo.X2, "Type[List[foo.Z[str]]]")\n        assert_type(foo.X3, "Type[Union[foo.Z[int], int]]")\n      ', pythonpath=[d.path])

    def test_bare_callable(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', '\n      import types\n      def f(x) -> types.FunctionType: ...\n    ')]):
            ty = self.Infer('\n        import foo\n        def f(x):\n          return foo.f(x)\n      ')
        self.assertTypesMatchPytd(ty, '\n      import foo\n      from typing import Callable\n      def f(x) -> Callable[..., nothing]: ...\n    ')

    def test_keyword_import(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.pyi', "\n      import importlib\n      my_module = importlib.import_module('regretting.decisions.in.naming')\n    ")]):
            self.Check('\n        from foo import my_module\n        print(my_module.whatever)\n      ')

class PYITestPython3Feature(test_base.BaseTest):
    """Tests for PYI."""

    def test_bytes(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f() -> bytes: ...\n      ')
            ty = self.Infer('\n        import foo\n        x = foo.f()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        x = ...  # type: bytes\n      ')

    def test_imported_literal_alias(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.pyi', '\n      from typing import Literal\n      X = Literal["a", "b"]\n    '), ('bar.pyi', '\n      import foo\n      from typing import Literal\n      Y = Literal[foo.X, "c", "d"]\n    ')]):
            self.Check('\n        import bar\n        assert_type(bar.Y, "Type[Literal[\'a\', \'b\', \'c\', \'d\']]")\n      ')

    def test_literal_in_dataclass(self):
        if False:
            for i in range(10):
                print('nop')
        self.options.tweak(use_enum_overlay=False)
        with self.DepTree([('foo.pyi', "\n      import enum\n      class Base: ...\n      class Foo(Base, enum.Enum):\n        FOO = 'FOO'\n    "), ('bar.pyi', '\n      import dataclasses\n      import foo\n      from typing import Literal, Optional\n      @dataclasses.dataclass\n      class Bar(foo.Base):\n        bar: Optional[Literal[foo.Foo.FOO]]\n    ')]):
            self.Check('\n        import bar\n        import dataclasses\n        import foo\n        @dataclasses.dataclass\n        class Baz(foo.Base):\n          baz: bar.Bar\n      ')

    def test_literal_quotes(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.py', '\n      from typing_extensions import Literal\n      def f(x: Literal[\'"\', "\'"]):\n        pass\n    ')]):
            self.CheckWithErrors('\n        import foo\n        foo.f(\'"\')\n        foo.f("\'")\n        foo.f(\'oops\')  # wrong-arg-types\n      ')

class PYITestAnnotated(test_base.BaseTest):
    """Tests for typing.Annotated."""

    @test_base.skip('We do not round-trip Annotated yet')
    def test_dict(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      from typing import Annotated\n      x: Annotated[int, 'str', {'x': 1, 'y': 'z'}]\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Annotated\n      x: Annotated[int, 'str', {'x': 1, 'y': 'z'}]\n    ")

    def test_invalid_pytype_metadata(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Annotated\n        x: Annotated[int, "pytype_metadata", 2]\n      ')
            err = self.CheckWithErrors('\n        import foo\n        a = foo.x  # invalid-annotation[e]\n      ', pythonpath=[d.path])
            self.assertErrorSequences(err, {'e': ['pytype_metadata']})

class PYITestAll(test_base.BaseTest):
    """Tests for __all__."""

    def test_star_import(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.pyi', "\n      import datetime\n      __all__ = ['f', 'g']\n      def f(x): ...\n      def h(x): ...\n    "), ('bar.pyi', '\n      from foo import *\n    ')]):
            self.CheckWithErrors('\n        import bar\n        a = bar.datetime  # module-attr\n        b = bar.f(1)\n        c = bar.h(1)  # module-attr\n      ')

    def test_http_client(self):
        if False:
            print('Hello World!')
        'Check that we can get unexported symbols from http.client.'
        self.Check('\n      from http import client\n      from six.moves import http_client\n      status = http_client.FOUND or client.FOUND\n    ')

class PYITestFuture(test_base.BaseTest):
    """Tests for __future__."""

    def test_skip_reexport(self):
        if False:
            i = 10
            return i + 15
        "Check that we don't reexport __future__ imports."
        ty = self.Infer('\n      from __future__ import annotations\n      class A:\n        pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      class A: ...\n    ')
if __name__ == '__main__':
    test_base.main()