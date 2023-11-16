"""Tests for the chex overlay."""
import contextlib
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class TestDataclass(test_base.BaseTest):
    """Tests for chex.dataclass."""

    @contextlib.contextmanager
    def _add_chex(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('chex.pyi', '\n        from typing import Any\n        def dataclass(\n            cls = ..., *, init = ..., repr = ..., eq = ..., order = ...,\n            unsafe_hash = ..., frozen = ..., mappable_dataclass = ...) -> Any: ...\n      ')
            yield d

    def Check(self, *args, **kwargs):
        if False:
            return 10
        if 'pythonpath' in kwargs:
            return super().Check(*args, **kwargs)
        else:
            with self._add_chex() as d:
                return super().Check(*args, **kwargs, pythonpath=[d.path])

    def Infer(self, *args, **kwargs):
        if False:
            return 10
        if 'pythonpath' in kwargs:
            return super().Infer(*args, **kwargs)
        else:
            with self._add_chex() as d:
                return super().Infer(*args, **kwargs, pythonpath=[d.path])

    def test_basic(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import chex\n      @chex.dataclass\n      class Foo:\n        x: int\n    ')
        self.assertTypesMatchPytd(ty, "\n      import chex\n      import dataclasses\n      from typing import Any, Dict, Iterator, Mapping, TypeVar\n      _TFoo = TypeVar('_TFoo', bound=Foo)\n      @dataclasses.dataclass\n      class Foo(Mapping, object):\n        x: int\n        def __init__(self, x: int) -> None: ...\n        def __getitem__(self, key) -> Any: ...\n        def __iter__(self) -> Iterator: ...\n        def __len__(self) -> int: ...\n        def replace(self: _TFoo, **changes) -> _TFoo: ...\n        @staticmethod\n        def from_tuple(args) -> Foo: ...\n        def to_tuple(self) -> tuple: ...\n    ")

    def test_not_mappable(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import chex\n      @chex.dataclass(mappable_dataclass=False)\n      class Foo:\n        x: int\n    ')
        self.assertTypesMatchPytd(ty, "\n      import chex\n      import dataclasses\n      from typing import Dict, TypeVar\n      _TFoo = TypeVar('_TFoo', bound=Foo)\n      @dataclasses.dataclass\n      class Foo:\n        x: int\n        def __init__(self, x: int) -> None: ...\n        def replace(self: _TFoo, **changes) -> _TFoo: ...\n        @staticmethod\n        def from_tuple(args) -> Foo: ...\n        def to_tuple(self) -> tuple: ...\n    ")

    def test_use_mappable(self):
        if False:
            print('Hello World!')
        self.Check('\n      import chex\n      from typing import Sequence\n\n      @chex.dataclass\n      class Foo:\n        x: int\n\n      def f(foos: Sequence[Foo]):\n        for foo in foos:\n          yield foo["x"]\n    ')

    def test_replace(self):
        if False:
            return 10
        ty = self.Infer('\n      import chex\n      @chex.dataclass\n      class Foo:\n        x: int\n      foo = Foo(0).replace(x=5)\n    ')
        self.assertTypesMatchPytd(ty, "\n      import chex\n      import dataclasses\n      from typing import Any, Dict, Iterator, Mapping, TypeVar\n      _TFoo = TypeVar('_TFoo', bound=Foo)\n      @dataclasses.dataclass\n      class Foo(Mapping, object):\n        x: int\n        def __init__(self, x: int) -> None: ...\n        def __getitem__(self, key) -> Any: ...\n        def __iter__(self) -> Iterator: ...\n        def __len__(self) -> int: ...\n        def replace(self: _TFoo, **changes) -> _TFoo: ...\n        @staticmethod\n        def from_tuple(args) -> Foo: ...\n        def to_tuple(self) -> tuple: ...\n      foo: Foo\n    ")

    def test_from_tuple(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import chex\n      @chex.dataclass\n      class Foo:\n        x: int\n      foo = Foo.from_tuple((0,))\n    ')
        self.assertTypesMatchPytd(ty, "\n      import chex\n      import dataclasses\n      from typing import Any, Dict, Iterator, Mapping, TypeVar\n      _TFoo = TypeVar('_TFoo', bound=Foo)\n      @dataclasses.dataclass\n      class Foo(Mapping, object):\n        x: int\n        def __init__(self, x: int) -> None: ...\n        def __getitem__(self, key) -> Any: ...\n        def __iter__(self) -> Iterator: ...\n        def __len__(self) -> int: ...\n        def replace(self: _TFoo, **changes) -> _TFoo: ...\n        @staticmethod\n        def from_tuple(args) -> Foo: ...\n        def to_tuple(self) -> tuple: ...\n      foo: Foo\n    ")

    def test_to_tuple(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import chex\n      @chex.dataclass\n      class Foo:\n        x: int\n      tup = Foo(0).to_tuple()\n    ')
        self.assertTypesMatchPytd(ty, "\n      import chex\n      import dataclasses\n      from typing import Any, Dict, Iterator, Mapping, TypeVar\n      _TFoo = TypeVar('_TFoo', bound=Foo)\n      @dataclasses.dataclass\n      class Foo(Mapping, object):\n        x: int\n        def __init__(self, x: int) -> None: ...\n        def __getitem__(self, key) -> Any: ...\n        def __iter__(self) -> Iterator: ...\n        def __len__(self) -> int: ...\n        def replace(self: _TFoo, **changes) -> _TFoo: ...\n        @staticmethod\n        def from_tuple(args) -> Foo: ...\n        def to_tuple(self) -> tuple: ...\n      tup: tuple\n    ")

    def test_multiple_dataclasses(self):
        if False:
            return 10
        foo = self.Infer('\n      import chex\n      @chex.dataclass\n      class A:\n        x: int\n      @chex.dataclass\n      class B:\n        x: str\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.Check("\n        import foo\n        print(foo.B(x='hello').replace(x='world'))\n      ", pythonpath=[d.path])

    def test_generic_dataclass(self):
        if False:
            i = 10
            return i + 15
        foo = self.Infer('\n      from typing import Generic, TypeVar\n      import chex\n      T = TypeVar("T")\n      @chex.dataclass\n      class A(Generic[T]):\n        x: T\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.Check('\n        import foo\n        a = foo.A(x=42)\n        assert_type(a.x, int)\n      ', pythonpath=[d.path])
if __name__ == '__main__':
    test_base.main()