"""Tests for the flax overlay."""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class TestStructDataclass(test_base.BaseTest):
    """Tests for flax.struct.dataclass."""

    def _setup_struct_pyi(self, d):
        if False:
            print('Hello World!')
        d.create_file('flax/struct.pyi', '\n      from typing import Type\n      def dataclass(_cls: Type[_T]) -> Type[_T]: ...\n    ')

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            self._setup_struct_pyi(d)
            ty = self.Infer('\n        import flax\n        @flax.struct.dataclass\n        class Foo:\n          x: bool\n          y: int\n          z: str\n        ', pythonpath=[d.path], module_name='foo')
            self.assertTypesMatchPytd(ty, "\n        import flax\n        from typing import Dict, TypeVar, Union\n\n        _TFoo = TypeVar('_TFoo', bound=Foo)\n\n        @dataclasses.dataclass\n        class Foo:\n          x: bool\n          y: int\n          z: str\n          def __init__(self, x: bool, y: int, z: str) -> None: ...\n          def replace(self: _TFoo, **kwargs) -> _TFoo: ...\n      ")

    def test_redefine_field(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import dataclasses\n      def field(**kwargs):\n        return dataclasses.field(**kwargs)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import dataclasses\n      from typing import Any\n      def field(**kwargs) -> Any: ...\n    ')

    def test_replace(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            self._setup_struct_pyi(d)
            self.Check('\n        import flax\n\n        @flax.struct.dataclass\n        class Foo:\n          x: int = 10\n          y: str = "hello"\n\n        Foo().replace(y="a")\n      ', pythonpath=[d.path])

class TestLinenModule(test_base.BaseTest):
    """Test dataclass construction in flax.linen.Module subclasses."""

    def _setup_linen_pyi(self, d):
        if False:
            print('Hello World!')
        d.create_file('flax/linen/__init__.pyi', '\n      from .module import Module\n    ')
        d.create_file('flax/linen/module.pyi', '\n      class Module:\n        def make_rng(self, x: str) -> None: ...\n    ')

    def test_constructor(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            self._setup_linen_pyi(d)
            ty = self.Infer('\n        from flax import linen as nn\n        class Foo(nn.Module):\n          x: bool\n          y: int = 10\n        ', pythonpath=[d.path], module_name='foo')
            self.assertTypesMatchPytd(ty, "\n        from flax import linen as nn\n        from typing import Dict, TypeVar\n        _TFoo = TypeVar('_TFoo', bound=Foo)\n        @dataclasses.dataclass\n        class Foo(nn.module.Module):\n          x: bool\n          y: int = ...\n          def __init__(self, x: bool, y: int = ..., name: str = ..., parent = ...) -> None: ...\n          def replace(self: _TFoo, **kwargs) -> _TFoo: ...\n      ")

    def test_unexported_constructor(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            self._setup_linen_pyi(d)
            ty = self.Infer('\n        from flax.linen import module\n        class Foo(module.Module):\n          x: bool\n          y: int = 10\n        ', pythonpath=[d.path], module_name='foo')
            self.assertTypesMatchPytd(ty, "\n        from flax.linen import module\n        from typing import Dict, TypeVar\n        _TFoo = TypeVar('_TFoo', bound=Foo)\n        @dataclasses.dataclass\n        class Foo(module.Module):\n          x: bool\n          y: int = ...\n          def __init__(self, x: bool, y: int = ..., name: str = ..., parent = ...) -> None: ...\n          def replace(self: _TFoo, **kwargs) -> _TFoo: ...\n      ")

    def test_relative_import_from_package_module(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            self._setup_linen_pyi(d)
            ty = self.Infer('\n        from .module import Module\n        class Foo(Module):\n          x: bool\n          y: int = 10\n        ', pythonpath=[d.path], module_name='flax.linen.foo')
            self.assertTypesMatchPytd(ty, "\n        from typing import Dict, Type, TypeVar\n        import flax.linen.module\n        Module: Type[flax.linen.module.Module]\n        _TFoo = TypeVar('_TFoo', bound=Foo)\n        @dataclasses.dataclass\n        class Foo(flax.linen.module.Module):\n          x: bool\n          y: int = ...\n          def __init__(self, x: bool, y: int = ..., name: str = ..., parent = ...) -> None: ...\n          def replace(self: _TFoo, **kwargs) -> _TFoo: ...\n      ")

    def test_parent_import_from_package_module(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            self._setup_linen_pyi(d)
            ty = self.Infer('\n        from .. import linen\n        class Foo(linen.Module):\n          x: bool\n          y: int = 10\n        ', pythonpath=[d.path], module_name='flax.linen.foo')
            self.assertTypesMatchPytd(ty, "\n        from flax import linen\n        from typing import Dict, TypeVar\n        _TFoo = TypeVar('_TFoo', bound=Foo)\n        @dataclasses.dataclass\n        class Foo(linen.module.Module):\n          x: bool\n          y: int = ...\n          def __init__(self, x: bool, y: int = ..., name: str = ..., parent = ...) -> None: ...\n          def replace(self: _TFoo, **kwargs) -> _TFoo: ...\n      ")

    def test_self_type(self):
        if False:
            while True:
                i = 10
        'Match self: f.l.module.Module even if imported as f.l.Module.'
        with test_utils.Tempdir() as d:
            self._setup_linen_pyi(d)
            self.Check('\n        from flax import linen\n        class Foo(linen.Module):\n          x: int\n        a = Foo(10)\n        b = a.make_rng("a")  # called on base class\n      ', pythonpath=[d.path])

    def test_invalid_field(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            self._setup_linen_pyi(d)
            errors = self.CheckWithErrors('\n        from flax import linen as nn\n        class Foo(nn.Module):  # invalid-annotation[e]\n          x: bool\n          name: str\n        ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'name.*implicitly'})

    def test_setup(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            self._setup_linen_pyi(d)
            self.Check('\n        from flax import linen\n        class Foo(linen.Module):\n          x: int\n          def setup(self):\n            self.y = 10\n        a = Foo(10)\n        b = a.y\n      ', pythonpath=[d.path])

    def test_reingest(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            self._setup_linen_pyi(d)
            foo_ty = self.Infer('\n        from flax import linen\n        class Foo(linen.Module):\n          pass\n      ', pythonpath=[d.path])
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            ty = self.Infer('\n        import foo\n        class Bar(foo.Foo):\n          x: int\n      ', pythonpath=[d.path])
        self.assertTypesMatchPytd(ty, "\n      import dataclasses\n      import foo\n      from typing import Any, Dict, TypeVar\n\n      _TBar = TypeVar('_TBar', bound=Bar)\n      @dataclasses.dataclass\n      class Bar(foo.Foo):\n        x: int\n        def __init__(\n            self, x: int, name: str = ..., parent: Any = ...) -> None: ...\n        def replace(self: _TBar, **kwargs) -> _TBar: ...\n    ")

    def test_reingest_and_subclass(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            self._setup_linen_pyi(d)
            foo_ty = self.Infer('\n        from flax import linen\n        class Foo(linen.Module):\n          pass\n      ', pythonpath=[d.path])
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            ty = self.Infer('\n        import foo\n        class Bar(foo.Foo):\n          pass\n        class Baz(Bar):\n          x: int\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, "\n        import dataclasses\n        import foo\n        from typing import Any, Dict, TypeVar\n\n        _TBar = TypeVar('_TBar', bound=Bar)\n        @dataclasses.dataclass\n        class Bar(foo.Foo):\n          def __init__(self, name: str = ..., parent: Any = ...) -> None: ...\n          def replace(self: _TBar, **kwargs) -> _TBar: ...\n\n        _TBaz = TypeVar('_TBaz', bound=Baz)\n        @dataclasses.dataclass\n        class Baz(Bar):\n          x: int\n          def __init__(\n              self, x: int, name: str = ..., parent: Any = ...) -> None: ...\n          def replace(self: _TBaz, **kwargs) -> _TBaz: ...\n      ")
if __name__ == '__main__':
    test_base.main()