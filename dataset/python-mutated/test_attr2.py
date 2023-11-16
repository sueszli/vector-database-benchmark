"""Tests for attrs library in attr_overlay.py."""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class TestAttrib(test_base.BaseTest):
    """Tests for attr.ib."""

    def test_factory_function(self):
        if False:
            return 10
        ty = self.Infer('\n      import attr\n      class CustomClass:\n        pass\n      def annotated_func() -> CustomClass:\n        return CustomClass()\n      @attr.s\n      class Foo:\n        x = attr.ib(factory=annotated_func)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      class CustomClass: ...\n      def annotated_func() -> CustomClass: ...\n      @attr.s\n      class Foo:\n        x: CustomClass = ...\n        def __init__(self, x: CustomClass = ...) -> None: ...\n    ')

    def test_attr_default_dict(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      import attr\n      @attr.s\n      class Dog2():\n        dog_attr = attr.ib(default='woofing', **dict())\n\n        def make_puppy(self) -> 'Dog2':\n          return Dog2()\n    ")

class TestAttribConverters(test_base.BaseTest):
    """Tests for attr.ib with converters."""

    def test_annotated_converter(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      import attr\n      def convert(input: str) -> int:\n        return int(input)\n      @attr.s\n      class Foo:\n        x = attr.ib(converter=convert)\n      Foo(x='123')\n    ")

    def test_type_and_converter(self):
        if False:
            return 10
        self.Check("\n      import attr\n      def convert(input: str):\n        return int(input)\n      @attr.s\n      class Foo:\n        x = attr.ib(type=int, converter=convert)\n      Foo(x='123')\n    ")

    def test_unannotated_converter_with_type(self):
        if False:
            while True:
                i = 10
        self.Check("\n      import attr\n      def convert(input):\n        return int(input)\n      @attr.s\n      class Foo:\n        x = attr.ib(type=int, converter=convert)\n      Foo(x='123')\n      Foo(x=[1,2,3])  # does not complain, input is treated as Any\n    ")

    def test_annotated_converter_with_mismatched_type(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      import attr\n      def convert(input: str) -> int:\n        return int(input)\n      @attr.s\n      class Foo:\n        x = attr.ib(type=str, converter=convert)  # annotation-type-mismatch\n      foo = Foo(x=123)  # wrong-arg-types\n      assert_type(foo.x, str)\n    ')

    def test_converter_without_return_annotation(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      import attr\n      def convert(input: str):\n        return int(input)\n      @attr.s\n      class Foo:\n        x = attr.ib(converter=convert)\n      foo = Foo(x=123) # wrong-arg-types\n      assert_type(foo.x, int)\n    ')

    def test_converter_with_union_type(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      import attr\n      from typing import Union\n      def convert(input: str):\n        if __random__:\n          return input\n        return int(input)\n      @attr.s\n      class Foo:\n        x = attr.ib(converter=convert)\n      foo = Foo(x='123')\n      assert_type(foo.x, Union[int, str])\n    ")

    def test_wrong_converter_arity(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      import attr\n      def convert(x, y) -> int:\n        return 42\n      @attr.s\n      class Foo:\n        x = attr.ib(type=str, converter=convert)  # wrong-arg-types\n    ')

    def test_converter_with_default_args(self):
        if False:
            return 10
        self.Check('\n      import attr\n      def convert(x, y=10) -> int:\n        return 42\n      @attr.s\n      class Foo:\n        x = attr.ib(converter=convert)\n    ')

    def test_converter_with_varargs(self):
        if False:
            return 10
        self.CheckWithErrors('\n      import attr\n      def convert(*args, **kwargs) -> int:\n        return 42\n      @attr.s\n      class Foo:\n        x = attr.ib(converter=convert)\n    ')

    def test_converter_conflicts_with_type(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors("\n      import attr\n      def convert(input: str) -> int:\n        return int(input)\n      @attr.s\n      class Foo:\n        x = attr.ib(type=list, converter=convert)  # annotation-type-mismatch\n      foo = Foo(x='123')\n      assert_type(foo.x, list)\n    ")

    def test_converter_conflicts_with_annotation(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors("\n      import attr\n      def convert(input: str) -> int:\n        return int(input)\n      @attr.s\n      class Foo:\n        x: list = attr.ib(converter=convert)  # annotation-type-mismatch\n      foo = Foo(x='123')\n      assert_type(foo.x, list)\n    ")

    def test_converter_conflicts_with_default(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors("\n      import attr\n      def convert(input: str) -> int:\n        return int(input)\n      @attr.s\n      class Foo:\n        x = attr.ib(converter=convert, default='a')  # annotation-type-mismatch\n      foo = Foo(x='123')\n      assert_type(foo.x, int)\n    ")

    def test_type_compatible_with_converter(self):
        if False:
            while True:
                i = 10
        self.Check("\n      import attr\n      from typing import Optional\n      def convert(input: str) -> int:\n        return int(input)\n      @attr.s\n      class Foo:\n        x = attr.ib(type=Optional[int], converter=convert)\n      foo = Foo(x='123')\n      assert_type(foo.x, Optional[int])\n    ")

    def test_callable_as_converter(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import attr\n      from typing import Callable\n      def f() -> Callable[[int], str]:\n        return __any_object__\n      @attr.s\n      class Foo:\n        x = attr.ib(converter=f())\n      foo = Foo(x=0)\n      assert_type(foo.x, str)\n    ')

    def test_partial_as_converter(self):
        if False:
            while True:
                i = 10
        self.Check("\n      import attr\n      import functools\n      def f(x: int) -> str:\n        return ''\n      @attr.s\n      class Foo:\n        x = attr.ib(converter=functools.partial(f))\n      # We don't yet infer the right type for Foo.x in this case, but we at\n      # least want to check that constructing a Foo doesn't generate errors.\n      Foo(x=0)\n    ")

class TestAttribPy3(test_base.BaseTest):
    """Tests for attr.ib using PEP526 syntax."""

    def test_variable_annotations(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import attr\n      @attr.s\n      class Foo:\n        x : int = attr.ib()\n        y = attr.ib(type=str)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: int\n        y: str\n        def __init__(self, x: int, y: str) -> None: ...\n    ')

    def test_late_annotations(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      import attr\n      @attr.s\n      class Foo:\n        x : 'Foo' = attr.ib()\n        y = attr.ib(type=str)\n    ")
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: Foo\n        y: str\n        def __init__(self, x: Foo, y: str) -> None: ...\n    ')

    def test_classvar(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import attr\n      @attr.s\n      class Foo:\n        x : int = attr.ib()\n        y = attr.ib(type=str)\n        z : int = 1 # class var, should not be in __init__\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: int\n        y: str\n        z: int\n        def __init__(self, x: int, y: str) -> None: ...\n    ')

    def test_type_clash(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      import attr\n      @attr.s\n      class Foo(object):  # invalid-annotation\n        x : int = attr.ib(type=str)\n    ')

    def test_defaults_with_annotation(self):
        if False:
            while True:
                i = 10
        (ty, err) = self.InferWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        x: int = attr.ib(default=42)\n        y: str = attr.ib(default=42)  # annotation-type-mismatch[e]\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: int = ...\n        y: str = ...\n        def __init__(self, x: int = ..., y: str = ...) -> None: ...\n    ')
        self.assertErrorRegexes(err, {'e': 'annotation for y'})

    def test_cannot_decorate(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Type\n        class Foo: ...\n        def decorate(cls: Type[Foo]) -> Type[Foo]: ...\n      ')
            ty = self.Infer('\n        import attr\n        import foo\n        @attr.s\n        @foo.decorate\n        class Bar(foo.Foo): ...\n      ', pythonpath=[d.path])
        self.assertTypesMatchPytd(ty, '\n      import attr\n      import foo\n      from typing import Type\n      Bar: Type[foo.Foo]\n    ')

    def test_conflicting_annotations(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        if __random__:\n          v: int = attr.ib()\n        else:\n          v: int = attr.ib()\n      @attr.s\n      class Bar:\n        if __random__:\n          v: int = attr.ib()\n        else:\n          v: str = attr.ib()  # invalid-annotation[e]\n    ')
        self.assertErrorRegexes(errors, {'e': "'int or str' for v"})

    def test_kw_only(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import attr\n      @attr.s(kw_only=False)\n      class Foo:\n        x = attr.ib(default=42)\n        y = attr.ib(type=int, kw_only=True)\n        z = attr.ib(type=str, default="hello")\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any, Union\n      @attr.s\n      class Foo:\n        x: int = ...\n        y: int\n        z: str = ...\n        def __init__(self, x: int = ..., z: str = ..., *, y: int) -> None: ...\n    ')

    def test_generic(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      import attr\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      @attr.s\n      class Foo(Generic[T]):\n        x: T = attr.ib()\n        y = attr.ib()  # type: T\n      foo1 = Foo[int](x=__any_object__, y=__any_object__)\n      x1, y1 = foo1.x, foo1.y\n      foo2 = Foo(x='', y='')\n      x2, y2 = foo2.x, foo2.y\n    ")
        self.assertTypesMatchPytd(ty, "\n      import attr\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      @attr.s\n      class Foo(Generic[T]):\n        x: T\n        y: T\n        def __init__(self, x: T, y: T) -> None:\n          self = Foo[T]\n      foo1: Foo[int]\n      x1: int\n      y1: int\n      foo2: Foo[str]\n      x2: str\n      y2: str\n    ")

    def test_generic_auto_attribs(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      import attr\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      @attr.s(auto_attribs=True)\n      class Foo(Generic[T]):\n        x: T\n      foo1 = Foo[int](x=__any_object__)\n      x1 = foo1.x\n      foo2 = Foo(x='')\n      x2 = foo2.x\n    ")
        self.assertTypesMatchPytd(ty, "\n      import attr\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      @attr.s\n      class Foo(Generic[T]):\n        x: T\n        def __init__(self, x: T) -> None:\n          self = Foo[T]\n      foo1: Foo[int]\n      x1: int\n      foo2: Foo[str]\n      x2: str\n    ")

    def test_typevar_in_type_arg_generic(self):
        if False:
            return 10
        self.Check("\n      import attr\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      @attr.s\n      class Foo(Generic[T]):\n        x = attr.ib(type=T)\n      assert_type(Foo[int](__any_object__).x, int)\n    ")

class TestAttrs(test_base.BaseTest):
    """Tests for attr.s."""

    def test_kw_only(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import attr\n      @attr.s(kw_only=True)\n      class Foo:\n        x = attr.ib()\n        y = attr.ib(type=int)\n        z = attr.ib(type=str)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any\n      @attr.s\n      class Foo:\n        x: Any\n        y: int\n        z: str\n        def __init__(self, *, x, y: int, z: str) -> None: ...\n    ')

    def test_kw_only_with_defaults(self):
        if False:
            return 10
        ty = self.Infer('\n      import attr\n      @attr.s(kw_only=True)\n      class Foo:\n        x = attr.ib(default=1)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any\n      @attr.s\n      class Foo:\n        x: int = ...\n        def __init__(self, *, x : int = ...) -> None: ...\n    ')

    def test_auto_attrs(self):
        if False:
            return 10
        ty = self.Infer("\n      import attr\n      @attr.s(auto_attribs=True)\n      class Foo:\n        x: int\n        y: 'Foo'\n        z = 10\n        a: str = 'hello'\n    ")
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: int\n        y: Foo\n        a: str = ...\n        z: int\n        def __init__(self, x: int, y: Foo, a: str = ...) -> None: ...\n    ')

    def test_redefined_auto_attrs(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      import attr\n      @attr.s(auto_attribs=True)\n      class Foo:\n        x = 10\n        y: int\n        x: str = 'hello'\n    ")
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        y: int\n        x: str = ...\n        def __init__(self, y: int, x: str = ...) -> None: ...\n    ')

    def test_non_attrs(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      import attr\n      @attr.s(auto_attribs=True)\n      class Foo:\n        @classmethod\n        def foo(cls):\n          pass\n        @staticmethod\n        def bar(x):\n          pass\n        _x = 10\n        y: str = 'hello'\n        @property\n        def x(self):\n          return self._x\n        @x.setter\n        def x(self, x: int):\n          self._x = x\n        def f(self):\n          pass\n    ")
        self.assertTypesMatchPytd(ty, "\n      import attr\n      from typing import Any, Annotated\n      @attr.s\n      class Foo:\n        y: str = ...\n        _x: int\n        x: Annotated[int, 'property']\n        def __init__(self, y: str = ...) -> None: ...\n        def f(self) -> None: ...\n        @staticmethod\n        def bar(x) -> None: ...\n        @classmethod\n        def foo(cls) -> None: ...\n    ")

    def test_callable_attrib(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import attr\n      from typing import Callable\n      @attr.s(auto_attribs=True)\n      class Foo:\n        x: Callable = lambda x: x\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Callable, Union\n      @attr.s\n      class Foo:\n        x: Callable = ...\n        def __init__(self, x: Callable = ...) -> None: ...\n    ')

    def test_auto_attrs_with_dataclass_constructor(self):
        if False:
            return 10
        ty = self.Infer("\n      import attr\n      @attr.dataclass\n      class Foo:\n        x: int\n        y: 'Foo'\n        z = 10\n        a: str = 'hello'\n    ")
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: int\n        y: Foo\n        a: str = ...\n        z: int\n        def __init__(self, x: int, y: Foo, a: str = ...) -> None: ...\n    ')

    def test_init_false_generates_attrs_init(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import attr\n      @attr.s(init=False)\n      class Foo:\n        x = attr.ib()\n        y: int = attr.ib()\n        z = attr.ib(type=str, default="bar")\n        t = attr.ib(init=False, default=5)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any\n      @attr.s\n      class Foo:\n        x: Any\n        y: int\n        z: str = ...\n        t: int = ...\n        def __attrs_init__(self, x, y: int, z: str = "bar") -> None: ...\n    ')

    def test_bad_default_param_order(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      import attr\n      @attr.s(auto_attribs=True)\n      class Foo(object):  # invalid-function-definition\n        x: int = 10\n        y: str\n    ')

    def test_subclass_auto_attribs(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import attr\n      @attr.s(auto_attribs=True)\n      class Foo:\n        x: bool\n        y: int = 42\n      class Bar(Foo):\n        def get_x(self):\n          return self.x\n        def get_y(self):\n          return self.y\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      @attr.s\n      class Foo:\n        x: bool\n        y: int = ...\n        def __init__(self, x: bool, y: int = ...) -> None: ...\n      class Bar(Foo):\n        def get_x(self) -> bool : ...\n        def get_y(self) -> int: ...\n    ')

    def test_partial_auto_attribs(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import attr\n      @attr.s(auto_attribs=True)\n      class Foo:\n        foo: str\n      @attr.s\n      class Bar:\n        bar: str = attr.ib()\n        baz = attr.ib()\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any\n      @attr.s\n      class Foo:\n        foo: str\n        def __init__(self, foo: str) -> None: ...\n      @attr.s\n      class Bar:\n        bar: str\n        baz: Any\n        def __init__(self, bar: str, baz) -> None: ...\n    ')

    def test_classvar_auto_attribs(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      from typing import ClassVar\n      import attr\n      @attr.s(auto_attribs=True)\n      class Foo:\n        x: ClassVar[int] = 10\n        y: str = 'hello'\n    ")
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import ClassVar\n      @attr.s\n      class Foo:\n        y: str = ...\n        x: ClassVar[int]\n        def __init__(self, y: str = ...) -> None: ...\n    ')

    def test_wrapper(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import attr\n      def s(*args, **kwargs):\n        return attr.s(*args, auto_attribs=True, **kwargs)\n      @s\n      class Foo:\n        x: int\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Callable\n      def s(*args, **kwargs) -> Callable: ...\n      @attr.s\n      class Foo:\n        x: int\n        def __init__(self, x: int) -> None: ...\n    ')

class TestAttrsNextGenApi(test_base.BaseTest):
    """Tests for attrs next generation API, added in attrs version 21.1.0.

  See: https://www.attrs.org/en/stable/api.html#next-gen
  """

    def test_define_auto_detects_auto_attrs_true(self):
        if False:
            while True:
                i = 10
        'Test whether @attr.define can detect auto_attrs will default to True.\n\n    This is determined by all variable declarations having a type annotation.\n    '
        ty = self.Infer('\n      from typing import Any\n      import attr\n      @attr.define\n      class Foo:\n        x: Any\n        y: int = attr.field()\n        z: str = attr.field(default="bar")\n        r: int = 43\n        t: int = attr.field(default=5, init=False)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any, Union\n      @attr.s(auto_attribs=True)\n      class Foo:\n        x: Any\n        y: int\n        z: str = ...\n        r: int = ...\n        t: int = ...\n        def __init__(self, x, y: int, z: str = "bar", r: int = 43) -> None: ...\n    ')

    def test_define_auto_detects_auto_attrs_false(self):
        if False:
            i = 10
            return i + 15
        'Test whether @attr.define can detect auto_attrs should default to False.\n\n    This is determined by at least one variable declaration not having a type\n    annotation.\n    '
        ty = self.Infer('\n      from typing import Any\n      import attr\n      @attr.define\n      class Foo:\n        x = None\n        y = attr.field(type=int)\n        z = attr.field()\n        r = attr.field(default="bar")\n        t: int = attr.field(default=5, init=False)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any\n      @attr.s\n      class Foo:\n        y: int\n        z: Any\n        r: str = ...\n        t: int = ...\n        x: None\n        def __init__(self, y: int, z, r: str = "bar") -> None: ...\n    ')

    def test_auto_attrs(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      import attr\n      @attr.define(auto_attribs=True)\n      class Foo:\n        x: int\n        y: 'Foo'\n        z = 10\n        a: str = 'hello'\n    ")
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: int\n        y: Foo\n        a: str = ...\n        z: int\n        def __init__(self, x: int, y: Foo, a: str = ...) -> None: ...\n    ')

    def test_redefined_auto_attrs(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      import attr\n      @attr.define(auto_attribs=True)\n      class Foo:\n        x = 10\n        y: int\n        x: str = 'hello'\n    ")
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        y: int\n        x: str = ...\n        def __init__(self, y: int, x: str = ...) -> None: ...\n    ')

    def test_non_attrs(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      import attr\n      @attr.define(auto_attribs=True)\n      class Foo:\n        @classmethod\n        def foo(cls):\n          pass\n        @staticmethod\n        def bar(x):\n          pass\n        _x = 10\n        y: str = 'hello'\n        @property\n        def x(self):\n          return self._x\n        @x.setter\n        def x(self, x: int):\n          self._x = x\n        def f(self):\n          pass\n    ")
        self.assertTypesMatchPytd(ty, "\n      import attr\n      from typing import Any, Annotated\n      @attr.s\n      class Foo:\n        y: str = ...\n        _x: int\n        x: Annotated[int, 'property']\n        def __init__(self, y: str = ...) -> None: ...\n        def f(self) -> None: ...\n        @staticmethod\n        def bar(x) -> None: ...\n        @classmethod\n        def foo(cls) -> None: ...\n    ")

    def test_subclass_auto_attribs(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import attr\n      @attr.define(auto_attribs=True)\n      class Foo:\n        x: bool\n        y: int = 42\n      class Bar(Foo):\n        def get_x(self):\n          return self.x\n        def get_y(self):\n          return self.y\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      @attr.s\n      class Foo:\n        x: bool\n        y: int = ...\n        def __init__(self, x: bool, y: int = ...) -> None: ...\n      class Bar(Foo):\n        def get_x(self) -> bool : ...\n        def get_y(self) -> int: ...\n    ')

    def test_partial_auto_attribs(self):
        if False:
            return 10
        ty = self.Infer('\n      import attr\n      @attr.define(auto_attribs=True)\n      class Foo:\n        foo: str\n      @attr.s  # Deliberately keeping this one @attr.s, test they work together.\n      class Bar:\n        bar: str = attr.ib()\n        baz = attr.ib()\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any\n      @attr.s\n      class Foo:\n        foo: str\n        def __init__(self, foo: str) -> None: ...\n      @attr.s\n      class Bar:\n        bar: str\n        baz: Any\n        def __init__(self, bar: str, baz) -> None: ...\n    ')

    def test_classvar_auto_attribs(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      from typing import ClassVar\n      import attr\n      @attr.define(auto_attribs=True)\n      class Foo:\n        x: ClassVar[int] = 10\n        y: str = 'hello'\n    ")
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import ClassVar\n      @attr.s\n      class Foo:\n        y: str = ...\n        x: ClassVar[int]\n        def __init__(self, y: str = ...) -> None: ...\n    ')

    def test_attrs_namespace(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import attrs\n      @attrs.define\n      class Foo:\n        x: int\n      @attrs.mutable\n      class Bar:\n        x: int\n      @attrs.frozen\n      class Baz:\n        x: int\n      @attrs.define\n      class Qux:\n        x: int = attrs.field(init=False)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      import attrs\n      @attr.s\n      class Foo:\n        x: int\n        def __init__(self, x: int) -> None: ...\n      @attr.s\n      class Bar:\n        x: int\n        def __init__(self, x: int) -> None: ...\n      @attr.s\n      class Baz:\n        x: int\n        def __init__(self, x: int) -> None: ...\n      @attr.s\n      class Qux:\n        x: int\n        def __init__(self) -> None: ...\n    ')

class TestPyiAttrs(test_base.BaseTest):
    """Tests for @attr.s in pyi files."""

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import attr\n        @attr.s\n        class A:\n          x: int\n          y: str\n      ')
            self.Check("\n        import foo\n        x = foo.A(10, 'hello')\n      ", pythonpath=[d.path])

    def test_docstring(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import attr\n        from typing import Union\n        @attr.s\n        class A:\n          __doc__: str  # should be filtered out\n          x: int\n          y: str\n      ')
            self.Check("\n        import foo\n        x = foo.A(10, 'hello')\n      ", pythonpath=[d.path])

    def test_type_mismatch(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import attr\n        @attr.s\n        class A:\n          x: int\n          y: str\n      ')
            self.CheckWithErrors('\n        import foo\n        x = foo.A(10, 20)  # wrong-arg-types\n      ', pythonpath=[d.path])

    def test_subclass(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import attr\n        @attr.s\n        class A:\n          x: bool\n          y: int\n      ')
            ty = self.Infer('\n        import attr\n        import foo\n        @attr.s(auto_attribs=True)\n        class Foo(foo.A):\n          z: str = "hello"\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import attr\n        from typing import Union\n        import foo\n        @attr.s\n        class Foo(foo.A):\n          z: str = ...\n          def __init__(self, x: bool, y: int, z: str = ...) -> None: ...\n      ')

    def test_subclass_from_same_pyi(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import attr\n        @attr.s\n        class A:\n          x: bool\n          y: int\n\n        @attr.s\n        class B(A):\n          z: str\n      ')
            ty = self.Infer('\n        import attr\n        import foo\n        @attr.s(auto_attribs=True)\n        class Foo(foo.B):\n          a: str = "hello"\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import attr\n        from typing import Union\n        import foo\n        @attr.s\n        class Foo(foo.B):\n          a: str = ...\n          def __init__(self, x: bool, y: int, z: str, a: str = ...) -> None: ...\n      ')

    def test_subclass_from_different_pyi(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('bar.pyi', '\n        import attr\n        @attr.s\n        class A:\n          x: bool\n          y: int\n      ')
            d.create_file('foo.pyi', '\n        import attr\n        import bar\n        @attr.s\n        class B(bar.A):\n          z: str\n      ')
            ty = self.Infer('\n        import attr\n        import foo\n        @attr.attrs(auto_attribs=True)\n        class Foo(foo.B):\n          a: str = "hello"\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import attr\n        from typing import Union\n        import foo\n        @attr.s\n        class Foo(foo.B):\n          a: str = ...\n          def __init__(self, x: bool, y: int, z: str, a: str = ...) -> None: ...\n      ')

    def test_subclass_with_kwonly(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        import attr\n        @attr.s\n        class A:\n          x: bool\n          y: int\n          def __init__(self, x: bool, *, y: int = ...): ...\n      ')
            ty = self.Infer('\n        import attr\n        import foo\n        @attr.s(auto_attribs=True)\n        class Foo(foo.A):\n          z: str\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import attr\n        from typing import Union\n        import foo\n        @attr.s\n        class Foo(foo.A):\n          z: str\n          def __init__(self, x: bool, z: str, *, y: int = ...) -> None: ...\n      ')

class TestPyiAttrsWrapper(test_base.BaseTest):
    """Tests for @attr.s wrappers in pyi files."""

    def test_basic(self):
        if False:
            print('Hello World!')
        foo_ty = self.Infer('\n      import attr\n      wrapper = attr.s(kw_only=True, auto_attribs=True)\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            ty = self.Infer('\n        import foo\n        @foo.wrapper\n        class Foo:\n          x: int\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Annotated, Callable\n\n        @attr.s\n        class Foo:\n          x: int\n          def __init__(self, *, x: int) -> None: ...\n      ')
if __name__ == '__main__':
    test_base.main()