"""Tests for attrs library in attr_overlay.py."""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class TestAttrib(test_base.BaseTest):
    """Tests for attr.ib."""

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib()\n        y = attr.ib(type=int)\n        z = attr.ib(type=str)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any\n      @attr.s\n      class Foo:\n        x: Any\n        y: int\n        z: str\n        def __init__(self, x, y: int, z: str) -> None: ...\n    ')

    def test_interpreter_class(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import attr\n      class A: pass\n      @attr.s\n      class Foo:\n        x = attr.ib(type=A)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      class A: ...\n      @attr.s\n      class Foo:\n        x: A\n        def __init__(self, x: A) -> None: ...\n    ')

    def test_typing(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import List\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(type=List[int])\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import List\n      @attr.s\n      class Foo:\n        x: List[int]\n        def __init__(self, x: List[int]) -> None: ...\n    ')

    def test_union_types(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import Union\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(type=Union[str, int])\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: Union[str, int]\n        def __init__(self, x: Union[str, int]) -> None: ...\n    ')

    def test_comment_annotations(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import Union\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib() # type: Union[str, int]\n        y = attr.ib(type=str)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: Union[str, int]\n        y: str\n        def __init__(self, x: Union[str, int], y: str) -> None: ...\n    ')

    def test_late_annotations(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib() # type: 'Foo'\n        y = attr.ib() # type: str\n    ")
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: Foo\n        y: str\n        def __init__(self, x: Foo, y: str) -> None: ...\n    ')

    def test_late_annotation_in_type(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(type='Foo')\n    ")
        self.assertTypesMatchPytd(ty, '\n      import attr\n      @attr.s\n      class Foo:\n        x: Foo\n        def __init__(self, x: Foo) -> None: ...\n    ')

    def test_classvar(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib() # type: int\n        y = attr.ib(type=str)\n        z = 1 # class var, should not be in __init__\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: int\n        y: str\n        z: int\n        def __init__(self, x: int, y: str) -> None: ...\n    ')

    def test_type_clash(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      import attr\n      @attr.s\n      class Foo(object):  # invalid-annotation\n        x = attr.ib(type=str) # type: int\n        y = attr.ib(type=str, default="")  # type: int\n      Foo(x="")  # should not report an error\n    ')

    def test_bad_type(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(type=10)  # invalid-annotation\n    ')

    def test_name_mangling(self):
        if False:
            return 10
        ty = self.Infer('\n      import attr\n      @attr.s\n      class Foo:\n        _x = attr.ib(type=int)\n        __y = attr.ib(type=int)\n        ___z = attr.ib(type=int)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      @attr.s\n      class Foo:\n        _x: int\n        _Foo__y: int\n        _Foo___z: int\n        def __init__(self, x: int, Foo__y: int, Foo___z: int) -> None: ...\n    ')

    def test_defaults(self):
        if False:
            print('Hello World!')
        (ty, err) = self.InferWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(default=42)\n        y = attr.ib(type=int, default=6)\n        z = attr.ib(type=str, default=28)  # annotation-type-mismatch[e]\n        a = attr.ib(type=str, default=None)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: int = ...\n        y: int = ...\n        z: str = ...\n        a: str = ...\n        def __init__(self, x: int = ..., y: int = ..., z: str = ...,\n                     a: str = ...) -> None: ...\n    ')
        self.assertErrorRegexes(err, {'e': 'annotation for z'})

    def test_defaults_with_typecomment(self):
        if False:
            while True:
                i = 10
        (ty, err) = self.InferWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(default=42) # type: int\n        y = attr.ib(default=42) # type: str  # annotation-type-mismatch[e]\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: int = ...\n        y: str = ...\n        def __init__(self, x: int = ..., y: str = ...) -> None: ...\n    ')
        self.assertErrorRegexes(err, {'e': 'annotation for y'})

    def test_factory_class(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import attr\n      class CustomClass:\n        pass\n      @attr.s\n      class Foo:\n        x = attr.ib(factory=list)\n        y = attr.ib(factory=CustomClass)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      class CustomClass: ...\n      @attr.s\n      class Foo:\n        x: list = ...\n        y: CustomClass = ...\n        def __init__(self, x: list = ..., y: CustomClass = ...) -> None: ...\n    ')

    def test_factory_function(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import attr\n      class CustomClass:\n        pass\n      def unannotated_func():\n        return CustomClass()\n      @attr.s\n      class Foo:\n        x = attr.ib(factory=locals)\n        y = attr.ib(factory=unannotated_func)\n    ')
        self.assertTypesMatchPytd(ty, "\n      import attr\n      from typing import Any, Dict, Union\n      class CustomClass: ...\n      def unannotated_func() -> CustomClass: ...\n      @attr.s\n      class Foo:\n        x: Dict[str, Any] = ...\n        y: Any = ...  # b/64832148: the return type isn't inferred early enough\n        def __init__(self, x: Dict[str, object] = ..., y = ...) -> None: ...\n    ")

    def test_verbose_factory(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(default=attr.Factory(list))\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: list = ...\n        def __init__(self, x: list = ...) -> None: ...\n    ')

    def test_bad_factory(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(default=attr.Factory(42))  # wrong-arg-types[e1]\n        y = attr.ib(factory=42)  # wrong-arg-types[e2]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Callable.*int', 'e2': 'Callable.*int'})

    def test_default_factory_clash(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(default=None, factory=list)  # duplicate-keyword-argument[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'default'})

    def test_takes_self(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(default=attr.Factory(len, takes_self=True))\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      @attr.s\n      class Foo:\n        x: int = ...\n        def __init__(self, x: int = ...) -> None: ...\n    ')

    def test_default_none(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(default=None)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any\n      @attr.s\n      class Foo:\n        x: Any = ...\n        def __init__(self, x: Any = ...) -> None: ...\n    ')

    def test_annotation_type(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing import List\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(type=List)\n      x = Foo([]).x\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      @attr.s\n      class Foo:\n        x: list\n        def __init__(self, x: list) -> None: ...\n      x: list\n    ')

    def test_instantiation(self):
        if False:
            print('Hello World!')
        self.Check('\n      import attr\n      class A:\n        def __init__(self):\n          self.w = None\n      @attr.s\n      class Foo:\n        x = attr.ib(type=A)\n        y = attr.ib()  # type: A\n        z = attr.ib(factory=A)\n      foo = Foo(A(), A())\n      foo.x.w\n      foo.y.w\n      foo.z.w\n    ')

    def test_init(self):
        if False:
            return 10
        self.Check("\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(init=False, default='')  # type: str\n        y = attr.ib()  # type: int\n      foo = Foo(42)\n      foo.x\n      foo.y\n    ")

    def test_init_type(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(init=False, default='')  # type: str\n        y = attr.ib()  # type: int\n    ")
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class Foo:\n        x: str = ...\n        y: int\n        def __init__(self, y: int) -> None: ...\n    ')

    def test_init_bad_constant(self):
        if False:
            return 10
        err = self.CheckWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(init=0)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(err, {'e': 'bool.*int'})

    def test_init_bad_kwarg(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(init=__random__)  # type: str  # not-supported-yet\n    ')

    def test_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNoCrash(self.Check, "\n      import attr\n      class X(attr.make_class('X', {'y': attr.ib(default=None)})):\n        pass\n    ")

    def test_base_class_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      import attr\n      @attr.s\n      class A:\n        a = attr.ib()  # type: int\n      @attr.s\n      class B:\n        b = attr.ib()  # type: str\n      @attr.s\n      class C(A, B):\n        c = attr.ib()  # type: int\n      x = C(10, 'foo', 42)\n      x.a\n      x.b\n      x.c\n    ")

    def test_base_class_attrs_type(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import attr\n      @attr.s\n      class A:\n        a = attr.ib()  # type: int\n      @attr.s\n      class B:\n        b = attr.ib()  # type: str\n      @attr.s\n      class C(A, B):\n        c = attr.ib()  # type: int\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class A:\n        a: int\n        def __init__(self, a: int) -> None: ...\n      @attr.s\n      class B:\n        b: str\n        def __init__(self, b: str) -> None: ...\n      @attr.s\n      class C(A, B):\n        c: int\n        def __init__(self, a: int, b: str, c: int) -> None: ...\n    ')

    def test_base_class_attrs_override_type(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import attr\n      @attr.s\n      class A:\n        a = attr.ib()  # type: int\n      @attr.s\n      class B:\n        b = attr.ib()  # type: str\n      @attr.s\n      class C(A, B):\n        a = attr.ib()  # type: str\n        c = attr.ib()  # type: int\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class A:\n        a: int\n        def __init__(self, a: int) -> None: ...\n      @attr.s\n      class B:\n        b: str\n        def __init__(self, b: str) -> None: ...\n      @attr.s\n      class C(A, B):\n        a: str\n        c: int\n        def __init__(self, b: str, a: str, c: int) -> None: ...\n    ')

    def test_base_class_attrs_init(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import attr\n      @attr.s\n      class A:\n        a = attr.ib(init=False)  # type: int\n      @attr.s\n      class B:\n        b = attr.ib()  # type: str\n      @attr.s\n      class C(A, B):\n        c = attr.ib()  # type: int\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Union\n      @attr.s\n      class A:\n        a: int\n        def __init__(self) -> None: ...\n      @attr.s\n      class B:\n        b: str\n        def __init__(self, b: str) -> None: ...\n      @attr.s\n      class C(A, B):\n        c: int\n        def __init__(self, b: str, c: int) -> None: ...\n    ')

    def test_base_class_attrs_abstract_type(self):
        if False:
            return 10
        ty = self.Infer('\n      import attr\n      @attr.s\n      class Foo(__any_object__):\n        a = attr.ib()  # type: int\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any\n      @attr.s\n      class Foo(Any):\n        a: int\n        def __init__(self, a: int) -> None: ...\n    ')

    def test_method_decorators(self):
        if False:
            print('Hello World!')
        (ty, err) = self.InferWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        a = attr.ib()\n        b = attr.ib()\n        c = attr.ib(type=str)  # annotation-type-mismatch[e]\n        @a.validator\n        def validate(self, attribute, value):\n          pass\n        @a.default\n        def default_a(self):\n          # type: (...) -> int\n          return 10\n        @b.default\n        def default_b(self):\n          return 10\n        @c.default\n        def default_c(self):\n          # type: (...) -> int\n          return 10\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any, Union\n      @attr.s\n      class Foo:\n        a: int = ...\n        b: int = ...\n        c: str = ...\n        def __init__(self, a: int = ..., b: int = ..., c: str = ...) -> None: ...\n        def default_a(self) -> int: ...\n        def default_b(self) -> int: ...\n        def default_c(self) -> int: ...\n        def validate(self, attribute, value) -> None: ...\n    ')
        self.assertErrorRegexes(err, {'e': 'annotation for c'})

    def test_default_decorator_using_self(self):
        if False:
            return 10
        ty = self.Infer('\n      import attr\n      @attr.s\n      class Foo:\n        a = attr.ib(default=42)\n        b = attr.ib()\n        c = attr.ib(type=str)\n        @b.default\n        def default_b(self):\n          return self.a\n        @c.default\n        def default_c(self):\n          return self.b\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any\n      @attr.s\n      class Foo:\n        a: int = ...\n        b: Any = ...\n        c: str = ...\n        def __init__(self, a: int = ..., b = ..., c: str = ...) -> None: ...\n        def default_b(self) -> int: ...\n        def default_c(self) -> Any: ...\n    ')

    def test_repeated_default(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import attr\n\n      class Call:\n        pass\n\n      @attr.s\n      class Function:\n        params = attr.ib(factory=list)\n        calls = attr.ib(factory=list)\n\n      class FunctionMap:\n\n        def __init__(self, index):\n          self.fmap = {"": Function()}\n\n        def print_params(self):\n          for param in self.fmap[""].params:\n            print(param.name)\n\n        def add_call(self, call):\n          self.fmap[""].calls.append(Call())\n    ')

    def test_empty_factory(self):
        if False:
            return 10
        ty = self.Infer('\n      import attr\n      FACTORIES = []\n      @attr.s\n      class Foo:\n        x = attr.ib(factory=FACTORIES[0])\n      Foo(x=0)  # should not be an error\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any, List\n      FACTORIES: List[nothing]\n      @attr.s\n      class Foo:\n        x: Any = ...\n        def __init__(self, x = ...) -> None: ...\n    ')

    def test_empty_tuple_default(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(default=())\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      @attr.s\n      class Foo:\n        x: tuple = ...\n        def __init__(self, x: tuple = ...) -> None: ...\n    ')

    def test_long_alias(self):
        if False:
            return 10
        self.Check('\n      import attr\n      @attr.s\n      class Foo:\n        x= attr.attrib(default=0)  # type: int\n    ')

    def test_typevar_in_type_arg(self):
        if False:
            return 10
        self.Check("\n      import attr\n      from typing import Callable, TypeVar\n      T = TypeVar('T')\n      @attr.s\n      class Foo:\n        f = attr.ib(type=Callable[[T], T])\n      assert_type(Foo(__any_object__).f(0), int)\n    ")

    def test_bad_typevar_in_type_arg(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n      import attr\n      from typing import TypeVar\n      T = TypeVar('T')\n      @attr.s\n      class Foo:\n        x = attr.ib(type=T)  # invalid-annotation\n    ")

    def test_bad_constructor(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(default=10, init=0)  # wrong-arg-types\n      a = Foo().x\n      assert_type(a, int)\n    ')

    def test_bad_factory_constructor(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(default=10)\n        y = attr.ib(factory=10, type=int)  # wrong-arg-types\n    ')

    def test_multiple_bad_constructor_args(self):
        if False:
            return 10
        self.CheckWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(init=0, validator=10, type=int)  # wrong-arg-types  # wrong-arg-types\n      a = Foo(10).x\n      assert_type(a, int)\n    ')

    def test_extra_constructor_args(self):
        if False:
            return 10
        self.CheckWithErrors('\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(bar=10, type=int)  # wrong-keyword-args\n      a = Foo(10).x\n      assert_type(a, int)\n    ')

    @test_base.skip('b/203591182')
    def test_duplicate_constructor_args(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors("\n      import attr\n      @attr.s\n      class Foo:\n        x = attr.ib(10, default='a')  # duplicate-keyword-argument\n      a = Foo().x\n      assert_type(a, int)\n    ")

class TestAttrs(test_base.BaseTest):
    """Tests for attr.s."""

    def test_basic(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import attr\n      @attr.s()\n      class Foo:\n        x = attr.ib()\n        y = attr.ib(type=int)\n        z = attr.ib(type=str)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any\n      @attr.s\n      class Foo:\n        x: Any\n        y: int\n        z: str\n        def __init__(self, x, y: int, z: str) -> None: ...\n    ')

    def test_no_init(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import attr\n      @attr.s(init=False)\n      class Foo:\n        x = attr.ib()\n        y = attr.ib(type=int)\n        z = attr.ib(type=str)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import attr\n      from typing import Any\n      @attr.s\n      class Foo:\n        x: Any\n        y: int\n        z: str\n        def __attrs_init__(self, x, y: int, z: str) -> None: ...\n    ')

    def test_init_bad_constant(self):
        if False:
            print('Hello World!')
        err = self.CheckWithErrors('\n      import attr\n      @attr.s(init=0)  # wrong-arg-types[e]\n      class Foo:\n        pass\n    ')
        self.assertErrorRegexes(err, {'e': 'bool.*int'})

    def test_bad_kwarg(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      import attr\n      @attr.s(init=__random__)  # not-supported-yet\n      class Foo:\n        pass\n    ')

    def test_depth(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import attr\n      def f():\n        @attr.s\n        class Foo:\n          pass\n    ', maximum_depth=1)

    def test_signature(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      import attr\n      @attr.s()\n      class A:\n        id = attr.ib(\n            default='', converter=str,\n            on_setattr=attr.setters.convert)\n    ")

class TestInheritedAttrib(test_base.BaseTest):
    """Tests for attrs in a different module."""

    def test_attrib_wrapper(self):
        if False:
            for i in range(10):
                print('nop')
        foo_ty = self.Infer('\n      import attr\n      def attrib_wrapper(*args, **kwargs):\n        return attr.ib(*args, **kwargs)\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            self.CheckWithErrors("\n        import attr\n        import foo\n        @attr.s()\n        class Foo:\n          x: int = foo.attrib_wrapper()\n          y = foo.attrib_wrapper(type=int)\n        a = Foo(10, 10)\n        b = Foo(10, '10')  # The wrapper returns attr.ib(Any) so y.type is lost\n        c = Foo(10, 20, 30)  # wrong-arg-count\n        d = Foo('10', 20)  # wrong-arg-types\n      ", pythonpath=[d.path])

    def test_attrib_wrapper_kwargs(self):
        if False:
            print('Hello World!')
        foo_ty = self.Infer('\n      import attr\n      def kw_attrib(typ):\n        return attr.ib(typ, kw_only=True)\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            self.CheckWithErrors('\n        import attr\n        import foo\n        @attr.s()\n        class Foo:\n          x = foo.kw_attrib(int)\n        a = Foo(10)  # missing-parameter\n        b = Foo(x=10)\n      ', pythonpath=[d.path])

    def test_wrapper_setting_type(self):
        if False:
            while True:
                i = 10
        foo_ty = self.Infer('\n      import attr\n      def int_attrib():\n        return attr.ib(type=int)\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            self.CheckWithErrors('\n        import attr\n        import foo\n        @attr.s()\n        class Foo(object):  # invalid-annotation\n          x: int = foo.int_attrib()\n      ', pythonpath=[d.path])

    def test_wrapper_setting_default(self):
        if False:
            print('Hello World!')
        foo_ty = self.Infer('\n      import attr\n      def default_attrib(typ):\n        return attr.ib(type=typ, default=None)\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            self.Check('\n        import attr\n        import foo\n        @attr.s()\n        class Foo:\n          y = attr.ib(default = 10)\n          x = foo.default_attrib(int)\n        a = Foo()\n      ', pythonpath=[d.path])

    def test_override_protected_member(self):
        if False:
            return 10
        foo_ty = self.Infer('\n      import attr\n      @attr.s\n      class A:\n        _x = attr.ib(type=str)\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo_ty))
            self.CheckWithErrors("\n        import attr\n        import foo\n        @attr.s()\n        class B(foo.A):\n          _x = attr.ib(init=False, default='')\n          y = attr.ib(type=int)\n        a = foo.A('10')\n        b = foo.A(x='10')\n        c = B(10)\n        d = B(y=10)\n        e = B('10', 10)  # wrong-arg-count\n      ", pythonpath=[d.path])
if __name__ == '__main__':
    test_base.main()