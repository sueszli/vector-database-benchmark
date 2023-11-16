"""Tests for classes."""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class ClassesTest(test_base.BaseTest):
    """Tests for classes."""

    def test_make_class(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Thing(tuple):\n        def __init__(self, x):\n          self.x = x\n      def f():\n        x = Thing(1)\n        x.y = 3\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n    from typing import Any\n    class Thing(tuple):\n      x = ...  # type: Any\n      y = ...  # type: int\n      def __init__(self, x) -> NoneType: ...\n    def f() -> Thing: ...\n    ')

    def test_load_classderef(self):
        if False:
            i = 10
            return i + 15
        'Exercises the Python 3 LOAD_CLASSDEREF opcode.\n\n    Serves as a simple test for Python 2.\n    '
        self.Check('\n      class A:\n        def foo(self):\n          x = 10\n          class B:\n            y = str(x)\n    ')

    def test_class_decorator(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      @__any_object__\n      class MyClass:\n        def method(self, response):\n          pass\n      def f():\n        return MyClass()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      MyClass = ...  # type: Any\n      def f() -> Any: ...\n    ')

    def test_class_name(self):
        if False:
            return 10
        ty = self.Infer('\n      class MyClass:\n        def __init__(self, name):\n          pass\n      def f():\n        factory = MyClass\n        return factory("name")\n      f()\n    ', deep=False, show_library_calls=True)
        self.assertTypesMatchPytd(ty, '\n    class MyClass:\n      def __init__(self, name: str) -> NoneType: ...\n\n    def f() -> MyClass: ...\n    ')

    def test_inherit_from_unknown(self):
        if False:
            return 10
        ty = self.Infer('\n      class A(__any_object__):\n        pass\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n    from typing import Any\n    class A(Any):\n      pass\n    ')

    def test_inherit_from_unknown_and_call(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      x = __any_object__\n      class A(x):\n        def __init__(self):\n          x.__init__(self)\n    ')
        self.assertTypesMatchPytd(ty, '\n    from typing import Any\n    x = ...  # type: Any\n    class A(Any):\n      def __init__(self) -> NoneType: ...\n    ')

    def test_inherit_from_unknown_and_set_attr(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class Foo(__any_object__):\n        def __init__(self):\n          setattr(self, "test", True)\n    ')
        self.assertTypesMatchPytd(ty, '\n    from typing import Any\n    class Foo(Any):\n      def __init__(self) -> NoneType: ...\n    ')

    def test_inherit_from_unknown_and_initialize(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Foo:\n        pass\n      class Bar(Foo, __any_object__):\n        pass\n      x = Bar(duration=0)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class Foo:\n        pass\n      class Bar(Foo, Any):\n        pass\n      x = ...  # type: Bar\n    ')

    def test_inherit_from_unsolvable(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Any\n        def __getattr__(name) -> Any: ...\n      ')
            ty = self.Infer('\n        import a\n        class Foo:\n          pass\n        class Bar(Foo, a.A):\n          pass\n        x = Bar(duration=0)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        from typing import Any\n        class Foo:\n          pass\n        class Bar(Foo, Any):\n          pass\n        x = ...  # type: Bar\n      ')

    def test_classmethod(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      module = __any_object__\n      class Foo:\n        @classmethod\n        def bar(cls):\n          module.bar("", \'%Y-%m-%d\')\n      def f():\n        return Foo.bar()\n    ')
        self.assertTypesMatchPytd(ty, '\n    from typing import Any\n    module = ...  # type: Any\n    def f() -> NoneType: ...\n    class Foo:\n      @classmethod\n      def bar(cls) -> None: ...\n    ')

    def test_factory_classmethod(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        @classmethod\n        def factory(cls, *args, **kwargs):\n          return object.__new__(cls)\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Type, TypeVar\n      _TFoo = TypeVar('_TFoo', bound=Foo)\n      class Foo:\n        @classmethod\n        def factory(cls: Type[_TFoo], *args, **kwargs) -> _TFoo: ...\n    ")

    def test_classmethod_return_inference(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        A = 10\n        @classmethod\n        def bar(cls):\n          return cls.A\n    ')
        self.assertTypesMatchPytd(ty, '\n    class Foo:\n      A: int\n      @classmethod\n      def bar(cls) -> int: ...\n    ')

    def test_inherit_from_unknown_attributes(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo(__any_object__):\n        def f(self):\n          self.x = [1]\n          self.y = list(self.x)\n    ')
        self.assertTypesMatchPytd(ty, '\n    from typing import List\n    from typing import Any\n    class Foo(Any):\n      x = ...  # type: List[int]\n      y = ...  # type: List[int]\n      def f(self) -> NoneType: ...\n    ')

    def test_inner_class(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        class Foo:\n          x = 3\n        l = Foo()\n        return l.x\n    ', show_library_calls=True)
        self.assertTypesMatchPytd(ty, '\n      def f() -> int: ...\n    ')

    def test_super(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      class Base:\n        def __init__(self, x, y):\n          pass\n      class Foo(Base):\n        def __init__(self, x):\n          super(Foo, self).__init__(x, y='foo')\n    ")
        self.assertTypesMatchPytd(ty, '\n    class Base:\n      def __init__(self, x, y) -> NoneType: ...\n    class Foo(Base):\n      def __init__(self, x) -> NoneType: ...\n    ')

    def test_super_error(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      class Base:\n        def __init__(self, x, y, z):\n          pass\n      class Foo(Base):\n        def __init__(self, x):\n          super(Foo, self).__init__()  # missing-parameter[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'x'})

    def test_super_in_init(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class A:\n        def __init__(self):\n          self.x = 3\n\n      class B(A):\n        def __init__(self):\n          super(B, self).__init__()\n\n        def get_x(self):\n          return self.x\n    ', show_library_calls=True)
        self.assertTypesMatchPytd(ty, '\n        class A:\n          x = ...  # type: int\n          def __init__(self) -> None: ...\n\n        class B(A):\n          x = ...  # type: int\n          def get_x(self) -> int: ...\n          def __init__(self) -> None: ...\n    ')

    def test_super_diamond(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class A:\n        x = 1\n      class B(A):\n        y = 4\n      class C(A):\n        y = "str"\n        z = 3j\n      class D(B, C):\n        def get_x(self):\n          return super(D, self).x\n        def get_y(self):\n          return super(D, self).y\n        def get_z(self):\n          return super(D, self).z\n    ')
        self.assertTypesMatchPytd(ty, '\n      class A:\n          x = ...  # type: int\n      class B(A):\n          y = ...  # type: int\n      class C(A):\n          y = ...  # type: str\n          z = ...  # type: complex\n      class D(B, C):\n          def get_x(self) -> int: ...\n          def get_y(self) -> int: ...\n          def get_z(self) -> complex: ...\n    ')

    def test_inherit_from_list(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      class MyList(list):\n        def foo(self):\n          return getattr(self, '__str__')\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class MyList(list):\n        def foo(self) -> Any: ...\n    ')

    def test_class_attr(self):
        if False:
            return 10
        ty = self.Infer('\n      class Foo:\n        pass\n      OtherFoo = Foo().__class__\n      Foo.x = 3\n      OtherFoo.x = "bar"\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Type\n      class Foo:\n        x: str\n      OtherFoo: Type[Foo]\n    ')

    def test_call_class_attr(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Flag:\n        convert_method = int\n        def convert(self, value):\n          return self.convert_method(value)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Type\n      class Flag:\n        convert_method = ...  # type: Type[int]\n        def convert(self, value) -> int: ...\n    ')

    def test_bound_method(self):
        if False:
            return 10
        ty = self.Infer('\n      class Random:\n          def seed(self):\n            pass\n\n      _inst = Random()\n      seed = _inst.seed\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Callable\n      class Random:\n         def seed(self) -> None: ...\n\n      _inst = ...  # type: Random\n      def seed() -> None: ...\n    ')

    def test_mro_with_unsolvables(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from nowhere import X, Y  # pytype: disable=import-error\n      class Foo(Y):\n        pass\n      class Bar(X, Foo, Y):\n        pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      X = ...  # type: Any\n      Y = ...  # type: Any\n      class Foo(Any):\n        ...\n      class Bar(Any, Foo, Any):\n        ...\n    ')

    def test_property(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        def __init__(self):\n          self._name = "name"\n        def test(self):\n          return self.name\n        name = property(fget=lambda self: self._name)\n    ')
        self.assertTypesMatchPytd(ty, "\n      from typing import Annotated\n      class Foo:\n        _name = ...  # type: str\n        name = ...  # type: Annotated[str, 'property']\n        def __init__(self) -> None: ...\n        def test(self) -> str: ...\n    ")

    def test_descriptor_self(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        def __init__(self):\n          self._name = "name"\n        def __get__(self, obj, objtype):\n          return self._name\n      class Bar:\n        def test(self):\n          return self.foo\n        foo = Foo()\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        _name = ...  # type: str\n        def __init__(self) -> None: ...\n        def __get__(self, obj, objtype) -> str: ...\n      class Bar:\n        foo = ...  # type: str\n        def test(self) -> str: ...\n    ')

    def test_descriptor_instance(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        def __get__(self, obj, objtype):\n          return obj._name\n      class Bar:\n        def __init__(self):\n          self._name = "name"\n        def test(self):\n          return self.foo\n        foo = Foo()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class Foo:\n        def __get__(self, obj, objtype) -> Any: ...\n      class Bar:\n        _name = ...  # type: str\n        foo = ...  # type: Any\n        def __init__(self) -> None: ...\n        def test(self) -> str: ...\n    ')

    def test_descriptor_class(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class Foo:\n        def __get__(self, obj, objtype):\n          return objtype._name\n      class Bar:\n        def test(self):\n          return self.foo\n        _name = "name"\n        foo = Foo()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class Foo:\n        def __get__(self, obj, objtype) -> Any: ...\n      class Bar:\n        _name = ...  # type: str\n        foo = ...  # type: Any\n        def test(self) -> str: ...\n    ')

    def test_descriptor_split(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      class Foo:\n        def __get__(self, obj, cls):\n          if obj is None:\n            return ''\n          else:\n            return 0\n      class Bar:\n        foo = Foo()\n      x1 = Bar.foo\n      x2 = Bar().foo\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      class Foo:\n        def __get__(self, obj, cls) -> Union[str, int]: ...\n      class Bar:\n        foo: Union[str, int]\n      x1: str\n      x2: int\n    ')

    def test_bad_descriptor(self):
        if False:
            return 10
        ty = self.Infer('\n      class Foo:\n        __get__ = None\n      class Bar:\n        foo = Foo()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class Foo:\n        __get__ = ...  # type: None\n      class Bar:\n        foo = ...  # type: Any\n    ')

    def test_not_descriptor(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Foo:\n        pass\n      foo = Foo()\n      foo.__get__ = None\n      class Bar:\n        foo = foo\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        __get__ = ...  # type: None\n      foo = ...  # type: Foo\n      class Bar:\n        foo = ...  # type: Foo\n    ')

    def test_getattr(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        def __getattr__(self, name):\n          return "attr"\n      def f():\n        return Foo().foo\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def __getattr__(self, name) -> str: ...\n      def f() -> str: ...\n    ')

    def test_getattr_pyi(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Foo:\n          def __getattr__(self, name) -> str: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f():\n          return foo.Foo().foo\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        def f() -> str: ...\n      ')

    def test_getattribute(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class A:\n        def __getattribute__(self, name):\n          return 42\n      x = A().x\n    ')
        self.assertTypesMatchPytd(ty, '\n      class A:\n        def __getattribute__(self, name) -> int: ...\n      x = ...  # type: int\n    ')

    def test_getattribute_pyi(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        class A:\n          def __getattribute__(self, name) -> int: ...\n      ')
            ty = self.Infer('\n        import a\n        x = a.A().x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: int\n      ')

    def test_inherit_from_classobj(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        class A():\n          pass\n      ')
            ty = self.Infer('\n        import a\n        class C(a.A):\n          pass\n        name = C.__name__\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        class C(a.A):\n          pass\n        name = ... # type: str\n      ')

    def test_metaclass_getattribute(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('menum.pyi', '\n        from typing import Any\n        class EnumMeta(type):\n          def __getattribute__(self, name) -> Any: ...\n        class Enum(metaclass=EnumMeta): ...\n        class IntEnum(int, Enum): ...\n      ')
            ty = self.Infer('\n        import menum\n        class A(menum.Enum):\n          x = 1\n        class B(menum.IntEnum):\n          x = 1\n        enum1 = A.x\n        name1 = A.x.name\n        enum2 = B.x\n        name2 = B.x.name\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import menum\n        from typing import Any\n        class A(menum.Enum):\n          x = ...  # type: int\n        class B(menum.IntEnum):\n          x = ...  # type: int\n        enum1 = ...  # type: Any\n        name1 = ...  # type: Any\n        enum2 = ...  # type: Any\n        name2 = ...  # type: Any\n      ')

    def test_return_class_type(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Type, Union\n        class A:\n          x = ...  # type: int\n        class B:\n          x = ...  # type: str\n        def f(x: Type[A]) -> Type[A]: ...\n        def g() -> Type[Union[A, B]]: ...\n        def h() -> Type[Union[int, B]]: ...\n      ')
            ty = self.Infer('\n        import a\n        x1 = a.f(a.A).x\n        x2 = a.g().x\n        x3 = a.h().x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        from typing import Union\n        x1 = ...  # type: int\n        x2 = ...  # type: Union[int, str]\n        x3 = ...  # type: str\n      ')

    def test_call_class_type(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Type\n        class A: ...\n        class B:\n          MyA = ...  # type: Type[A]\n      ')
            ty = self.Infer('\n        import a\n        x = a.B.MyA()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x = ...  # type: a.A\n      ')

    def test_call_alias(self):
        if False:
            return 10
        ty = self.Infer('\n      class A: pass\n      B = A\n      x = B()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Type\n      class A: ...\n      B: Type[A]\n      x: A\n    ')

    def test_new(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        class A:\n          def __new__(cls, x: int) -> B: ...\n        class B: ...\n      ')
            ty = self.Infer('\n        import a\n        class C:\n          def __new__(cls):\n            return "hello world"\n        x1 = a.A(42)\n        x2 = C()\n        x3 = object.__new__(bool)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        class C:\n          def __new__(cls) -> str: ...\n        x1 = ...  # type: a.B\n        x2 = ...  # type: str\n        x3 = ...  # type: bool\n      ')

    def test_new_and_init(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class A:\n        def __new__(cls, a, b):\n          return super(A, cls).__new__(cls, a, b)\n        def __init__(self, a, b):\n          self.x = a + b\n      class B:\n        def __new__(cls, x):\n          v = A(x, 0)\n          v.y = False\n          return v\n        # __init__ should not be called\n        def __init__(self, x):\n          pass\n      x1 = A("hello", "world")\n      x2 = x1.x\n      x3 = B(3.14)\n      x4 = x3.x\n      x5 = x3.y\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Type, TypeVar\n      _TA = TypeVar("_TA", bound=A)\n      class A:\n        x = ...  # type: Any\n        y = ...  # type: bool\n        def __new__(cls: Type[_TA], a, b) -> _TA: ...\n        def __init__(self, a, b) -> None: ...\n      class B:\n        def __new__(cls, x) -> A: ...\n        def __init__(self, x) -> None: ...\n      x1 = ...  # type: A\n      x2 = ...  # type: str\n      x3 = ...  # type: A\n      x4 = ...  # type: float\n      x5 = ...  # type: bool\n    ')

    def test_new_and_init_pyi(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        N = TypeVar("N")\n        class A(Generic[T]):\n          def __new__(cls, x) -> A[nothing]: ...\n          def __init__(self, x: N):\n            self = A[N]\n        class B:\n          def __new__(cls) -> A[str]: ...\n          # __init__ should not be called\n          def __init__(self, x, y) -> None: ...\n      ')
            ty = self.Infer('\n        import a\n        x1 = a.A(0)\n        x2 = a.B()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        x1 = ...  # type: a.A[int]\n        x2 = ...  # type: a.A[str]\n      ')

    def test_get_type(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class A:\n        x = 3\n      def f():\n        return A() if __random__ else ""\n      B = type(A())\n      C = type(f())\n      D = type(int)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Type, Union\n      class A:\n        x = ...  # type: int\n      def f() -> Union[A, str]: ...\n      B: Type[A]\n      C: Type[Union[A, str]]\n      D: Type[type]\n    ')

    def test_type_attribute(self):
        if False:
            return 10
        ty = self.Infer('\n      class A:\n        x = 3\n      B = type(A())\n      x = B.x\n      mro = B.mro()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Type\n      class A:\n        x: int\n      B: Type[A]\n      x: int\n      mro: list\n    ')

    def test_type_subclass(self):
        if False:
            return 10
        ty = self.Infer('\n      class A(type):\n        def __init__(self, name, bases, dict):\n          super(A, self).__init__(name, bases, dict)\n        def f(self):\n          return 3.14\n      Int = A(0)\n      X = A("X", (int, object), {"a": 1})\n      x = X()\n      a = X.a\n      v = X.f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Type\n      class A(type):\n        def __init__(self, name, bases, dict) -> None: ...\n        def f(self) -> float: ...\n      Int = ...  # type: Type[int]\n      class X(int, object, metaclass=A):\n        a = ...  # type: int\n      x = ...  # type: X\n      a = ...  # type: int\n      v = ...  # type: float\n    ')

    def test_union_base_class(self):
        if False:
            return 10
        self.Check('\n      import typing\n      class A(tuple): pass\n      class B(tuple): pass\n      class Foo(typing.Union[A,B]): pass\n      ')

    def test_metaclass_pyi(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        class A(type):\n          def f(self) -> float: ...\n        class X(metaclass=A): ...\n      ')
            ty = self.Infer('\n        import a\n        v = a.X.f()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import a\n        v = ...  # type: float\n      ')

    def test_unsolvable_metaclass(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('a.pyi', '\n        from typing import Any\n        def __getattr__(name) -> Any: ...\n      ')
            d.create_file('b.pyi', '\n        from a import A\n        class B(metaclass=A): ...\n      ')
            ty = self.Infer('\n        import b\n        x = b.B.x\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import b\n        from typing import Any\n        x = ...  # type: Any\n      ')

    def test_make_type(self):
        if False:
            return 10
        ty = self.Infer('\n      X = type("X", (int, object), {"a": 1})\n      x = X()\n      a = X.a\n    ')
        self.assertTypesMatchPytd(ty, '\n      class X(int, object):\n        a = ...  # type: int\n      x = ...  # type: X\n      a = ...  # type: int\n    ')

    def test_make_simple_type(self):
        if False:
            return 10
        ty = self.Infer('\n      X = type("X", (), {})\n      x = X()\n    ')
        self.assertTypesMatchPytd(ty, '\n      class X: ...\n      x = ...  # type: X\n    ')

    def test_make_ambiguous_type(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      if __random__:\n        name = "A"\n      else:\n        name = "B"\n      X = type(name, (int, object), {"a": 1})\n      x = X()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      name = ...  # type: str\n      X = ...  # type: Any\n      x = ...  # type: Any\n    ')

    def test_type_init(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import six\n      class A(type):\n        def __init__(self, name, bases, members):\n          self.x = 42\n          super(A, self).__init__(name, bases, members)\n      B = A("B", (), {})\n      class C(six.with_metaclass(A, object)):\n        pass\n      x1 = B.x\n      x2 = C.x\n    ')
        self.assertTypesMatchPytd(ty, '\n      import six\n      class A(type):\n        x: int\n        def __init__(self, name, bases, members) -> None: ...\n      class B(object, metaclass=A):\n        x: int\n      class C(object, metaclass=A):\n        x: int\n      x1: int\n      x2: int\n    ')

    def test_bad_mro_parameterized_class(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, TypeVar\n        T = TypeVar("T")\n        class A(Generic[T]): ...\n        class B(A[T]): ...\n        class C(A[T], B[T]): ...\n        def f() -> C[int]: ...\n      ')
            (_, errors) = self.InferWithErrors('\n        import foo\n        foo.f()  # mro-error[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'C'})

    def test_call_parameterized_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.InferWithErrors('\n      from typing import Deque\n      list[str]()\n      Deque[str]()\n    ')

    def test_errorful_constructors(self):
        if False:
            return 10
        (ty, _) = self.InferWithErrors('\n      class Foo:\n        attr = 42\n        def __new__(cls):\n          return name_error  # name-error\n        def __init__(self):\n          self.attribute_error  # attribute-error\n          self.instance_attr = self.attr\n        def f(self):\n          return self.instance_attr\n    ', deep=True)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class Foo:\n        attr = ...  # type: int\n        instance_attr = ...  # type: int\n        def __new__(cls) -> Any: ...\n        def __init__(self) -> None: ...\n        def f(self) -> int: ...\n    ')

    def test_new_false(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        def __new__(cls):\n          return False\n        def __init__(self):\n          self.instance_attr = ""\n        def f(self):\n          return self.instance_attr\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        instance_attr = ...  # type: str\n        def __new__(cls) -> bool: ...\n        def __init__(self) -> None: ...\n        def f(self) -> str: ...\n    ')

    def test_new_ambiguous(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Foo:\n        def __new__(cls):\n          if __random__:\n            return super(Foo, cls).__new__(cls)\n          else:\n            return "hello world"\n        def __init__(self):\n          self.instance_attr = ""\n        def f(self):\n          return self.instance_attr\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      class Foo:\n        instance_attr = ...  # type: str\n        def __new__(cls) -> Union[str, Foo]: ...\n        def __init__(self) -> None: ...\n        def f(self) -> str: ...\n    ')

    def test_new_extra_arg(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def __new__(cls, _):\n          return super(Foo, cls).__new__(cls)\n      Foo("Foo")\n    ')

    def test_new_extra_none_return(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        def __new__(cls):\n          if __random__:\n            return super(Foo, cls).__new__(cls)\n        def foo(self):\n          return self\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypeVar, Union\n      _TFoo = TypeVar("_TFoo", bound=Foo)\n      class Foo:\n        def __new__(cls) -> Union[Foo, None]: ...\n        def foo(self: _TFoo) -> _TFoo: ...\n    ')

    def test_super_new_extra_arg(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class Foo:\n        def __init__(self, x):\n          pass\n        def __new__(cls, x):\n          # The extra arg is okay because __init__ is defined.\n          return super(Foo, cls).__new__(cls, x)\n    ')

    def test_super_init_extra_arg(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Foo:\n        def __init__(self, x):\n          # The extra arg is okay because __new__ is defined.\n          super(Foo, self).__init__(x)\n        def __new__(cls, x):\n          return super(Foo, cls).__new__(cls)\n    ')

    def test_super_init_extra_arg2(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Foo:\n          def __new__(cls, a, b) -> Foo: ...\n      ')
            self.Check('\n        import foo\n        class Bar(foo.Foo):\n          def __init__(self, a, b):\n            # The extra args are okay because __new__ is defined on Foo.\n            super(Bar, self).__init__(a, b)\n      ', pythonpath=[d.path])

    def test_super_new_wrong_arg_count(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      class Foo:\n        def __new__(cls, x):\n          return super(Foo, cls).__new__(cls, x)  # wrong-arg-count[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': '1.*2'})

    def test_super_init_wrong_arg_count(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      class Foo:\n        def __init__(self, x):\n          super(Foo, self).__init__(x)  # wrong-arg-count[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': '1.*2'})

    def test_super_new_missing_parameter(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      class Foo:\n        def __new__(cls, x):\n          # Even when __init__ is defined, too few args is an error.\n          return super(Foo, cls).__new__()  # missing-parameter[e]\n        def __init__(self, x):\n          pass\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'cls.*__new__'})

    def test_new_kwarg(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      class Foo:\n        def __new__(cls):\n          # ok because __init__ is defined.\n          return super(Foo, cls).__new__(cls, x=42)\n        def __init__(self):\n          pass\n      class Bar:\n        def __new__(cls):\n          return super(Bar, cls).__new__(cls, x=42)  # wrong-keyword-args[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'x.*__new__'})

    def test_init_kwarg(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      class Foo:\n        def __init__(self):\n          # ok because __new__ is defined.\n          super(Foo, self).__init__(x=42)\n        def __new__(cls):\n          return super(Foo, cls).__new__(cls)\n      class Bar:\n        def __init__(self):\n          super(Bar, self).__init__(x=42)  # wrong-keyword-args[e]\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'x.*__init__'})

    def test_alias_inner_class(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        class Bar:\n          def __new__(cls, _):\n            return super(Bar, cls).__new__(cls)\n        return Bar\n      Baz = f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Type, TypeVar\n      def f() -> Type[Baz]: ...\n      _TBaz = TypeVar("_TBaz", bound=Baz)\n      class Baz:\n        def __new__(cls: Type[_TBaz], _) -> _TBaz: ...\n    ')

    def test_module_in_class_definition_scope(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Bar: ...\n      ')
            self.Check('\n        import foo\n        class ConstStr(str):\n          foo.Bar # testing that this does not affect inference.\n          def __new__(cls, x):\n            obj = super(ConstStr, cls).__new__(cls, x)\n            return obj\n      ', pythonpath=[d.path])

    def test_init_with_no_params(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def __init__():\n          pass\n      ')

    def test_instantiate_with_abstract_dict(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      X = type("", (), dict())\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      class X: ...\n    ')

    def test_not_instantiable(self):
        if False:
            return 10
        self.CheckWithErrors('\n      class Foo:\n        def __new__(cls):\n          assert cls is not Foo, "not instantiable"\n        def foo(self):\n          name_error  # name-error\n    ')

    def test_metaclass_on_unknown_class(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import six\n      class Foo(type):\n        pass\n      def decorate(cls):\n        return __any_object__\n      @six.add_metaclass(Foo)\n      @decorate\n      class Bar:\n        pass\n    ')

    def test_subclass_contains_base(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def get_c():\n        class C:\n          def __init__(self, z):\n            self.a = 3\n            self.c = z\n          def baz(self): pass\n        return C\n      class DC(get_c()):\n        def __init__(self, z):\n          super(DC, self).__init__(z)\n          self.b = "hello"\n        def bar(self, x): pass\n      x = DC(1)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class DC:\n          a = ...  # type: int\n          b = ...  # type: str\n          c = ...  # type: Any\n          def __init__(self, z) -> None: ...\n          def bar(self, x) -> None: ...\n          def baz(self) -> None: ...\n      def get_c() -> type: ...\n      x = ...  # type: DC\n    ')

    def test_subclass_multiple_base_options(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class A: pass\n      def get_base():\n        class B: pass\n        return B\n      Base = A if __random__ else get_base()\n      class C(Base): pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Union\n      def get_base() -> type: ...\n      class A: pass\n      Base = ...  # type: type\n      class C(Any): pass\n    ')

    def test_subclass_contains_generic_base(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import typing\n      def get_base():\n        class C(typing.List[str]):\n          def get_len(self): return len(self)\n        return C\n      class DL(get_base()): pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      import typing\n      class DL(List[str]):\n          def get_len(self) -> int: ...\n      def get_base() -> type: ...\n    ')

    def test_subclass_overrides_base_attributes(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def get_base():\n        class B:\n          def __init__(self):\n            self.a = 1\n            self.b = 2\n          def bar(self, x): pass\n          def baz(self): pass\n        return B\n      class C(get_base()):\n        def __init__(self):\n          super(C, self).__init__()\n          self.b = "hello"\n          self.c = "world"\n        def bar(self, x): pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      def get_base() -> type: ...\n      class C:\n        a = ...  # type: int\n        b = ...  # type: str\n        c = ...  # type: str\n        def __init__(self) -> None: ...\n        def bar(self, x) -> None: ...\n        def baz(self) -> None: ...\n    ')

    def test_subclass_make_base(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def make_base(x):\n        class C(x):\n          def __init__(self):\n            self.x = 1\n        return C\n      class BX(make_base(list)): pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      def make_base(x) -> type: ...\n      class BX(list):\n        x = ...  # type: int\n        def __init__(self) -> None: ...\n    ')

    def test_subclass_bases_overlap(self):
        if False:
            return 10
        ty = self.Infer('\n      def make_a():\n        class A:\n          def __init__(self):\n            self.x = 1\n        return A\n      def make_b():\n        class B:\n          def __init__(self):\n            self.x = "hello"\n        return B\n      class C(make_a(), make_b()):\n        pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      def make_a() -> type: ...\n      def make_b() -> type: ...\n      class C:\n        x = ...  # type: int\n        def __init__(self) -> None: ...\n    ')

    def test_pyi_nested_class(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class X:\n          class Y: ...\n      ')
            ty = self.Infer('\n        import foo\n        Y = foo.X.Y\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Type\n        Y: Type[foo.X.Y]\n      ')
            d.create_file('bar.pyi', pytd_utils.Print(ty))
            ty = self.Infer('\n        import bar\n        Y = bar.Y\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import bar\n        from typing import Type\n        Y: Type[foo.X.Y]\n      ')

    def test_pyi_nested_class_alias(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class X:\n          class Y: ...\n          Z = X.Y\n      ')
            ty = self.Infer('\n        import foo\n        Z = foo.X.Z\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Type\n        import foo\n        Z: Type[foo.X.Y]\n      ')

    def test_pyi_deeply_nested_class(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class X:\n          class Y:\n            class Z: ...\n      ')
            ty = self.Infer('\n        import foo\n        Z = foo.X.Y.Z\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Type\n        import foo\n        Z: Type[foo.X.Y.Z]\n      ')

    def test_late_annotation(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      class Foo:\n        bar = None  # type: 'Bar'\n      class Bar:\n        def __init__(self):\n          self.x = 0\n      class Baz(Foo):\n        def f(self):\n          return self.bar.x\n    ")
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        bar: Bar\n      class Bar:\n        x: int\n        def __init__(self) -> None: ...\n      class Baz(Foo):\n        def f(self) -> int: ...\n    ')

    def test_iterate_ambiguous_base_class(self):
        if False:
            return 10
        self.Check('\n      from typing import Any\n      class Foo(Any):\n        pass\n      list(Foo())\n    ')

    def test_instantiate_class(self):
        if False:
            while True:
                i = 10
        self.Check('\n      import abc\n      import six\n      @six.add_metaclass(abc.ABCMeta)\n      class Foo:\n        def __init__(self, x):\n          if x > 0:\n            print(self.__class__(x-1))\n        @abc.abstractmethod\n        def f(self):\n          pass\n    ')
if __name__ == '__main__':
    test_base.main()