"""Test instance and class attributes."""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class TestStrictNone(test_base.BaseTest):
    """Tests for strict attribute checking on None."""

    def test_module_constant(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      x = None\n      def f():\n        return x.upper()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*None'})

    def test_class_constant(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      class Foo:\n        x = None\n        def f(self):\n          return self.x.upper()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*None'})

    def test_class_constant_error(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      x = None\n      class Foo:\n        x = x.upper()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*None'})

    def test_multiple_paths(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      x = None\n      def f():\n        z = None if __random__ else x\n        y = z\n        return y.upper()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*None'})

    def test_late_initialization(self):
        if False:
            return 10
        (ty, _) = self.InferWithErrors('\n      class Foo:\n        def __init__(self):\n          self.x = None\n        def f(self):\n          return self.x.upper()  # attribute-error\n        def set_x(self):\n          self.x = ""\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Optional\n      class Foo:\n        x = ...  # type: Optional[str]\n        def __init__(self) -> None: ...\n        def f(self) -> Any: ...\n        def set_x(self) -> None: ...\n    ')

    def test_pyi_constant(self):
        if False:
            i = 10
            return i + 15
        self.options.tweak(strict_none_binding=False)
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        x = ...  # type: None\n      ')
            self.Check('\n        import foo\n        def f():\n          return foo.x.upper()\n      ', pythonpath=[d.path])

    def test_pyi_attribute(self):
        if False:
            return 10
        self.options.tweak(strict_none_binding=False)
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Foo:\n          x = ...  # type: None\n      ')
            self.Check('\n        import foo\n        def f():\n          return foo.Foo.x.upper()\n      ', pythonpath=[d.path])

    def test_return_value(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      def f():\n        pass\n      def g():\n        return f().upper()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*None'})

    def test_method_return_value(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      class Foo:\n        def f(self):\n          pass\n      def g():\n        return Foo().f().upper()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*None'})

    def test_pyi_return_value(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', 'def f() -> None: ...')
            errors = self.CheckWithErrors('\n        import foo\n        def g():\n          return foo.f().upper()  # attribute-error[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'upper.*None'})

    def test_pass_through_none(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      def f(x):\n        return x\n      def g():\n        return f(None).upper()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*None'})

    def test_shadowed_local_origin(self):
        if False:
            print('Hello World!')
        self.options.tweak(strict_none_binding=False)
        self.Check('\n      x = None\n      def f():\n        y = None\n        y = "hello"\n        return x if __random__ else y\n      def g():\n        return f().upper()\n    ')

    @test_base.skip("has_strict_none_origins can't tell if an origin is blocked.")
    def test_blocked_local_origin(self):
        if False:
            print('Hello World!')
        self.Check('\n      x = None\n      def f():\n        v = __random__\n        if v:\n          y = None\n        return x if v else y\n      def g():\n        return f().upper()\n    ')

    def test_return_constant(self):
        if False:
            print('Hello World!')
        self.options.tweak(strict_none_binding=False)
        self.Check('\n      x = None\n      def f():\n        return x\n      def g():\n        return f().upper()\n    ')

    def test_unpacked_none(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      _, a = 42, None\n      b = a.upper()  # attribute-error[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*None'})

    def test_function_default(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      class Foo:\n        def __init__(self, v=None):\n          v.upper()  # attribute-error[e]\n      def f():\n        Foo()\n    ')
        self.assertErrorRegexes(errors, {'e': 'upper.*None.*traceback.*line 5'})

    def test_keep_none_return(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> None: ...\n    ')

    def test_keep_none_yield(self):
        if False:
            return 10
        ty = self.Infer('\n      def f():\n        yield None\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Generator, Any\n      def f() -> Generator[None, Any, None]: ...\n    ')

    def test_keep_contained_none_return(self):
        if False:
            return 10
        ty = self.Infer('\n      def f():\n        return [None]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      def f() -> List[None]: ...\n    ')

    def test_discard_none_return(self):
        if False:
            while True:
                i = 10
        self.options.tweak(strict_none_binding=False)
        ty = self.Infer('\n      x = None\n      def f():\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      x = ...  # type: None\n      def f() -> Any: ...\n    ')

    def test_discard_none_yield(self):
        if False:
            for i in range(10):
                print('nop')
        self.options.tweak(strict_none_binding=False)
        ty = self.Infer('\n      x = None\n      def f():\n        yield x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Generator\n      x = ...  # type: None\n      def f() -> Generator[Any, Any, None]: ...\n    ')

    def test_discard_contained_none_return(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      x = None\n      def f():\n        return [x]\n    ')
        self.assertTypesMatchPytd(ty, '\n      x: None\n      def f() -> list[None]: ...\n    ')

    def test_discard_attribute_none_return(self):
        if False:
            print('Hello World!')
        self.options.tweak(strict_none_binding=False)
        ty = self.Infer('\n      class Foo:\n        x = None\n      def f():\n        return Foo.x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class Foo:\n        x = ...  # type: None\n      def f() -> Any: ...\n    ')

    def test_getitem(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      def f():\n        x = None\n        return x[0]  # unsupported-operands[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'item retrieval.*None.*int'})

    def test_ignore_getitem(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      x = None\n      def f():\n        return x[0]  # unsupported-operands\n    ')

    def test_ignore_iter(self):
        if False:
            return 10
        self.CheckWithErrors('\n      x = None\n      def f():\n        return [y for y in x]  # attribute-error\n    ')

    def test_contains(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      def f():\n        x = None\n        return 42 in x  # unsupported-operands[e]\n    ')
        self.assertErrorRegexes(errors, {'e': "'in'.*None.*int"})

    def test_ignore_contains(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      x = None\n      def f():\n        return 42 in x  # unsupported-operands\n    ')

    def test_property(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def __init__(self):\n          self._dofoo = __random__\n        @property\n        def foo(self):\n          return "hello" if self._dofoo else None\n      foo = Foo()\n      if foo.foo:\n        foo.foo.upper()\n    ')

    def test_isinstance(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Foo:\n        def f(self):\n          instance = None if __random__ else {}\n          if instance is not None:\n            self.g(instance)\n        def g(self, instance):\n          if isinstance(instance, str):\n            instance.upper()  # line 10\n    ')

    def test_impossible_return_type(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Dict\n      def f():\n        d = None  # type: Dict[str, str]\n        instance = d.get("hello")\n        return instance if instance else "world"\n      def g():\n        return f().upper()\n    ')

    def test_no_return(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def f():\n        text_value = "hello" if __random__ else None\n        if not text_value:\n          missing_value()\n        return text_value.strip()\n      def missing_value():\n        raise ValueError()\n    ')

class TestAttributes(test_base.BaseTest):
    """Tests for attributes."""

    def test_simple_attribute(self):
        if False:
            return 10
        ty = self.Infer('\n      class A:\n        def method1(self):\n          self.a = 3\n        def method2(self):\n          self.a = 3j\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      class A:\n        a = ...  # type: Union[complex, int]\n        def method1(self) -> NoneType: ...\n        def method2(self) -> NoneType: ...\n    ')

    def test_outside_attribute_access(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class A:\n        pass\n      def f1():\n        A().a = 3\n      def f2():\n        A().a = 3j\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      class A:\n        a = ...  # type: Union[complex, int]\n      def f1() -> NoneType: ...\n      def f2() -> NoneType: ...\n    ')

    def test_private(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class C:\n        def __init__(self):\n          self._x = 3\n        def foo(self):\n          return self._x\n    ')
        self.assertTypesMatchPytd(ty, '\n      class C:\n        _x = ...  # type: int\n        def __init__(self) -> None: ...\n        def foo(self) -> int: ...\n    ')

    def test_public(self):
        if False:
            return 10
        ty = self.Infer('\n      class C:\n        def __init__(self):\n          self.x = 3\n        def foo(self):\n          return self.x\n    ')
        self.assertTypesMatchPytd(ty, '\n      class C:\n        x = ...  # type: int\n        def __init__(self) -> None: ...\n        def foo(self) -> int: ...\n    ')

    def test_crosswise(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class A:\n        def __init__(self):\n          if id(self):\n            self.b = B()\n        def set_on_b(self):\n          self.b.x = 3\n      class B:\n        def __init__(self):\n          if id(self):\n            self.a = A()\n        def set_on_a(self):\n          self.a.x = 3j\n    ')
        self.assertTypesMatchPytd(ty, '\n      class A:\n        b = ...  # type: B\n        x = ...  # type: complex\n        def __init__(self) -> None: ...\n        def set_on_b(self) -> NoneType: ...\n      class B:\n        a = ...  # type: A\n        x = ...  # type: int\n        def __init__(self) -> None: ...\n        def set_on_a(self) -> NoneType: ...\n    ')

    def test_attr_with_bad_getattr(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      class AttrA:\n        def __getattr__(self, name2):\n          pass\n      class AttrB:\n        def __getattr__(self):\n          pass\n      class AttrC:\n        def __getattr__(self, x, y):\n          pass\n      class Foo:\n        A = AttrA\n        B = AttrB\n        C = AttrC\n        def foo(self):\n          self.A\n          self.B\n          self.C\n    ')

    def test_inherit_getattribute(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class MyClass1:\n        def __getattribute__(self, name):\n          return super(MyClass1, self).__getattribute__(name)\n\n      class MyClass2:\n        def __getattribute__(self, name):\n          return object.__getattribute__(self, name)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class MyClass1:\n        def __getattribute__(self, name) -> Any: ...\n      class MyClass2:\n        def __getattribute__(self, name) -> Any: ...\n    ')

    def test_getattribute(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class A:\n        def __getattribute__(self, name):\n          return 42\n      a = A()\n      a.x = "hello world"\n      x = a.x\n    ')
        self.assertTypesMatchPytd(ty, '\n      class A:\n        x = ...  # type: str\n        def __getattribute__(self, name) -> int: ...\n      a = ...  # type: A\n      x = ...  # type: int\n    ')

    def test_getattribute_branch(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class A:\n        x = "hello world"\n      class B:\n        def __getattribute__(self, name):\n          return False\n      def f(x):\n        v = A()\n        if x:\n          v.__class__ = B\n        return v.x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      class A:\n        x = ...  # type: str\n      class B:\n        def __getattribute__(self, name) -> bool: ...\n      def f(x) -> Any: ...\n    ')

    def test_set_class(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(x):\n        y = None\n        y.__class__ = x.__class__\n        return set([x, y])\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f(x) -> set: ...\n    ')

    def test_get_mro(self):
        if False:
            return 10
        ty = self.Infer('\n      x = int.mro()\n    ')
        self.assertTypesMatchPytd(ty, '\n      x = ...  # type: list\n    ')

    def test_call(self):
        if False:
            return 10
        ty = self.Infer('\n      class A:\n        def __call__(self):\n          return 42\n      x = A()()\n    ')
        self.assertTypesMatchPytd(ty, '\n      class A:\n        def __call__(self) -> int: ...\n      x = ...  # type: int\n    ')

    @test_base.skip("Magic methods aren't computed")
    def test_call_computed(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class A:\n        def __getattribute__(self, name):\n          return int\n      x = A().__call__()\n    ')
        self.assertTypesMatchPytd(ty, '\n      class A:\n        def __getattribute__(self, name) -> int: ...\n      x = ...  # type: int\n    ')

    def test_has_dynamic_attributes(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class Foo1:\n        has_dynamic_attributes = True\n      class Foo2:\n        HAS_DYNAMIC_ATTRIBUTES = True\n      class Foo3:\n        _HAS_DYNAMIC_ATTRIBUTES = True\n      Foo1().baz\n      Foo2().baz\n      Foo3().baz\n    ')

    def test_has_dynamic_attributes_subclass(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        _HAS_DYNAMIC_ATTRIBUTES = True\n      class Bar(Foo):\n        pass\n      Foo().baz\n      Bar().baz\n    ')

    def test_has_dynamic_attributes_class_attr(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        _HAS_DYNAMIC_ATTRIBUTES = True\n      Foo.CONST\n    ')

    def test_has_dynamic_attributes_metaclass(self):
        if False:
            return 10
        self.Check('\n      import six\n      class Metaclass(type):\n        _HAS_DYNAMIC_ATTRIBUTES = True\n      class Foo(six.with_metaclass(Metaclass, object)):\n        pass\n      @six.add_metaclass(Metaclass)\n      class Bar:\n        pass\n      Foo.CONST\n      Foo().baz\n      Bar.CONST\n      Bar().baz\n    ')

    def test_has_dynamic_attributes_pyi(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        class Foo:\n          has_dynamic_attributes = True\n      ')
            self.Check('\n        import mod\n        mod.Foo().baz\n      ', pythonpath=[d.path])

    def test_has_dynamic_attributes_metaclass_pyi(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('mod.pyi', '\n        class Metaclass(type):\n          _HAS_DYNAMIC_ATTRIBUTES: bool\n        class Foo(metaclass=Metaclass): ...\n      ')
            self.Check('\n        import mod\n        class Bar(mod.Foo):\n          pass\n        mod.Foo.CONST\n        mod.Foo().baz\n        Bar.CONST\n        Bar().baz\n      ', pythonpath=[d.path])

    def test_attr_on_static_method(self):
        if False:
            print('Hello World!')
        self.Check('\n      import collections\n\n      X = collections.namedtuple("X", "a b")\n      X.__new__.__defaults__ = (1, 2)\n      ')

    def test_module_type_attribute(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import types\n      v = None  # type: types.ModuleType\n      v.some_attribute\n    ')

    def test_attr_on_none(self):
        if False:
            for i in range(10):
                print('nop')
        self.InferWithErrors('\n      def f(arg):\n        x = "foo" if arg else None\n        if not x:\n          x.upper()  # attribute-error\n    ')

    def test_iterator_on_none(self):
        if False:
            i = 10
            return i + 15
        self.InferWithErrors('\n      def f():\n        pass\n      a, b = f()  # attribute-error\n    ')

    def test_overloaded_builtin(self):
        if False:
            print('Hello World!')
        self.Check('\n      if __random__:\n        getattr = None\n      else:\n        getattr(__any_object__, __any_object__)\n    ')

    def test_callable_return(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Callable\n      class Foo:\n        def __init__(self):\n          self.x = 42\n      v = None  # type: Callable[[], Foo]\n      w = v().x\n    ')

    def test_property_on_union(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class A:\n        def __init__(self):\n          self.foo = 1\n      class B:\n        def __init__(self):\n          self.bar = 2\n        @property\n        def foo(self):\n          return self.bar\n      x = A() if __random__ else B()\n      a = x.foo\n    ', deep=False)
        self.assertTypesMatchPytd(ty, "\n      from typing import Annotated, Union\n      a = ...  # type: int\n      x = ...  # type: Union[A, B]\n      class A:\n        foo = ...  # type: int\n        def __init__(self) -> None: ...\n      class B:\n        bar = ...  # type: int\n        foo = ...  # type: Annotated[int, 'property']\n        def __init__(self) -> None: ...\n    ")

    def test_reuse_annotated(self):
        if False:
            for i in range(10):
                print('nop')
        foo = self.Infer("\n      class Annotated:\n        @property\n        def name(self):\n          return ''\n    ")
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.Check('import foo', pythonpath=[d.path])

    @test_base.skip('Needs vm._get_iter() to iterate over individual bindings.')
    def test_bad_iter(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      v = [] if __random__ else 42\n      for _ in v:  # attribute-error[e]\n        pass\n    ')
        self.assertErrorRegexes(errors, {'e': '__iter__.*int'})

    def test_bad_getitem(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      class Foo:\n        def __getitem__(self, x):\n          return 0\n      v = Foo() if __random__ else 42\n      for _ in v:  # attribute-error[e]\n        pass\n    ')
        self.assertErrorRegexes(errors, {'e': '__iter__.*int.*Union\\[Foo, int\\]'})

    def test_bad_contains(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      class Foo:\n        def __iter__(self):\n          return iter([])\n      v = Foo() if __random__ else 42\n      if 42 in v:  # unsupported-operands[e]\n        pass\n    ')
        self.assertErrorRegexes(errors, {'e': "'in'.*'.*Union\\[Foo, int\\]' and 'int'"})

    def test_subclass_shadowing(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class X:\n          b = ...  # type: int\n        ')
            self.Check("\n        import foo\n        a = foo.X()\n        a.b  # The attribute exists\n        if __random__:\n          a.b = 1  # A new value is assigned\n        else:\n          a.b  # The original attribute isn't overwritten by the assignment\n        ", pythonpath=[d.path])

    def test_generic_property(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Generic, Optional, TypeVar\n        T = TypeVar("T")\n        class Foo(Generic[T]):\n          @property\n          def x(self) -> Optional[T]: ...\n        def f() -> Foo[str]: ...\n      ')
            ty = self.Infer('\n        import foo\n        def f():\n          return foo.f().x\n      ', pythonpath=[d.path])
        self.assertTypesMatchPytd(ty, '\n      import foo\n      from typing import Optional\n      def f() -> Optional[str]: ...\n    ')

    def test_bad_instance_assignment(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors("\n      class Foo:\n        x = None  # type: int\n        def foo(self):\n          self.x = 'hello, world'  # annotation-type-mismatch[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Annotation: int.*Assignment: str'})

    def test_bad_cls_assignment(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors("\n      class Foo:\n        x = None  # type: int\n      Foo.x = 'hello, world'  # annotation-type-mismatch[e]\n    ")
        self.assertErrorRegexes(errors, {'e': 'Annotation: int.*Assignment: str'})

    def test_any_annotation(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Any\n      class Foo:\n        x = None  # type: Any\n        def foo(self):\n          print(self.x.some_attr)\n          self.x = 0\n          print(self.x.some_attr)\n    ')

    def test_preserve_annotation_in_pyi(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Foo:\n        x = None  # type: float\n        def __init__(self):\n          self.x = 0\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        x: float\n        def __init__(self) -> None: ...\n    ')

    def test_annotation_in_init(self):
        if False:
            print('Hello World!')
        (ty, errors) = self.InferWithErrors("\n      class Foo:\n        def __init__(self):\n          self.x = 0  # type: int\n        def oops(self):\n          self.x = ''  # annotation-type-mismatch[e]\n    ")
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        x: int\n        def __init__(self) -> None: ...\n        def oops(self) -> None: ...\n    ')
        self.assertErrorRegexes(errors, {'e': 'Annotation: int.*Assignment: str'})

    def test_split(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import Union\n      class Foo:\n        pass\n      class Bar:\n        pass\n      def f(x):\n        # type: (Union[Foo, Bar]) -> None\n        if isinstance(x, Foo):\n          x.foo = 42\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      class Foo:\n        foo: int\n      class Bar: ...\n      def f(x: Union[Foo, Bar]) -> None: ...\n    ')

    def test_separate_instances(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        from typing import Any\n        _T = TypeVar('_T')\n\n        class Foo:\n          return_value: Any\n\n        def patch() -> Foo: ...\n      ")
            self.Check('\n        import foo\n\n        x = foo.patch()\n        y = foo.patch()\n\n        x.return_value = 0\n        y.return_value.rumpelstiltskin = 1\n      ', pythonpath=[d.path])

    def test_typevar(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', "\n        from typing import Generic, Type, TypeVar\n        T = TypeVar('T')\n        class Foo(Generic[T]):\n          x: Type[T]\n      ")
            self.Check('\n        import foo\n        from typing import Any, Type\n        class Bar:\n          x = None  # type: Type[Any]\n          def __init__(self, foo):\n            self.x = foo.x\n        def f():\n          return Bar(foo.Foo())\n      ', pythonpath=[d.path])
if __name__ == '__main__':
    test_base.main()