"""Tests for typing.TypedDict."""
from pytype.tests import test_base
from pytype.tests import test_utils

class TypedDictTest(test_base.BaseTest):
    """Tests for typing.TypedDict."""

    def test_init(self):
        if False:
            print('Hello World!')
        err = self.CheckWithErrors("\n      from typing_extensions import TypedDict\n      class A(TypedDict):\n        x: int\n        y: str\n      a = A(x=1, y='2')\n      b = A(x=1, y=2)  # wrong-arg-types[e1]\n      c = A(x=1)  # missing-parameter[e2]\n      d = A(y='1')  # missing-parameter\n      e = A(1, '2')  # missing-parameter\n    ")
        self.assertErrorSequences(err, {'e1': ['Expected', '(*, x, y: str)', 'Actual', '(x, y: int)'], 'e2': ['Expected', '(*, x, y)', 'Actual', '(x)']})

    def test_key_error(self):
        if False:
            i = 10
            return i + 15
        self.options.tweak(strict_parameter_checks=False)
        err = self.CheckWithErrors('\n      from typing_extensions import TypedDict\n      class A(TypedDict):\n        x: int\n        y: str\n      a = A(x=1, y="2")\n      a["z"] = 10  # typed-dict-error[e1]\n      a[10] = 10  # typed-dict-error[e2]\n      b = a["z"]  # typed-dict-error\n      del a["z"]  # typed-dict-error\n    ')
        self.assertErrorSequences(err, {'e1': ['TypedDict A', 'key z'], 'e2': ['TypedDict A', 'requires all keys', 'strings']})

    def test_value_error(self):
        if False:
            while True:
                i = 10
        err = self.CheckWithErrors('\n      from typing_extensions import TypedDict\n      class A(TypedDict):\n        x: int\n        y: str\n      a = A(x=1, y="2")\n      a["x"] = "10"  # annotation-type-mismatch[e]\n    ')
        self.assertErrorSequences(err, {'e': ['Type annotation', 'key x', 'TypedDict A', 'Annotation: int', 'Assignment: str']})

    def test_union_type(self):
        if False:
            i = 10
            return i + 15
        err = self.CheckWithErrors('\n      from typing_extensions import TypedDict\n      from typing import Union\n      class A(TypedDict):\n        x: Union[int, str]\n        y: Union[int, str]\n      a = A(x=1, y="2")\n      a["x"] = "10"\n      a["y"] = []  # annotation-type-mismatch[e]\n    ')
        self.assertErrorSequences(err, {'e': ['Type annotation', 'key y', 'TypedDict A', 'Annotation: Union[int, str]', 'Assignment: List[nothing]']})

    def test_bad_base_class(self):
        if False:
            print('Hello World!')
        err = self.CheckWithErrors('\n      from typing_extensions import TypedDict\n      class Foo: pass\n      class Bar(TypedDict, Foo):  # base-class-error[e]\n        x: int\n    ')
        self.assertErrorSequences(err, {'e': ['Invalid base class', 'Foo', 'TypedDict Bar', 'cannot inherit']})

    def test_inheritance(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n      from typing_extensions import TypedDict\n      class Foo(TypedDict):\n        x: int\n      class Bar(TypedDict):\n        y: str\n      class Baz(Foo, Bar):\n        z: bool\n      a = Baz(x=1, y='2', z=False)\n      a['x'] = 1\n      a['y'] = 2  # annotation-type-mismatch\n      a['z'] = True\n      a['w'] = True  # typed-dict-error\n    ")

    def test_inheritance_clash(self):
        if False:
            print('Hello World!')
        err = self.CheckWithErrors('\n      from typing_extensions import TypedDict\n      class Foo(TypedDict):\n        x: int\n      class Bar(TypedDict):\n        y: str\n      class Baz(Foo, Bar):  # base-class-error[e]\n        x: bool\n    ')
        self.assertErrorSequences(err, {'e': ['Duplicate', 'key x', 'Foo', 'Baz']})

    def test_annotation(self):
        if False:
            print('Hello World!')
        err = self.CheckWithErrors("\n      from typing_extensions import TypedDict\n      class A(TypedDict):\n        x: int\n        y: str\n      a: A = {'x': '10', 'z': 20}  # annotation-type-mismatch[e]\n    ")
        self.assertErrorSequences(err, {'e': ['Annotation: A(TypedDict)', 'extra keys', 'z', 'type errors', "{'x': ...}", 'expected int', 'got str']})

    def test_annotated_global_var(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      from typing_extensions import TypedDict\n      class A(TypedDict):\n        x: int\n      a: A = {'x': 10}\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import TypedDict\n\n      class A(TypedDict):\n        x: int\n\n      a: A\n    ')

    def test_annotated_local_var(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      from typing_extensions import TypedDict\n      class A(TypedDict):\n        x: int\n      def f():\n        a: A = {'x': 10}\n        return a\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import TypedDict\n\n      class A(TypedDict):\n        x: int\n\n      def f() -> A: ...\n    ')

    def test_return_type(self):
        if False:
            while True:
                i = 10
        err = self.CheckWithErrors("\n      from typing_extensions import TypedDict\n      class A(TypedDict):\n        x: int\n        y: str\n      def f() -> A:\n        return {'x': '10', 'z': 20}  # bad-return-type[e]\n    ")
        self.assertErrorSequences(err, {'e': ['Expected: A(TypedDict)', 'extra keys', 'z', 'type errors', "{'x': ...}", 'expected int', 'got str']})

    def test_total_with_constructor(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n      from typing_extensions import TypedDict\n      class Foo(TypedDict, total=True):\n        w: int\n        x: int\n      class Bar(TypedDict, total=False):\n        y: str\n        z: bool\n      class Baz(Foo, Bar):\n        a: int\n      a = Baz(w=1, x=1, y='2', z=False, a=2)\n      b = Baz(w=1, x=1, a=2)\n      c = Baz(w=1, x=1, y='2')  # missing-parameter\n      d = Baz(w=1, x=1, a=2, b=3)  # wrong-keyword-args\n    ")

    def test_total_with_annotation(self):
        if False:
            while True:
                i = 10
        err = self.CheckWithErrors("\n      from typing_extensions import TypedDict\n      class Foo(TypedDict, total=True):\n        w: int\n        x: int\n      class Bar(TypedDict, total=False):\n        y: str\n        z: bool\n      class Baz(Foo, Bar):\n        a: int\n      a: Baz = {'w': 1, 'x': 1, 'y': '2', 'z': False, 'a': 2}\n      b: Baz = {'w': 1, 'x': 1, 'a': 2}\n      c: Baz = {'w': 1, 'y': '2', 'z': False, 'a': 2}  # annotation-type-mismatch[e1]\n      d: Baz = {'w': 1, 'x': 1, 'y': '2', 'b': False, 'a': 2}  # annotation-type-mismatch[e2]\n    ")
        self.assertErrorSequences(err, {'e1': ['missing keys', 'x'], 'e2': ['extra keys', 'b']})

    def test_function_arg_matching(self):
        if False:
            return 10
        err = self.CheckWithErrors("\n      from typing_extensions import TypedDict\n      class A(TypedDict):\n        x: int\n        y: str\n      def f(a: A):\n        pass\n      a: A = {'x': 10, 'y': 'a'}\n      b = {'x': 10, 'y': 'a'}\n      c = {'x': 10}\n      f(a)\n      f(b)\n      f(c)  # wrong-arg-types[e]\n    ")
        self.assertErrorSequences(err, {'e': ['TypedDict', 'missing keys', 'y']})

    def test_function_arg_instantiation(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n      from typing_extensions import TypedDict\n      class A(TypedDict):\n        x: int\n        y: str\n      def f(a: A):\n        a['z'] = 10  # typed-dict-error\n    ")

    def test_function_arg_getitem(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors("\n      from typing import Union\n      from typing_extensions import TypedDict\n      class A(TypedDict):\n        x: int\n        y: Union[int, str]\n      def f(a: A) -> int:\n        assert_type(a['x'], int)\n        assert_type(a['y'], Union[int, str])\n        return a['z']  # typed-dict-error\n    ")

    def test_output_type(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing_extensions import TypedDict\n      class Foo(TypedDict):\n        x: int\n        y: str\n\n      def f(x: Foo) -> None:\n        pass\n\n      foo = Foo(x=1, y="2")\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypedDict\n\n      foo: Foo\n\n      class Foo(TypedDict):\n        x: int\n        y: str\n\n      def f(x: Foo) -> None: ...\n    ')

    def test_instantiate(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing_extensions import TypedDict\n      class Foo(TypedDict):\n        x: int\n      def f(x: Foo):\n        pass\n      x: Foo\n      f(x)\n    ')

    def test_key_existence_check(self):
        if False:
            return 10
        self.Check("\n      from typing import Union\n      from typing_extensions import TypedDict\n\n      class Foo(TypedDict):\n        a: int\n      class Bar(TypedDict):\n        b: str\n      class Baz(TypedDict):\n        c: Union[Foo, Bar]\n\n      baz: Baz = {'c': {'a': 0}}\n      assert 'a' in baz['c']\n      print(baz['c']['a'])\n    ")

    def test_get(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing_extensions import TypedDict\n      class X(TypedDict):\n        a: int\n        b: str\n      def f(x: X):\n        assert_type(x.get('a'), int)\n        assert_type(x.get('c'), None)\n        assert_type(x.get('c', ''), str)\n    ")

    def test_generic_holder(self):
        if False:
            return 10
        self.Check("\n      from dataclasses import dataclass\n      from typing import Generic, TypeVar\n      from typing_extensions import TypedDict\n\n      T = TypeVar('T')\n\n      class Animal(TypedDict):\n        name: str\n\n      @dataclass\n      class GenericHolder(Generic[T]):\n        a: T\n        def get(self) -> T:\n          return self.a\n\n      class AnimalHolder(GenericHolder[Animal]):\n        def get2(self) -> Animal:\n          return self.get()\n    ")

    def test_match_mapping(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      from typing import Mapping\n      from typing_extensions import TypedDict\n      class A(TypedDict):\n        x: int\n      def f1(a: Mapping[str, int]):\n        pass\n      def f2(a: Mapping[int, str]):\n        pass\n      f1(A(x=0))  # ok\n      f2(A(x=0))  # wrong-arg-types\n    ')

    def test_typed_dict_dataclass(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import dataclasses\n      from typing_extensions import TypedDict\n      @dataclasses.dataclass\n      class A(TypedDict):\n        x: int\n      def f():\n        return A(x=0)\n    ')

    def test_iterable_generic_class_and_recursive_type_interaction(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', "\n      from typing import Any, Generic, Iterable, TypeVar, Union\n      _ShapeType = TypeVar('_ShapeType')\n      _DType = TypeVar('_DType')\n      class ndarray(Generic[_ShapeType, _DType]):\n        def __iter__(self) -> Any: ...\n      ArrayTree = Union[Iterable[ArrayTree], ndarray]\n    ")]):
            self.Check('\n        import foo\n        from typing_extensions import TypedDict\n        class TD(TypedDict):\n          x: foo.ArrayTree\n        def f() -> TD:\n          return __any_object__\n      ')

class TypedDictFunctionalTest(test_base.BaseTest):
    """Tests for typing.TypedDict functional constructor."""

    def test_constructor(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      from typing_extensions import TypedDict\n      A = TypedDict("A", {"x": int, "y": str})\n      B = TypedDict("B", "b")  # wrong-arg-types\n      C = TypedDict("C")  # wrong-arg-count\n    ')

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        err = self.CheckWithErrors('\n      from typing_extensions import TypedDict\n      A = TypedDict("A", {"x": int, "y": str})\n      a = A(x=1, y=\'2\')\n      b = A(x=1, y=2)  # wrong-arg-types[e1]\n      c = A(x=1)  # missing-parameter[e2]\n      d = A(y=\'1\')  # missing-parameter\n      e = A(1, \'2\')  # missing-parameter\n    ')
        self.assertErrorSequences(err, {'e1': ['Expected', '(*, x, y: str)', 'Actual', '(x, y: int)'], 'e2': ['Expected', '(*, x, y)', 'Actual', '(x)']})

    def test_annotation(self):
        if False:
            i = 10
            return i + 15
        err = self.CheckWithErrors('\n      from typing_extensions import TypedDict\n      A = TypedDict("A", {"x": int, "y": str})\n      a: A = {\'x\': \'10\', \'z\': 20}  # annotation-type-mismatch[e]\n    ')
        self.assertErrorSequences(err, {'e': ['Annotation: A(TypedDict)', 'extra keys', 'z', 'type errors', "{'x': ...}", 'expected int', 'got str']})

    def test_keyword_field_name(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.py', '\n      from typing_extensions import TypedDict\n      A = TypedDict("A", {"in": int})\n    ')]):
            self.Check('\n        import foo\n        a: foo.A\n        assert_type(a["in"], int)\n      ')

    def test_colon_field_name(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.py', '\n      from typing_extensions import TypedDict\n      XMLDict = TypedDict("XMLDict", {"xml:name": str})\n    ')]):
            self.Check('\n        import foo\n        d: foo.XMLDict\n        assert_type(d["xml:name"], str)\n      ')

    def test_total(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      from typing_extensions import TypedDict\n      X = TypedDict('X', {'name': str}, total=False)\n      X()\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import TypedDict\n      class X(TypedDict, total=False):\n        name: str\n    ')
_SINGLE = '\n  from typing import TypedDict\n  class A(TypedDict):\n    x: int\n    y: str\n'
_MULTIPLE = '\n  from typing import TypedDict\n  class A(TypedDict):\n    x: int\n    y: str\n\n  class B(A):\n    z: int\n'

class PyiTypedDictTest(test_base.BaseTest):
    """Tests for typing.TypedDict in pyi files."""

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', _SINGLE)]):
            self.CheckWithErrors("\n        from foo import A\n        a = A(x=1, y='2')\n        b = A(x=1, y=2)  # wrong-arg-types\n      ")

    def test_function_arg(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', _SINGLE)]):
            self.CheckWithErrors("\n        from foo import A\n        def f(d: A) -> str:\n          a = d['x']\n          assert_type(a, int)\n          b = d['z']  # typed-dict-error\n          return d['y']\n      ")

    def test_function_return_type(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', _SINGLE)]):
            self.Check("\n        from foo import A\n        def f() -> A:\n          return {'x': 1, 'y': '2'}\n      ")

    def test_inheritance(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', _SINGLE)]):
            self.CheckWithErrors("\n        from foo import A\n        class B(A):\n          z: int\n        def f() -> B:\n          return {'x': 1, 'y': '2', 'z': 3}\n        def g() -> B:\n          return {'x': 1, 'y': '2'}  # bad-return-type\n      ")

    def test_pyi_inheritance(self):
        if False:
            return 10
        with self.DepTree([('foo.pyi', _MULTIPLE)]):
            self.CheckWithErrors("\n        from foo import A, B\n        def f() -> B:\n          return {'x': 1, 'y': '2', 'z': 3}\n        def g() -> B:\n          return {'x': 1, 'y': '2'}  # bad-return-type\n      ")

    def test_multi_module_pyi_inheritance(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.pyi', _MULTIPLE), ('bar.pyi', '\n         from foo import B\n         class C(B):\n           w: int\n         ')]):
            self.CheckWithErrors("\n        from bar import C\n        def f() -> C:\n          return {'x': 1, 'y': '2', 'z': 3, 'w': 4}\n        a = C(x=1, y='2', z=3, w='4')  # wrong-arg-types\n      ")

    def test_typing_extensions_import(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', '\n         from typing_extensions import TypedDict\n         class A(TypedDict):\n           x: int\n           y: str\n         ')]):
            self.CheckWithErrors("\n        from foo import A\n        a = A(x=1, y='2')\n        b = A(x=1, y=2)  # wrong-arg-types\n      ")

    def test_full_name(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.pyi', _SINGLE)]):
            err = self.CheckWithErrors("\n        import foo\n        from typing_extensions import TypedDict\n        class A(TypedDict):\n          z: int\n        def f(x: A):\n          pass\n        def g() -> foo.A:\n          return {'x': 1, 'y': '2'}\n        a = g()\n        f(a)  # wrong-arg-types[e]\n      ")
            self.assertErrorSequences(err, {'e': ['Expected', 'x: A', 'Actual', 'x: foo.A']})

    def test_setitem(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', '\n      from typing import TypedDict\n      class Foo(TypedDict):\n        x: int\n    '), ('bar.pyi', '\n      import foo\n      def f() -> foo.Foo: ...\n    ')]):
            self.Check("\n        import bar\n        foo = bar.f()\n        foo['x'] = 42\n      ")

    def test_match(self):
        if False:
            i = 10
            return i + 15
        with self.DepTree([('foo.pyi', '\n      from typing import TypedDict\n      class Foo(TypedDict):\n        x: int\n      def f(x: Foo) -> None: ...\n      def g() -> Foo: ...\n    ')]):
            self.Check('\n        import foo\n        foo.f(foo.g())\n      ')

    def test_nested(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.py', '\n      from typing_extensions import TypedDict\n      class Foo:\n        class Bar(TypedDict):\n          x: str\n    ')]):
            self.CheckWithErrors("\n        import foo\n        foo.Foo.Bar(x='')  # ok\n        foo.Foo.Bar(x=0)  # wrong-arg-types\n      ")

    def test_imported_and_nested(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.py', '\n      from typing_extensions import TypedDict\n      class Foo(TypedDict):\n        x: str\n    ')]):
            ty = self.Infer('\n        import foo\n        class Bar:\n          Foo = foo.Foo\n      ')
        self.assertTypesMatchPytd(ty, '\n      import foo\n      class Bar:\n        Foo: type[foo.Foo]\n    ')

    def test_nested_alias(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing_extensions import TypedDict\n      class Foo(TypedDict):\n        x: str\n      class Bar:\n        Foo = Foo\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypedDict\n      class Foo(TypedDict):\n        x: str\n      class Bar:\n        Foo: type[Foo]\n    ')

    def test_total_false(self):
        if False:
            for i in range(10):
                print('nop')
        with self.DepTree([('foo.py', '\n      from typing_extensions import TypedDict\n      class Foo(TypedDict, total=False):\n        x: str\n        y: int\n    '), ('bar.pyi', '\n      from typing import TypedDict\n      class Bar(TypedDict, total=False):\n        x: str\n        y: int\n    ')]):
            self.Check("\n        import foo\n        import bar\n        foo.Foo(x='hello')\n        bar.Bar(x='world')\n      ")

    def test_total_inheritance(self):
        if False:
            while True:
                i = 10
        with self.DepTree([('foo.pyi', '\n      from typing import TypedDict\n      class Parent1(TypedDict, total=True):\n        x: str\n      class Child1(Parent1, total=False):\n        y: int\n      class Parent2(TypedDict, total=False):\n        x: str\n      class Child2(Parent2, total=True):\n        y: int\n    ')]):
            self.CheckWithErrors("\n        import foo\n        foo.Child1(x='')\n        foo.Child1(y=0)  # missing-parameter\n        foo.Child2(x='')  # missing-parameter\n        foo.Child2(y=0)\n      ")

class IsTypedDictTest(test_base.BaseTest):
    """Tests for typing.is_typeddict.

  These tests define variables based on the result of is_typeddict, allowing us
  to verify the result based on whether the corresponding variable appears in
  the pytd.
  """

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing_extensions import is_typeddict, TypedDict\n      class X(TypedDict):\n        x: str\n      class Y:\n        y: int\n      if is_typeddict(X):\n        X_is_typeddict = True\n      else:\n        X_is_not_typeddict = True\n      if is_typeddict(Y):\n        Y_is_typeddict = True\n      else:\n        Y_is_not_typeddict = True\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypedDict\n      class X(TypedDict):\n        x: str\n      class Y:\n        y: int\n      X_is_typeddict: bool\n      Y_is_not_typeddict: bool\n    ')

    def test_pyi(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.pyi', '\n      from typing import TypedDict\n      class X(TypedDict):\n        x: str\n      class Y:\n        y: int\n    ')]):
            ty = self.Infer('\n        import foo\n        from typing_extensions import is_typeddict\n        if is_typeddict(foo.X):\n          X_is_typeddict = True\n        else:\n          X_is_not_typeddict = True\n        if is_typeddict(foo.Y):\n          Y_is_typeddict = True\n        else:\n          Y_is_not_typeddict = True\n      ')
            self.assertTypesMatchPytd(ty, '\n        import foo\n        X_is_typeddict: bool\n        Y_is_not_typeddict: bool\n      ')

    @test_utils.skipBeforePy((3, 10), 'is_typeddict is new in Python 3.10.')
    def test_from_typing(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import is_typeddict, TypedDict\n      class X(TypedDict):\n        x: str\n      class Y:\n        y: int\n      if is_typeddict(X):\n        X_is_typeddict = True\n      else:\n        X_is_not_typeddict = True\n      if is_typeddict(Y):\n        Y_is_typeddict = True\n      else:\n        Y_is_not_typeddict = True\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypedDict\n      class X(TypedDict):\n        x: str\n      class Y:\n        y: int\n      X_is_typeddict: bool\n      Y_is_not_typeddict: bool\n    ')

    def test_union(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing import Union\n      from typing_extensions import is_typeddict, TypedDict\n      class X(TypedDict):\n        x: str\n      class Y(TypedDict):\n        y: int\n      class Z:\n        z: bytes\n      if is_typeddict(Union[X, Y]):\n        XY_is_typeddict = True\n      else:\n        XY_is_not_typeddict = True\n      if is_typeddict(Union[X, Z]):\n        XZ_is_typeddict = True\n      else:\n        XZ_is_not_typeddict = True\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypedDict\n      class X(TypedDict):\n        x: str\n      class Y(TypedDict):\n        y: int\n      class Z:\n        z: bytes\n      XY_is_typeddict: bool\n      XZ_is_not_typeddict: bool\n    ')

    def test_split(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing_extensions import is_typeddict, TypedDict\n      class X(TypedDict):\n        x: str\n      class Y:\n        y: int\n      cls = X if __random__ else Y\n      if is_typeddict(cls):\n        XY_may_be_typeddict = True\n      else:\n        XY_may_not_be_typeddict = True\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypedDict\n      class X(TypedDict):\n        x: str\n      class Y:\n        y: int\n      cls: type[X | Y]\n      XY_may_be_typeddict: bool\n      XY_may_not_be_typeddict: bool\n    ')

    def test_namedarg(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing_extensions import is_typeddict, TypedDict\n      class X(TypedDict):\n        x: str\n      class Y:\n        y: int\n      if is_typeddict(tp=X):\n        X_is_typeddict = True\n      else:\n        X_is_not_typeddict = True\n      if is_typeddict(tp=Y):\n        Y_is_typeddict = True\n      else:\n        Y_is_not_typeddict = True\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypedDict\n      class X(TypedDict):\n        x: str\n      class Y:\n        y: int\n      X_is_typeddict: bool\n      Y_is_not_typeddict: bool\n    ')

    def test_ambiguous(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing_extensions import is_typeddict\n      if is_typeddict(*__any_object__, **__any_object__):\n        ambiguous_may_be_typeddict = True\n      else:\n        ambiguous_may_not_be_typeddict = True\n    ')
        self.assertTypesMatchPytd(ty, '\n      ambiguous_may_be_typeddict: bool\n      ambiguous_may_not_be_typeddict: bool\n    ')

    def test_subclass(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      from typing_extensions import is_typeddict, TypedDict\n      class X(TypedDict):\n        x: str\n      class Y(X):\n        pass\n      if is_typeddict(Y):\n        Y_is_typeddict = True\n      else:\n        Y_is_not_typeddict = True\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import TypedDict\n      class X(TypedDict):\n        x: str\n      class Y(TypedDict):\n        x: str\n      Y_is_typeddict: bool\n    ')

    def test_bad_args(self):
        if False:
            return 10
        self.CheckWithErrors('\n      from typing_extensions import is_typeddict\n      is_typeddict()  # missing-parameter\n      is_typeddict(__any_object__, __any_object__)  # wrong-arg-count\n      is_typeddict(toilet_paper=True)  # wrong-keyword-args\n    ')
if __name__ == '__main__':
    test_base.main()