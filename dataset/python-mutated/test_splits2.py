"""Tests for if-splitting."""
from pytype.tests import test_base

class AmbiguousIsInstanceTest(test_base.BaseTest):
    """Tests that isinstance() checks work with Any.

  Concretely, the following should not produce any errors:

  X: Any
  def f(x: X | str):
    if isinstance(x, X):
      # This attribute access should not be an error, even though we don't know
      # that x isn't a str.
      return x.y

  This is needed to support gradual typing:
  - Typed and untyped code should be able to interoperate seamlessly. X may
    originate from a untyped or partially typed library.
  - Converting code from more to less typed should never introduce new type
    errors. In this example, no error would be reported if X were a class with a
    'y' attribute, so downgrading X to Any should not introduce errors.
  """

    def test_basic(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Any, Union\n      X: Any\n      def f(x: Union[X, str]):\n        if isinstance(x, X):\n          return x.y\n    ')

    def test_multiple_classes(self):
        if False:
            return 10
        self.CheckWithErrors('\n      from typing import Any, Union\n      X: Any\n      def f(x: Union[X, str]):\n        if isinstance(x, (X, str)):\n          print(x.y)  # attribute-error\n        if isinstance(x, (X, int)):\n          print(x.real)  # ok\n    ')

    def test_inversion(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      from typing import Any, Union\n      X: Any\n      def f(x: Union[X, str]):\n        if not isinstance(x, X):\n          print(x.y)  # attribute-error\n        else:\n          print(x.z)  # ok\n    ')

class SplitTest(test_base.BaseTest):
    """Tests for if-splitting."""

    def test_hasattr(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Foo():\n        def bar(self):\n          pass\n      class Baz(Foo):\n        def quux(self):\n          pass\n      def d1(x: Foo): return "y" if hasattr(x, "bar") else 0\n      def d2(x: Foo): return "y" if hasattr(x, "unknown") else 0\n      def d3(x: Baz): return "y" if hasattr(x, "quux") else 0\n      def d4(x: Baz): return "y" if hasattr(x, "bar") else 0\n      def a1(x): return "y" if hasattr(x, "bar") else 0\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      class Baz(Foo):\n        def quux(self) -> None: ...\n      class Foo:\n        def bar(self) -> None: ...\n      def d1(x: Foo) -> str: ...\n      def d2(x: Foo) -> int: ...\n      def d3(x: Baz) -> str: ...\n      def d4(x: Baz) -> str: ...\n      def a1(x) -> Union[int, str]: ...\n    ')

    def test_union(self):
        if False:
            return 10
        self.Check('\n      from typing import Union\n      def f(data: str):\n        pass\n      def as_my_string(data: Union[str, int]):\n        if isinstance(data, str):\n          f(data)\n    ')

    def test_union2(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Union\n      class MyString:\n        def __init__(self, arg: str):\n          self.arg = arg\n      def as_my_string(data: Union[str, MyString]) -> MyString:\n        if isinstance(data, str):\n          result = MyString(data)\n        else:\n          # data has type MyString\n          result = data\n        return result\n    ')

    def test_load_attr(self):
        if False:
            return 10
        self.Check('\n      class A:\n        def __init__(self):\n          self.initialized = False\n          self.data = None\n        def f1(self, x: int):\n          self.initialized = True\n          self.data = x\n        def f2(self) -> int:\n          if self.initialized:\n            return self.data\n          else:\n            return 0\n    ')

    def test_guarding_is(self):
        if False:
            return 10
        'Assert that conditions are remembered for is.'
        self.Check("\n      from typing import Optional\n      def f(x: Optional[str]) -> str:\n        if x is None:\n          x = ''\n        return x\n      ")

    def test_conditions_are_ordered(self):
        if False:
            for i in range(10):
                print('nop')
        'Assert that multiple conditions on a path work.'
        self.Check('\n      from typing import Optional\n      def f(x: Optional[NoneType]) -> int:\n        if x is not None:\n          x = None\n        if x is None:\n          x = 1  # type: int\n        return x\n      ')

    def test_guarding_is_not(self):
        if False:
            while True:
                i = 10
        'Assert that conditions are remembered for is not.'
        self.Check('\n      from typing import Optional\n      def f(x: Optional[str]) -> NoneType:\n        if x is not None:\n          x = None\n        return x\n      ')

    def test_guarding_is_not_else(self):
        if False:
            while True:
                i = 10
        'Assert that conditions are remembered for else if.'
        self.Check('\n      from typing import Optional\n      def f(x: Optional[str]) -> int:\n        if x is None:\n          x = 1  # type: int\n        else:\n          x = 1  # type: int\n        return x\n      ')

    def test_simple_or(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import Optional\n      def f(self, x: Optional[str] = None) -> str:\n        return x or "foo"\n    ')

    def test_or(self):
        if False:
            return 10
        self.Check('\n      from typing import Optional\n      def f(foo: Optional[int] = None) -> int:\n        if foo is None:\n          return 1\n        return foo\n      def g(foo: Optional[int] = None) -> int:\n        return foo or 1\n      def h(foo: Optional[int] = None) -> int:\n        foo = foo or 1\n        return foo\n      def j(foo: Optional[int] = None) -> int:\n        if foo is None:\n          foo = 1\n        return foo\n    ')

    def test_hidden_conflict(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      import typing\n      def f(obj: typing.Union[int, dict, list, float, str, complex]):\n        if isinstance(obj, int):\n          return\n        if isinstance(obj, dict):\n          obj.values\n    ')

    def test_isinstance_list(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import List\n      def f(x: List[float]):\n        if not isinstance(x, list):\n          return float(x)\n    ')

    def test_long_signature(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Optional\n\n      class Foo:\n\n        def __init__(\n            self, x1: Optional[str] = None, x2: Optional[str] = None,\n            x3: Optional[str] = None, x4: Optional[str] = None,\n            x5: Optional[str] = None, x6: Optional[str] = None,\n            x7: Optional[str] = None, x8: Optional[str] = None,\n            x9: Optional[str] = None, credentials: Optional[str] = None):\n          if not credentials:\n            credentials = ""\n          self.credentials = credentials.upper()\n    ')

    def test_create_list(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import List, Optional\n      def _CreateList(opt: Optional[str]) -> List[str]:\n        if opt is not None:\n          return [opt]\n        return ["foo"]\n    ')

    def test_create_tuple(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Optional, Tuple\n      def _CreateTuple(opt: Optional[str]) -> Tuple[str]:\n        if opt is not None:\n          return (opt,)\n        return ("foo",)\n    ')

    def test_closure(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Optional\n      def foo(arg: Optional[str]):\n        if arg is None:\n          raise TypeError()\n        def nested():\n          print(arg.upper())\n    ')

    def test_annotated_closure(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Optional\n      def foo(arg: Optional[str]):\n        if arg is None:\n          raise TypeError()\n        def nested() -> None:\n          print(arg.upper())\n    ')

    def test_iterable_truthiness(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      from typing import Iterable\n      def f(x: Iterable[int]):\n        return 0 if x else ''\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Iterable, Union\n      def f(x: Iterable[int]) -> Union[int, str]: ...\n    ')

    def test_custom_container_truthiness(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      from typing import Iterable, TypeVar\n      T = TypeVar('T')\n      class MyIterable(Iterable[T]):\n        pass\n      def f(x: MyIterable[int]):\n        return 0 if x else ''\n    ")
        self.assertTypesMatchPytd(ty, "\n      from typing import Iterable, TypeVar, Union\n      T = TypeVar('T')\n      class MyIterable(Iterable[T]): ...\n      def f(x: MyIterable[int]) -> Union[int, str]: ...\n    ")

    def test_str_none_eq(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Optional\n      def f(x: str, y: Optional[str]) -> str:\n        if x == y:\n          return y\n        return x\n    ')

    def test_annotation_class(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      from typing import Sequence, Union\n      def f(x: Union[int, Sequence[int]]):\n        if isinstance(x, Sequence):\n          return x[0]\n        else:\n          return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Sequence, Union\n      def f(x: Union[int, Sequence[int]]) -> int: ...\n    ')

    def test_isinstance_tuple(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from typing import AbstractSet, Sequence, Union\n\n      def yo(collection: Union[AbstractSet[int], Sequence[int]]):\n        if isinstance(collection, (Sequence,)):\n          return collection.index(5)\n        else:\n          return len(collection)\n    ')

    def test_ordered_dict_value(self):
        if False:
            return 10
        self.Check("\n      import collections\n      from typing import Iterator, Optional, Tuple\n\n      _DEFINITIONS = {'a': 'b'}\n\n      class Foo:\n        def __init__(self):\n          self.d = collections.OrderedDict()\n        def add(self, k: str, v: Optional[str]):\n          if v is None:\n            v = _DEFINITIONS[k]\n          self.d[k] = v\n        def generate(self) -> Iterator[Tuple[str, str]]:\n          for k, v in self.d.items():\n            yield k, v\n    ")

    @test_base.skip('b/256934562')
    def test_set_add_in_if(self):
        if False:
            return 10
        self.Check("\n      from typing import Optional, Set\n      class Foo:\n        def add(self, s: Optional[Set['Foo']], skip: bool) -> Set['Foo']:\n          if s is None:\n            s = set()\n          if __random__:\n            if not skip:\n              s.add(self)\n            return s\n          return s\n    ")

class SplitTestPy3(test_base.BaseTest):
    """Tests for if-splitting in Python 3."""

    def test_isinstance_multiple(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import Union\n      def UpperIfString(value: Union[bytes, str, int]):\n        if isinstance(value, (bytes, str)):\n          return value.upper()\n      def ReturnIfNumeric(value: Union[str, int]):\n        if isinstance(value, (int, (float, complex))):\n          return value\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Optional, Union\n      def UpperIfString(value: Union[bytes, int, str]) -> Optional[Union[bytes, str]]: ...\n      def ReturnIfNumeric(value: Union[str, int]) -> Optional[int]: ...\n    ')

    def test_isinstance_aliased(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import Union\n      myisinstance = isinstance\n      def UpperIfString(value: Union[bytes, str, int]):\n        if myisinstance(value, (bytes, str)):\n          return value.upper()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Callable, Optional, Tuple, Union\n      def myisinstance(object, class_or_type_or_tuple: Union[Tuple[Union[Tuple[type, ...], type], ...], type]) -> bool: ...\n      def UpperIfString(value: Union[bytes, int, str]) -> Optional[Union[bytes, str]]: ...\n    ')

    def test_shadow_none(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import Optional, Union\n      def f(x: Optional[Union[str, bytes]]):\n        if x is None:\n          x = ''\n        return x.upper()\n    ")

    def test_override_bool(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class A:\n        def __bool__(self):\n          return __random__\n\n      x = A() and True\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      class A:\n        def __bool__(self) -> bool: ...\n      x: Union[A, bool]\n    ')

    def test_ordered_dict_list_value(self):
        if False:
            return 10
        self.Check("\n      import collections\n      from typing import List, Optional\n\n      class A:\n        def __init__(self):\n          self._arg_dict = collections.OrderedDict()\n        def Set(self, arg_name: str, arg_value: Optional[str] = None):\n          if arg_value is not None:\n            self._arg_dict[arg_name] = [arg_value]\n        def Get(self) -> List[str]:\n          arg_list: List[str] = []\n          for key, value in self._arg_dict.items():\n            arg_list.append(key)\n            if __random__:\n              arg_list.append('')\n            else:\n              arg_list.extend(value)\n          return arg_list\n        def __str__(self) -> str:\n          return ''.join(self.Get())\n\n      class B:\n        def Build(self) -> List[str]:\n          a = A()\n          return a.Get()\n    ")

    def test_frametype(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import inspect\n      current = inspect.currentframe()\n      assert current is not None\n      caller = current.f_back\n      assert caller is not None\n      code = caller.f_code\n    ')
if __name__ == '__main__':
    test_base.main()