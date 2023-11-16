"""Tests for displaying errors."""
from pytype.tests import test_base
from pytype.tests import test_utils

class ErrorTest(test_base.BaseTest):
    """Tests for errors."""

    def test_union(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      def f(x: int):\n        pass\n      if __random__:\n        i = 0\n      else:\n        i = 1\n      x = (3.14, "")\n      f(x[i])  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Actually passed:.*Union\\[float, str\\]'})

    def test_invalid_annotations(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      from typing import Dict, List, Union\n      def f1(x: Dict):  # okay\n        pass\n      def f2(x: Dict[str]):  # invalid-annotation[e1]\n        pass\n      def f3(x: List[int, str]):  # invalid-annotation[e2]\n        pass\n      def f4(x: Union):  # invalid-annotation[e3]\n        pass\n    ')
        self.assertErrorSequences(errors, {'e1': ['dict[str]', 'dict[_K, _V]', '2', '1'], 'e2': ['list[int, str]', 'list[_T]', '1', '2'], 'e3': ['Union', 'x']})

    def test_print_unsolvable(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      from typing import List\n      def f(x: List[nonsense], y: str, z: float):  # name-error\n        pass\n      f({nonsense}, "", "")  # name-error  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Expected:.*x: list.*Actual.*x: set'})

    def test_print_union_of_containers(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      def f(x: str):\n        pass\n      if __random__:\n        x = dict\n      else:\n        x = [float]\n      f(x)  # wrong-arg-types[e]\n    ')
        error = ['Actual', 'Union[List[Type[float]], Type[dict]]']
        self.assertErrorSequences(errors, {'e': error})

    def test_wrong_brackets(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      from typing import List\n      def f(x: List(str)):  # invalid-annotation[e]\n        pass\n    ')
        self.assertErrorSequences(errors, {'e': ['<instance of list>']})

    def test_interpreter_class_printing(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      class Foo: pass\n      def f(x: str): pass\n      f(Foo())  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['str', 'Foo']})

    def test_print_dict_and_tuple(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      from typing import Tuple\n      tup = None  # type: Tuple[int, ...]\n      dct = None  # type: dict[str, int]\n      def f1(x: (int, str)):  # invalid-annotation[e1]\n        pass\n      def f2(x: tup):  # invalid-annotation[e2]\n        pass\n      def g1(x: {"a": 1}):  # invalid-annotation[e3]\n        pass\n      def g2(x: dct):  # invalid-annotation[e4]\n        pass\n    ')
        self.assertErrorSequences(errors, {'e1': ['(int, str)', 'Not a type'], 'e2': ['instance of Tuple[int, ...]', 'Not a type'], 'e3': ["{'a': 1}", 'Not a type'], 'e4': ['instance of Dict[str, int]', 'Not a type']})

    def test_move_union_inward(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      def f() -> str:  # bad-yield-annotation[e]\n        y = "hello" if __random__ else 42\n        yield y\n    ')
        self.assertErrorSequences(errors, {'e': ['Generator, Iterable or Iterator']})

    def test_inner_class_error(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      def f(x: str): pass\n      def g():\n        class Foo: pass\n        f(Foo())  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['x: str', 'x: Foo']})

    def test_inner_class_error2(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      def f():\n        class Foo: pass\n        def g(x: Foo): pass\n        g("")  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['x: Foo', 'x: str']})

    def test_clean_namedtuple_names(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      import collections\n      X = collections.namedtuple("X", "a b c d")\n      Y = collections.namedtuple("Z", "")\n      W = collections.namedtuple("W", "abc def ghi abc", rename=True)\n      def bar(x: str):\n        return x\n      bar(X(1,2,3,4))  # wrong-arg-types[e1]\n      bar(Y())         # wrong-arg-types[e2]\n      bar(W(1,2,3,4))  # wrong-arg-types[e3]\n      bar({1: 2}.__iter__())  # wrong-arg-types[e4]\n      if __random__:\n        a = X(1,2,3,4)\n      else:\n        a = 1\n      bar(a)  # wrong-arg-types[e5]\n      ')
        self.assertErrorSequences(errors, {'e1': ['x: X'], 'e2': ['x: Y'], 'e3': ['x: W'], 'e4': ['Iterator'], 'e5': ['Union[X, int]']})

    def test_argument_order(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      def g(f: str, a, b, c, d, e,):\n        pass\n      g(a=1, b=2, c=3, d=4, e=5, f=6)  # wrong-arg-types[e]\n      ')
        self.assertErrorSequences(errors, {'e': ['Expected', 'f: str, ...', 'Actual', 'f: int, ...']})

    def test_conversion_of_generic(self):
        if False:
            for i in range(10):
                print('nop')
        self.InferWithErrors('\n      import os\n      def f() -> None:\n        return os.walk("/tmp")  # bad-return-type\n    ')

    def test_inner_class(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      def f() -> int:\n        class Foo:\n          pass\n        return Foo()  # bad-return-type[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['int', 'Foo']})

    def test_nested_proto_class(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo_bar.pyi', '\n        from typing import Type\n        class _Foo_DOT_Bar: ...\n        class Foo:\n          Bar = ...  # type: Type[_Foo_DOT_Bar]\n      ')
            errors = self.CheckWithErrors('\n        import foo_bar\n        def f(x: foo_bar.Foo.Bar): ...\n        f(42)  # wrong-arg-types[e]\n      ', pythonpath=[d.path])
            self.assertErrorSequences(errors, {'e': ['foo_bar.Foo.Bar']})

    def test_staticmethod_in_error(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class A:\n          @staticmethod\n          def t(a: str) -> None: ...\n        ')
            errors = self.CheckWithErrors('\n        from typing import Callable\n        import foo\n        def f(x: Callable[[int], None], y: int) -> None:\n          return x(y)\n        f(foo.A.t, 1)  # wrong-arg-types[e]\n        ', pythonpath=[d.path])
            self.assertErrorSequences(errors, {'e': ['Actually passed: (x: Callable[[str], None]']})

    def test_generator_send(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      from typing import Generator, Any\n      def f(x) -> Generator[Any, int, Any]:\n        if x == 1:\n          yield 1\n        else:\n          yield "1"\n\n      x = f(2)\n      x.send("123")  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['(self, value: int)']})

    def test_generator_iterator_ret_type(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      from typing import Iterator\n      def f() -> Iterator[str]:\n        yield 1  # bad-return-type[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['str', 'int']})

    def test_generator_iterable_ret_type(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      from typing import Iterable\n      def f() -> Iterable[str]:\n        yield 1  # bad-return-type[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['str', 'int']})

    def test_bad_self_annot(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      class Foo:\n        def f(self: int):\n          pass  # wrong-arg-types\n    ')

    def test_union_with_any(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing import Any, Union\n      X = Union[Any, int]\n      Y = X[str]  # invalid-annotation[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['Union[Any, int][str]', 'Union[Any, int]', '0', '1']})

    def test_optional_union(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing import Union\n      X = Union[int, str, None]\n      Y = X[float]  # invalid-annotation[e]\n    ')
        self.assertErrorSequences(errors, {'e': 'Optional[Union[int, str]'})

    def test_nested_class(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors("\n      from typing import Protocol, TypeVar\n      T = TypeVar('T')\n      def f():\n        class X(Protocol[T]):\n          def __len__(self) -> T: ...\n        class Y:\n          def __len__(self) -> int:\n            return 0\n        def g(x: X[str]):\n          pass\n        g(Y())  # wrong-arg-types[e]\n    ")
        self.assertErrorSequences(errors, {'e': ['Method __len__ of protocol X[str] has the wrong signature in Y']})

    def test_variables_not_printed(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      class A:\n        pass\n      {\'a\': 1, \'b\': "hello"} + [1, 2]  # unsupported-operands[e1]\n      {\'a\': 1, 2: A()} + [A(), 2]  # unsupported-operands[e2]\n    ')
        self.assertErrorSequences(errors, {'e1': ["{'a': 1, 'b': 'hello'}: Dict[str, Union[int, str]]", '[1, 2]: List[int]'], 'e2': ['{...: ...}: Dict[Union[int, str], Union[A, int]', '[..., 2]: List[Union[A, int]]']})

    def test_wrong_self_type(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      from typing import MutableMapping\n      import unittest\n      def f(self: int) -> None:\n        foo: MutableMapping[str, str] = {}  # wrong-arg-types[e]\n      class C(unittest.TestCase):\n        f = f\n    ')
        (e,) = errors.errorlog
        self.assertEqual(e.filename, '<inline>')
        self.assertEqual(e.methodname, 'f')

    def test_starargs(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      def f(*args: int, **kwargs: str):\n        pass\n      f("")  # wrong-arg-types[e1]\n      f(x=0)  # wrong-arg-types[e2]\n    ')
        self.assertErrorSequences(errors, {'e1': ['Expected: (_0: int, ...)', 'Actual', '(_0: str)'], 'e2': ['Expected: (x: str, ...)', 'Actual', '(x: int)']})

    def test_starargs_pyi(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.pyi', '\n      def f(*args: int, **kwargs: str): ...\n    ')]):
            errors = self.CheckWithErrors('\n        import foo\n        foo.f("")  # wrong-arg-types[e1]\n        foo.f(x=0)  # wrong-arg-types[e2]\n      ')
            self.assertErrorSequences(errors, {'e1': ['Expected: (_0: int, ...)', 'Actual', '(_0: str)'], 'e2': ['Expected: (x: str, ...)', 'Actual', '(x: int)']})

class AssertTypeTest(test_base.BaseTest):
    """Tests for pseudo-builtin assert_type()."""

    def test_assert_type(self):
        if False:
            while True:
                i = 10
        (_, errors) = self.InferWithErrors('\n      from typing import Union\n      class A: pass\n      def f(x: int, y: str, z):\n        assert_type(x, int)\n        assert_type(y, int)  # assert-type[e]\n        if __random__:\n          x = A()\n        assert_type(x, Union[A, int])\n    ')
        self.assertErrorSequences(errors, {'e': ['Expected', 'int', 'Actual', 'str']})

    def test_assert_type_str(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors("\n      class A: pass\n      def f(x: int, y: str, z):\n        assert_type(x, 'int')\n        assert_type(y, 'int')  # assert-type[e]\n        if __random__:\n          x = A()\n        assert_type(x, 'Union[A, int]')\n    ")
        self.assertErrorSequences(errors, {'e': ['Expected', 'int', 'Actual', 'str']})

    def test_assert_type_import(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('pytype_extensions.pyi', '\n        def assert_type(*args): ...\n      ')
            (_, errors) = self.InferWithErrors('\n        from typing import Union\n        from pytype_extensions import assert_type\n        class A: pass\n        def f(x: int, y: str, z):\n          assert_type(x, int)\n          assert_type(y, int)  # assert-type[e]\n          if __random__:\n            x = A()\n          assert_type(x, Union[A, int])\n      ', pythonpath=[d.path])
            self.assertErrorSequences(errors, {'e': ['Expected', 'int', 'Actual', 'str']})

    def test_combine_containers(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Set, Union\n      x: Set[Union[int, str]]\n      y: Set[Union[str, bytes]]\n      assert_type(x | y, "Set[Union[bytes, int, str]]")\n    ')

class InPlaceOperationsTest(test_base.BaseTest):
    """Test in-place operations."""

    def _testOp(self, op, symbol):
        if False:
            return 10
        errors = self.CheckWithErrors(f'\n      class A:\n        def __{op}__(self, x: "A"):\n          return None\n      def f():\n        v = A()\n        v {symbol} 3  # unsupported-operands[e]\n    ')
        self.assertErrorSequences(errors, {'e': [symbol, 'A', 'int', f'__{op}__ on A', 'A']})

    def test_isub(self):
        if False:
            return 10
        self._testOp('isub', '-=')

    def test_imul(self):
        if False:
            return 10
        self._testOp('imul', '*=')

    def test_idiv(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      class A:\n        def __idiv__(self, x: "A"):\n          return None\n        def __itruediv__(self, x: "A"):\n          return None\n      def f():\n        v = A()\n        v /= 3  # unsupported-operands[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '\\/\\=.*A.*int.*__i(true)?div__ on A.*A'})

    def test_imod(self):
        if False:
            print('Hello World!')
        self._testOp('imod', '%=')

    def test_ipow(self):
        if False:
            return 10
        self._testOp('ipow', '**=')

    def test_ilshift(self):
        if False:
            print('Hello World!')
        self._testOp('ilshift', '<<=')

    def test_irshift(self):
        if False:
            return 10
        self._testOp('irshift', '>>=')

    def test_iand(self):
        if False:
            for i in range(10):
                print('nop')
        self._testOp('iand', '&=')

    def test_ixor(self):
        if False:
            for i in range(10):
                print('nop')
        self._testOp('ixor', '^=')

    def test_ior(self):
        if False:
            i = 10
            return i + 15
        self._testOp('ior', '|=')

    def test_ifloordiv(self):
        if False:
            print('Hello World!')
        self._testOp('ifloordiv', '//=')

class ErrorTestPy3(test_base.BaseTest):
    """Tests for errors."""

    def test_nis_wrong_arg_types(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      from typing import Iterable\n      def f(x: Iterable[str]): ...\n      f("abc")  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['str does not match string iterables by default']})

    def test_nis_bad_return(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      from typing import Iterable\n      def f() -> Iterable[str]:\n        return "abc" # bad-return-type[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['str does not match string iterables by default']})

    def test_protocol_mismatch(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      class Foo: pass\n      next(Foo())  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['__next__']})

    def test_protocol_mismatch_partial(self):
        if False:
            print('Hello World!')
        (_, errors) = self.InferWithErrors('\n      class Foo:\n        def __iter__(self):\n          return self\n      next(Foo())  # wrong-arg-types[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['not implemented on Foo: __next__']})

    def test_generator_send_ret_type(self):
        if False:
            i = 10
            return i + 15
        (_, errors) = self.InferWithErrors('\n      from typing import Generator\n      def f() -> Generator[int, str, int]:\n        x = yield 1\n        return x  # bad-return-type[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['int', 'str']})

class MatrixOperationsTest(test_base.BaseTest):
    """Test matrix operations."""

    def test_matmul(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors("\n      def f():\n        return 'foo' @ 3  # unsupported-operands[e]\n    ")
        self.assertErrorSequences(errors, {'e': ['@', 'str', 'int', "'__matmul__' on ", 'str', "'__rmatmul__' on ", 'int']})

    def test_imatmul(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      class A:\n        def __imatmul__(self, x: "A"):\n          pass\n      def f():\n        v = A()\n        v @= 3  # unsupported-operands[e]\n    ')
        self.assertErrorSequences(errors, {'e': ['@', 'A', 'int', '__imatmul__ on A', 'A']})

class UnboundLocalErrorTest(test_base.BaseTest):
    """Tests for UnboundLocalError.

  It is often confusing to users when a name error is logged due to a local
  variable shadowing one from an outer scope and being referenced before its
  local definition, e.g.:

  def f():
    x = 0
    def g():
      print(x)  # name error!
      x = 1

  In this case, we add some more details to the error message.
  """

    def test_function_in_function(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      def f(x):\n        def g():\n          print(x)  # name-error[e]\n          x = 0\n    ')
        self.assertErrorSequences(errors, {'e': ["Add `nonlocal x` in function 'f.g' to", "reference 'x' from function 'f'"]})

    def test_global(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('\n      x = 0\n      def f():\n        print(x)  # name-error[e]\n        x = 1\n    ')
        self.assertErrorSequences(errors, {'e': ["Add `global x` in function 'f' to", "reference 'x' from global scope"]})

    def test_class_in_function(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      def f():\n        x = 0\n        class C:\n          print(x)  # name-error[e]\n          x = 1\n    ')
        self.assertErrorSequences(errors, {'e': ["Add `nonlocal x` in class 'f.C' to", "reference 'x' from function 'f'"]})

    def test_deep_nesting(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      def f():\n        def g():\n          x = 0\n          class C:\n            class D:\n              print(x)  # name-error[e]\n              x = 1\n    ')
        self.assertErrorSequences(errors, {'e': ["Add `nonlocal x` in class 'f.g.C.D' to", "reference 'x' from function 'f.g'"]})

    def test_duplicate_names(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors('\n      def f1():\n        def f2():\n          def f3():\n            x = 0\n        def f3():\n          def f4():\n            print(x)  # name-error[e]\n    ')
        self.assertErrorSequences(errors, {'e': ["Name 'x' is not defined"]})

    def test_precedence(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      def f():\n        x = 0\n        def g():\n          x = 1\n          def h():\n            print(x)  # name-error[e]\n            x = 2\n    ')
        self.assertErrorSequences(errors, {'e': ["Add `nonlocal x` in function 'f.g.h' to", "reference 'x' from function 'f.g'"]})

class ClassAttributeNameErrorTest(test_base.BaseTest):
    """Tests for name errors on class attributes.

  For code like:
    class C:
      x = 0
      def f(self):
        print(x)  # name error!
  it's non-obvious that 'C.x' needs to be used to reference attribute 'x' from
  class 'C', so we add a hint to the error message.
  """

    def test_nested_classes(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      class C:\n        x = 0\n        class D:\n          y = 1\n          def f(self):\n            print(x)  # name-error[e1]\n            print(y)  # name-error[e2]\n    ')
        self.assertErrorSequences(errors, {'e1': ["Use 'C.x' to reference 'x' from class 'C'"], 'e2': ["Use 'C.D.y' to reference 'y' from class 'C.D'"]})

    def test_outer_function(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      def f():\n        class C:\n          x = 0\n          def f(self):\n            print(x)  # name-error[e]\n    ')
        self.assertErrorSequences(errors, {'e': ["Use 'C.x' to reference 'x' from class 'f.C'"]})

class PartiallyDefinedClassNameErrorTest(test_base.BaseTest):
    """Test for name errors on the attributes of partially defined classes.

  For code like:
    class C:
      x = 0
      class D:
        print(x)  # name error!
  unlike the similar examples in ClassAttributeNameErrorTest, using 'C.x' does
  not work because 'C' has not yet been fully defined. We add this explanation
  to the error message.
  """
    POST = 'before the class is fully defined'

    def test_nested_classes(self):
        if False:
            i = 10
            return i + 15
        errors = self.CheckWithErrors('\n      class C:\n        x = 0\n        class D:\n          y = 1\n          class E:\n            print(x)  # name-error[e1]\n            print(y)  # name-error[e2]\n    ')
        self.assertErrorSequences(errors, {'e1': ["Cannot reference 'x' from class 'C'", self.POST], 'e2': ["Cannot reference 'y' from class 'C.D'", self.POST]})

    def test_nested_classes_in_function(self):
        if False:
            return 10
        errors = self.CheckWithErrors('\n      def f():\n        class C:\n          x = 0\n          class D:\n            print(x)  # name-error[e]\n    ')
        self.assertErrorSequences(errors, {'e': ["Cannot reference 'x' from class 'f.C'", self.POST]})

    def test_unbound_local_precedence(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors('\n      def f():\n        x = 0\n        class C:\n          x = 1\n          class D:\n            print(x)  # name-error[e]\n            x = 2\n    ')
        self.assertErrorSequences(errors, {'e': ["Add `nonlocal x` in class 'f.C.D' to", "reference 'x' from function 'f'"]})
if __name__ == '__main__':
    test_base.main()