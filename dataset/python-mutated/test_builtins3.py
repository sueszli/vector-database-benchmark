"""Tests of builtins (in stubs/builtins/{version}/__builtins__.pytd).

File 3/3. Split into parts to enable better test parallelism.
"""
from pytype.abstract import abstract_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class BuiltinTests3(test_base.BaseTest):
    """Tests for builtin methods and classes."""

    def test_super_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      x = super.__name__\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      x = ...  # type: str\n    ')

    def test_slice(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      x1 = [1,2,3][1:None]\n      x2 = [1,2,3][None:2]\n      x3 = [1,2,3][None:None]\n      x4 = [1,2,3][1:3:None]\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      x1 = ...  # type: List[int]\n      x2 = ...  # type: List[int]\n      x3 = ...  # type: List[int]\n      x4 = ...  # type: List[int]\n    ')

    def test_slice_attributes(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      v = slice(1)\n      start = v.start\n      stop = v.stop\n      step = v.step\n      indices = v.indices(0)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Optional, Tuple\n      v = ...  # type: slice\n      start = ...  # type: Optional[int]\n      stop = ...  # type: Optional[int]\n      step = ...  # type: Optional[int]\n      indices = ...  # type: Tuple[int, int, int]\n    ')

    def test_next_function(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      a = next(iter([1, 2, 3]))\n      b = next(iter([1, 2, 3]), default = 4)\n      c = next(iter([1, 2, 3]), "hello")\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      a = ...  # type: int\n      b = ...  # type: int\n      c = ...  # type: Union[int, str]\n    ')

    def test_implicit_typevar_import(self):
        if False:
            print('Hello World!')
        (ty, _) = self.InferWithErrors(f'\n      v = {abstract_utils.T}  # name-error\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      v = ...  # type: Any\n    ')

    def test_explicit_typevar_import(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      from builtins import _T\n      _T\n    ')

    def test_class_of_type(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      v = int.__class__\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Type\n      v = ...  # type: Type[type]\n    ')

    @test_base.skip('broken')
    def test_clear(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      x = {1, 2}\n      x.clear()\n      y = {"foo": 1}\n      y.clear()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Set\n      x = ...  # type: Set[nothing]\n      y = ...  # type: Dict[nothing, nothing]\n    ')

    def test_cmp(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      if not cmp(4, 4):\n        x = 42\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      x = ...  # type: int\n    ')

    def test_repr(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      if repr("hello world"):\n        x = 42\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      x = ...  # type: int\n    ')

    def test_int_init(self):
        if False:
            return 10
        (_, errors) = self.InferWithErrors('\n      int()\n      int(0)\n      int("0")\n      int("0", 10)\n      int(u"0")\n      int(u"0", 10)\n      int(0, 1, 2)  # wrong-arg-count[e]\n    ')
        self.assertErrorRegexes(errors, {'e': '1.*4'})

    def test_newlines(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('newlines.txt', '\n          1\n          2\n          3\n          ')
            self.Check('\n          l = []\n          with open("newlines.txt", "rU") as f:\n            for line in f:\n              l.append(line)\n            newlines = f.newlines\n          ')

    def test_init_with_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n        int(u"123.0")\n        float(u"123.0")\n        complex(u"123.0")\n    ')

    def test_io_write(self):
        if False:
            while True:
                i = 10
        self.Check('\n        import sys\n        sys.stdout.write("hello world")\n    ')

    def test_binary_io_write(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      with open('foo', 'wb') as f:\n        f.write(bytearray([1, 2, 3]))\n    ")

    def test_hasattr_none(self):
        if False:
            return 10
        self.assertNoCrash(self.Check, 'hasattr(int, None)')

    def test_number_attrs(self):
        if False:
            return 10
        ty = self.Infer('\n      a = (42).denominator\n      b = (42).numerator\n      c = (42).real\n      d = (42).imag\n      e = (3.14).conjugate()\n      f = (3.14).is_integer()\n      g = (3.14).real\n      h = (3.14).imag\n      i = (2j).conjugate()\n      j = (2j).real\n      k = (2j).imag\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      a = ...  # type: int\n      b = ...  # type: int\n      c = ...  # type: int\n      d = ...  # type: int\n      e = ...  # type: float\n      f = ...  # type: bool\n      g = ...  # type: float\n      h = ...  # type: float\n      i = ...  # type: complex\n      j = ...  # type: float\n      k = ...  # type: float\n    ')

    def test_builtins(self):
        if False:
            return 10
        self.Check('\n      import builtins  # pytype: disable=import-error\n    ')

    def test_special_builtin_types(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n      isinstance(1, int)\n      isinstance(1, "no")  # wrong-arg-types\n      issubclass(int, object)\n      issubclass(0, 0)  # wrong-arg-types\n      issubclass(int, 0)  # wrong-arg-types\n      hasattr(str, "upper")\n      hasattr(int, int)  # wrong-arg-types\n      ')

    def test_unpack_list(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      x = [1, ""]\n      a, b = x\n      x.append(2)\n      c, d, e = x\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Union\n      x = ...  # type: List[Union[int, str]]\n      a = ...  # type: int\n      b = ...  # type: str\n      c = ...  # type: Union[int, str]\n      d = ...  # type: Union[int, str]\n      e = ...  # type: Union[int, str]\n    ')

    def test_bytearray_setitem(self):
        if False:
            print('Hello World!')
        self.Check('\n      ba = bytearray(b"hello")\n      ba[0] = 106\n      ba[4:] = [121, 102, 105, 115, 104]\n      ba[4:] = b"yfish"\n      ba[4:] = bytearray("yfish")\n      ba[:5] = b""\n      ba[1:2] = b"la"\n      ba[2:3:2] = b"u"\n    ')

    def test_bytearray_setitem_py3(self):
        if False:
            print('Hello World!')
        self.Check('\n      ba = bytearray(b"hello")\n      ba[0] = 106\n      ba[:1] = [106]\n      ba[:1] = b"j"\n      ba[:1] = bytearray(b"j")\n      ba[:1] = memoryview(b"j")\n      ba[4:] = b"yfish"\n      ba[0:5] = b""\n      ba[1:4:2] = b"at"\n    ')

    def test_bytearray_contains(self):
        if False:
            return 10
        self.Check('\n      ba = bytearray(b"test")\n      1 in ba\n      "world" in ba\n      b"world" in ba\n      bytearray(b"t") in ba\n    ')

    def test_from_hex(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      f = float.fromhex("feed")\n      b = bytearray.fromhex("beef")\n    ')
        self.assertTypesMatchPytd(ty, '\n      f = ...  # type: float\n      b = ...  # type: bytearray\n    ')

    def test_none_length(self):
        if False:
            for i in range(10):
                print('nop')
        errors = self.CheckWithErrors('len(None)  # wrong-arg-types[e]')
        self.assertErrorRegexes(errors, {'e': 'Sized.*None'})

    def test_sequence_length(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      len("")\n      len(u"")\n      len(bytearray())\n      len([])\n      len(())\n      len(frozenset())\n      len(range(0))\n    ')

    def test_mapping_length(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      len({})\n    ')

    def test_print_bare_type(self):
        if False:
            return 10
        ty = self.Infer('\n      from typing import Any, Dict, Type\n      d1 = {}  # type: Dict[str, type]\n      d2 = {}  # type: Dict[str, Type[Any]]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict\n      d1 = ...  # type: Dict[str, type]\n      d2 = ...  # type: Dict[str, type]\n    ')

    def test_get_function_attr(self):
        if False:
            i = 10
            return i + 15
        self.Check("getattr(lambda: None, '__defaults__')")

    def test_str_startswith(self):
        if False:
            while True:
                i = 10
        self.Check('\n      s = "some str"\n      s.startswith("s")\n      s.startswith(("s", "t"))\n      s.startswith("a", start=1, end=2)\n    ')

    def test_str_endswith(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      s = "some str"\n      s.endswith("r")\n      s.endswith(("r", "t"))\n      s.endswith("a", start=1, end=2)\n    ')

    def test_path(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo/__init__.py')
            self.Check('\n        import foo\n        __path__, foo.__path__\n      ', pythonpath=[d.path])

    def test_del_byte_array_slice(self):
        if False:
            return 10
        self.Check('\n      ba = bytearray(b"hello")\n      del ba[0:2]\n    ')

    def test_input(self):
        if False:
            print('Hello World!')
        self.Check("\n      input()\n      input('input: ')\n    ")

    def test_set_default_error(self):
        if False:
            print('Hello World!')
        (ty, errors) = self.InferWithErrors('\n      x = {}\n      y = x.setdefault()  # wrong-arg-count[e1]\n      z = x.setdefault(1, 2, 3, *[])  # wrong-arg-count[e2]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Dict\n      x = ...  # type: Dict[nothing, nothing]\n      y = ...  # type: Any\n      z = ...  # type: Any\n    ')
        self.assertErrorRegexes(errors, {'e1': '2.*0', 'e2': '2.*3'})

    def test_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x, y):\n        return y\n      def g():\n        args = (4, )\n        return f(3, *args)\n      g()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, TypeVar\n      _T1 = TypeVar("_T1")\n      def f(x, y: _T1) -> _T1: ...\n      def g() -> int: ...\n    ')

    def test_str_join_error(self):
        if False:
            print('Hello World!')
        errors = self.CheckWithErrors("', '.join([1, 2, 3])  # wrong-arg-types[e]")
        self.assertErrorRegexes(errors, {'e': 'Expected.*Iterable\\[str\\].*Actual.*List\\[int\\]'})

    def test_int_protocols(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class Foo:\n        def __int__(self):\n          return 0\n      class Bar:\n        def __trunc__(self):\n          return 0\n      int(Foo())\n      int(Bar())\n    ')

    def test_bool_methods(self):
        if False:
            return 10
        ty = self.Infer('\n      x = True\n      print((not x) * (1,))\n      print((not x) * [1])\n      print((1,) * (not x))\n      print([1] * (not x))\n      a = True ** True\n      b = True ** 1.0\n    ')
        self.assertTypesMatchPytd(ty, '\n      a: int\n      b: float\n      x: bool\n    ')

    def test_delattr(self):
        if False:
            print('Hello World!')
        self.Check('\n      class Foo:\n        def __delattr__(self, name):\n          super(Foo, self).__delattr__(name)\n    ')
if __name__ == '__main__':
    test_base.main()