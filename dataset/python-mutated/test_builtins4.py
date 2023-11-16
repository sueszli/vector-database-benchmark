"""Tests of builtins (in stubs/builtins/{version}/__builtins__.pytd)."""
from pytype.tests import test_base
from pytype.tests import test_utils

class MapTest(test_base.BaseTest):
    """Tests for builtin.map."""

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      v = map(int, ("0",))\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Iterator\n      v : Iterator[int]\n    ')

    def test_lambda(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        pass\n\n      def f():\n        return map(lambda x: x, [Foo()])\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Iterator\n      class Foo:\n        pass\n\n      def f() -> Iterator: ...\n    ')

    def test_join(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      def f(input_string, sub):\n        return ''.join(map(lambda ch: ch, input_string))\n    ")
        self.assertTypesMatchPytd(ty, 'def f(input_string, sub) -> str: ...')

    def test_empty(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      lst1 = []\n      lst2 = [x for x in lst1]\n      lst3 = map(str, lst2)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, List, Iterator\n      lst1 : List[nothing]\n      lst2 : List[nothing]\n      lst3 : Iterator[nothing]\n    ')

    def test_heterogeneous(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      from typing import Union\n      def func(a: Union[int, str, float, bool]) -> str:\n        return str(a)\n      map(func, [1, 'pi', 3.14, True])\n    ")
        self.Check("\n      from typing import Iterable, Union\n      def func(\n          first: Iterable[str], second: str, third: Union[int, bool, float]\n      ) -> str:\n        return ' '.join(first) + second + str(third)\n      map(func,\n          [('one', 'two'), {'three', 'four'}, ['five', 'six']],\n          'abc',\n          [1, False, 3.14])\n    ")

    def test_error_message(self):
        if False:
            while True:
                i = 10
        errors = self.CheckWithErrors("\n      def func(a: int) -> float:\n        return float(a)\n      map(func, ['str'])  # wrong-arg-types[e]\n    ")
        self.assertErrorSequences(errors, {'e': ['Expected', 'Iterable[int]', 'Actual', 'List[str]']})

    def test_abspath(self):
        if False:
            return 10
        self.Check("\n      import os.path\n      map(os.path.abspath, [''])\n    ")

    def test_protocol(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def __len__(self) -> int:\n          return 0\n      map(len, [Foo()])\n    ')

class BuiltinTests(test_base.BaseTest):
    """Tests for builtin methods and classes."""

    def test_bool_return_value(self):
        if False:
            return 10
        ty = self.Infer('\n      def f():\n        return True\n      def g() -> bool:\n        return f()\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> bool: ...\n      def g() -> bool: ...\n    ')

    def test_sum_return(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import List\n      def f(x: List[float]) -> float:\n        return sum(x)\n    ')

    def test_sum_custom(self):
        if False:
            i = 10
            return i + 15
        self.Check("\n      class Foo:\n        def __init__(self, v):\n          self.v = v\n        def __add__(self, other: 'Foo') -> 'Foo':\n          return Foo(self.v + other.v)\n      assert_type(sum([Foo(0), Foo(1)]), Foo)\n    ")

    def test_sum_bad(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      class Foo:\n        pass\n      sum([Foo(), Foo()])  # wrong-arg-types\n    ')

    def test_print_function(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      import sys\n      print(file=sys.stderr)\n    ')

    def test_filename(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      def foo(s: str) -> str:\n        return s\n      foo(__file__)\n      ', filename='foobar.py')

    def test_super(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Type\n        def f(x: type): ...\n        def g(x: Type[super]): ...\n      ')
            ty = self.Infer('\n        from typing import Any, Type\n        import foo\n        def f(x): ...\n        def g(x: object): ...\n        def h(x: Any): ...\n        def i(x: type): ...\n        def j(x: Type[super]): ...\n        f(super)\n        g(super)\n        h(super)\n        i(super)\n        j(super)\n        foo.f(super)\n        foo.g(super)\n        v = super\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any, Type\n        def f(x) -> None: ...\n        def g(x: object) -> None: ...\n        def h(x: Any) -> None: ...\n        def i(x: type) -> None: ...\n        def j(x: Type[super]) -> None: ...\n        v : Type[super]\n      ')

    def test_bytearray_slice(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      def f(x: bytearray) -> bytearray:\n        return x[1:]\n      def g(x: bytearray) -> bytearray:\n        return x[1:5:2]\n    ')

    def test_set_length(self):
        if False:
            return 10
        self.Check('\n      from typing import Set\n      x : Set[int]\n      len(x)\n      len(set())\n    ')

    def test_sequence_length(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Sequence\n      x : Sequence\n      len(x)\n    ')

    def test_mapping_length(self):
        if False:
            return 10
        self.Check('\n      from typing import Mapping\n      x : Mapping\n      len(x)\n    ')

    def test_dict_copy(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      import collections\n      from typing import Dict\n      def f1(x: Dict[int, str]):\n        return x.copy()\n      def f2(x: 'collections.OrderedDict[int, str]'):\n        return x.copy()\n    ")
        self.assertTypesMatchPytd(ty, '\n      import collections\n      from typing import Dict, OrderedDict\n      def f1(x: Dict[int, str]) -> Dict[int, str]: ...\n      def f2(\n          x: OrderedDict[int, str]\n      ) -> OrderedDict[int, str]: ...\n    ')

    def test_format_self(self):
        if False:
            while True:
                i = 10
        self.Check('\n      "{self}".format(self="X")\n    ')

class BuiltinPython3FeatureTest(test_base.BaseTest):
    """Tests for builtin methods and classes."""

    def test_builtins(self):
        if False:
            return 10
        self.Check('\n      import builtins\n    ')

    def test_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      unicode("foo")  # name-error\n    ')

    def test_bytes_iteration(self):
        if False:
            return 10
        self.CheckWithErrors('\n      def f():\n        for x in bytes():\n          return bytes() + x  # unsupported-operands\n    ')

    def test_inplace_division(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      x, y = 24, 3\n      x /= y\n      assert x == 8.0 and y == 3\n      assert isinstance(x, float)\n      x /= y\n      assert x == (8.0/3.0) and y == 3\n      assert isinstance(x, float)\n    ')

    def test_removed_dict_methods(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('\n      {}.iteritems  # attribute-error\n      {}.iterkeys  # attribute-error\n      {}.itervalues  # attribute-error\n      {}.viewitems  # attribute-error\n      {}.viewkeys  # attribute-error\n      {}.viewvalues  # attribute-error\n    ')

    def test_dict_views(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import KeysView, ItemsView, ValuesView\n      def f(x: KeysView): ...\n      def g(x: ItemsView): ...\n      def h(x: ValuesView): ...\n      f({}.keys())\n      g({}.items())\n      h({}.values())\n    ')

    def test_str_join(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      b = u",".join([])\n      d = u",".join(["foo"])\n      e = ",".join([u"foo"])\n      f = u",".join([u"foo"])\n      g = ",".join([u"foo", "bar"])\n      h = u",".join([u"foo", "bar"])\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      b : str\n      d : str\n      e : str\n      f : str\n      g : str\n      h : str\n    ')

    @test_utils.skipBeforePy((3, 9), 'removeprefix and removesuffix new in 3.9')
    def test_str_remove_prefix_suffix(self):
        if False:
            return 10
        ty = self.Infer('\n      a = "prefix_suffix"\n      b = a.removeprefix("prefix_")\n      c = a.removesuffix("_suffix")\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      a : str\n      b : str\n      c : str\n    ')

    def test_str_is_hashable(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      from typing import Any, Dict, Hashable\n      def f(x: Dict[Hashable, Any]):\n        return x["foo"]\n      f({\'foo\': 1})\n    ')

    def test_bytearray_join(self):
        if False:
            return 10
        ty = self.Infer('\n      b = bytearray()\n      x2 = b.join([b"x"])\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      b : bytearray\n      x2 : bytearray\n    ')

    def test_iter1(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      a = next(iter([1, 2, 3]))\n      b = next(iter([1, 2, 3]), default = 4)\n      c = next(iter([1, 2, 3]), "hello")\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      a : int\n      b : int\n      c : Union[int, str]\n    ')

    def test_dict_keys(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      m = {"x": None}\n      a = m.keys() & {1, 2, 3}\n      b = m.keys() - {1, 2, 3}\n      c = m.keys() | {1, 2, 3}\n      d = m.keys() ^ {1, 2, 3}\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Set, Union\n      m : Dict[str, None]\n      a : Set[str]\n      b : Set[str]\n      c : Set[Union[int, str]]\n      d : Set[Union[int, str]]\n    ')

    def test_open(self):
        if False:
            return 10
        ty = self.Infer('\n      f1 = open("foo.py", "r")\n      f2 = open("foo.pickled", "rb")\n      v1 = f1.read()\n      v2 = f2.read()\n      def open_file1(mode):\n        f = open("foo.x", mode)\n        return f, f.read()\n      def open_file2(mode: str):\n        f = open("foo.x", mode)\n        return f, f.read()\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, BinaryIO, IO, TextIO, Tuple, Union\n      f1: TextIO\n      f2: BinaryIO\n      v1: str\n      v2: bytes\n      def open_file1(mode) -> Tuple[Any, Any]: ...\n      def open_file2(mode: str) -> Tuple[IO[Union[bytes, str]], Union[bytes, str]]: ...\n    ')

    def test_open_extended_file_modes(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      f1 = open("f1", "rb+")\n      f2 = open("f2", "w+t")\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import BinaryIO, TextIO\n      f1: BinaryIO\n      f2: TextIO\n    ')

    def test_filter(self):
        if False:
            return 10
        ty = self.Infer('\n      import re\n      def f(x: int):\n        pass\n      x1 = filter(None, bytearray(""))\n      x2 = filter(None, (True, False))\n      x3 = filter(None, {True, False})\n      x4 = filter(f, {1: None}.keys())\n      x5 = filter(None, {1: None}.keys())\n      x6 = filter(re.compile("").search, ("",))\n    ')
        self.assertTypesMatchPytd(ty, '\n      import re\n      from typing import Iterator\n      def f(x: int) -> None: ...\n      x1 : Iterator[int]\n      x2 : Iterator[bool]\n      x3 : Iterator[bool]\n      x4 : Iterator[int]\n      x5 : Iterator[int]\n      x6 : Iterator[str]\n      ')

    def test_filter_types(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Iterator, List, Optional, Tuple, Union\n      def f(xs: List[Optional[int]]) -> Iterator[int]:\n        return filter(None, xs)\n      def g(x: Tuple[int, str, None]) -> Iterator[Union[int, str]]:\n        return filter(None, x)\n    ')

    def test_zip(self):
        if False:
            return 10
        ty = self.Infer('\n      a = zip(())\n      b = zip((1, 2j))\n      c = zip((1, 2, 3), ())\n      d = zip((), (1, 2, 3))\n      e = zip((1j, 2j), (1, 2))\n      assert zip([], [], [])\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Iterator, Tuple, Union\n      a : Iterator[nothing]\n      b : Iterator[Tuple[Union[int, complex]]]\n      c : Iterator[nothing]\n      d : Iterator[nothing]\n      e : Iterator[Tuple[complex, int]]\n      ')

    def test_dict(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      def t_testDict():\n        d = {}\n        d['a'] = 3\n        d[3j] = 1.0\n        return _i1_(list(_i2_(d).values()))[0]\n      def _i1_(x):\n        return x\n      def _i2_(x):\n        return x\n      t_testDict()\n    ", deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, List, Union\n      def t_testDict() -> Union[float, int]: ...\n      # _i1_, _i2_ capture the more precise definitions of the ~dict, ~list\n      def _i1_(x: List[float]) -> List[Union[float, int]]: ...\n      def _i2_(x: dict[Union[complex, str], Union[float, int]]) -> Dict[Union[complex, str], Union[float, int]]: ...\n    ')

    def test_list_init(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      l3 = list({"a": 1}.keys())\n      l4 = list({"a": 1}.values())\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      l3 : List[str]\n      l4 : List[int]\n    ')

    def test_tuple_init(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      t3 = tuple({"a": 1}.keys())\n      t4 = tuple({"a": 1}.values())\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Tuple\n      t3 : Tuple[str, ...]\n      t4 : Tuple[int, ...]\n    ')

    def test_items(self):
        if False:
            return 10
        ty = self.Infer('\n      lst = list({"a": 1}.items())\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Tuple\n      lst : List[Tuple[str, int]]\n    ')

    def test_int_init(self):
        if False:
            for i in range(10):
                print('nop')
        (_, errors) = self.InferWithErrors('\n      int(0, 1)  # wrong-arg-types[e]\n    ')
        self.assertErrorRegexes(errors, {'e': 'Union\\[bytes, str\\].*int'})

    def test_removed_builtins(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      long  # name-error\n      {}.has_key  # attribute-error\n    ')

    def test_range(self):
        if False:
            while True:
                i = 10
        (ty, _) = self.InferWithErrors('\n      xrange(3)  # name-error\n      v = range(3)\n      v[0]\n      v[:]\n      x, y, z = v.start, v.stop, v.step\n    ')
        self.assertTypesMatchPytd(ty, '\n      v: range\n      x: int\n      y: int\n      z: int\n    ')

    def test_create_str(self):
        if False:
            return 10
        self.Check('\n      str(b"foo", "utf-8")\n    ')

    def test_bytes_constant(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("v = b'foo'")
        self.assertTypesMatchPytd(ty, 'v : bytes')

    def test_unicode_constant(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("v = 'foo\\u00e4'")
        self.assertTypesMatchPytd(ty, 'v : str')

    def test_memoryview(self):
        if False:
            while True:
                i = 10
        self.Check("\n      v = memoryview(b'abc')\n      v.format\n      v.itemsize\n      v.shape\n      v.strides\n      v.suboffsets\n      v.readonly\n      v.ndim\n      v[1]\n      v[1:]\n      98 in v\n      [x for x in v]\n      len(v)\n      v[1] = 98\n      v[1:] = b'bc'\n    ")

    def test_memoryview_methods(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      v1 = memoryview(b'abc')\n      v2 = v1.tobytes()\n      v3 = v1.tolist()\n      v4 = v1.hex()\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      v1: memoryview\n      v2: bytes\n      v3: List[int]\n      v4: str\n    ')

    def test_bytes_hex(self):
        if False:
            print('Hello World!')
        self.Check('\n      b = b\'abc\'\n      b.hex(",", 3)\n      m = memoryview(b)\n      m.hex(",", 4)\n      ba = bytearray([1,2,3])\n      ba.hex(b",", 5)\n    ')

    def test_memoryview_contextmanager(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      with memoryview(b'abc') as v:\n        pass\n    ")
        self.assertTypesMatchPytd(ty, '\n      v : memoryview\n    ')

    def test_array_tobytes(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      import array\n      def t_testTobytes():\n        return array.array('B').tobytes()\n    ")
        self.assertTypesMatchPytd(ty, '\n      import array\n      def t_testTobytes() -> bytes: ...\n    ')

    def test_iterator_builtins(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      v1 = map(int, ["0"])\n      v2 = zip([0], [1])\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Iterator, Tuple\n      v1 : Iterator[int]\n      v2 : Iterator[Tuple[int, int]]\n    ')

    def test_next(self):
        if False:
            return 10
        ty = self.Infer('\n      itr = iter((1, 2))\n      v1 = itr.__next__()\n      v2 = next(itr)\n    ')
        self.assertTypesMatchPytd(ty, '\n      itr : tupleiterator[int]\n      v1 : int\n      v2 : int\n    ')

    def test_aliased_error(self):
        if False:
            print('Hello World!')
        self.Check('\n      def f(e: OSError): ...\n      def g(e: IOError): ...\n      f(EnvironmentError())\n      g(EnvironmentError())\n    ')

    def test_os_error_subclasses(self):
        if False:
            while True:
                i = 10
        self.Check('\n      BlockingIOError\n      ChildProcessError\n      ConnectionError\n      FileExistsError\n      FileNotFoundError\n      InterruptedError\n      IsADirectoryError\n      NotADirectoryError\n      PermissionError\n      ProcessLookupError\n      TimeoutError\n    ')

    def test_raw_input(self):
        if False:
            while True:
                i = 10
        self.CheckWithErrors('raw_input  # name-error')

    def test_clear(self):
        if False:
            while True:
                i = 10
        self.Check('\n      bytearray().clear()\n      [].clear()\n    ')

    def test_copy(self):
        if False:
            print('Hello World!')
        self.Check('\n      bytearray().copy()\n      [].copy()\n    ')

    def test_round(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      v1 = round(4.2)\n      v2 = round(4.2, 1)\n    ')
        self.assertTypesMatchPytd(ty, '\n      v1: int\n      v2: float\n    ')

    def test_int_bytes_conversion(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      bytes_obj = (42).to_bytes(1, "little")\n      int_obj = int.from_bytes(b"*", "little")\n    ')
        self.assertTypesMatchPytd(ty, '\n      bytes_obj: bytes\n      int_obj: int\n    ')

    def test_unicode_error(self):
        if False:
            print('Hello World!')
        self.Check('\n      UnicodeDecodeError("", b"", 0, 0, "")\n      UnicodeEncodeError("", u"", 0, 0, "")\n    ')

    def test_min_max(self):
        if False:
            return 10
        ty = self.Infer("\n      x1 = min([1, 2, 3], default=3)\n      x2 = min((), default='')\n      y1 = max([1, 2, 3], default=3)\n      y2 = max((), default='')\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      x1 : int\n      x2 : Any\n      y1 : int\n      y2 : Any\n    ')

    def test_str_is_not_int(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      from typing import SupportsInt\n      def f(x: SupportsInt): pass\n      f("")  # wrong-arg-types\n    ')

    def test_str_is_not_float(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors('\n      from typing import SupportsFloat\n      def f(x: SupportsFloat): pass\n      f("")  # wrong-arg-types\n    ')

    def test_int_from_index(self):
        if False:
            return 10
        self.Check('\n      class Foo:\n        def __index__(self):\n          return 0\n      int(Foo())\n    ')

    def test_bytearray_compatible_with_bytes(self):
        if False:
            return 10
        self.Check('\n      def f(x):\n        # type: (bytes) -> None\n        pass\n      f(bytearray())\n    ')

    def test_breakpoint(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      breakpoint()\n    ')

    def test_range_with_index(self):
        if False:
            while True:
                i = 10
        self.Check('\n      class C:\n        def __index__(self) -> int:\n          return 2\n      range(C())\n    ')

    def test_getitem_with_index(self):
        if False:
            print('Hello World!')
        self.Check('\n      class C:\n        def __index__(self) -> int:\n          return 2\n      x = [7, 8, 9]\n      print(x[C()])\n    ')

class SetMethodsTest(test_base.BaseTest):
    """Tests for methods of the `set` class."""

    def test_union(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Set, Union\n      x: Set[int]\n      y: Set[str]\n      assert_type(x.union(y), Set[Union[int, str]])\n      assert_type(set.union(x, y), Set[Union[int, str]])\n    ')

    def test_difference(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Set\n      x: Set[int]\n      assert_type(x.difference({None}), Set[int])\n      assert_type(set.difference(x, {None}), Set[int])\n    ')

    def test_intersection(self):
        if False:
            print('Hello World!')
        self.Check('\n      from typing import Set\n      x: Set[int]\n      y: Set[str]\n      assert_type(x.intersection(y), Set[int])\n      assert_type(set.intersection(x, y), Set[int])\n    ')

    def test_symmetric_difference(self):
        if False:
            while True:
                i = 10
        self.Check('\n      from typing import Set, Union\n      x: Set[int]\n      y: Set[str]\n      assert_type(x.symmetric_difference(y), Set[Union[int, str]])\n      assert_type(set.symmetric_difference(x, y), Set[Union[int, str]])\n    ')

    def test_functools_reduce(self):
        if False:
            i = 10
            return i + 15
        self.CheckWithErrors('\n      import functools\n      from typing import Set\n      def f1(x: Set[str]) -> Set[str]:\n        return functools.reduce(set.union, x, set())\n      def f2(x: Set[str]) -> Set[str]:\n        return functools.reduce(set().union, x, set())  # bad-return-type\n    ')

class TypesNoneTypeTest(test_base.BaseTest):
    """Tests for types.NoneType."""

    @test_utils.skipBeforePy((3, 10), 'types.NoneType is new in 3.10')
    def test_function_param(self):
        if False:
            return 10
        self.Check('\n      import types\n      def f(x: types.NoneType) -> None:\n        return x\n      f(None)\n    ')

    @test_utils.skipBeforePy((3, 10), 'types.NoneType is new in 3.10')
    def test_if_splitting(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      import types\n      def f(x: types.NoneType) -> int:\n        if x:\n          return 'a'\n        else:\n          return 42\n      a = f(None)\n    ")
if __name__ == '__main__':
    test_base.main()