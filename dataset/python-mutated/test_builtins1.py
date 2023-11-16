"""Tests of builtins (in stubs/builtins/{version}/__builtins__.pytd).

File 1/3. Split into parts to enable better test parallelism.
"""
from pytype.tests import test_base

class BuiltinTests(test_base.BaseTest):
    """Tests for builtin methods and classes."""

    def test_repr1(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def t_testRepr1(x):\n        return repr(x)\n      t_testRepr1(4)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def t_testRepr1(x: int) -> str: ...\n    ')

    @test_base.skip('b/238794928: Function inference will be removed.')
    def test_repr2(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      def t_testRepr2(x):\n        return repr(x)\n      t_testRepr2(4)\n      t_testRepr2(1.234)\n      t_testRepr2('abc')\n    ", deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def t_testRepr2(x: Union[float, int, str]) -> str: ...\n    ')

    def test_repr3(self):
        if False:
            return 10
        ty = self.Infer('\n      def t_testRepr3(x):\n        return repr(x)\n      t_testRepr3(__any_object__())\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def t_testRepr3(x) -> str: ...\n    ')

    def test_eval_solve(self):
        if False:
            return 10
        ty = self.Infer('\n      def t_testEval(x):\n        return eval(x)\n      t_testEval(4)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def t_testEval(x: int) -> Any: ...\n    ')

    def test_isinstance1(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def t_testIsinstance1(x):\n        return isinstance(x, int)\n    ')
        self.assertTypesMatchPytd(ty, '\n      def t_testIsinstance1(x) -> bool: ...\n    ')

    def test_isinstance2(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Bar:\n        def foo(self):\n          return isinstance(self, Baz)\n\n      class Baz(Bar):\n        pass\n    ')
        self.assertTypesMatchPytd(ty, '\n    class Bar:\n      def foo(self) -> bool: ...\n\n    class Baz(Bar):\n      pass\n    ')

    def test_pow1(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      def t_testPow1():\n        # pow(int, int) returns int, or float if the exponent is negative.\n        # Hence, it's a handy function for testing UnionType returns.\n        return pow(1, -2)\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def t_testPow1() -> Union[float, int]: ...\n    ')

    def test_max1(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def t_testMax1():\n        # max is a parameterized function\n        return max(1, 2)\n    ')
        self.assertTypesMatchPytd(ty, '\n      def t_testMax1() -> int: ...\n      ')

    def test_max2(self):
        if False:
            return 10
        ty = self.Infer('\n      def t_testMax2(x, y):\n        # max is a parameterized function\n        return max(x, y)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def t_testMax2(x, y) -> Any: ...\n      ')

    def test_zip_error(self):
        if False:
            return 10
        errors = self.CheckWithErrors('zip([], [], [], 42)  # wrong-arg-types[e]')
        self.assertErrorRegexes(errors, {'e': 'Iterable.*int'})

    def test_dict_defaults(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n    def t_testDictDefaults(x):\n      d = {}\n      res = d.setdefault(x, str(x))\n      _i_(d)\n      return res\n    def _i_(x):\n      return x\n    t_testDictDefaults(3)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def t_testDictDefaults(x: int) -> str: ...\n      # _i_ captures the more precise definition of the dict\n      def _i_(x: dict[int, str]) -> dict[int, str]: ...\n    ')

    def test_dict_get(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        mydict = {"42": 42}\n        return mydict.get("42")\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def f() -> Union[int, NoneType]: ...\n    ')

    def test_dict_get_or_default(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f():\n        mydict = {"42": 42}\n        return mydict.get("42", False)\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> int: ...\n    ')

    def test_list_init0(self):
        if False:
            return 10
        ty = self.Infer('\n    def t_testListInit0(x):\n      return list(x)\n    ')
        self.assertTypesMatchPytd(ty, '\n      def t_testListInit0(x) -> list: ...\n    ')

    def test_list_init1(self):
        if False:
            return 10
        ty = self.Infer('\n    def t_testListInit1(x, y):\n      return x + [y]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def t_testListInit1(x, y) -> Any: ...\n    ')

    def test_list_init2(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n    def t_testListInit2(x, i):\n      return x[i]\n    z = __any_object__\n    t_testListInit2(__any_object__, z)\n    z + 1\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      z = ...  # type: Any\n\n      def t_testListInit2(x, i) -> Any: ...\n    ')

    def test_list_init3(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n    def t_testListInit3(x, i):\n      return x[i]\n    t_testListInit3([1,2,3,'abc'], 0)\n    ", deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Union\n      def t_testListInit3(x: List[Union[int, str]], i: int) -> int: ...\n    ')

    def test_list_init4(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n    def t_testListInit4(x):\n      return _i_(list(x))[0]\n    def _i_(x):\n      return x\n    t_testListInit4(__any_object__)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def t_testListInit4(x) -> Any: ...\n      def _i_(x: list) -> list: ...\n    ')

    def test_abs_int(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def t_testAbsInt(x):\n        return abs(x)\n      t_testAbsInt(1)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def t_testAbsInt(x: int) -> int: ...\n  ')

    def test_abs(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def t_testAbs(x):\n        return abs(x)\n      t_testAbs(__any_object__)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      # Since SupportsAbs.__abs__ returns a type parameter, the return type\n      # of abs(...) can be anything.\n      def t_testAbs(x) -> Any: ...\n    ')

    def test_abs_union(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        def __abs__(self):\n          return "hello"\n      class Bar:\n        def __abs__(self):\n          return 42\n      x = Foo() if __random__ else Bar()\n      y = abs(x)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Union\n      x = ...  # type: Union[Bar, Foo]\n      y = ...  # type: Union[str, int]\n      class Bar:\n          def __abs__(self) -> int: ...\n      class Foo:\n          def __abs__(self) -> str: ...\n    ')

    def test_cmp(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def t_testCmp(x, y):\n        return cmp(x, y)\n    ')
        self.assertTypesMatchPytd(ty, '\n    def t_testCmp(x, y) -> int: ...\n    ')

    @test_base.skip('b/238794928: Function inference will be removed.')
    def test_cmp_multi(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def t_testCmpMulti(x, y):\n        return cmp(x, y)\n      t_testCmpMulti(1, 2)\n      t_testCmpMulti(1, 2.0)\n      t_testCmpMulti(1.0, 2)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def t_testCmpMulti(x: Union[float, int], y: int) -> int: ...\n      def t_testCmpMulti(x: int, y: float) -> int: ...\n    ')

    def test_cmp_str(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def t_testCmpStr(x, y):\n        return cmp(x, y)\n      t_testCmpStr("abc", "def")\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def t_testCmpStr(x: str, y: str) -> int: ...\n    ')

    def test_cmp_str2(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def t_testCmpStr2(x, y):\n        return cmp(x, y)\n      t_testCmpStr2("abc", __any_object__)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      def t_testCmpStr2(x: str, y) -> int: ...\n    ')

    def test_tuple(self):
        if False:
            i = 10
            return i + 15
        self.Infer('\n      def f(x):\n        return x\n      def g(args):\n        f(*tuple(args))\n    ', show_library_calls=True)

    def test_open(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x):\n        with open(x, "r") as fi:\n          return fi.read()\n      ')
        self.assertTypesMatchPytd(ty, '\n      def f(x) -> str: ...\n    ')

    def test_open_error(self):
        if False:
            print('Hello World!')
        src = 'open(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)  # wrong-arg-count'
        self.CheckWithErrors(src)

    def test_signal(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import signal\n      def f():\n        signal.signal(signal.SIGTERM, 0)\n    ')
        self.assertTypesMatchPytd(ty, '\n      import signal\n\n      def f() -> NoneType: ...\n    ')

    def test_sys_argv(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      import sys\n      def args():\n        return ' '.join(sys.argv)\n      args()\n    ", deep=False, show_library_calls=True)
        self.assertTypesMatchPytd(ty, '\n      import sys\n      def args() -> str: ...\n    ')

    def test_setattr(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        def __init__(self, x):\n          for attr in x.__dict__:\n            setattr(self, attr, getattr(x, attr))\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo:\n        def __init__(self, x) -> NoneType: ...\n    ')

    def test_array_smoke(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      import array\n      class Foo:\n        def __init__(self):\n          array.array('i')\n    ")
        ty.Lookup('Foo')

    def test_array(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      import array\n      class Foo:\n        def __init__(self):\n          self.bar = array.array('i', [1, 2, 3])\n    ")
        self.assertTypesMatchPytd(ty, '\n      import array\n      class Foo:\n        bar = ...  # type: array.array[int]\n        def __init__(self) -> None: ...\n    ')

    def test_inherit_from_builtin(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo(list):\n        pass\n    ')
        self.assertTypesMatchPytd(ty, '\n      class Foo(list):\n        pass\n    ')

    def test_os_path(self):
        if False:
            return 10
        ty = self.Infer("\n      import os\n      class Foo:\n        bar = os.path.join('hello', 'world')\n    ")
        ty.Lookup('Foo')

    def test_hasattr(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      class Bar:\n        pass\n      a = hasattr(Bar, 'foo')\n    ")
        self.assertTypesMatchPytd(ty, '\n    class Bar:\n      pass\n    a : bool\n    ')

    def test_time(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import time\n      def f(x):\n        if x:\n          return time.mktime(time.struct_time((1, 2, 3, 4, 5, 6, 7, 8, 9)))\n        else:\n          return 3j\n    ')
        self.assertTypesMatchPytd(ty, '\n      import time\n      from typing import Union\n      def f(x) -> Union[complex, float]: ...\n    ')

    def test_div_mod(self):
        if False:
            return 10
        ty = self.Infer('\n      def seed(self, a=None):\n        a = int(0)\n        divmod(a, 30268)\n    ')
        self.assertTypesMatchPytd(ty, '\n      def seed(self, a=...) -> NoneType: ...\n    ')

    def test_div_mod2(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def seed(self, a=None):\n        if a is None:\n          a = int(16)\n        return divmod(a, 30268)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Tuple\n      def seed(self, a = ...) -> Tuple[Any, Any]: ...\n    ')

    def test_join(self):
        if False:
            return 10
        ty = self.Infer('\n      def f(elements):\n        return ",".join(t for t in elements)\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f(elements) -> str: ...\n    ')

    def test_version_info(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      import sys\n      def f():\n        return 'py%d' % sys.version_info[0]\n    ")
        self.assertTypesMatchPytd(ty, '\n      import sys\n      def f() -> str: ...\n    ')

    def test_inherit_from_namedtuple(self):
        if False:
            print('Hello World!')
        self.Check("\n      import collections\n\n      class Foo(\n          collections.namedtuple('_Foo', 'x y z')):\n        pass\n      a = Foo(1, 2, 3)\n    ")

    @test_base.skip('Does not work - x, y and z all get set to Any')
    def test_store_and_load_from_namedtuple(self):
        if False:
            print('Hello World!')
        self.Check('\n      import collections\n      t = collections.namedtuple(\'t\', [\'x\', \'y\', \'z\'])\n      t.x = 3\n      t.y = "foo"\n      t.z = 1j\n      x = t.x\n      y = t.y\n      z = t.z\n      assert_type(x, int)\n      assert_type(y, str)\n      assert_type(z, complex)\n    ')

    def test_type_equals(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(n):\n        return type(n) == type(0)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any\n      def f(n) -> Any: ...\n    ')

    def test_type_equals2(self):
        if False:
            return 10
        ty = self.Infer('\n      import types\n      def f(mod):\n        return type(mod) == types.ModuleType\n    ')
        self.assertTypesMatchPytd(ty, '\n      import types\n      from typing import Any\n      def f(mod) -> Any: ...\n    ')

    def test_date_time(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      import datetime\n\n      def f(date):\n        return date.ctime()\n    ')
        self.assertTypesMatchPytd(ty, '\n      import datetime\n      from typing import Any\n      def f(date) -> Any: ...\n  ')

    def test_from_utc(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import datetime\n\n      def f(tz):\n        tz.fromutc(datetime.datetime(1929, 10, 29))\n    ')
        self.assertTypesMatchPytd(ty, '\n      import datetime\n      def f(tz) -> NoneType: ...\n  ')
if __name__ == '__main__':
    test_base.main()