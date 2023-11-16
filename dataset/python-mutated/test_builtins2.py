"""Tests of builtins (in stubs/builtins/{version}/__builtins__.pytd).

File 2/3. Split into parts to enable better test parallelism.
"""
from pytype.tests import test_base
from pytype.tests import test_utils

class BuiltinTests2(test_base.BaseTest):
    """Tests for builtin methods and classes."""

    def test_div_mod_with_unknown(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f(x, y):\n        divmod(x, __any_object__)\n        return divmod(3, y)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Tuple\n      def f(x, y) -> Tuple[Any, Any]: ...\n    ')

    def test_defaultdict(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      import collections\n      r = collections.defaultdict()\n      r[3] = 3\n    ')
        self.assertTypesMatchPytd(ty, '\n      import collections\n      r = ...  # type: collections.defaultdict[int, int]\n    ')

    def test_dict_update(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      x = {}\n      x.update(a=3, b=4)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Dict\n      x = ...  # type: Dict[str, int]\n    ')

    def test_import_lib(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      import importlib\n    ')
        self.assertTypesMatchPytd(ty, '\n      import importlib\n    ')

    def test_set_union(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def f(y):\n        return set.union(*y)\n      def g(y):\n        return set.intersection(*y)\n      def h(y):\n        return set.difference(*y)\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f(y) -> set: ...\n      def g(y) -> set: ...\n      def h(y) -> set: ...\n    ')

    def test_set_init(self):
        if False:
            return 10
        ty = self.Infer('\n      data = set(x for x in [""])\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Set\n      data = ...  # type: Set[str]\n    ')

    def test_frozenset_inheritance(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo(frozenset):\n        pass\n      Foo([])\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      class Foo(frozenset):\n        pass\n    ')

    def test_old_style_class(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class Foo:\n        def get_dict(self):\n          return self.__dict__\n        def get_name(self):\n          return self.__name__\n        def get_class(self):\n          return self.__class__\n        def get_doc(self):\n          return self.__doc__\n        def get_module(self):\n          return self.__module__\n        def get_bases(self):\n          return self.__bases__\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Dict, Type\n      class Foo:\n        def get_dict(self) -> Dict[str, Any]: ...\n        def get_name(self) -> str: ...\n        def get_class(self) -> Type[Foo]: ...\n        def get_doc(self) -> str: ...\n        def get_module(self) -> str: ...\n        def get_bases(self) -> tuple: ...\n    ')

    def test_new_style_class(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      class Foo:\n        def get_dict(self):\n          return self.__dict__\n        def get_name(self):\n          return self.__name__\n        def get_class(self):\n          return self.__class__\n        def get_doc(self):\n          return self.__doc__\n        def get_module(self):\n          return self.__module__\n        def get_bases(self):\n          return self.__bases__\n        def get_hash(self):\n          return self.__hash__()\n        def get_mro(self):\n          return self.__mro__\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Dict, Type\n      class Foo:\n        def get_dict(self) -> Dict[str, Any]: ...\n        def get_name(self) -> str: ...\n        def get_class(self) -> Type[Foo]: ...\n        def get_doc(self) -> str: ...\n        def get_module(self) -> str: ...\n        def get_hash(self) -> int: ...\n        def get_mro(self) -> list: ...\n        def get_bases(self) -> tuple: ...\n    ')

    def test_dict_init(self):
        if False:
            return 10
        ty = self.Infer('\n      x1 = dict(u=3, v=4, w=5)\n      x2 = dict([(3, "")])\n      x3 = dict(((3, ""),))\n      x4 = dict({(3, "")})\n      x5 = dict({})\n      x6 = dict({3: ""})\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict\n      x1 = ...  # type: Dict[str, int]\n      x2 = ...  # type: Dict[int, str]\n      x3 = ...  # type: Dict[int, str]\n      x4 = ...  # type: Dict[int, str]\n      x5 = ...  # type: Dict[nothing, nothing]\n      x6 = ...  # type: Dict[int, str]\n    ')

    def test_dict(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      x = dict(u=3, v=4, w=5)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict\n      x: Dict[str, int]\n    ')

    def test_module(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        x = ...  # type: module\n      ')
            ty = self.Infer('\n        import foo\n        foo.x.bar()\n        x = foo.__name__\n        y = foo.x.baz\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        from typing import Any\n        x = ...  # type: str\n        y = ...  # type: Any\n      ')

    def test_classmethod(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class A:\n          x = ...  # type: classmethod\n      ')
            ty = self.Infer('\n        from foo import A\n        y = A.x()\n        z = A().x()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any, Type\n        A = ...  # type: Type[foo.A]\n        y = ...  # type: Any\n        z = ...  # type: Any\n      ')

    def test_staticmethod(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class A:\n          x = ...  # type: staticmethod\n      ')
            ty = self.Infer('\n        from foo import A\n        y = A.x()\n        z = A().x()\n      ', deep=False, pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        from typing import Any, Type\n        A = ...  # type: Type[foo.A]\n        y = ...  # type: Any\n        z = ...  # type: Any\n      ')

    def test_min_max(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      x1 = min(x for x in range(3))\n      x2 = min([3.1, 4.1], key=lambda n: n)\n      x3 = min((1, 2, 3), key=int)\n      y1 = max(x for x in range(3))\n      y2 = max([3.1, 4.1], key=lambda n: n)\n      y3 = max((1, 2, 3), key=int)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      x1 = ...  # type: int\n      x2 = ...  # type: float\n      x3 = ...  # type: int\n      y1 = ...  # type: int\n      y2 = ...  # type: float\n      y3 = ...  # type: int\n    ')

    def test_max_different_types(self):
        if False:
            return 10
        ty = self.Infer('\n      a = max(1, None)\n      b = max(1, None, 3j)\n      c = max(1, None, 3j, "str")\n      d = max(1, 2, 3, 4, 5, 6, 7)\n      e = max(1, None, key=int)\n      f = max(1, None, 3j, key=int)\n      g = max(1, None, 3j, "str", key=int)\n      h = max(1, 2, 3, 4, 5, 6, 7, key=int)\n      i = max([1,2,3,4])\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Optional, Union\n      a = ...  # type: Optional[int]\n      b = ...  # type: Optional[Union[complex, int]]\n      c = ...  # type: Optional[Union[complex, int, str]]\n      d = ...  # type: int\n      e = ...  # type: Optional[int]\n      f = ...  # type: Optional[Union[complex, int]]\n      g = ...  # type: Optional[Union[complex, int, str]]\n      h = ...  # type: int\n      i = ...  # type: int\n      ')

    def test_min_different_types(self):
        if False:
            return 10
        ty = self.Infer('\n      a = min(1, None)\n      b = min(1, None, 3j)\n      c = min(1, None, 3j, "str")\n      d = min(1, 2, 3, 4, 5, 6, 7)\n      e = min(1, None, key=int)\n      f = min(1, None, 3j, key=int)\n      g = min(1, None, 3j, "str", key=int)\n      h = min(1, 2, 3, 4, 5, 6, 7, key=int)\n      i = min([1,2,3,4])\n      ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Optional, Union\n      a = ...  # type: Optional[int]\n      b = ...  # type: Optional[Union[complex, int]]\n      c = ...  # type: Optional[Union[complex, int, str]]\n      d = ...  # type: int\n      e = ...  # type: Optional[int]\n      f = ...  # type: Optional[Union[complex, int]]\n      g = ...  # type: Optional[Union[complex, int, str]]\n      h = ...  # type: int\n      i = ...  # type: int\n      ')

    def test_from_keys(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      d1 = dict.fromkeys([1])\n      d2 = dict.fromkeys([1], 0)\n      d3 = dict.fromkeys(bytearray("x"))\n      d4 = dict.fromkeys({True: False})\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict\n      d1 = ...  # type: Dict[int, None]\n      d2 = ...  # type: Dict[int, int]\n      d3 = ...  # type: Dict[int, None]\n      d4 = ...  # type: Dict[bool, None]\n    ')

    def test_redefined_builtin(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class BaseException(Exception): pass\n      class CryptoException(BaseException, ValueError): pass\n    ', deep=False)
        (p1, p2) = ty.Lookup('CryptoException').bases
        self.assertEqual(p1.name, 'BaseException')
        self.assertEqual(p2.name, 'builtins.ValueError')
        self.assertTypesMatchPytd(ty, '\n      class BaseException(Exception): ...\n      class CryptoException(BaseException, ValueError): ...\n    ')

    def test_sum(self):
        if False:
            return 10
        ty = self.Infer('\n      x1 = sum([1, 2])\n      x2 = sum([1, 2], 0)\n      x3 = sum([1.0, 3j])\n      x4 = sum([1.0, 3j], 0)\n      x5 = sum([[1], ["2"]], [])\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Union\n      x1 = ...  # type: int\n      x2 = ...  # type: int\n      x3 = ...  # type: Union[float, complex]\n      x4 = ...  # type: Union[int, float, complex]\n      x5 = ...  # type: List[Union[int, str]]\n    ')

    def test_reversed(self):
        if False:
            for i in range(10):
                print('nop')
        (ty, errors) = self.InferWithErrors('\n      x1 = reversed(range(42))\n      x2 = reversed([42])\n      x3 = reversed((4, 2))\n      x4 = reversed("hello")\n      x5 = reversed({42})  # wrong-arg-types[e1]\n      x6 = reversed(frozenset([42]))  # wrong-arg-types[e2]\n      x7 = reversed({True: 42})  # wrong-arg-types[e3]\n      x8 = next(reversed([42]))\n      x9 = list(reversed([42]))\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      x1 = ...  # type: reversed[int]\n      x2 = ...  # type: reversed[int]\n      x3 = ...  # type: reversed[int]\n      x4 = ...  # type: reversed[str]\n      x5 = ...  # type: reversed[nothing]\n      x6 = ...  # type: reversed[nothing]\n      x7 = ...  # type: reversed[nothing]\n      x8 = ...  # type: int\n      x9 = ...  # type: List[int]\n    ')
        self.assertErrorRegexes(errors, {'e1': 'Set\\[int\\]', 'e2': 'FrozenSet\\[int\\]', 'e3': 'Dict\\[bool, int\\]'})

    def test_str_join(self):
        if False:
            return 10
        ty = self.Infer('\n      a = ",".join([])\n      c = ",".join(["foo"])\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      a = ...  # type: str\n      c = ...  # type: str\n    ')

    def test_bytearray_join(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      b = bytearray()\n      x1 = b.join([])\n      x3 = b.join([b])\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      b = ...  # type: bytearray\n      x1 = ...  # type: bytearray\n      x3 = ...  # type: bytearray\n    ')

    def test_reduce(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      reduce(lambda x, y: x+y, [1,2,3]).real\n      reduce(lambda x, y: x+y, ["foo"]).upper()\n      reduce(lambda x, y: 4, [], "foo").upper()\n      reduce(lambda x, y: "s", [1,2,3], 0).upper()\n    ')

    def test_dict_pop_item(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      v = {"a": 1}.popitem()\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Tuple\n      v = ...  # type: Tuple[str, int]\n    ')

    def test_long_constant(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      MAX_VALUE = 2**64\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      MAX_VALUE = ...  # type: int\n    ')

    def test_iter(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      x3 = iter(bytearray(42))\n      x4 = iter(x for x in [42])\n      x5 = iter([42])\n      x6 = iter((42,))\n      x7 = iter({42})\n      x8 = iter({"a": 1})\n      x9 = iter(int, 42)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Generator, Iterator\n      x3 = ...  # type: bytearray_iterator\n      x4 = ...  # type: Generator[int, Any, Any]\n      x5 = ...  # type: listiterator[int]\n      x6 = ...  # type: tupleiterator[int]\n      x7 = ...  # type: setiterator[int]\n      x8 = ...  # type: Iterator[str]\n      # The "nothing" is due to pytype ignoring Callable parameters and\n      # therefore not seeing the type parameter value tucked away in _RET.\n      x9 = ...  # type: Iterator[int]\n    ')

    def test_list_init(self):
        if False:
            return 10
        ty = self.Infer('\n      l1 = list()\n      l2 = list([42])\n      l5 = list(iter([42]))\n      l6 = list(reversed([42]))\n      l7 = list(iter((42,)))\n      l8 = list(iter({42}))\n      l9 = list((42,))\n      l10 = list({42})\n      l11 = list("hello")\n      l12 = list(iter(bytearray(42)))\n      l13 = list(iter(range(42)))\n      l14 = list(x for x in [42])\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      l1 = ...  # type: List[nothing]\n      l2 = ...  # type: List[int]\n      l5 = ...  # type: List[int]\n      l6 = ...  # type: List[int]\n      l7 = ...  # type: List[int]\n      l8 = ...  # type: List[int]\n      l9 = ...  # type: List[int]\n      l10 = ...  # type: List[int]\n      l11 = ...  # type: List[str]\n      l12 = ...  # type: List[int]\n      l13 = ...  # type: List[int]\n      l14 = ...  # type: List[int]\n    ')

    def test_tuple_init(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      t1 = tuple()\n      t2 = tuple([42])\n      t5 = tuple(iter([42]))\n      t6 = tuple(reversed([42]))\n      t7 = tuple(iter((42,)))\n      t8 = tuple(iter({42}))\n      t9 = tuple((42,))\n      t10 = tuple({42})\n      t12 = tuple(iter(bytearray(42)))\n      t13 = tuple(iter(range(42)))\n      t14 = tuple(x for x in [42])\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Tuple\n      t1 = ...  # type: Tuple[()]\n      t2 = ...  # type: Tuple[int, ...]\n      t5 = ...  # type: Tuple[int, ...]\n      t6 = ...  # type: Tuple[int, ...]\n      t7 = ...  # type: Tuple[int, ...]\n      t8 = ...  # type: Tuple[int, ...]\n      t9 = ...  # type: Tuple[int, ...]\n      t10 = ...  # type: Tuple[int, ...]\n      t12 = ...  # type: Tuple[int, ...]\n      t13 = ...  # type: Tuple[int, ...]\n      t14 = ...  # type: Tuple[int, ...]\n    ')

    def test_empty_tuple(self):
        if False:
            return 10
        self.Check('\n      isinstance(42, ())\n      issubclass(int, ())\n      type("X", (), {"foo": 42})\n      type("X", (), {})\n    ')

    def test_list_extend(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      x1 = [42]\n      x1.extend([""])\n      x2 = [42]\n      x2.extend(("",))\n      x3 = [42]\n      x3.extend({""})\n      x4 = [42]\n      x4.extend(frozenset({""}))\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List, Union\n      x1 = ...  # type: List[Union[int, str]]\n      x2 = ...  # type: List[Union[int, str]]\n      x3 = ...  # type: List[Union[int, str]]\n      x4 = ...  # type: List[Union[int, str]]\n    ')

    def test_sorted(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      x3 = sorted(bytearray("hello"))\n      x4 = sorted([])\n      x5 = sorted([42], reversed=True)\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      x3 = ...  # type: List[int]\n      x4 = ...  # type: List[nothing]\n      x5 = ...  # type: List[int]\n    ')

    def test_enumerate(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      x1 = enumerate([42])\n      x2 = enumerate((42,))\n      x3 = enumerate(x for x in range(5))\n      x4 = list(enumerate(['']))\n    ", deep=False)
        self.assertTypesMatchPytd(ty, '\n      x1 = ...  # type: enumerate[int]\n      x2 = ...  # type: enumerate[int]\n      x3 = ...  # type: enumerate[int]\n      x4 = ...  # type: list[tuple[int, str]]\n    ')

    def test_frozenset_init(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      x1 = frozenset([42])\n      x2 = frozenset({42})\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      x1 = ...  # type: frozenset[int]\n      x2 = ...  # type: frozenset[int]\n    ')

    def test_frozenset_literal(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      a = "foo" in {"foo"}\n    ')
        self.assertTypesMatchPytd(ty, '\n      a = ...  # type: bool\n    ')

    def test_func_tools(self):
        if False:
            return 10
        self.Check('\n      import functools\n    ')

    def test_abc(self):
        if False:
            print('Hello World!')
        self.Check('\n      import abc\n    ')

    def test_set_default(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      x = {}\n      x['bar'] = 3\n      y = x.setdefault('foo', 3.14)\n      z = x['foo']\n    ", deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Union\n      x = ...  # type: Dict[str, Union[float, int]]\n      y = ...  # type: Union[float, int]\n      z = ...  # type: float\n    ')

    def test_set_default_one_arg(self):
        if False:
            print('Hello World!')
        ty = self.Infer("\n      x = {}\n      x['bar'] = 3\n      y = x.setdefault('foo')\n      z = x['foo']\n    ", deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Optional\n      x = ...  # type: Dict[str, Optional[int]]\n      y = ...  # type: Optional[int]\n      z = ...  # type: None\n    ')

    def test_set_default_varargs(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      x1 = {}\n      y1 = x1.setdefault(*("foo", 42))\n\n      x2 = {}\n      y2 = x2.setdefault(*["foo", 42])\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Dict\n\n      x1 = ...  # type: Dict[str, int]\n      y1 = ...  # type: int\n\n      x2 = ...  # type: Dict[str, int]\n      y2 = ...  # type: int\n    ')

    def test_redefine_next(self):
        if False:
            return 10
        ty = self.Infer('\n      next = 42\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      next = ...  # type: int\n    ')

    def test_os_environ_copy(self):
        if False:
            return 10
        self.Check('\n      import os\n      os.environ.copy()["foo"] = "bar"\n    ')

    def test_bytearray_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      bytearray(42)\n      bytearray([42])\n      bytearray(u"hello", "utf-8")\n      bytearray(u"hello", "utf-8", "")\n    ')

    def test_compile(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      code = compile("1 + 2", "foo.py", "single")\n    ')

    def test_int_init(self):
        if False:
            while True:
                i = 10
        self.Check('\n      int(42)\n      int(42.0)\n      int("42")\n      int(u"42")\n      int()\n    ')

    def test_exec(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        x = exec\n      ')
            self.Check('\n        import foo\n        foo.x("a = 2")\n      ', pythonpath=[d.path])

    def test_format_string(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class A:\n        def __format__(self, format_spec):\n          return "hard_coded".__format__(format_spec)\n      a = A()\n      print(f"{a}")\n    ')
if __name__ == '__main__':
    test_base.main()