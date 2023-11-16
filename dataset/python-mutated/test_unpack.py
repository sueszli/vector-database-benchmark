"""Test unpacking."""
from pytype.tests import test_base
from pytype.tests import test_utils

class TestUnpack(test_base.BaseTest):
    """Test unpacking of sequences via *xs."""

    def test_build_with_unpack_indefinite(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      from typing import List\n      class A: pass\n      a: List[A] = []\n      b: List[str] = []\n      c = [*a, *b, 1]\n      d = {*a, *b, 1}\n      e = (*a, *b, 1)\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, List, Set, Tuple, Union\n\n      class A: ...\n      a = ...  # type: List[A]\n      b = ...  # type: List[str]\n      c = ...  # type: List[Union[A, str, int]]\n      d = ...  # type: Set[Union[A, str, int]]\n      e = ...  # type: Tuple[Union[A, str, int], ...]\n    ')

    def test_empty(self):
        if False:
            print('Hello World!')
        (ty, err) = self.InferWithErrors('\n      a, *b = []  # bad-unpacking[e]\n      c, *d = [1]\n      *e, f = [2]\n      g, *h, i = [1, 2]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, List\n      a: Any\n      b: List[nothing]\n      c: int\n      d: List[nothing]\n      e: List[nothing]\n      f: int\n      g: int\n      h: List[nothing]\n      i: int\n    ')
        self.assertErrorSequences(err, {'e': ['0 values', '1 variable']})

    def test_unpack_indefinite_from_pytd(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Tuple\n        a: Tuple[int, ...]\n        b: Tuple[str, ...]\n      ')
            ty = self.Infer('\n        import foo\n        c = (*foo.a, *foo.b)\n      ', pythonpath=[d.path])
        self.assertTypesMatchPytd(ty, '\n      import foo\n      from typing import Tuple, Union\n      c: Tuple[Union[int, str], ...]\n    ')

    def test_unpack_in_function_args(self):
        if False:
            for i in range(10):
                print('nop')
        self.options.tweak(strict_parameter_checks=False)
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Tuple\n        a: Tuple[int, ...]\n        b: Tuple[str, ...]\n      ')
            errors = self.CheckWithErrors('\n        import foo\n        class A: pass\n        def f(w: A, x: int, y: str, z: str):\n          pass\n        c = (*foo.a, *foo.b)\n        f(A(), *c, "hello")\n        f(A(), *c)\n        f(*c, "hello")  # wrong-arg-types[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'w: A.*w: Union.int,.str.'})

    def test_unpack_concrete_in_function_args(self):
        if False:
            return 10
        self.CheckWithErrors('\n      def f(x: int, y: str):\n        pass\n      a = (1, 2)\n      f(*a)  # wrong-arg-types\n      f(1, *("x", "y"))  # wrong-arg-count\n    ')

    def test_match_typed_starargs(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        def f(x:int, *args: str): ...\n        a: list\n        b: Any\n      ')
            self.Check('\n        import foo\n        foo.f(1, *foo.a)\n        foo.f(1, *foo.b)\n        foo.f(*foo.a)\n      ', pythonpath=[d.path])

    def test_path_join(self):
        if False:
            while True:
                i = 10
        self.Check("\n      import os\n      xs: list\n      os.path.join('x', *xs)\n    ")

    def test_overloaded_function(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any\n        @overload\n        def f(x:int, *args: str): ...\n        @overload\n        def f(x:str, *args: str): ...\n        a: list\n        b: Any\n      ')
            self.Check('\n        import foo\n        foo.f(1, *foo.a)\n        foo.f(1, *foo.b)\n        foo.f(*foo.a)\n      ', pythonpath=[d.path])

    def test_unpack_kwargs_without_starargs(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any, Dict, Optional\n        def f(x: int, y: str, z: bool = True, a: Optional[object] = None ): ...\n        a: Dict[str, Any]\n        b: dict\n      ')
            self.Check("\n        import foo\n        foo.f(1, 'a', **foo.a)\n        foo.f(1, 'a', **foo.b)\n        def g(x: int, y: str, **kwargs):\n          foo.f(x, y, **kwargs)\n      ", pythonpath=[d.path])

    def test_set_length_one_nondeterministic_unpacking(self):
        if False:
            return 10
        self.Check("\n    (x,) = {'a'}\n    ")

    def test_frozenset_length_one_nondeterministic_unpacking(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n    (x,) = frozenset(['a'])\n    ")

    def test_set_nondeterministic_unpacking(self):
        if False:
            for i in range(10):
                print('nop')
        self.CheckWithErrors("\n    (x, y) = {'a', 'b'}   # bad-unpacking\n    ")

    def test_frozenset_nondeterministic_unpacking(self):
        if False:
            print('Hello World!')
        self.CheckWithErrors("\n    (x, y) = frozenset(['a', 'b'])   # bad-unpacking\n    ")

    def test_str(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Optional, Text\n        class A: ...\n        def f(\n            x: Text,\n            y: int,\n            k: bool = ...,\n            l: Optional[Text] = ...,\n            m: Optional[A] = ...,\n        ) -> None: ...\n      ')
            self.Check('\n        import foo\n        from typing import Text\n        def g(self, x: str, **kwargs) -> None:\n          foo.f(x, 1, **kwargs)\n      ', pythonpath=[d.path])

    def test_unknown_length_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check("\n      from typing import Tuple\n      def f(*args: str):\n        pass\n      x: Tuple[str, ...]\n      f(*x, 'a', 'b', 'c')\n    ")

    def test_dont_unpack_iterable(self):
        if False:
            return 10
        self.Check('\n      class Foo(list):\n        pass\n\n      def f(x: Foo, y: int, z: bool = True):\n        pass\n\n      def g(x: Foo, **kwargs):\n        f(x, 10, **kwargs)\n    ')

    def test_erroneous_splat(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        from typing import Any, Sequence\n        def f(x: Sequence[Any], y: str): ...\n        def g(x: Sequence[Any], y: Sequence[str]): ...\n      ')
            self.CheckWithErrors('\n        import itertools\n        from typing import List\n        import foo\n        x: list\n        y: List[int]\n        foo.f(*x, "a")\n        foo.f(*x, *y)  # wrong-arg-types\n        foo.g(*x, *y)  # wrong-arg-types\n        a = itertools.product(*x, *y)\n      ', pythonpath=[d.path])

    def test_unpack_namedtuple(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(a, b, c, d, e, f): ...\n      ')
            self.Check("\n        import collections\n        import foo\n        X = collections.namedtuple('X', ('a', 'b', 'c'))\n        foo.f(*X(0, 1, 2), 3, 4, 5)\n\n        def g() -> X:\n          return X(0, 1, 2)\n        p = X(*g())\n        q = X(*g())\n        f = X(*(x - y for x, y in zip(p, q)))\n      ", pythonpath=[d.path])

    def test_posargs_and_namedargs(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def f(x, y=1, z=2, a=3):\n        pass\n\n      def g(b=None):\n        f(*b, y=2, z=3)\n    ')

    def test_dont_unpack_into_optional(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def f(x: int, y: int, z: str = ...):\n        pass\n\n      def g(*args: int):\n        f(*args)\n    ')

    def test_multiple_tuple_bindings(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer("\n      from typing import Tuple\n\n      class C:\n        def __init__(self, p, q):\n          self.p = p\n          self.q = q\n\n      x = [('a', 1), ('c', 3j), (2, 3)]\n      y = [C(*a).q for a in x]\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, List, Tuple, Union\n      class C:\n        p: Any\n        q: Any\n        def __init__(self, p, q): ...\n      x: List[Tuple[Union[int, str], Union[complex, int]]]\n      y: List[Union[complex, int]]\n    ')

    def test_type_parameter_instance(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      from typing import Dict, Tuple\n\n      class Key:\n        pass\n      class Value:\n        pass\n\n      def foo(x: Dict[Tuple[Key, Value], str]):\n        ret = []\n        for k, v in sorted(x.items()):\n          key, value = k\n          ret.append(key)\n        return ret\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, List, Tuple\n\n      class Key: ...\n      class Value: ...\n\n      def foo(x: Dict[Tuple[Key, Value], str]) -> List[Key]: ...\n    ')

    def test_unpack_any_subclass_instance(self):
        if False:
            print('Hello World!')
        with self.DepTree([('foo.pyi', '\n      from typing import Any\n\n      Base: Any\n    ')]):
            self.Check('\n        import foo\n        class A(foo.Base):\n          @classmethod\n          def make(cls, hello, world):\n            return cls(hello, world)\n\n        a = A.make(1, 2)\n        b = A.make(*a)\n      ')
if __name__ == '__main__':
    test_base.main()