"""Tests for if-splitting."""
from pytype.tests import test_base
from pytype.tests import test_utils

class SplitTest(test_base.BaseTest):
    """Tests for if-splitting."""

    def test_restrict_none(self):
        if False:
            while True:
                i = 10
        ty = self.Infer("\n      def foo(x):\n        y = str(x) if x else None\n\n        if y:\n          # y can't be None here!\n          return y\n        else:\n          return 123\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def foo(x) -> Union[int, str]: ...\n    ')

    def test_restrict_true(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def foo(x):\n        y = str(x) if x else True\n\n        if y:\n          return 123\n        else:\n          return y\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def foo(x) -> Union[int, str]: ...\n    ')

    def test_related_variable(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def foo(x):\n        # y is Union[str, None]\n        # z is Union[float, True]\n        if x:\n          y = str(x)\n          z = 1.23\n        else:\n          y = None\n          z = True\n\n        if y:\n          # We only return z when y is true, so z must be a float here.\n          return z\n\n        return 123\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def foo(x) -> Union[float, int]: ...\n    ')

    def test_nested_conditions(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def foo(x1, x2):\n        y1 = str(x1) if x1 else 0\n\n        if y1:\n          if x2:\n            return y1  # The y1 condition is still active here.\n\n        return "abc"\n    ')
        self.assertTypesMatchPytd(ty, '\n      def foo(x1, x2) -> str: ...\n    ')

    def test_remove_condition_after_merge(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n      def foo(x):\n        y = str(x) if x else None\n\n        if y:\n          # y can't be None here.\n          z = 123\n        # But y can be None here.\n        return y\n    ")
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def foo(x) -> Union[None, str]: ...\n    ')

    def test_unsatisfiable_condition(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f1(x):\n        c = 0\n        if c:\n          unknown_method()\n          return 123\n        else:\n          return "hello"\n\n      def f2(x):\n        c = 1\n        if c:\n          return 123\n        else:\n          unknown_method()\n          return "hello"\n\n      def f3(x):\n        c = 0\n        if c:\n          return 123\n        else:\n          return "hello"\n\n      def f4(x):\n        c = 1\n        if c:\n          return 123\n        else:\n          return "hello"\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f1(x) -> str: ...\n      def f2(x) -> int: ...\n      def f3(x) -> str: ...\n      def f4(x) -> int: ...\n    ')

    def test_sources_propagated_through_call(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class Foo:\n        def method(self):\n          return 1\n\n      class Bar:\n        def method(self):\n          return "x"\n\n      def foo(x):\n        if x:\n          obj = Foo()\n        else:\n          obj = Bar()\n\n        if isinstance(obj, Foo):\n          return obj.method()\n        return None\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      class Foo:\n        def method(self) -> int: ...\n\n      class Bar:\n        def method(self) -> str: ...\n\n      def foo(x) -> Union[None, int]: ...\n    ')

    def test_short_circuit(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def int_t(x): return 1 or x\n      def int_f(x): return 0 and x\n      def str_t(x): return "s" or x\n      def str_f(x): return "" and x\n      def bool_t(x): return True or x\n      def bool_f(x): return False and x\n      def tuple_t(x): return (1, ) or x\n      def tuple_f(x): return () and x\n      def dict_f(x): return {} and x\n      def list_f(x): return [] and x\n      def set_f(x): return set() and x\n      def frozenset_f(x): return frozenset() and x\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, List, Tuple\n      def int_t(x) -> int: ...\n      def int_f(x) -> int: ...\n      def str_t(x) -> str: ...\n      def str_f(x) -> str: ...\n      def bool_t(x) -> bool: ...\n      def bool_f(x) -> bool: ...\n      def tuple_t(x) -> Tuple[int]: ...\n      def tuple_f(x) -> Tuple[()]: ...\n      def dict_f(x) -> Dict[nothing, nothing]: ...\n      def list_f(x) -> List[nothing]: ...\n      def set_f(x) -> set[nothing]: ...\n      def frozenset_f(x) -> frozenset[nothing]: ...\n    ')

    def test_dict(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f1():\n        d = {}\n        return 123 if d else "hello"\n\n      def f2(x):\n        d = {}\n        d[x] = x\n        return 123 if d else "hello"\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def f1() -> str: ...\n      def f2(x) -> Union[int, str]: ...\n    ')

    def test_dict_update(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f1():\n        d = {}\n        d.update({})\n        return 123 if d else "hello"\n\n      def f2():\n        d = {}\n        d.update({"a": 1})\n        return 123 if d else "hello"\n\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def f1() -> str: ...\n      def f2() -> int: ...\n    ')

    def test_dict_update_from_kwargs(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f1():\n        d = {}\n        d.update()\n        return 123 if d else "hello"\n\n      def f2():\n        d = {}\n        d.update(a=1)\n        return 123 if d else "hello"\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f1() -> str: ...\n      def f2() -> int: ...\n    ')

    def test_dict_update_wrong_count(self):
        if False:
            print('Hello World!')
        (ty, _) = self.InferWithErrors('\n      def f1():\n        d = {}\n        d.update({"a": 1}, {"b": 2})  # wrong-arg-count\n        return 123 if d else "hello"\n\n      def f2():\n        d = {}\n        d.update({"a": 1}, {"b": 2}, c=3)  # wrong-arg-count\n        return 123 if d else "hello"\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def f1() -> Union[str, int]: ...\n      def f2() -> Union[str, int]: ...\n    ')

    def test_dict_update_wrong_type(self):
        if False:
            i = 10
            return i + 15
        (ty, _) = self.InferWithErrors('\n      def f():\n        d = {}\n        d.update(1)  # wrong-arg-types\n        return 123 if d else "hello"\n    ')
        self.assertTypesMatchPytd(ty, '\n      def f() -> int | str: ...\n    ')

    def test_isinstance(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      # Always returns a bool.\n      def sig(x): return isinstance(x, str)\n      # Cases where isinstance() can be determined, if-split will\n      # narrow the return to a single type.\n      def d1(): return "y" if isinstance("s", str) else 0\n      def d2(): return "y" if isinstance("s", object) else 0\n      def d3(): return "y" if isinstance("s", int) else 0\n      def d4(): return "y" if isinstance("s", (float, str)) else 0\n      # Cases where isinstance() is ambiguous.\n      def a1(x): return "y" if isinstance(x, str) else 0\n      def a2(x):\n        cls = int if __random__ else str\n        return "y" if isinstance("a", cls) else 0\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def sig(x) -> bool: ...\n      def d1() -> str: ...\n      def d2() -> str: ...\n      def d3() -> int: ...\n      def d4() -> str: ...\n      def a1(x) -> Union[int, str]: ...\n      def a2(x) -> Union[int, str]: ...\n    ')

    def test_is_subclass(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      # Always return a bool\n      def sig(x): return issubclass(x, object)\n      # Classes for testing\n      class A: pass\n      class B(A): pass\n      class C: pass\n      # Check the if-splitting based on issubclass\n      def d1(): return "y" if issubclass(B, A) else 0\n      def d2(): return "y" if issubclass(B, object) else 0\n      def d3(): return "y" if issubclass(B, C) else 0\n      def d4(): return "y" if issubclass(B, (C, A)) else 0\n      def d5(): return "y" if issubclass(B, ((C, str), A, (int, object))) else 0\n      def d6(): return "y" if issubclass(B, ((C, str), int, (float, A))) else 0\n      # Ambiguous results\n      def a1(x): return "y" if issubclass(x, A) else 0\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def sig(x) -> bool: ...\n      def d1() -> str: ...\n      def d2() -> str: ...\n      def d3() -> int: ...\n      def d4() -> str: ...\n      def d5() -> str: ...\n      def d6() -> str: ...\n      def a1(x) -> Union[int, str]: ...\n\n      class A:\n        pass\n\n      class B(A):\n        pass\n\n      class C:\n        pass\n      ')

    def test_hasattr_builtin(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      # Always returns a bool.\n      def sig(x): return hasattr(x, "upper")\n      # Cases where hasattr() can be determined, if-split will\n      # narrow the return to a single type.\n      def d1(): return "y" if hasattr("s", "upper") else 0\n      def d2(): return "y" if hasattr("s", "foo") else 0\n      # We should follow the chain of superclasses\n      def d3(): return "y" if hasattr("s", "__repr__") else 0\n      # Cases where hasattr() is ambiguous.\n      def a1(x): return "y" if hasattr(x, "upper") else 0\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def sig(x) -> bool: ...\n      def d1() -> str: ...\n      def d2() -> int: ...\n      def d3() -> str: ...\n      def a1(x) -> Union[int, str]: ...\n    ')

    def test_split(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def f2(x):\n        if x:\n          return x\n        else:\n          return 3j\n\n      def f1(x):\n        y = 1 if x else 0\n        if y:\n          return f2(y)\n        else:\n          return None\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Any, Optional, TypeVar, Union\n      _T0 = TypeVar("_T0")\n      def f2(x: _T0) -> Union[_T0, complex]: ...\n      def f1(x) -> Optional[int]: ...\n    ')

    def test_dead_if(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def foo(x):\n        x = None\n        if x is not None:\n          x.foo()\n        return x\n    ')
        self.assertTypesMatchPytd(ty, '\n      def foo(x) -> None: ...\n    ')

    def test_unary_not(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      def not_t(x):\n        x = None\n        if not x:\n          return 1\n        else:\n          x.foo()\n          return "a"\n\n      def not_f(x):\n        x = True\n        if not x:\n          x.foo()\n          return 1\n        else:\n          return "a"\n\n      def not_ambiguous(x):\n        if not x:\n          return 1\n        else:\n          return "a"\n\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      def not_t(x) -> int: ...\n      def not_f(x) -> str: ...\n      def not_ambiguous(x) -> Union[int, str]: ...\n    ')

    def test_isinstance_object_without_class(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      def foo(x):\n        return 1 if isinstance(dict, type) else "x"\n    ')
        self.assertTypesMatchPytd(ty, '\n      def foo(x) -> int: ...\n    ')

    def test_double_assign(self):
        if False:
            return 10
        self.Check('\n      x = 1\n      x = None\n      if x is not None:\n        x.foo()\n    ')

    def test_infinite_loop(self):
        if False:
            i = 10
            return i + 15
        self.Check('\n      class A:\n        def __init__(self):\n          self.members = []\n        def add(self):\n          self.members.append(42)\n\n      class B:\n        def __init__(self):\n          self._map = {}\n        def _foo(self):\n          self._map[0] = A()\n          while True:\n            pass\n        def add2(self):\n          self._map[0].add()\n\n      b = B()\n      b._foo()\n      b.add2()\n    ')

    def test_dict_contains(self):
        if False:
            while True:
                i = 10
        'Assert that we can determine whether a dict contains a key.'
        self.Check('\n      d1 = {"x": 42}\n      if "x" in d1:\n        d1["x"]\n      else:\n        d1["nonsense"]  # Dead code\n\n      d2 = {}\n      if "x" in d2:\n        d2["nonsense"]  # Dead code\n\n      d3 = {__any_object__: __any_object__}\n      if "x" in d3:\n        d3["x"]\n      else:\n        d3["y"]\n    ')

    def test_dict_does_not_contain(self):
        if False:
            print('Hello World!')
        'Assert that we can determine whether a dict does not contain a key.'
        self.Check('\n      d1 = {"x": 42}\n      if "x" not in d1:\n        d1["nonsense"]  # Dead code\n      else:\n        d1["x"]\n\n      d2 = {}\n      if "x" not in d2:\n        pass\n      else:\n        d2["nonsense"]  # Dead code\n\n      d3 = {__any_object__: __any_object__}\n      if "x" not in d3:\n        d3["y"]\n      else:\n        d3["x"]\n    ')

    def test_dict_maybe_contains(self):
        if False:
            return 10
        'Test that we can handle more complex cases involving dict membership.'
        ty = self.Infer('\n      if __random__:\n        x = {"a": 1, "b": 2}\n      else:\n        x = {"b": 42j}\n      if "a" in x:\n        v1 = x["b"]\n      if "a" not in x:\n        v2 = x["b"]\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Union\n      x = ...  # type: Dict[str, Union[int, complex]]\n      v1 = ...  # type: int\n      v2 = ...  # type: complex\n    ')

    def test_contains_coerce_to_bool(self):
        if False:
            return 10
        ty = self.Infer('\n      class A:\n        def __contains__(self, x):\n          return 1\n      class B:\n        def __contains__(self, x):\n          return 0\n      x1 = "" if "a" in A() else u""\n      x2 = 3 if "a" not in A() else 42j\n      y1 = 3.14 if "b" in B() else 16j\n      y2 = True if "b" not in B() else 4.2\n    ')
        self.assertTypesMatchPytd(ty, '\n      class A:\n        def __contains__(self, x) -> int: ...\n      class B:\n        def __contains__(self, x) -> int: ...\n      x1 = ...  # type: str\n      x2 = ...  # type: complex\n      y1 = ...  # type: complex\n      y2 = ...  # type: bool\n    ')

    def test_skip_over_midway_if(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      def f(r):\n        y = "foo"\n        if __random__:\n          x = True\n        else:\n          x = False\n        if x:\n          return y\n        else:\n          return None\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Optional\n      def f(r) -> Optional[str]: ...\n    ')

    def test_dict_eq(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      if __random__:\n        x = {"a": 1}\n        z = 42\n      else:\n        x = {"b": 1}\n        z = 42j\n      y = {"b": 1}\n      if x == y:\n        v1 = z\n      if x != y:\n        v2 = z\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Union\n      x = ...  # type: Dict[str, int]\n      y = ...  # type: Dict[str, int]\n      z = ...  # type: Union[int, complex]\n      v1 = ...  # type: complex\n      v2 = ...  # type: Union[int, complex]\n    ')

    def test_tuple_eq(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      if __random__:\n        x = (1,)\n        z = ""\n      else:\n        x = (1, 2)\n        z = 3.14\n      y = (1, 2)\n      if x == y:\n        v1 = z\n      if x != y:\n        v2 = z\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Tuple, Union\n      x: Tuple[int, ...]\n      y: Tuple[int, int]\n      z: Union[str, float]\n      v1: float\n      v2: str\n    ')

    def test_primitive_eq(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      if __random__:\n        x = "a"\n        z = 42\n      else:\n        x = "b"\n        z = 3.14\n      y = "a"\n      if x == y:\n        v1 = z\n      if x != y:\n        v2 = z\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      x = ...  # type: str\n      y = ...  # type: str\n      z = ...  # type: Union[int, float]\n      v1 = ...  # type: int\n      v2 = ...  # type: float\n    ')

    def test_primitive_not_eq(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      x = "foo" if __random__ else 42\n      if x == "foo":\n        x.upper()\n    ')

    def test_builtin_full_name_check(self):
        if False:
            while True:
                i = 10
        self.InferWithErrors('\n      class int():\n        pass\n      x = "foo" if __random__ else int()\n      if x == "foo":\n        x.upper()  # attribute-error\n    ')

    def test_type_parameter_in_branch(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      if __random__:\n        x = {"a": 1, "b": 42}\n      else:\n        x = {"b": 42j}\n    ', deep=False)
        self.assertTypesMatchPytd(ty, '\n      from typing import Dict, Union\n      x = ...  # type: Dict[str, Union[int, complex]]\n    ')

    def test_none_or_tuple(self):
        if False:
            print('Hello World!')
        self.Check('\n      foo = (0, 0)\n      if __random__:\n        foo = None\n      if foo:\n        a, b = foo\n    ')

    def test_cmp_is_pytd_class(self):
        if False:
            while True:
                i = 10
        self.Check('\n      x = bool\n      if x is str:\n        name_error\n      if x is not bool:\n        name_error\n    ')

    def test_cmp_is_tuple_type(self):
        if False:
            while True:
                i = 10
        self.Check('\n      x = (1,)\n      y = (1, 2)\n      z = None  # type: type[tuple]\n      if type(x) is not type(y):\n        name_error\n      if type(x) is not z:\n        name_error\n    ')

    def test_cmp_is_function_type(self):
        if False:
            print('Hello World!')
        self.Check('\n      def f(): pass\n      def g(x): return x\n      if type(f) is not type(g):\n        name_error\n    ')

    def test_cmp_is_interpreter_class(self):
        if False:
            print('Hello World!')
        self.Check('\n      class X: pass\n      class Y: pass\n      if X is Y:\n        name_error\n      if X is not X:\n        name_error\n    ')

    def test_cmp_is_class_name_collision(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class X: ...\n      ')
            self.Check('\n        import foo\n        class X: pass\n        if foo.X is X:\n          name_error\n      ', pythonpath=[d.path])

    def test_get_iter(self):
        if False:
            while True:
                i = 10
        self.Check('\n      def f():\n        z = (1,2) if __random__ else None\n        if not z:\n          return\n          x, y = z\n    ')

    def test_list_comprehension(self):
        if False:
            return 10
        self.Check("\n      widgets = [None, 'hello']\n      wotsits = [x for x in widgets if x]\n      for x in wotsits:\n        x.upper()\n    ")

    def test_primitive(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Value(int):\n            pass\n        value1 = ...  # type: int\n        value2 = ...  # type: Value\n      ')
            self.CheckWithErrors('\n        import foo\n        if foo.value1 == foo.value2:\n          name_error  # name-error\n      ', pythonpath=[d.path])

    def test_list_element(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      def f():\n        x = None if __random__ else 42\n        return [x] if x else [42]\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import List\n      def f() -> List[int]: ...\n    ')

    def test_keep_constant(self):
        if False:
            while True:
                i = 10
        self.Check('\n      use_option = False\n      if use_option:\n        name_error\n    ')

    def test_function_and_class_truthiness(self):
        if False:
            for i in range(10):
                print('nop')
        self.Check('\n      def f(x):\n        return {} if x else []\n      def g():\n        return f(lambda: True).values()\n      def h():\n        return f(object).values()\n    ')

    def test_object_truthiness(self):
        if False:
            while True:
                i = 10
        ty = self.Infer('\n      x = object() and True\n    ')
        self.assertTypesMatchPytd(ty, '\n      x: bool\n    ')

    def test_override_len(self):
        if False:
            print('Hello World!')
        ty = self.Infer('\n      class A:\n        def __len__(self):\n          return 42\n\n      x = A() and True\n    ')
        self.assertTypesMatchPytd(ty, '\n      from typing import Union\n      class A:\n        def __len__(self) -> int: ...\n      x: Union[A, bool]\n    ')

    def test_container_loop(self):
        if False:
            print('Hello World!')
        self.Check("\n      from typing import Optional\n      def f(x):\n        # type: (Optional[str]) -> str\n        lst = []\n        if x:\n          lst.append(x)\n        for _ in range(5):\n          lst.append('hello')\n        return lst[0]\n    ")
if __name__ == '__main__':
    test_base.main()