import re
import textwrap
from pytype import config
from pytype import load_pytd
from pytype.pytd import optimize
from pytype.pytd import pytd
from pytype.pytd import pytd_utils
from pytype.pytd import visitors
from pytype.pytd.parse import parser_test_base
import unittest

def pytd_src(text):
    if False:
        return 10
    'Add a typing.Union import if needed.'
    text = textwrap.dedent(text)
    if 'Union' in text and (not re.search('typing.*Union', text)):
        return 'from typing import Union\n' + text
    else:
        return text

class TestOptimize(parser_test_base.ParserTest):
    """Test the visitors in optimize.py."""

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super().setUpClass()
        cls.loader = load_pytd.Loader(config.Options.create(python_version=cls.python_version))
        cls.builtins = cls.loader.builtins
        cls.typing = cls.loader.typing

    def ParseAndResolve(self, src):
        if False:
            while True:
                i = 10
        ast = self.Parse(src)
        return ast.Visit(visitors.LookupBuiltins(self.builtins))

    def Optimize(self, ast, **kwargs):
        if False:
            return 10
        return optimize.Optimize(ast, self.builtins, **kwargs)

    def OptimizedString(self, data):
        if False:
            while True:
                i = 10
        tree = self.Parse(data) if isinstance(data, str) else data
        new_tree = self.Optimize(tree)
        return pytd_utils.Print(new_tree)

    def AssertOptimizeEquals(self, src, new_src):
        if False:
            print('Hello World!')
        self.AssertSourceEquals(self.OptimizedString(src), new_src)

    def test_one_function(self):
        if False:
            i = 10
            return i + 15
        src = pytd_src('\n        def foo(a: int, c: bool) -> int:\n          raise AssertionError()\n          raise ValueError()\n    ')
        self.AssertOptimizeEquals(src, src)

    def test_function_duplicate(self):
        if False:
            print('Hello World!')
        src = pytd_src('\n        def foo(a: int, c: bool) -> int:\n          raise AssertionError()\n          raise ValueError()\n        def foo(a: int, c: bool) -> int:\n          raise AssertionError()\n          raise ValueError()\n    ')
        new_src = pytd_src('\n        def foo(a: int, c: bool) -> int:\n          raise AssertionError()\n          raise ValueError()\n    ')
        self.AssertOptimizeEquals(src, new_src)

    def test_complex_function_duplicate(self):
        if False:
            return 10
        src = pytd_src('\n        def foo(a: Union[int, float], c: bool) -> list[int]:\n          raise IndexError()\n        def foo(a: str, c: str) -> str: ...\n        def foo(a: int, *args) -> Union[int, float]:\n          raise ValueError()\n        def foo(a: Union[int, float], c: bool) -> list[int]:\n          raise IndexError()\n        def foo(a: int, *args) -> Union[int, float]:\n          raise ValueError()\n    ')
        new_src = pytd_src('\n        def foo(a: float, c: bool) -> list[int]:\n          raise IndexError()\n        def foo(a: str, c: str) -> str: ...\n        def foo(a: int, *args) -> Union[int, float]:\n          raise ValueError()\n    ')
        self.AssertOptimizeEquals(src, new_src)

    def test_combine_returns(self):
        if False:
            print('Hello World!')
        src = pytd_src('\n        def foo(a: int) -> int: ...\n        def foo(a: int) -> float: ...\n    ')
        new_src = pytd_src('\n        def foo(a: int) -> Union[int, float]: ...\n    ')
        self.AssertOptimizeEquals(src, new_src)

    def test_combine_redundant_returns(self):
        if False:
            return 10
        src = pytd_src('\n        def foo(a: int) -> int: ...\n        def foo(a: int) -> float: ...\n        def foo(a: int) -> Union[int, float]: ...\n    ')
        new_src = pytd_src('\n        def foo(a: int) -> Union[int, float]: ...\n    ')
        self.AssertOptimizeEquals(src, new_src)

    def test_combine_union_returns(self):
        if False:
            i = 10
            return i + 15
        src = pytd_src('\n        def foo(a: int) -> Union[int, float]: ...\n        def bar(a: str) -> str: ...\n        def foo(a: int) -> Union[str, bytes]: ...\n    ')
        new_src = pytd_src('\n        def foo(a: int) -> Union[int, float, str, bytes]: ...\n        def bar(a: str) -> str: ...\n    ')
        self.AssertOptimizeEquals(src, new_src)

    def test_combine_exceptions(self):
        if False:
            i = 10
            return i + 15
        src = pytd_src('\n        def foo(a: int) -> int:\n          raise ValueError()\n        def foo(a: int) -> int:\n          raise IndexError()\n        def foo(a: float) -> int:\n          raise IndexError()\n        def foo(a: int) -> int:\n          raise AttributeError()\n    ')
        new_src = pytd_src('\n        def foo(a: int) -> int:\n          raise ValueError()\n          raise IndexError()\n          raise AttributeError()\n        def foo(a: float) -> int:\n          raise IndexError()\n    ')
        self.AssertOptimizeEquals(src, new_src)

    def test_mixed_combine(self):
        if False:
            return 10
        src = pytd_src('\n        def foo(a: int) -> int:\n          raise ValueError()\n        def foo(a: int) -> float:\n          raise ValueError()\n        def foo(a: int) -> int:\n          raise IndexError()\n    ')
        new_src = pytd_src('\n        def foo(a: int) -> Union[int, float]:\n          raise ValueError()\n          raise IndexError()\n    ')
        self.AssertOptimizeEquals(src, new_src)

    def test_lossy(self):
        if False:
            print('Hello World!')
        src = pytd_src('\n        def foo(a: int) -> float:\n          raise IndexError()\n        def foo(a: str) -> complex:\n          raise AssertionError()\n    ')
        optimized = self.Optimize(self.Parse(src), lossy=True, use_abcs=False)
        self.AssertSourceEquals(optimized, src)

    @unittest.skip('Needs ABCs to be included in the builtins')
    def test_abcs(self):
        if False:
            return 10
        src = pytd_src('\n        def foo(a: Union[int, float]) -> NoneType: ...\n        def foo(a: Union[int, complex, float]) -> NoneType: ...\n    ')
        new_src = pytd_src('\n        def foo(a: Real) -> NoneType: ...\n        def foo(a: Complex) -> NoneType: ...\n    ')
        optimized = self.Optimize(self.Parse(src), lossy=True, use_abcs=True)
        self.AssertSourceEquals(optimized, new_src)

    def test_duplicates_in_unions(self):
        if False:
            while True:
                i = 10
        src = pytd_src('\n      def a(x: Union[int, float, complex]) -> bool: ...\n      def b(x: Union[int, float]) -> bool: ...\n      def c(x: Union[int, int, int]) -> bool: ...\n      def d(x: Union[int, int]) -> bool: ...\n      def e(x: Union[float, int, int, float]) -> bool: ...\n      def f(x: Union[float, int]) -> bool: ...\n    ')
        new_src = pytd_src('\n      def a(x) -> builtins.bool: ...  # max_union=2 makes this object\n      def b(x: Union[builtins.int, builtins.float]) -> builtins.bool: ...\n      def c(x: builtins.int) -> builtins.bool: ...\n      def d(x: builtins.int) -> builtins.bool: ...\n      def e(x: Union[builtins.float, builtins.int]) -> builtins.bool: ...\n      def f(x: Union[builtins.float, builtins.int]) -> builtins.bool: ...\n    ')
        ast = self.ParseAndResolve(src)
        optimized = self.Optimize(ast, lossy=False, max_union=2)
        self.AssertSourceEquals(optimized, new_src)

    def test_simplify_unions(self):
        if False:
            return 10
        src = pytd_src('\n      from typing import Any\n      a = ...  # type: Union[int, int]\n      b = ...  # type: Union[int, Any]\n      c = ...  # type: Union[int, int, float]\n    ')
        new_src = pytd_src('\n      from typing import Any\n      a = ...  # type: int\n      b = ...  # type: Any\n      c = ...  # type: Union[int, float]\n    ')
        self.AssertSourceEquals(self.ApplyVisitorToString(src, optimize.SimplifyUnions()), new_src)

    def test_builtin_superclasses(self):
        if False:
            for i in range(10):
                print('nop')
        src = pytd_src('\n        def f(x: Union[list, object], y: Union[complex, slice]) -> Union[int, bool]: ...\n    ')
        expected = pytd_src('\n        def f(x: builtins.object, y: builtins.object) -> builtins.int: ...\n    ')
        hierarchy = self.builtins.Visit(visitors.ExtractSuperClassesByName())
        hierarchy.update(self.typing.Visit(visitors.ExtractSuperClassesByName()))
        visitor = optimize.FindCommonSuperClasses(optimize.SuperClassHierarchy(hierarchy))
        ast = self.ParseAndResolve(src)
        ast = ast.Visit(visitor)
        ast = ast.Visit(visitors.CanonicalOrderingVisitor())
        self.AssertSourceEquals(ast, expected)

    def test_user_superclass_hierarchy(self):
        if False:
            for i in range(10):
                print('nop')
        class_data = pytd_src('\n        class AB:\n            pass\n\n        class EFG:\n            pass\n\n        class A(AB, EFG):\n            pass\n\n        class B(AB):\n            pass\n\n        class E(EFG, AB):\n            pass\n\n        class F(EFG):\n            pass\n\n        class G(EFG):\n            pass\n    ')
        src = pytd_src('\n        from typing import Any\n        def f(x: Union[A, B], y: A, z: B) -> Union[E, F, G]: ...\n        def g(x: Union[E, F, G, B]) -> Union[E, F]: ...\n        def h(x) -> Any: ...\n    ') + class_data
        expected = pytd_src('\n        from typing import Any\n        def f(x: AB, y: A, z: B) -> EFG: ...\n        def g(x: object) -> EFG: ...\n        def h(x) -> Any: ...\n    ') + class_data
        hierarchy = self.Parse(src).Visit(visitors.ExtractSuperClassesByName())
        visitor = optimize.FindCommonSuperClasses(optimize.SuperClassHierarchy(hierarchy))
        new_src = self.ApplyVisitorToString(src, visitor)
        self.AssertSourceEquals(new_src, expected)

    def test_find_common_superclasses(self):
        if False:
            print('Hello World!')
        src = pytd_src('\n        x = ...  # type: Union[int, other.Bar]\n    ')
        expected = pytd_src('\n        x = ...  # type: Union[int, other.Bar]\n    ')
        ast = self.Parse(src)
        ast = ast.Visit(visitors.ReplaceTypesByName({'other.Bar': pytd.LateType('other.Bar')}))
        hierarchy = ast.Visit(visitors.ExtractSuperClassesByName())
        ast = ast.Visit(optimize.FindCommonSuperClasses(optimize.SuperClassHierarchy(hierarchy)))
        ast = ast.Visit(visitors.LateTypeToClassType())
        self.AssertSourceEquals(ast, expected)

    def test_simplify_unions_with_superclasses(self):
        if False:
            while True:
                i = 10
        src = pytd_src('\n        x = ...  # type: Union[int, bool]\n        y = ...  # type: Union[int, bool, float]\n        z = ...  # type: Union[list[int], int]\n    ')
        expected = pytd_src('\n        x = ...  # type: int\n        y = ...  # type: Union[int, float]\n        z = ...  # type: Union[list[int], int]\n    ')
        hierarchy = self.builtins.Visit(visitors.ExtractSuperClassesByName())
        visitor = optimize.SimplifyUnionsWithSuperclasses(optimize.SuperClassHierarchy(hierarchy))
        ast = self.Parse(src)
        ast = visitors.LookupClasses(ast, self.builtins)
        ast = ast.Visit(visitor)
        self.AssertSourceEquals(ast, expected)

    @unittest.skip('Needs better handling of GenericType')
    def test_simplify_unions_with_superclasses_generic(self):
        if False:
            while True:
                i = 10
        src = pytd_src('\n        x = ...  # type: Union[frozenset[int], AbstractSet[int]]\n    ')
        expected = pytd_src('\n        x = ...  # type: AbstractSet[int]\n    ')
        hierarchy = self.builtins.Visit(visitors.ExtractSuperClassesByName())
        visitor = optimize.SimplifyUnionsWithSuperclasses(optimize.SuperClassHierarchy(hierarchy))
        ast = self.Parse(src)
        ast = visitors.LookupClasses(ast, self.builtins)
        ast = ast.Visit(visitor)
        self.AssertSourceEquals(ast, expected)

    def test_collapse_long_unions(self):
        if False:
            while True:
                i = 10
        src = pytd_src('\n        from typing import Any\n        def f(x: Union[A, B, C, D]) -> X: ...\n        def g(x: Union[A, B, C, D, E]) -> X: ...\n        def h(x: Union[A, Any]) -> X: ...\n    ')
        expected = pytd_src('\n        def f(x: Union[A, B, C, D]) -> X: ...\n        def g(x) -> X: ...\n        def h(x) -> X: ...\n    ')
        ast = self.ParseAndResolve(src)
        ast = ast.Visit(optimize.CollapseLongUnions(max_length=4))
        self.AssertSourceEquals(ast, expected)

    def test_collapse_long_constant_unions(self):
        if False:
            while True:
                i = 10
        src = pytd_src('\n      x = ...  # type: Union[A, B, C, D]\n      y = ...  # type: Union[A, B, C, D, E]\n    ')
        expected = pytd_src('\n      from typing import Any\n      x = ...  # type: Union[A, B, C, D]\n      y = ...  # type: Any\n    ')
        ast = self.ParseAndResolve(src)
        ast = ast.Visit(optimize.CollapseLongUnions(max_length=4))
        ast = ast.Visit(optimize.AdjustReturnAndConstantGenericType())
        self.AssertSourceEquals(ast, expected)

    def test_combine_containers(self):
        if False:
            while True:
                i = 10
        src = pytd_src('\n        from typing import Any\n        def f(x: Union[list[int], list[float]]) -> Any: ...\n        def g(x: Union[list[int], str, list[float], set[int], long]) -> Any: ...\n        def h(x: Union[list[int], list[str], set[int], set[float]]) -> Any: ...\n        def i(x: Union[list[int], list[int]]) -> Any: ...\n        def j(x: Union[dict[int, float], dict[float, int]]) -> Any: ...\n        def k(x: Union[dict[int, bool], list[int], dict[bool, int], list[bool]]) -> Any: ...\n    ')
        expected = pytd_src('\n        from typing import Any\n        def f(x: list[float]) -> Any: ...\n        def g(x: Union[list[float], str, set[int], long]) -> Any: ...\n        def h(x: Union[list[Union[int, str]], set[float]]) -> Any: ...\n        def i(x: list[int]) -> Any: ...\n        def j(x: dict[float, float]) -> Any: ...\n        def k(x: Union[dict[Union[int, bool], Union[bool, int]], list[Union[int, bool]]]) -> Any: ...\n    ')
        new_src = self.ApplyVisitorToString(src, optimize.CombineContainers())
        self.AssertSourceEquals(new_src, expected)

    def test_combine_containers_multi_level(self):
        if False:
            for i in range(10):
                print('nop')
        src = pytd_src('\n      v = ...  # type: Union[list[tuple[Union[long, int], ...]], list[tuple[Union[float, bool], ...]]]\n    ')
        expected = pytd_src('\n      v = ...  # type: list[tuple[Union[long, int, float, bool], ...]]\n    ')
        new_src = self.ApplyVisitorToString(src, optimize.CombineContainers())
        self.AssertSourceEquals(new_src, expected)

    def test_combine_same_length_tuples(self):
        if False:
            for i in range(10):
                print('nop')
        src = pytd_src('\n      x = ...  # type: Union[tuple[int], tuple[str]]\n    ')
        expected = pytd_src('\n      x = ...  # type: tuple[Union[int, str]]\n    ')
        new_src = self.ApplyVisitorToString(src, optimize.CombineContainers())
        self.AssertSourceEquals(new_src, expected)

    def test_combine_different_length_tuples(self):
        if False:
            return 10
        src = pytd_src('\n      x = ...  # type: Union[tuple[int], tuple[int, str]]\n    ')
        expected = pytd_src('\n      x = ...  # type: tuple[Union[int, str], ...]\n    ')
        new_src = self.ApplyVisitorToString(src, optimize.CombineContainers())
        self.AssertSourceEquals(new_src, expected)

    def test_combine_different_length_callables(self):
        if False:
            i = 10
            return i + 15
        src = pytd_src('\n      from typing import Callable\n      x = ...  # type: Union[Callable[[int], str], Callable[[int, int], str]]\n    ')
        expected = pytd_src('\n      from typing import Callable\n      x = ...  # type: Callable[..., str]\n    ')
        new_src = self.ApplyVisitorToString(src, optimize.CombineContainers())
        self.AssertSourceEquals(new_src, expected)

    def test_pull_in_method_classes(self):
        if False:
            for i in range(10):
                print('nop')
        src = pytd_src('\n        from typing import Any\n        class A:\n            mymethod1 = ...  # type: Method1\n            mymethod2 = ...  # type: Method2\n            member = ...  # type: Method3\n            mymethod4 = ...  # type: Method4\n        class Method1:\n            def __call__(self: A, x: int) -> Any: ...\n        class Method2:\n            def __call__(self: object, x: int) -> Any: ...\n        class Method3:\n            def __call__(x: bool, y: int) -> Any: ...\n        class Method4:\n            def __call__(self: Any) -> Any: ...\n        class B(Method4):\n            pass\n    ')
        expected = pytd_src('\n        from typing import Any\n        class A:\n            member = ...  # type: Method3\n            def mymethod1(self, x: int) -> Any: ...\n            def mymethod2(self, x: int) -> Any: ...\n            def mymethod4(self) -> Any: ...\n\n        class Method3:\n            def __call__(x: bool, y: int) -> Any: ...\n\n        class Method4:\n            def __call__(self) -> Any: ...\n\n        class B(Method4):\n            pass\n    ')
        new_src = self.ApplyVisitorToString(src, optimize.PullInMethodClasses())
        self.AssertSourceEquals(new_src, expected)

    def test_add_inherited_methods(self):
        if False:
            return 10
        src = pytd_src('\n        from typing import Any\n        class A():\n            foo = ...  # type: bool\n            def f(self, x: int) -> float: ...\n            def h(self) -> complex: ...\n\n        class B(A):\n            bar = ...  # type: int\n            def g(self, y: int) -> bool: ...\n            def h(self, z: float) -> Any: ...\n    ')
        ast = self.Parse(src)
        ast = visitors.LookupClasses(ast, self.builtins)
        self.assertCountEqual(('g', 'h'), [m.name for m in ast.Lookup('B').methods])
        ast = ast.Visit(optimize.AddInheritedMethods())
        self.assertCountEqual(('f', 'g', 'h'), [m.name for m in ast.Lookup('B').methods])

    def test_adjust_inherited_method_self(self):
        if False:
            for i in range(10):
                print('nop')
        src = pytd_src('\n      class A():\n        def f(self: object) -> float: ...\n      class B(A):\n        pass\n    ')
        ast = self.Parse(src)
        ast = visitors.LookupClasses(ast, self.builtins)
        ast = ast.Visit(optimize.AddInheritedMethods())
        self.assertMultiLineEqual(pytd_utils.Print(ast.Lookup('B')), pytd_src('\n        class B(A):\n            def f(self) -> float: ...\n    ').lstrip())

    def test_absorb_mutable_parameters(self):
        if False:
            i = 10
            return i + 15
        src = pytd_src('\n        from typing import Any\n        def popall(x: list[Any]) -> Any:\n            x = list[nothing]\n        def add_float(x: list[int]) -> Any:\n            x = list[Union[int, float]]\n        def f(x: list[int]) -> Any:\n            x = list[Union[int, float]]\n    ')
        expected = pytd_src('\n        from typing import Any\n        def popall(x: list[Any]) -> Any: ...\n        def add_float(x: list[Union[int, float]]) -> Any: ...\n        def f(x: list[Union[int, float]]) -> Any: ...\n    ')
        tree = self.Parse(src)
        new_tree = tree.Visit(optimize.AbsorbMutableParameters())
        new_tree = new_tree.Visit(optimize.CombineContainers())
        self.AssertSourceEquals(new_tree, expected)

    def test_absorb_mutable_parameters_from_methods(self):
        if False:
            while True:
                i = 10
        src = pytd_src("\n        from typing import Any\n        T = TypeVar('T')\n        NEW = TypeVar('NEW')\n        class MyClass(typing.Generic[T], object):\n            def append(self, x: NEW) -> Any:\n                self = MyClass[Union[T, NEW]]\n    ")
        tree = self.Parse(src)
        new_tree = tree.Visit(optimize.AbsorbMutableParameters())
        new_tree = new_tree.Visit(optimize.CombineContainers())
        self_type = new_tree.Lookup('MyClass').Lookup('append').signatures[0].params[0].type
        self.assertEqual(pytd_utils.Print(self_type), 'MyClass[Union[T, NEW]]')

    def test_merge_type_parameters(self):
        if False:
            print('Hello World!')
        src = pytd_src("\n      from typing import Any\n      T = TypeVar('T')\n      T2 = TypeVar('T2')\n      T3 = TypeVar('T3')\n      class A(typing.Generic[T], object):\n          def foo(self, x: Union[T, T2]) -> T2: ...\n          def bar(self, x: Union[T, T2, T3]) -> T3: ...\n          def baz(self, x: Union[T, T2], y: Union[T2, T3]) -> Any: ...\n\n      K = TypeVar('K')\n      V = TypeVar('V')\n      class D(typing.Generic[K, V], object):\n          def foo(self, x: T) -> Union[K, T]: ...\n          def bar(self, x: T) -> Union[V, T]: ...\n          def baz(self, x: Union[K, V]) -> Union[K, V]: ...\n          def lorem(self, x: T) -> Union[T, K, V]: ...\n          def ipsum(self, x: T) -> Union[T, K]: ...\n    ")
        expected = pytd_src("\n      from typing import Any\n      T = TypeVar('T')\n      T2 = TypeVar('T2')\n      T3 = TypeVar('T3')\n      class A(typing.Generic[T], object):\n          def foo(self, x: T) -> T: ...\n          def bar(self, x: T) -> T: ...\n          def baz(self, x: T, y: T) -> Any: ...\n\n      K = TypeVar('K')\n      V = TypeVar('V')\n      class D(typing.Generic[K, V], object):\n          def foo(self, x: K) -> K: ...\n          def bar(self, x: V) -> V: ...\n          def baz(self, x: Union[K, V]) -> Union[K, V]: ...\n          def lorem(self, x: Union[K, V]) -> Union[K, V]: ...\n          def ipsum(self, x: K) -> K: ...\n      ")
        tree = self.Parse(src)
        new_tree = tree.Visit(optimize.MergeTypeParameters())
        self.AssertSourceEquals(new_tree, expected)

    def test_overloads_not_flattened(self):
        if False:
            for i in range(10):
                print('nop')
        src = pytd_src('\n      from typing import overload\n      @overload\n      def f(x: int) -> str: ...\n      @overload\n      def f(x: str) -> str: ...\n    ')
        self.AssertOptimizeEquals(src, src)
if __name__ == '__main__':
    unittest.main()