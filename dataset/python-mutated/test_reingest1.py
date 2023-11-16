"""Tests for reloading generated pyi."""
from pytype.pytd import pytd_utils
from pytype.tests import test_base
from pytype.tests import test_utils

class ReingestTest(test_base.BaseTest):
    """Tests for reloading the pyi we generate."""

    def test_container(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n      class Container:\n        def Add(self):\n          pass\n      class A(Container):\n        pass\n    ', deep=False)
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(ty))
            self.Check('\n        # u.py\n        from foo import A\n        A().Add()\n      ', pythonpath=[d.path])

    def test_union(self):
        if False:
            for i in range(10):
                print('nop')
        ty = self.Infer('\n      class Union:\n        pass\n      x = {"Union": Union}\n    ', deep=False)
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(ty))
            self.Check('\n        from foo import Union\n      ', pythonpath=[d.path])

    def test_identity_decorators(self):
        if False:
            while True:
                i = 10
        foo = self.Infer('\n      def decorate(f):\n        return f\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            ty = self.Infer('\n        import foo\n        @foo.decorate\n        def f():\n          return 3\n        def g():\n          return f()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        def f() -> int: ...\n        def g() -> int: ...\n      ')

    @test_base.skip('Needs better handling of Union[Callable, f] in output.py.')
    def test_maybe_identity_decorators(self):
        if False:
            while True:
                i = 10
        foo = self.Infer('\n      def maybe_decorate(f):\n        return f or (lambda *args: 42)\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            ty = self.Infer('\n        import foo\n        @foo.maybe_decorate\n        def f():\n          return 3\n        def g():\n          return f()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        def f() -> int: ...\n        def g() -> int: ...\n      ')

    def test_namedtuple(self):
        if False:
            i = 10
            return i + 15
        foo = self.Infer('\n      import collections\n      X = collections.namedtuple("X", ["a", "b"])\n    ', deep=False)
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.Check('\n        import foo\n        foo.X(0, 0)\n        foo.X(a=0, b=0)\n      ', pythonpath=[d.path])

    def test_new_chain(self):
        if False:
            print('Hello World!')
        foo = self.Infer('\n      class X:\n        def __new__(cls, x):\n          return super(X, cls).__new__(cls)\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.Check('\n        import foo\n        class Y(foo.X):\n          def __new__(cls, x):\n            return super(Y, cls).__new__(cls, x)\n          def __init__(self, x):\n            self.x = x\n        Y("x").x\n      ', pythonpath=[d.path])

    def test_namedtuple_subclass(self):
        if False:
            i = 10
            return i + 15
        foo = self.Infer('\n      import collections\n      class X(collections.namedtuple("X", ["a"])):\n        def __new__(cls, a, b):\n          _ = b\n          return super(X, cls).__new__(cls, a)\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            (_, errors) = self.InferWithErrors('\n        import foo\n        foo.X("hello", "world")\n        foo.X(42)  # missing-parameter[e]\n      ', pythonpath=[d.path])
            self.assertErrorRegexes(errors, {'e': 'b.*__new__'})

    def test_alias(self):
        if False:
            print('Hello World!')
        foo = self.Infer('\n      class _Foo:\n        def __new__(cls, _):\n          return super(_Foo, cls).__new__(cls)\n      Foo = _Foo\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.Check('\n        import foo\n        foo.Foo("hello world")\n      ', pythonpath=[d.path])

    def test_dynamic_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        foo1 = self.Infer('\n      HAS_DYNAMIC_ATTRIBUTES = True\n    ')
        foo2 = self.Infer('\n      has_dynamic_attributes = True\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo1.pyi', pytd_utils.Print(foo1))
            d.create_file('foo2.pyi', pytd_utils.Print(foo2))
            d.create_file('bar.pyi', '\n        from foo1 import xyz\n        from foo2 import zyx\n      ')
            self.Check('\n        import foo1\n        import foo2\n        import bar\n        foo1.abc\n        foo2.abc\n        bar.xyz\n        bar.zyx\n      ', pythonpath=[d.path])

    def test_inherited_mutation(self):
        if False:
            i = 10
            return i + 15
        foo = self.Infer('\n      class MyList(list):\n        write = list.append\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            ty = self.Infer('\n        import foo\n        lst = foo.MyList()\n        lst.write(42)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        lst = ...  # type: foo.MyList\n      ')

    @test_base.skip('Need to give MyList.write the right self mutation.')
    def test_inherited_mutation_in_generic_class(self):
        if False:
            for i in range(10):
                print('nop')
        foo = self.Infer('\n      from typing import List, TypeVar\n      T = TypeVar("T")\n      class MyList(List[T]):\n        write = list.append\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            ty = self.Infer('\n        import foo\n        lst = foo.MyList()\n        lst.write(42)\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        lst = ...  # type: foo.MyList[int]\n      ')

    def test_instantiate_imported_generic(self):
        if False:
            for i in range(10):
                print('nop')
        foo = self.Infer("\n      from typing import Generic, TypeVar\n      T = TypeVar('T')\n      class Foo(Generic[T]):\n        def __init__(self):\n          pass\n    ")
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            ty = self.Infer('\n        import foo\n        x = foo.Foo[int]()\n      ', pythonpath=[d.path])
            self.assertTypesMatchPytd(ty, '\n        import foo\n        x: foo.Foo[int]\n      ')

class StrictNoneTest(test_base.BaseTest):
    """Tests for strict none."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.options.tweak(strict_none_binding=False)

    def test_pyi_return_constant(self):
        if False:
            for i in range(10):
                print('nop')
        foo = self.Infer('\n      x = None\n      def f():\n        return x\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.Check('\n        import foo\n        def g():\n          return foo.f().upper()\n      ', pythonpath=[d.path])

    def test_pyi_yield_constant(self):
        if False:
            i = 10
            return i + 15
        foo = self.Infer('\n      x = None\n      def f():\n        yield x\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.Check('\n        import foo\n        def g():\n          return [v.upper() for v in foo.f()]\n      ', pythonpath=[d.path])

    def test_pyi_return_contained_constant(self):
        if False:
            while True:
                i = 10
        foo = self.Infer('\n      x = None\n      def f():\n        return [x]\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.Check('\n        import foo\n        def g():\n          return [v.upper() for v in foo.f()]\n      ', pythonpath=[d.path])

    def test_pyi_return_attribute(self):
        if False:
            i = 10
            return i + 15
        foo = self.Infer('\n      class Foo:\n        x = None\n      def f():\n        return Foo.x\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.Check('\n        import foo\n        def g():\n          return foo.f().upper()\n      ', pythonpath=[d.path])

    def test_no_return(self):
        if False:
            while True:
                i = 10
        foo = self.Infer('\n      def fail():\n        raise ValueError()\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.Check('\n        import foo\n        def g():\n          x = "hello" if __random__ else None\n          if x is None:\n            foo.fail()\n          return x.upper()\n      ', pythonpath=[d.path])

    def test_context_manager_subclass(self):
        if False:
            for i in range(10):
                print('nop')
        foo = self.Infer('\n      class Foo:\n        def __enter__(self):\n          return self\n        def __exit__(self, type, value, traceback):\n          return None\n    ')
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', pytd_utils.Print(foo))
            self.Check('\n        import foo\n        class Bar(foo.Foo):\n          x = None\n        with Bar() as bar:\n          bar.x\n      ', pythonpath=[d.path])
if __name__ == '__main__':
    test_base.main()