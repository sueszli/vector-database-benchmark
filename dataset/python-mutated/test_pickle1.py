"""Tests for loading and saving pickled files."""
import pickle
from pytype.imports import pickle_utils
from pytype.pytd import visitors
from pytype.tests import test_base
from pytype.tests import test_utils

class PickleTest(test_base.BaseTest):
    """Tests for loading and saving pickled files."""

    def _verifyDeps(self, module, immediate_deps, late_deps):
        if False:
            print('Hello World!')
        if isinstance(module, bytes):
            data = pickle.loads(module)
            self.assertCountEqual(dict(data.dependencies), immediate_deps)
            self.assertCountEqual(dict(data.late_dependencies), late_deps)
        else:
            c = visitors.CollectDependencies()
            module.Visit(c)
            self.assertCountEqual(c.dependencies, immediate_deps)
            self.assertCountEqual(c.late_dependencies, late_deps)

    def test_type(self):
        if False:
            while True:
                i = 10
        pickled = self.Infer('\n      x = type\n    ', deep=False, pickle=True, module_name='foo')
        with test_utils.Tempdir() as d:
            u = d.create_file('u.pickled', pickled)
            ty = self.Infer('\n        import u\n        r = u.x\n      ', deep=False, pythonpath=[''], imports_map={'u': u})
            self.assertTypesMatchPytd(ty, '\n        import u\n        from typing import Type\n        r = ...  # type: Type[type]\n      ')

    def test_copy_class_into_output(self):
        if False:
            print('Hello World!')
        pickled_foo = self.Infer('\n      import datetime\n      a = 42\n      timedelta = datetime.timedelta  # copy class\n    ', deep=False, pickle=True, module_name='foo')
        self._verifyDeps(pickled_foo, ['builtins'], ['datetime'])
        with test_utils.Tempdir() as d:
            foo = d.create_file('foo.pickled', pickled_foo)
            pickled_bar = self.Infer('\n        import foo\n        timedelta = foo.timedelta  # copy class\n      ', pickle=True, pythonpath=[''], imports_map={'foo': foo}, module_name='bar')
            self._verifyDeps(pickled_bar, ['builtins'], ['datetime'])
            bar = d.create_file('bar.pickled', pickled_bar)
            ty = self.Infer('\n        import bar\n        r = bar.timedelta(0)\n      ', deep=False, pythonpath=[''], imports_map={'foo': foo, 'bar': bar})
            self._verifyDeps(ty, ['datetime'], [])
            self.assertTypesMatchPytd(ty, '\n        import datetime\n        import bar\n        r = ...  # type: datetime.timedelta\n      ')

    def test_optimize_on_late_types(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            pickled_foo = self.Infer('\n        class X: pass\n      ', deep=False, pickle=True, module_name='foo')
            self._verifyDeps(pickled_foo, ['builtins'], [])
            foo = d.create_file('foo.pickled', pickled_foo)
            pickled_bar = self.Infer('\n        import foo\n        def f():\n          return foo.X()\n      ', pickle=True, pythonpath=[''], imports_map={'foo': foo}, module_name='bar', deep=True)
            bar = d.create_file('bar.pickled', pickled_bar)
            self._verifyDeps(pickled_bar, [], ['foo'])
            self.Infer('\n        import bar\n        f = bar.f\n      ', deep=False, imports_map={'foo': foo, 'bar': bar})

    def test_file_change(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            pickled_xy = self.Infer('\n        class X: pass\n        class Y: pass\n      ', deep=False, pickle=True, module_name='foo')
            pickled_x = self.Infer('\n        class X: pass\n      ', deep=False, pickle=True, module_name='foo')
            foo = d.create_file('foo.pickled', pickled_xy)
            pickled_bar = self.Infer('\n        import foo\n        class A(foo.X): pass\n        class B(foo.Y): pass\n      ', deep=False, pickle=True, imports_map={'foo': foo}, module_name='bar')
            self._verifyDeps(pickled_bar, [], ['foo'])
            bar = d.create_file('bar.pickled', pickled_bar)
            foo = d.create_file('foo.pickled', pickled_x)
            self.Infer('\n        import bar\n        a = bar.A()\n        b = bar.B()\n      ', deep=False, imports_map={'foo': foo, 'bar': bar})
            d.delete_file('foo.pickled')
            self.Infer('\n        import bar\n        a = bar.A()\n        b = bar.B()\n      ', deep=False, imports_map={'foo': foo, 'bar': bar})

    def test_file_rename(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            pickled_other_foo = self.Infer('\n        class Foo: pass\n      ', deep=False, pickle=True, module_name='bar')
            other_foo = d.create_file('empty.pickled', pickled_other_foo)
            pickled_foo = self.Infer('\n        class Foo:\n          def __init__(self): pass\n        x = Foo()\n      ', deep=False, pickle=True, module_name='foo')
            foo = d.create_file('foo.pickled', pickled_foo)
            self.Infer('\n        import bar\n        bar.Foo()\n      ', pickle=True, imports_map={'bar': foo, 'foo': other_foo}, module_name='baz')

    def test_optimize(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            pickled_foo = self._PickleSource('\n        import UserDict\n        class Foo: ...\n        @overload\n        def f(self, x: Foo, y: UserDict.UserDict): ...\n        @overload\n        def f(self, x: UserDict.UserDict, y: Foo): ...\n      ', module_name='foo')
            self._verifyDeps(pickled_foo, ['builtins', 'foo'], ['UserDict'])
            foo = d.create_file('foo.pickled', pickled_foo)
            self.assertNoCrash(self.Infer, '\n        import foo\n        class Bar:\n          f = foo.f\n      ', imports_map={'foo': foo}, module_name='bar')

    def test_function_type(self):
        if False:
            return 10
        self.ConfigureOptions(module_name='bar', pythonpath=[''], use_pickled_files=True)
        pickled_foo = self._PickleSource('\n        import UserDict\n        def f(x: UserDict.UserDict) -> None: ...\n      ', module_name='foo')
        with test_utils.Tempdir() as d:
            foo = d.create_file('foo.pickled', pickled_foo)
            self.options.tweak(imports_map={'foo': foo})
            pickled_bar = self._PickleSource('\n        from foo import f  # Alias(name="f", type=Function("foo.f", ...))\n      ', module_name='bar')
            bar = d.create_file('bar.pickled', pickled_bar)
            self.assertNoCrash(self.Infer, '\n        import bar\n        bar.f(42)\n      ', imports_map={'foo': foo, 'bar': bar}, module_name='baz')

    def test_class_decorator(self):
        if False:
            print('Hello World!')
        foo = '\n      from typing_extensions import final\n      @final\n      class A:\n        def f(self): ...\n    '
        with self.DepTree([('foo.py', foo, {'pickle': True})]):
            self.CheckWithErrors('\n        import foo\n        class B(foo.A):  # final-error\n          pass\n      ')

    def test_exception(self):
        if False:
            for i in range(10):
                print('nop')
        old = pickle.load

        def load_with_error(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            raise ValueError('error!')
        foo = '\n      class A: pass\n    '
        pickle.load = load_with_error
        with self.DepTree([('foo.py', foo, {'pickle': True})]):
            with self.assertRaises(pickle_utils.LoadPickleError):
                self.Check('\n          import foo\n          x = foo.A()\n        ')
        pickle.load = old
if __name__ == '__main__':
    test_base.main()