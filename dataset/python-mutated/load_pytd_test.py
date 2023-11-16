"""Tests for load_pytd.py."""
import contextlib
import dataclasses
import io
import os
import sys
import textwrap
from pytype import config
from pytype import file_utils
from pytype import load_pytd
from pytype import module_utils
from pytype.imports import pickle_utils
from pytype.platform_utils import path_utils
from pytype.pytd import pytd
from pytype.pytd import pytd_utils
from pytype.pytd import visitors
from pytype.tests import test_base
from pytype.tests import test_utils
import unittest

class ModuleTest(test_base.UnitTest):
    """Tests for load_pytd.Module."""

    def test_is_package(self):
        if False:
            while True:
                i = 10
        for (filename, is_package) in [('foo/bar.pyi', False), ('foo/__init__.pyi', True), ('foo/__init__.pyi-1', True), ('foo/__init__.pickled', True), (os.devnull, True)]:
            with self.subTest(filename=filename):
                mod = load_pytd.Module(module_name=None, filename=filename, ast=None)
                self.assertEqual(mod.is_package(), is_package)

class _LoaderTest(test_base.UnitTest):

    @contextlib.contextmanager
    def _setup_loader(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            for (name, contents) in kwargs.items():
                d.create_file(f'{name}.pyi', contents)
            yield load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=d.path))

    def _import(self, **kwargs):
        if False:
            i = 10
            return i + 15
        with self._setup_loader(**kwargs) as loader:
            return loader.import_name(kwargs.popitem()[0])

class ImportPathsTest(_LoaderTest):
    """Tests for load_pytd.py."""

    def test_filepath_to_module(self):
        if False:
            for i in range(10):
                print('nop')
        test_cases = [('foo/bar/baz.py', [''], 'foo.bar.baz'), ('foo/bar/baz.py', ['foo'], 'bar.baz'), ('foo/bar/baz.py', ['fo'], 'foo.bar.baz'), ('foo/bar/baz.py', ['foo/'], 'bar.baz'), ('foo/bar/baz.py', ['foo', 'bar'], 'bar.baz'), ('foo/bar/baz.py', ['foo/bar', 'foo'], 'baz'), ('foo/bar/baz.py', ['foo', 'foo/bar'], 'bar.baz'), ('./foo/bar.py', [''], 'foo.bar'), ('./foo.py', [''], 'foo'), ('../foo.py', [''], None), ('../foo.py', ['.'], None), ('foo/bar/../baz.py', [''], 'foo.baz'), ('../foo.py', ['..'], 'foo'), ('../../foo.py', ['../..'], 'foo'), ('../../foo.py', ['..'], None)]
        replaced_test_cased = []
        for (a, b, c) in test_cases:
            replaced_test_cased.append((file_utils.replace_separator(a), list(map(file_utils.replace_separator, b)), c))
        test_cases = replaced_test_cased
        for (filename, pythonpath, expected) in test_cases:
            module = module_utils.get_module_name(filename, pythonpath)
            self.assertEqual(module, expected)

    def test_builtin_sys(self):
        if False:
            print('Hello World!')
        loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version))
        ast = loader.import_name('sys')
        self.assertTrue(ast.Lookup('sys.exit'))

    def test_basic(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('path/to/some/module.pyi'), 'def foo(x:int) -> str: ...')
            loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version, pythonpath=d.path))
            ast = loader.import_name('path.to.some.module')
            self.assertTrue(ast.Lookup('path.to.some.module.foo'))

    def test_path(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d1:
            with test_utils.Tempdir() as d2:
                d1.create_file(file_utils.replace_separator('dir1/module1.pyi'), 'def foo1() -> str: ...')
                d2.create_file(file_utils.replace_separator('dir2/module2.pyi'), 'def foo2() -> str: ...')
                loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version, pythonpath=f'{d1.path}{os.pathsep}{d2.path}'))
                module1 = loader.import_name('dir1.module1')
                module2 = loader.import_name('dir2.module2')
                self.assertTrue(module1.Lookup('dir1.module1.foo1'))
                self.assertTrue(module2.Lookup('dir2.module2.foo2'))

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d1:
            d1.create_file(file_utils.replace_separator('baz/__init__.pyi'), 'x = ... # type: int')
            loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version, pythonpath=d1.path))
            self.assertTrue(loader.import_name('baz').Lookup('baz.x'))

    def test_builtins(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', 'x = ... # type: int')
            loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version, pythonpath=d.path))
            mod = loader.import_name('foo')
            self.assertEqual('builtins.int', mod.Lookup('foo.x').type.cls.name)
            self.assertEqual('builtins.int', mod.Lookup('foo.x').type.name)

    def test_no_init(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_directory('baz')
            loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version, pythonpath=d.path))
            self.assertTrue(loader.import_name('baz'))

    def test_no_init_imports_map(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_directory('baz')
            with file_utils.cd(d.path):
                loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version, pythonpath=''))
                loader.options.tweak(imports_map={})
                self.assertFalse(loader.import_name('baz'))

    def test_stdlib(self):
        if False:
            for i in range(10):
                print('nop')
        loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version))
        ast = loader.import_name('io')
        self.assertTrue(ast.Lookup('io.StringIO'))

    def test_deep_dependency(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('module1.pyi', 'def get_bar() -> module2.Bar: ...')
            d.create_file('module2.pyi', 'class Bar:\n  pass')
            loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version, pythonpath=d.path))
            module1 = loader.import_name('module1')
            (f,) = module1.Lookup('module1.get_bar').signatures
            self.assertEqual('module2.Bar', f.return_type.cls.name)

    def test_circular_dependency(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def get_bar() -> bar.Bar: ...\n        class Foo:\n          pass\n      ')
            d.create_file('bar.pyi', '\n        def get_foo() -> foo.Foo: ...\n        class Bar:\n          pass\n      ')
            loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version, pythonpath=d.path))
            foo = loader.import_name('foo')
            bar = loader.import_name('bar')
            (f1,) = foo.Lookup('foo.get_bar').signatures
            (f2,) = bar.Lookup('bar.get_foo').signatures
            self.assertEqual('bar.Bar', f1.return_type.cls.name)
            self.assertEqual('foo.Foo', f2.return_type.cls.name)

    def test_circular_dependency_complicated(self):
        if False:
            while True:
                i = 10
        with self._setup_loader(target='\n      from dep1 import PathLike\n      from dep3 import AnyPath\n      def abspath(path: PathLike[str]) -> str: ...\n    ', dep1="\n      from dep2 import Popen\n      from typing import Generic, TypeVar\n      _T = TypeVar('_T')\n      class PathLike(Generic[_T]): ...\n    ", dep2='\n      from dep3 import AnyPath\n      class Popen: ...\n    ', dep3='\n      from dep1 import PathLike\n      AnyPath = PathLike[str]\n    ') as loader:
            loader.finish_and_verify_ast(loader.load_file('target', path_utils.join(loader.options.pythonpath[0], 'target.pyi')))

    def test_relative(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('__init__.pyi', 'base = ...  # type: str')
            d.create_file(file_utils.replace_separator('path/__init__.pyi'), 'path = ...  # type: str')
            d.create_file(file_utils.replace_separator('path/to/__init__.pyi'), 'to = ...  # type: str')
            d.create_file(file_utils.replace_separator('path/to/some/__init__.pyi'), 'some = ...  # type: str')
            d.create_file(file_utils.replace_separator('path/to/some/module.pyi'), '')
            loader = load_pytd.Loader(config.Options.create(module_name='path.to.some.module', python_version=self.python_version, pythonpath=d.path))
            some = loader.import_relative(1)
            to = loader.import_relative(2)
            path = loader.import_relative(3)
            self.assertTrue(some.Lookup('path.to.some.some'))
            self.assertTrue(to.Lookup('path.to.to'))
            self.assertTrue(path.Lookup('path.path'))

    def test_typeshed(self):
        if False:
            while True:
                i = 10
        loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version))
        self.assertTrue(loader.import_name('urllib.request'))

    def test_prefer_typeshed(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('typing_extensions/__init__.pyi'), 'foo: str = ...')
            d.create_file(file_utils.replace_separator('crypt/__init__.pyi'), 'foo: str = ...')
            loader = load_pytd.Loader(config.Options.create(module_name='x', python_version=self.python_version, pythonpath=d.path))
            ast1 = loader.import_name('typing_extensions')
            ast2 = loader.import_name('crypt')
            self.assertTrue(ast1.Lookup('typing_extensions.Literal'))
            self.assertTrue(ast2.Lookup('crypt.foo'))
            with self.assertRaises(KeyError):
                ast1.Lookup('typing_extensions.foo')
            with self.assertRaises(KeyError):
                ast2.Lookup('crypt.crypt')

    def test_resolve_alias(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('module1.pyi', '\n          from typing import List\n          x = List[int]\n      ')
            d.create_file('module2.pyi', '\n          def f() -> module1.x: ...\n      ')
            loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version, pythonpath=d.path))
            module2 = loader.import_name('module2')
            (f,) = module2.Lookup('module2.f').signatures
            self.assertEqual('List[int]', pytd_utils.Print(f.return_type))

    def test_import_map_congruence(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            foo_path = d.create_file('foo.pyi', 'class X: ...')
            bar_path = d.create_file('bar.pyi', 'X = ...  # type: another.foo.X')
            null_device = '/dev/null' if sys.platform != 'win32' else 'NUL'
            imports_map = {'foo': foo_path, file_utils.replace_separator('another/foo'): foo_path, 'bar': bar_path, 'empty1': null_device, 'empty2': null_device}
            loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version, pythonpath=''))
            loader.options.tweak(imports_map=imports_map)
            normal = loader.import_name('foo')
            self.assertEqual('foo', normal.name)
            loader.import_name('bar')
            another = loader.import_name('another.foo')
            self.assertIsNot(normal, another)
            self.assertTrue([c.name.startswith('foo') for c in normal.classes])
            self.assertTrue([c.name.startswith('another.foo') for c in another.classes])
            empty1 = loader.import_name('empty1')
            empty2 = loader.import_name('empty2')
            self.assertIsNot(empty1, empty2)
            self.assertEqual('empty1', empty1.name)
            self.assertEqual('empty2', empty2.name)

    def test_package_relative_import(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('pkg/foo.pyi'), 'class X: ...')
            d.create_file(file_utils.replace_separator('pkg/bar.pyi'), '\n          from .foo import X\n          y = ...  # type: X')
            loader = load_pytd.Loader(config.Options.create(module_name='pkg.bar', python_version=self.python_version, pythonpath=d.path))
            bar = loader.import_name('pkg.bar')
            f = bar.Lookup('pkg.bar.y')
            self.assertEqual('pkg.foo.X', f.type.name)

    def test_directory_import(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('pkg/sub/__init__.pyi'), '\n          from .foo import *\n          from .bar import *')
            d.create_file(file_utils.replace_separator('pkg/sub/foo.pyi'), '\n          class X: pass')
            d.create_file(file_utils.replace_separator('pkg/sub/bar.pyi'), '\n          from .foo import X\n          y = ...  # type: X')
            loader = load_pytd.Loader(config.Options.create(module_name='pkg', python_version=self.python_version, pythonpath=d.path))
            ast = loader.import_name('pkg.sub')
            self.assertTrue(ast.Lookup('pkg.sub.X'))

    def test_diamond_import(self):
        if False:
            print('Hello World!')
        'Should not fail on importing a module via two paths.'
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('pkg/sub/__init__.pyi'), '\n          from .foo import *\n          from .bar import *')
            d.create_file(file_utils.replace_separator('pkg/sub/foo.pyi'), '\n          from .baz import X')
            d.create_file(file_utils.replace_separator('pkg/sub/bar.pyi'), '\n          from .baz import X')
            d.create_file(file_utils.replace_separator('pkg/sub/baz.pyi'), '\n          class X: ...')
            loader = load_pytd.Loader(config.Options.create(module_name='pkg', python_version=self.python_version, pythonpath=d.path))
            ast = loader.import_name('pkg.sub')
            self.assertTrue(ast.Lookup('pkg.sub.X'))

    def test_get_resolved_modules(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            filename = d.create_file(file_utils.replace_separator('dir/module.pyi'), 'def foo() -> str: ...')
            loader = load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=d.path))
            ast = loader.import_name('dir.module')
            modules = loader.get_resolved_modules()
            self.assertEqual(set(modules), {'builtins', 'typing', 'dir.module'})
            module = modules['dir.module']
            self.assertEqual(module.module_name, 'dir.module')
            self.assertEqual(module.filename, filename)
            self.assertEqual(module.ast, ast)

    def test_circular_import(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('os2/__init__.pyi'), '\n        from . import path as path\n        _PathType = path._PathType\n        def utime(path: _PathType) -> None: ...\n        class stat_result: ...\n      ')
            d.create_file(file_utils.replace_separator('os2/path.pyi'), '\n        import os2\n        _PathType = bytes\n        def samestat(stat1: os2.stat_result) -> bool: ...\n      ')
            loader = load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=d.path))
            ast = loader.import_name('os2.path')
            self.assertEqual(ast.Lookup('os2.path._PathType').type.name, 'builtins.bytes')

    def test_circular_import_with_external_type(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('os2/__init__.pyi'), '\n        from posix2 import stat_result as stat_result\n        from . import path as path\n        _PathType = path._PathType\n        def utime(path: _PathType) -> None: ...\n      ')
            d.create_file(file_utils.replace_separator('os2/path.pyi'), '\n        import os2\n        _PathType = bytes\n        def samestate(stat1: os2.stat_result) -> bool: ...\n      ')
            d.create_file('posix2.pyi', 'class stat_result: ...')
            loader = load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=d.path))
            loader.import_name('os2')
            loader.import_name('os2.path')
            loader.import_name('posix2')

    def test_union_alias(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('test.pyi', '\n        from typing import Union as _UnionT\n        x: _UnionT[int, str]\n      ')
            loader = load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=d.path))
            ast = loader.import_name('test')
            x = ast.Lookup('test.x')
            self.assertIsInstance(x.type, pytd.UnionType)

    def test_optional_alias(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file('test.pyi', '\n        from typing import Optional as _OptionalT\n        x: _OptionalT[int]\n      ')
            loader = load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=d.path))
            ast = loader.import_name('test')
            x = ast.Lookup('test.x')
            self.assertIsInstance(x.type, pytd.UnionType)

    def test_intersection_alias(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file('test.pyi', '\n        from typing import Intersection as _IntersectionT\n        x: _IntersectionT[int, str]\n      ')
            loader = load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=d.path))
            ast = loader.import_name('test')
            x = ast.Lookup('test.x')
            self.assertIsInstance(x.type, pytd.IntersectionType)

    def test_open_function(self):
        if False:
            for i in range(10):
                print('nop')

        def mock_open(*unused_args, **unused_kwargs):
            if False:
                return 10
            return io.StringIO('x: int')
        loader = load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version, open_function=mock_open))
        a = loader.load_file('a', 'a.pyi')
        self.assertEqual('int', pytd_utils.Print(a.Lookup('a.x').type))

    def test_submodule_reexport(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/bar.pyi'), '')
            d.create_file(file_utils.replace_separator('foo/__init__.pyi'), '\n        from . import bar as bar\n      ')
            loader = load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=d.path))
            foo = loader.import_name('foo')
            self.assertEqual(pytd_utils.Print(foo), 'import foo.bar')

    def test_submodule_rename(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/bar.pyi'), '')
            d.create_file(file_utils.replace_separator('foo/__init__.pyi'), '\n        from . import bar as baz\n      ')
            loader = load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=d.path))
            foo = loader.import_name('foo')
            self.assertEqual(pytd_utils.Print(foo), 'from foo import bar as foo.baz')

    def test_typing_reexport(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo.pyi'), '\n        from typing import List as List\n      ')
            d.create_file(file_utils.replace_separator('bar.pyi'), '\n        from foo import *\n        def f() -> List[int]: ...\n      ')
            loader = load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=d.path))
            foo = loader.import_name('foo')
            bar = loader.import_name('bar')
            self.assertEqual(pytd_utils.Print(foo), 'from builtins import list as List')
            self.assertEqual(pytd_utils.Print(bar), textwrap.dedent('\n        import typing\n        from builtins import list as List\n\n        def bar.f() -> typing.List[int]: ...\n      ').strip())

    def test_reuse_builtin_name(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        class Ellipsis: ...\n      ')
            d.create_file('bar.pyi', '\n        from foo import *\n        def f(x: Ellipsis): ...\n      ')
            loader = load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=d.path))
            loader.import_name('foo')
            bar = loader.import_name('bar')
            self.assertEqual(pytd_utils.Print(bar.Lookup('bar.f')), 'def bar.f(x: foo.Ellipsis) -> Any: ...')

    def test_import_typevar(self):
        if False:
            print('Hello World!')
        self._import(a="\n      from typing import TypeVar\n      T = TypeVar('T')\n    ", b='\n      from a import T\n      def f(x: T) -> T: ...\n    ', c='\n      from b import *\n    ')

    def test_import_class_from_parent_module(self):
        if False:
            return 10
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('foo/__init__.pyi'), 'class Foo: ...')
            d.create_file(file_utils.replace_separator('foo/bar.pyi'), '\n        from . import Foo\n        class Bar(Foo): ...\n      ')
            loader = load_pytd.Loader(config.Options.create(python_version=self.python_version, pythonpath=d.path))
            loader.import_name('foo.bar')

    def test_module_alias(self):
        if False:
            while True:
                i = 10
        ast = self._import(foo='\n      import subprocess as _subprocess\n      x: _subprocess.Popen\n    ')
        expected = textwrap.dedent('\n      import subprocess as foo._subprocess\n\n      foo.x: foo._subprocess.Popen\n    ').strip()
        self.assertMultiLineEqual(pytd_utils.Print(ast), expected)

    def test_star_import_in_circular_dep(self):
        if False:
            i = 10
            return i + 15
        stub3_ast = self._import(stub1='\n      from stub2 import Foo\n      from typing import Mapping as Mapping\n    ', stub2='\n      from stub3 import Mapping\n      class Foo: ...\n    ', stub3='\n      from stub1 import *\n    ')
        self.assertEqual(stub3_ast.Lookup('stub3.Foo').type, pytd.ClassType('stub2.Foo'))
        self.assertEqual(stub3_ast.Lookup('stub3.Mapping').type, pytd.ClassType('typing.Mapping'))

    def test_import_all(self):
        if False:
            while True:
                i = 10
        ast = self._import(foo="__all__ = ['foo']", bar="__all__ = ['bar']", baz='\n      from foo import *\n      from bar import *\n    ')
        self.assertFalse(ast.aliases)

    def test_import_private_typevar(self):
        if False:
            while True:
                i = 10
        ast = self._import(foo="\n      from typing import TypeVar\n      _T = TypeVar('_T')\n    ", bar="\n      from typing import TypeVar\n      _T = TypeVar('_T')\n    ", baz='\n      from foo import *\n      from bar import *\n    ')
        self.assertFalse(ast.type_params)

    def test_use_class_alias(self):
        if False:
            while True:
                i = 10
        ast = self._import(foo='\n      class A:\n        class B: ...\n        x: A2.B\n      A2 = A\n    ')
        a = ast.Lookup('foo.A')
        self.assertEqual(a.Lookup('x').type.cls, a.Lookup('foo.A.B'))

    def test_alias_typevar(self):
        if False:
            i = 10
            return i + 15
        ast = self._import(foo="\n      from typing import TypeVar as _TypeVar\n      T = _TypeVar('T')\n    ")
        self.assertEqual(ast.Lookup('foo.T'), pytd.TypeParameter(name='T', scope='foo'))

    def test_alias_property_with_setter(self):
        if False:
            print('Hello World!')
        ast = self._import(foo='\n      class X:\n        @property\n        def f(self) -> int: ...\n        @f.setter\n        def f(self, value: int) -> None: ...\n        g = f\n    ')
        x = ast.Lookup('foo.X')
        self.assertEqual(pytd_utils.Print(x.Lookup('f')), "f: Annotated[int, 'property']")
        self.assertEqual(pytd_utils.Print(x.Lookup('g')), "g: Annotated[int, 'property']")

    def test_typing_alias(self):
        if False:
            i = 10
            return i + 15
        ast = self._import(foo='\n      from typing import _Alias, TypeAlias\n      X = _Alias()\n      Y: TypeAlias = _Alias()\n    ')
        self.assertEqual(pytd_utils.Print(ast), 'from typing import _Alias as X, _Alias as Y')

class ImportTypeMacroTest(_LoaderTest):

    def test_container(self):
        if False:
            for i in range(10):
                print('nop')
        ast = self._import(a="\n      from typing import List, TypeVar\n      T = TypeVar('T')\n      Alias = List[T]\n    ", b='\n      import a\n      Strings = a.Alias[str]\n    ')
        self.assertEqual(pytd_utils.Print(ast.Lookup('b.Strings').type), 'List[str]')

    def test_union(self):
        if False:
            for i in range(10):
                print('nop')
        ast = self._import(a="\n      from typing import List, TypeVar, Union\n      T = TypeVar('T')\n      Alias = Union[T, List[T]]\n    ", b='\n      import a\n      Strings = a.Alias[str]\n    ')
        self.assertEqual(pytd_utils.Print(ast.Lookup('b.Strings').type), 'Union[str, List[str]]')

    def test_bad_parameterization(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(load_pytd.BadDependencyError, 'Union\\[T, List\\[T\\]\\] expected 1 parameters, got 2'):
            self._import(a="\n        from typing import List, TypeVar, Union\n        T = TypeVar('T')\n        Alias = Union[T, List[T]]\n      ", b='\n        import a\n        Strings = a.Alias[str, str]\n      ')

    def test_parameterize_twice(self):
        if False:
            print('Hello World!')
        ast = self._import(a='\n      from typing import AnyStr, Generic\n      class Foo(Generic[AnyStr]): ...\n    ', b='\n      import a\n      from typing import AnyStr\n      x: Foo[str]\n      Foo = a.Foo[AnyStr]\n    ')
        self.assertEqual(pytd_utils.Print(ast.Lookup('b.x').type), 'a.Foo[str]')

@dataclasses.dataclass(eq=True, frozen=True)
class _Module:
    module_name: str
    file_name: str

class PickledPyiLoaderTest(test_base.UnitTest):

    def _create_files(self, tempdir):
        if False:
            for i in range(10):
                print('nop')
        src = '\n        import module2\n        from typing import List\n\n        constant = True\n\n        x = List[int]\n        b = List[int]\n\n        class SomeClass:\n          def __init__(self, a: module2.ObjectMod2):\n            pass\n\n        def ModuleFunction():\n          pass\n    '
        tempdir.create_file('module1.pyi', src)
        tempdir.create_file('module2.pyi', '\n        class ObjectMod2:\n          def __init__(self):\n            pass\n    ')

    def _get_path(self, tempdir, filename):
        if False:
            return 10
        return path_utils.join(tempdir.path, filename)

    def _load_ast(self, tempdir, module):
        if False:
            print('Hello World!')
        loader = load_pytd.Loader(config.Options.create(module_name=module.module_name, python_version=self.python_version, pythonpath=tempdir.path))
        return (loader, loader.load_file(module.module_name, self._get_path(tempdir, module.file_name)))

    def _pickle_modules(self, loader, tempdir, *modules):
        if False:
            print('Hello World!')
        for module in modules:
            pickle_utils.StoreAst(loader._modules[module.module_name].ast, self._get_path(tempdir, module.file_name + '.pickled'))

    def _load_pickled_module(self, tempdir, module):
        if False:
            for i in range(10):
                print('nop')
        pickle_loader = load_pytd.PickledPyiLoader(config.Options.create(python_version=self.python_version, pythonpath=tempdir.path))
        return pickle_loader.load_file(module.module_name, self._get_path(tempdir, module.file_name))

    def test_load_with_same_module_name(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            self._create_files(tempdir=d)
            module1 = _Module(module_name='foo.bar.module1', file_name='module1.pyi')
            module2 = _Module(module_name='module2', file_name='module2.pyi')
            (loader, ast) = self._load_ast(tempdir=d, module=module1)
            self._pickle_modules(loader, d, module1, module2)
            pickled_ast_filename = self._get_path(d, module1.file_name + '.pickled')
            result = pickle_utils.StoreAst(ast, pickled_ast_filename)
            self.assertIsNone(result)
            loaded_ast = self._load_pickled_module(d, module1)
            self.assertTrue(loaded_ast)
            self.assertIsNot(loaded_ast, ast)
            self.assertTrue(pytd_utils.ASTeq(ast, loaded_ast))
            loaded_ast.Visit(visitors.VerifyLookup())

    def test_star_import(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', 'class A: ...')
            d.create_file('bar.pyi', 'from foo import *')
            foo = _Module(module_name='foo', file_name='foo.pyi')
            bar = _Module(module_name='bar', file_name='bar.pyi')
            (loader, _) = self._load_ast(d, module=bar)
            self._pickle_modules(loader, d, foo, bar)
            loaded_ast = self._load_pickled_module(d, bar)
            loaded_ast.Visit(visitors.VerifyLookup())
            self.assertEqual(pytd_utils.Print(loaded_ast), 'from foo import A')

    def test_function_alias(self):
        if False:
            while True:
                i = 10
        with test_utils.Tempdir() as d:
            d.create_file('foo.pyi', '\n        def f(): ...\n        g = f\n      ')
            foo = _Module(module_name='foo', file_name='foo.pyi')
            (loader, _) = self._load_ast(d, module=foo)
            self._pickle_modules(loader, d, foo)
            loaded_ast = self._load_pickled_module(d, foo)
            g = loaded_ast.Lookup('foo.g')
            self.assertEqual(g.type, loaded_ast.Lookup('foo.f'))

    def test_package_relative_import(self):
        if False:
            i = 10
            return i + 15
        with test_utils.Tempdir() as d:
            d.create_file(file_utils.replace_separator('pkg/foo.pyi'), 'class X: ...')
            d.create_file(file_utils.replace_separator('pkg/bar.pyi'), '\n          from .foo import X\n          y = ...  # type: X')
            foo = _Module(module_name='pkg.foo', file_name=file_utils.replace_separator('pkg/foo.pyi'))
            bar = _Module(module_name='pkg.bar', file_name=file_utils.replace_separator('pkg/bar.pyi'))
            (loader, _) = self._load_ast(d, module=bar)
            self._pickle_modules(loader, d, foo, bar)
            loaded_ast = self._load_pickled_module(d, bar)
            loaded_ast.Visit(visitors.VerifyLookup())

    def test_pickled_builtins(self):
        if False:
            print('Hello World!')
        with test_utils.Tempdir() as d:
            filename = d.create_file('builtins.pickle')
            foo_path = d.create_file('foo.pickle', '\n        import datetime\n        tz = ...  # type: datetime.tzinfo\n      ')
            load_pytd.Loader(config.Options.create(module_name='base', python_version=self.python_version)).save_to_pickle(filename)
            loader = load_pytd.PickledPyiLoader.load_from_pickle(filename, config.Options.create(module_name='base', python_version=self.python_version, pythonpath=''))
            loader.options.tweak(imports_map={'foo': foo_path})
            self.assertTrue(loader.import_name('sys'))
            self.assertTrue(loader.import_name('__future__'))
            self.assertTrue(loader.import_name('datetime'))
            self.assertTrue(loader.import_name('foo'))
            self.assertTrue(loader.import_name('ctypes'))

class MethodAliasTest(_LoaderTest):

    def test_import_class(self):
        if False:
            while True:
                i = 10
        b_ast = self._import(a='\n      class Foo:\n        def f(self) -> int: ...\n    ', b='\n      import a\n      f = a.Foo.f\n    ')
        self.assertEqual(pytd_utils.Print(b_ast.Lookup('b.f')), 'def b.f(self: a.Foo) -> int: ...')

    def test_import_class_instance(self):
        if False:
            i = 10
            return i + 15
        b_ast = self._import(a='\n      class Foo:\n        def f(self) -> int: ...\n      foo: Foo\n    ', b='\n      import a\n      f = a.foo.f\n    ')
        self.assertEqual(pytd_utils.Print(b_ast.Lookup('b.f')), 'def b.f() -> int: ...')

    def test_create_instance_after_import(self):
        if False:
            print('Hello World!')
        b_ast = self._import(a='\n      class Foo:\n        def f(self) -> int: ...\n    ', b='\n      import a\n      foo: a.Foo\n      f = foo.f\n    ')
        self.assertEqual(pytd_utils.Print(b_ast.Lookup('b.f')), 'def b.f() -> int: ...')

    def test_function(self):
        if False:
            for i in range(10):
                print('nop')
        ast = self._import(a='\n      def f(x: int) -> int: ...\n      g = f\n    ')
        self.assertEqual(pytd_utils.Print(ast.Lookup('a.g')), 'def a.g(x: int) -> int: ...')

    def test_imported_function(self):
        if False:
            for i in range(10):
                print('nop')
        b_ast = self._import(a='\n      def f(x: int) -> int: ...\n    ', b='\n      import a\n      f = a.f\n    ')
        self.assertEqual(pytd_utils.Print(b_ast.Lookup('b.f')), 'def b.f(x: int) -> int: ...')

    def test_base_class(self):
        if False:
            for i in range(10):
                print('nop')
        a_ast = self._import(a='\n      class Foo:\n        def f(self) -> int: ...\n      class Bar(Foo): ...\n      x: Bar\n      f = x.f\n    ')
        self.assertEqual(pytd_utils.Print(a_ast.Lookup('a.f')), 'def a.f() -> int: ...')

    def test_base_class_imported(self):
        if False:
            while True:
                i = 10
        b_ast = self._import(a='\n      class Foo:\n        def f(self) -> int: ...\n      class Bar(Foo): ...\n      x: Bar\n    ', b='\n      import a\n      f = a.x.f\n    ')
        self.assertEqual(pytd_utils.Print(b_ast.Lookup('b.f')), 'def b.f() -> int: ...')

class RecursiveAliasTest(_LoaderTest):

    def test_basic(self):
        if False:
            while True:
                i = 10
        ast = self._import(a='\n      from typing import List\n      X = List[X]\n    ')
        actual_x = ast.Lookup('a.X')
        expected_x = pytd.Alias(name='a.X', type=pytd.GenericType(base_type=pytd.ClassType('builtins.list'), parameters=(pytd.LateType('a.X', recursive=True),)))
        self.assertEqual(actual_x, expected_x)

    def test_mutual_recursion(self):
        if False:
            for i in range(10):
                print('nop')
        ast = self._import(a='\n      from typing import List\n      X = List[Y]\n      Y = List[X]\n    ')
        actual_x = ast.Lookup('a.X')
        expected_x = pytd.Alias(name='a.X', type=pytd.GenericType(base_type=pytd.ClassType('builtins.list'), parameters=(pytd.LateType('a.Y', recursive=True),)))
        self.assertEqual(actual_x, expected_x)
        actual_y = ast.Lookup('a.Y')
        expected_y = pytd.Alias(name='a.Y', type=pytd.GenericType(base_type=pytd.ClassType('builtins.list'), parameters=(pytd.GenericType(base_type=pytd.ClassType('builtins.list'), parameters=(pytd.LateType('a.Y', recursive=True),)),)))
        self.assertEqual(actual_y, expected_y)

    def test_very_mutual_recursion(self):
        if False:
            return 10
        ast = self._import(a='\n      from typing import List\n      X = List[Y]\n      Y = List[Z]\n      Z = List[X]\n    ')
        actual_x = ast.Lookup('a.X')
        expected_x = pytd.Alias(name='a.X', type=pytd.GenericType(base_type=pytd.ClassType('builtins.list'), parameters=(pytd.LateType('a.Y', recursive=True),)))
        self.assertEqual(actual_x, expected_x)
        actual_y = ast.Lookup('a.Y')
        expected_y = pytd.Alias(name='a.Y', type=pytd.GenericType(base_type=pytd.ClassType('builtins.list'), parameters=(pytd.LateType('a.Z', recursive=True),)))
        self.assertEqual(actual_y, expected_y)
        actual_z = ast.Lookup('a.Z')
        expected_z = pytd.Alias(name='a.Z', type=pytd.GenericType(base_type=pytd.ClassType('builtins.list'), parameters=(pytd.GenericType(base_type=pytd.ClassType('builtins.list'), parameters=(pytd.LateType('a.Y', recursive=True),)),)))
        self.assertEqual(actual_z, expected_z)

class NestedClassTest(_LoaderTest):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        ast = self._import(a='\n      class A:\n        class B:\n          def f(self) -> C: ...\n        class C(B): ...\n    ')
        actual_f = pytd.LookupItemRecursive(ast, 'A.B.f')
        self.assertEqual(pytd_utils.Print(actual_f), 'def f(self: a.A.B) -> a.A.C: ...')
        actual_c = pytd.LookupItemRecursive(ast, 'A.C')
        self.assertEqual(pytd_utils.Print(actual_c).rstrip(), 'class a.A.C(a.A.B): ...')

    @test_base.skip('This does not work yet')
    def test_shadowing(self):
        if False:
            print('Hello World!')
        ast = self._import(a='\n      class A:\n        class A(A):\n          def f(self) -> A: ...\n        class C(A): ...\n    ')
        self.assertEqual(pytd_utils.Print(ast).rstrip(), textwrap.dedent('\n      class a.A:\n          class a.A.A(a.A):\n              def f(self) -> a.A.A: ...\n          class a.A.C(a.A.A): ...\n      ').strip())
if __name__ == '__main__':
    unittest.main()