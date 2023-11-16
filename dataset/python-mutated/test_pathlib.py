import errno
import os.path
import pickle
import sys
import unittest.mock
from pathlib import Path
from textwrap import dedent
from types import ModuleType
from typing import Any
from typing import Generator
from typing import Iterator
import pytest
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import commonpath
from _pytest.pathlib import ensure_deletable
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import get_extended_length_path_str
from _pytest.pathlib import get_lock_path
from _pytest.pathlib import import_path
from _pytest.pathlib import ImportMode
from _pytest.pathlib import ImportPathMismatchError
from _pytest.pathlib import insert_missing_modules
from _pytest.pathlib import maybe_delete_a_numbered_dir
from _pytest.pathlib import module_name_from_path
from _pytest.pathlib import resolve_package_path
from _pytest.pathlib import safe_exists
from _pytest.pathlib import symlink_or_skip
from _pytest.pathlib import visit
from _pytest.pytester import Pytester
from _pytest.tmpdir import TempPathFactory

class TestFNMatcherPort:
    """Test our port of py.common.FNMatcher (fnmatch_ex)."""
    if sys.platform == 'win32':
        drv1 = 'c:'
        drv2 = 'd:'
    else:
        drv1 = '/c'
        drv2 = '/d'

    @pytest.mark.parametrize('pattern, path', [('*.py', 'foo.py'), ('*.py', 'bar/foo.py'), ('test_*.py', 'foo/test_foo.py'), ('tests/*.py', 'tests/foo.py'), (f'{drv1}/*.py', f'{drv1}/foo.py'), (f'{drv1}/foo/*.py', f'{drv1}/foo/foo.py'), ('tests/**/test*.py', 'tests/foo/test_foo.py'), ('tests/**/doc/test*.py', 'tests/foo/bar/doc/test_foo.py'), ('tests/**/doc/**/test*.py', 'tests/foo/doc/bar/test_foo.py')])
    def test_matching(self, pattern: str, path: str) -> None:
        if False:
            print('Hello World!')
        assert fnmatch_ex(pattern, path)

    def test_matching_abspath(self) -> None:
        if False:
            while True:
                i = 10
        abspath = os.path.abspath(os.path.join('tests/foo.py'))
        assert fnmatch_ex('tests/foo.py', abspath)

    @pytest.mark.parametrize('pattern, path', [('*.py', 'foo.pyc'), ('*.py', 'foo/foo.pyc'), ('tests/*.py', 'foo/foo.py'), (f'{drv1}/*.py', f'{drv2}/foo.py'), (f'{drv1}/foo/*.py', f'{drv2}/foo/foo.py'), ('tests/**/test*.py', 'tests/foo.py'), ('tests/**/test*.py', 'foo/test_foo.py'), ('tests/**/doc/test*.py', 'tests/foo/bar/doc/foo.py'), ('tests/**/doc/test*.py', 'tests/foo/bar/test_foo.py')])
    def test_not_matching(self, pattern: str, path: str) -> None:
        if False:
            print('Hello World!')
        assert not fnmatch_ex(pattern, path)

class TestImportPath:
    """

    Most of the tests here were copied from py lib's tests for "py.local.path.pyimport".

    Having our own pyimport-like function is inline with removing py.path dependency in the future.
    """

    @pytest.fixture(scope='session')
    def path1(self, tmp_path_factory: TempPathFactory) -> Generator[Path, None, None]:
        if False:
            print('Hello World!')
        path = tmp_path_factory.mktemp('path')
        self.setuptestfs(path)
        yield path
        assert path.joinpath('samplefile').exists()

    @pytest.fixture(autouse=True)
    def preserve_sys(self):
        if False:
            while True:
                i = 10
        with unittest.mock.patch.dict(sys.modules):
            with unittest.mock.patch.object(sys, 'path', list(sys.path)):
                yield

    def setuptestfs(self, path: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        samplefile = path / 'samplefile'
        samplefile.write_text('samplefile\n', encoding='utf-8')
        execfile = path / 'execfile'
        execfile.write_text('x=42', encoding='utf-8')
        execfilepy = path / 'execfile.py'
        execfilepy.write_text('x=42', encoding='utf-8')
        d = {1: 2, 'hello': 'world', 'answer': 42}
        path.joinpath('samplepickle').write_bytes(pickle.dumps(d, 1))
        sampledir = path / 'sampledir'
        sampledir.mkdir()
        sampledir.joinpath('otherfile').touch()
        otherdir = path / 'otherdir'
        otherdir.mkdir()
        otherdir.joinpath('__init__.py').touch()
        module_a = otherdir / 'a.py'
        module_a.write_text('from .b import stuff as result\n', encoding='utf-8')
        module_b = otherdir / 'b.py'
        module_b.write_text('stuff="got it"\n', encoding='utf-8')
        module_c = otherdir / 'c.py'
        module_c.write_text(dedent('\n            import pluggy;\n            import otherdir.a\n            value = otherdir.a.result\n        '), encoding='utf-8')
        module_d = otherdir / 'd.py'
        module_d.write_text(dedent('\n            import pluggy;\n            from otherdir import a\n            value2 = a.result\n        '), encoding='utf-8')

    def test_smoke_test(self, path1: Path) -> None:
        if False:
            print('Hello World!')
        obj = import_path(path1 / 'execfile.py', root=path1)
        assert obj.x == 42
        assert obj.__name__ == 'execfile'

    def test_import_path_missing_file(self, path1: Path) -> None:
        if False:
            return 10
        with pytest.raises(ImportPathMismatchError):
            import_path(path1 / 'sampledir', root=path1)

    def test_renamed_dir_creates_mismatch(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        if False:
            i = 10
            return i + 15
        tmp_path.joinpath('a').mkdir()
        p = tmp_path.joinpath('a', 'test_x123.py')
        p.touch()
        import_path(p, root=tmp_path)
        tmp_path.joinpath('a').rename(tmp_path.joinpath('b'))
        with pytest.raises(ImportPathMismatchError):
            import_path(tmp_path.joinpath('b', 'test_x123.py'), root=tmp_path)
        monkeypatch.setenv('PY_IGNORE_IMPORTMISMATCH', '1')
        import_path(tmp_path.joinpath('b', 'test_x123.py'), root=tmp_path)
        monkeypatch.setenv('PY_IGNORE_IMPORTMISMATCH', '0')
        with pytest.raises(ImportPathMismatchError):
            import_path(tmp_path.joinpath('b', 'test_x123.py'), root=tmp_path)

    def test_messy_name(self, tmp_path: Path) -> None:
        if False:
            i = 10
            return i + 15
        path = tmp_path / 'foo__init__.py'
        path.touch()
        module = import_path(path, root=tmp_path)
        assert module.__name__ == 'foo__init__'

    def test_dir(self, tmp_path: Path) -> None:
        if False:
            while True:
                i = 10
        p = tmp_path / 'hello_123'
        p.mkdir()
        p_init = p / '__init__.py'
        p_init.touch()
        m = import_path(p, root=tmp_path)
        assert m.__name__ == 'hello_123'
        m = import_path(p_init, root=tmp_path)
        assert m.__name__ == 'hello_123'

    def test_a(self, path1: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        otherdir = path1 / 'otherdir'
        mod = import_path(otherdir / 'a.py', root=path1)
        assert mod.result == 'got it'
        assert mod.__name__ == 'otherdir.a'

    def test_b(self, path1: Path) -> None:
        if False:
            i = 10
            return i + 15
        otherdir = path1 / 'otherdir'
        mod = import_path(otherdir / 'b.py', root=path1)
        assert mod.stuff == 'got it'
        assert mod.__name__ == 'otherdir.b'

    def test_c(self, path1: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        otherdir = path1 / 'otherdir'
        mod = import_path(otherdir / 'c.py', root=path1)
        assert mod.value == 'got it'

    def test_d(self, path1: Path) -> None:
        if False:
            print('Hello World!')
        otherdir = path1 / 'otherdir'
        mod = import_path(otherdir / 'd.py', root=path1)
        assert mod.value2 == 'got it'

    def test_import_after(self, tmp_path: Path) -> None:
        if False:
            print('Hello World!')
        tmp_path.joinpath('xxxpackage').mkdir()
        tmp_path.joinpath('xxxpackage', '__init__.py').touch()
        mod1path = tmp_path.joinpath('xxxpackage', 'module1.py')
        mod1path.touch()
        mod1 = import_path(mod1path, root=tmp_path)
        assert mod1.__name__ == 'xxxpackage.module1'
        from xxxpackage import module1
        assert module1 is mod1

    def test_check_filepath_consistency(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        if False:
            return 10
        name = 'pointsback123'
        p = tmp_path.joinpath(name + '.py')
        p.touch()
        with monkeypatch.context() as mp:
            for ending in ('.pyc', '.pyo'):
                mod = ModuleType(name)
                pseudopath = tmp_path.joinpath(name + ending)
                pseudopath.touch()
                mod.__file__ = str(pseudopath)
                mp.setitem(sys.modules, name, mod)
                newmod = import_path(p, root=tmp_path)
                assert mod == newmod
        mod = ModuleType(name)
        pseudopath = tmp_path.joinpath(name + '123.py')
        pseudopath.touch()
        mod.__file__ = str(pseudopath)
        monkeypatch.setitem(sys.modules, name, mod)
        with pytest.raises(ImportPathMismatchError) as excinfo:
            import_path(p, root=tmp_path)
        (modname, modfile, orig) = excinfo.value.args
        assert modname == name
        assert modfile == str(pseudopath)
        assert orig == p
        assert issubclass(ImportPathMismatchError, ImportError)

    def test_issue131_on__init__(self, tmp_path: Path) -> None:
        if False:
            print('Hello World!')
        tmp_path.joinpath('proja').mkdir()
        p1 = tmp_path.joinpath('proja', '__init__.py')
        p1.touch()
        tmp_path.joinpath('sub', 'proja').mkdir(parents=True)
        p2 = tmp_path.joinpath('sub', 'proja', '__init__.py')
        p2.touch()
        m1 = import_path(p1, root=tmp_path)
        m2 = import_path(p2, root=tmp_path)
        assert m1 == m2

    def test_ensuresyspath_append(self, tmp_path: Path) -> None:
        if False:
            return 10
        root1 = tmp_path / 'root1'
        root1.mkdir()
        file1 = root1 / 'x123.py'
        file1.touch()
        assert str(root1) not in sys.path
        import_path(file1, mode='append', root=tmp_path)
        assert str(root1) == sys.path[-1]
        assert str(root1) not in sys.path[:-1]

    def test_invalid_path(self, tmp_path: Path) -> None:
        if False:
            i = 10
            return i + 15
        with pytest.raises(ImportError):
            import_path(tmp_path / 'invalid.py', root=tmp_path)

    @pytest.fixture
    def simple_module(self, tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[Path]:
        if False:
            return 10
        name = f'mymod_{request.node.name}'
        fn = tmp_path / f'_src/tests/{name}.py'
        fn.parent.mkdir(parents=True)
        fn.write_text('def foo(x): return 40 + x', encoding='utf-8')
        module_name = module_name_from_path(fn, root=tmp_path)
        yield fn
        sys.modules.pop(module_name, None)

    def test_importmode_importlib(self, simple_module: Path, tmp_path: Path, request: pytest.FixtureRequest) -> None:
        if False:
            while True:
                i = 10
        '`importlib` mode does not change sys.path.'
        module = import_path(simple_module, mode='importlib', root=tmp_path)
        assert module.foo(2) == 42
        assert str(simple_module.parent) not in sys.path
        assert module.__name__ in sys.modules
        assert module.__name__ == f'_src.tests.mymod_{request.node.name}'
        assert '_src' in sys.modules
        assert '_src.tests' in sys.modules

    def test_remembers_previous_imports(self, simple_module: Path, tmp_path: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        '`importlib` mode called remembers previous module (#10341, #10811).'
        module1 = import_path(simple_module, mode='importlib', root=tmp_path)
        module2 = import_path(simple_module, mode='importlib', root=tmp_path)
        assert module1 is module2

    def test_no_meta_path_found(self, simple_module: Path, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        if False:
            while True:
                i = 10
        'Even without any meta_path should still import module.'
        monkeypatch.setattr(sys, 'meta_path', [])
        module = import_path(simple_module, mode='importlib', root=tmp_path)
        assert module.foo(2) == 42
        import importlib.util
        del sys.modules[module.__name__]
        monkeypatch.setattr(importlib.util, 'spec_from_file_location', lambda *args: None)
        with pytest.raises(ImportError):
            import_path(simple_module, mode='importlib', root=tmp_path)

def test_resolve_package_path(tmp_path: Path) -> None:
    if False:
        while True:
            i = 10
    pkg = tmp_path / 'pkg1'
    pkg.mkdir()
    (pkg / '__init__.py').touch()
    (pkg / 'subdir').mkdir()
    (pkg / 'subdir/__init__.py').touch()
    assert resolve_package_path(pkg) == pkg
    assert resolve_package_path(pkg / 'subdir/__init__.py') == pkg

def test_package_unimportable(tmp_path: Path) -> None:
    if False:
        i = 10
        return i + 15
    pkg = tmp_path / 'pkg1-1'
    pkg.mkdir()
    pkg.joinpath('__init__.py').touch()
    subdir = pkg / 'subdir'
    subdir.mkdir()
    (pkg / 'subdir/__init__.py').touch()
    assert resolve_package_path(subdir) == subdir
    xyz = subdir / 'xyz.py'
    xyz.touch()
    assert resolve_package_path(xyz) == subdir
    assert not resolve_package_path(pkg)

def test_access_denied_during_cleanup(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    if False:
        while True:
            i = 10
    'Ensure that deleting a numbered dir does not fail because of OSErrors (#4262).'
    path = tmp_path / 'temp-1'
    path.mkdir()

    def renamed_failed(*args):
        if False:
            for i in range(10):
                print('nop')
        raise OSError('access denied')
    monkeypatch.setattr(Path, 'rename', renamed_failed)
    lock_path = get_lock_path(path)
    maybe_delete_a_numbered_dir(path)
    assert not lock_path.is_file()

def test_long_path_during_cleanup(tmp_path: Path) -> None:
    if False:
        return 10
    'Ensure that deleting long path works (particularly on Windows (#6775)).'
    path = (tmp_path / ('a' * 250)).resolve()
    if sys.platform == 'win32':
        assert len(str(path)) > 260
        extended_path = '\\\\?\\' + str(path)
    else:
        extended_path = str(path)
    os.mkdir(extended_path)
    assert os.path.isdir(extended_path)
    maybe_delete_a_numbered_dir(path)
    assert not os.path.isdir(extended_path)

def test_get_extended_length_path_str() -> None:
    if False:
        i = 10
        return i + 15
    assert get_extended_length_path_str('c:\\foo') == '\\\\?\\c:\\foo'
    assert get_extended_length_path_str('\\\\share\\foo') == '\\\\?\\UNC\\share\\foo'
    assert get_extended_length_path_str('\\\\?\\UNC\\share\\foo') == '\\\\?\\UNC\\share\\foo'
    assert get_extended_length_path_str('\\\\?\\c:\\foo') == '\\\\?\\c:\\foo'

def test_suppress_error_removing_lock(tmp_path: Path) -> None:
    if False:
        return 10
    'ensure_deletable should be resilient if lock file cannot be removed (#5456, #7491)'
    path = tmp_path / 'dir'
    path.mkdir()
    lock = get_lock_path(path)
    lock.touch()
    mtime = lock.stat().st_mtime
    with unittest.mock.patch.object(Path, 'unlink', side_effect=OSError) as m:
        assert not ensure_deletable(path, consider_lock_dead_if_created_before=mtime + 30)
        assert m.call_count == 1
    assert lock.is_file()
    with unittest.mock.patch.object(Path, 'is_file', side_effect=OSError) as m:
        assert not ensure_deletable(path, consider_lock_dead_if_created_before=mtime + 30)
        assert m.call_count == 1
    assert lock.is_file()
    assert ensure_deletable(path, consider_lock_dead_if_created_before=mtime + 30)
    assert not lock.is_file()

def test_bestrelpath() -> None:
    if False:
        for i in range(10):
            print('nop')
    curdir = Path('/foo/bar/baz/path')
    assert bestrelpath(curdir, curdir) == '.'
    assert bestrelpath(curdir, curdir / 'hello' / 'world') == 'hello' + os.sep + 'world'
    assert bestrelpath(curdir, curdir.parent / 'sister') == '..' + os.sep + 'sister'
    assert bestrelpath(curdir, curdir.parent) == '..'
    assert bestrelpath(curdir, Path('hello')) == 'hello'

def test_commonpath() -> None:
    if False:
        while True:
            i = 10
    path = Path('/foo/bar/baz/path')
    subpath = path / 'sampledir'
    assert commonpath(path, subpath) == path
    assert commonpath(subpath, path) == path
    assert commonpath(Path(str(path) + 'suffix'), path) == path.parent
    assert commonpath(path, path.parent.parent) == path.parent.parent

def test_visit_ignores_errors(tmp_path: Path) -> None:
    if False:
        while True:
            i = 10
    symlink_or_skip('recursive', tmp_path / 'recursive')
    tmp_path.joinpath('foo').write_bytes(b'')
    tmp_path.joinpath('bar').write_bytes(b'')
    assert [entry.name for entry in visit(str(tmp_path), recurse=lambda entry: False)] == ['bar', 'foo']

@pytest.mark.skipif(not sys.platform.startswith('win'), reason='Windows only')
def test_samefile_false_negatives(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    if False:
        print('Hello World!')
    '\n    import_file() should not raise ImportPathMismatchError if the paths are exactly\n    equal on Windows. It seems directories mounted as UNC paths make os.path.samefile\n    return False, even when they are clearly equal.\n    '
    module_path = tmp_path.joinpath('my_module.py')
    module_path.write_text('def foo(): return 42', encoding='utf-8')
    monkeypatch.syspath_prepend(tmp_path)
    with monkeypatch.context() as mp:
        mp.setattr(os.path, 'samefile', lambda x, y: False)
        module = import_path(module_path, root=tmp_path)
    assert getattr(module, 'foo')() == 42

class TestImportLibMode:

    def test_importmode_importlib_with_dataclass(self, tmp_path: Path) -> None:
        if False:
            return 10
        'Ensure that importlib mode works with a module containing dataclasses (#7856).'
        fn = tmp_path.joinpath('_src/tests/test_dataclass.py')
        fn.parent.mkdir(parents=True)
        fn.write_text(dedent('\n                from dataclasses import dataclass\n\n                @dataclass\n                class Data:\n                    value: str\n                '), encoding='utf-8')
        module = import_path(fn, mode='importlib', root=tmp_path)
        Data: Any = getattr(module, 'Data')
        data = Data(value='foo')
        assert data.value == 'foo'
        assert data.__module__ == '_src.tests.test_dataclass'

    def test_importmode_importlib_with_pickle(self, tmp_path: Path) -> None:
        if False:
            while True:
                i = 10
        'Ensure that importlib mode works with pickle (#7859).'
        fn = tmp_path.joinpath('_src/tests/test_pickle.py')
        fn.parent.mkdir(parents=True)
        fn.write_text(dedent('\n                import pickle\n\n                def _action():\n                    return 42\n\n                def round_trip():\n                    s = pickle.dumps(_action)\n                    return pickle.loads(s)\n                '), encoding='utf-8')
        module = import_path(fn, mode='importlib', root=tmp_path)
        round_trip = getattr(module, 'round_trip')
        action = round_trip()
        assert action() == 42

    def test_importmode_importlib_with_pickle_separate_modules(self, tmp_path: Path) -> None:
        if False:
            return 10
        '\n        Ensure that importlib mode works can load pickles that look similar but are\n        defined in separate modules.\n        '
        fn1 = tmp_path.joinpath('_src/m1/tests/test.py')
        fn1.parent.mkdir(parents=True)
        fn1.write_text(dedent('\n                import dataclasses\n                import pickle\n\n                @dataclasses.dataclass\n                class Data:\n                    x: int = 42\n                '), encoding='utf-8')
        fn2 = tmp_path.joinpath('_src/m2/tests/test.py')
        fn2.parent.mkdir(parents=True)
        fn2.write_text(dedent('\n                import dataclasses\n                import pickle\n\n                @dataclasses.dataclass\n                class Data:\n                    x: str = ""\n                '), encoding='utf-8')
        import pickle

        def round_trip(obj):
            if False:
                for i in range(10):
                    print('nop')
            s = pickle.dumps(obj)
            return pickle.loads(s)
        module = import_path(fn1, mode='importlib', root=tmp_path)
        Data1 = getattr(module, 'Data')
        module = import_path(fn2, mode='importlib', root=tmp_path)
        Data2 = getattr(module, 'Data')
        assert round_trip(Data1(20)) == Data1(20)
        assert round_trip(Data2('hello')) == Data2('hello')
        assert Data1.__module__ == '_src.m1.tests.test'
        assert Data2.__module__ == '_src.m2.tests.test'

    def test_module_name_from_path(self, tmp_path: Path) -> None:
        if False:
            while True:
                i = 10
        result = module_name_from_path(tmp_path / 'src/tests/test_foo.py', tmp_path)
        assert result == 'src.tests.test_foo'
        result = module_name_from_path(Path('/home/foo/test_foo.py'), Path('/bar'))
        assert result == 'home.foo.test_foo'
        result = module_name_from_path(tmp_path / 'src/app/__init__.py', tmp_path)
        assert result == 'src.app'
        result = module_name_from_path(tmp_path / '__init__.py', tmp_path)
        assert result == '__init__'

    def test_insert_missing_modules(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        if False:
            i = 10
            return i + 15
        monkeypatch.chdir(tmp_path)
        modules = {'xxx.tests.foo': ModuleType('xxx.tests.foo')}
        insert_missing_modules(modules, 'xxx.tests.foo')
        assert sorted(modules) == ['xxx', 'xxx.tests', 'xxx.tests.foo']
        mod = ModuleType('mod', doc='My Module')
        modules = {'xxy': mod}
        insert_missing_modules(modules, 'xxy')
        assert modules == {'xxy': mod}
        modules = {}
        insert_missing_modules(modules, '')
        assert modules == {}

    def test_parent_contains_child_module_attribute(self, monkeypatch: MonkeyPatch, tmp_path: Path):
        if False:
            for i in range(10):
                print('nop')
        monkeypatch.chdir(tmp_path)
        modules = {'xxx.tests.foo': ModuleType('xxx.tests.foo')}
        insert_missing_modules(modules, 'xxx.tests.foo')
        assert sorted(modules) == ['xxx', 'xxx.tests', 'xxx.tests.foo']
        assert modules['xxx'].tests is modules['xxx.tests']
        assert modules['xxx.tests'].foo is modules['xxx.tests.foo']

    def test_importlib_package(self, monkeypatch: MonkeyPatch, tmp_path: Path):
        if False:
            while True:
                i = 10
        "\n        Importing a package using --importmode=importlib should not import the\n        package's __init__.py file more than once (#11306).\n        "
        monkeypatch.chdir(tmp_path)
        monkeypatch.syspath_prepend(tmp_path)
        package_name = 'importlib_import_package'
        tmp_path.joinpath(package_name).mkdir()
        init = tmp_path.joinpath(f'{package_name}/__init__.py')
        init.write_text(dedent('\n                from .singleton import Singleton\n\n                instance = Singleton()\n                '), encoding='ascii')
        singleton = tmp_path.joinpath(f'{package_name}/singleton.py')
        singleton.write_text(dedent('\n                class Singleton:\n                    INSTANCES = []\n\n                    def __init__(self) -> None:\n                        self.INSTANCES.append(self)\n                        if len(self.INSTANCES) > 1:\n                            raise RuntimeError("Already initialized")\n                '), encoding='ascii')
        mod = import_path(init, root=tmp_path, mode=ImportMode.importlib)
        assert len(mod.instance.INSTANCES) == 1

    def test_importlib_root_is_package(self, pytester: Pytester) -> None:
        if False:
            return 10
        '\n        Regression for importing a `__init__`.py file that is at the root\n        (#11417).\n        '
        pytester.makepyfile(__init__='')
        pytester.makepyfile('\n            def test_my_test():\n                assert True\n            ')
        result = pytester.runpytest('--import-mode=importlib')
        result.stdout.fnmatch_lines('* 1 passed *')

def test_safe_exists(tmp_path: Path) -> None:
    if False:
        i = 10
        return i + 15
    d = tmp_path.joinpath('some_dir')
    d.mkdir()
    assert safe_exists(d) is True
    f = tmp_path.joinpath('some_file')
    f.touch()
    assert safe_exists(f) is True
    p = tmp_path.joinpath('some long filename' * 100)
    with unittest.mock.patch.object(Path, 'exists', autospec=True, side_effect=OSError(errno.ENAMETOOLONG, 'name too long')):
        assert safe_exists(p) is False
    with unittest.mock.patch.object(Path, 'exists', autospec=True, side_effect=ValueError('name too long')):
        assert safe_exists(p) is False