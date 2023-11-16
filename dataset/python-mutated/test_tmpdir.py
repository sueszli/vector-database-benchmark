import dataclasses
import os
import stat
import sys
import warnings
from pathlib import Path
from typing import Callable
from typing import cast
from typing import List
from typing import Union
import pytest
from _pytest import pathlib
from _pytest.config import Config
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pathlib import cleanup_numbered_dir
from _pytest.pathlib import create_cleanup_lock
from _pytest.pathlib import make_numbered_dir
from _pytest.pathlib import maybe_delete_a_numbered_dir
from _pytest.pathlib import on_rm_rf_error
from _pytest.pathlib import register_cleanup_lock_removal
from _pytest.pathlib import rm_rf
from _pytest.pytester import Pytester
from _pytest.tmpdir import get_user
from _pytest.tmpdir import TempPathFactory

def test_tmp_path_fixture(pytester: Pytester) -> None:
    if False:
        return 10
    p = pytester.copy_example('tmpdir/tmp_path_fixture.py')
    results = pytester.runpytest(p)
    results.stdout.fnmatch_lines(['*1 passed*'])

@dataclasses.dataclass
class FakeConfig:
    basetemp: Union[str, Path]

    @property
    def trace(self):
        if False:
            while True:
                i = 10
        return self

    def get(self, key):
        if False:
            return 10
        return lambda *k: None

    def getini(self, name):
        if False:
            i = 10
            return i + 15
        if name == 'tmp_path_retention_count':
            return 3
        elif name == 'tmp_path_retention_policy':
            return 'all'
        else:
            assert False

    @property
    def option(self):
        if False:
            while True:
                i = 10
        return self

class TestTmpPathHandler:

    def test_mktemp(self, tmp_path: Path) -> None:
        if False:
            i = 10
            return i + 15
        config = cast(Config, FakeConfig(tmp_path))
        t = TempPathFactory.from_config(config, _ispytest=True)
        tmp = t.mktemp('world')
        assert str(tmp.relative_to(t.getbasetemp())) == 'world0'
        tmp = t.mktemp('this')
        assert str(tmp.relative_to(t.getbasetemp())).startswith('this')
        tmp2 = t.mktemp('this')
        assert str(tmp2.relative_to(t.getbasetemp())).startswith('this')
        assert tmp2 != tmp

    def test_tmppath_relative_basetemp_absolute(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        if False:
            while True:
                i = 10
        '#4425'
        monkeypatch.chdir(tmp_path)
        config = cast(Config, FakeConfig('hello'))
        t = TempPathFactory.from_config(config, _ispytest=True)
        assert t.getbasetemp().resolve() == (tmp_path / 'hello').resolve()

class TestConfigTmpPath:

    def test_getbasetemp_custom_removes_old(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        mytemp = pytester.path.joinpath('xyz')
        p = pytester.makepyfile('\n            def test_1(tmp_path):\n                pass\n        ')
        pytester.runpytest(p, '--basetemp=%s' % mytemp)
        assert mytemp.exists()
        mytemp.joinpath('hello').touch()
        pytester.runpytest(p, '--basetemp=%s' % mytemp)
        assert mytemp.exists()
        assert not mytemp.joinpath('hello').exists()

    def test_policy_failed_removes_only_passed_dir(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = pytester.makepyfile('\n            def test_1(tmp_path):\n                assert 0 == 0\n            def test_2(tmp_path):\n                assert 0 == 1\n        ')
        pytester.makepyprojecttoml('\n            [tool.pytest.ini_options]\n            tmp_path_retention_policy = "failed"\n        ')
        pytester.inline_run(p)
        root = pytester._test_tmproot
        for child in root.iterdir():
            base_dir = list(filter(lambda x: x.is_dir() and (not x.is_symlink()), child.iterdir()))
            assert len(base_dir) == 1
            test_dir = list(filter(lambda x: x.is_dir() and (not x.is_symlink()), base_dir[0].iterdir()))
            assert len(test_dir) == 1
            assert test_dir[0].name == 'test_20'

    def test_policy_failed_removes_basedir_when_all_passed(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = pytester.makepyfile('\n            def test_1(tmp_path):\n                assert 0 == 0\n        ')
        pytester.makepyprojecttoml('\n            [tool.pytest.ini_options]\n            tmp_path_retention_policy = "failed"\n        ')
        pytester.inline_run(p)
        root = pytester._test_tmproot
        for child in root.iterdir():
            base_dir = filter(lambda x: not x.is_symlink(), child.iterdir())
            assert len(list(base_dir)) == 0

    def test_policy_failed_removes_dir_when_skipped_from_fixture(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        p = pytester.makepyfile('\n            import pytest\n\n            @pytest.fixture\n            def fixt(tmp_path):\n                pytest.skip()\n\n            def test_fixt(fixt):\n                pass\n        ')
        pytester.makepyprojecttoml('\n            [tool.pytest.ini_options]\n            tmp_path_retention_policy = "failed"\n        ')
        pytester.inline_run(p)
        root = pytester._test_tmproot
        for child in root.iterdir():
            base_dir = list(filter(lambda x: x.is_dir() and (not x.is_symlink()), child.iterdir()))
            assert len(base_dir) == 0

    def test_policy_all_keeps_dir_when_skipped_from_fixture(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        p = pytester.makepyfile('\n            import pytest\n\n            @pytest.fixture\n            def fixt(tmp_path):\n                pytest.skip()\n\n            def test_fixt(fixt):\n                pass\n        ')
        pytester.makepyprojecttoml('\n            [tool.pytest.ini_options]\n            tmp_path_retention_policy = "all"\n        ')
        pytester.inline_run(p)
        root = pytester._test_tmproot
        for child in root.iterdir():
            base_dir = list(filter(lambda x: x.is_dir() and (not x.is_symlink()), child.iterdir()))
            assert len(base_dir) == 1
            test_dir = list(filter(lambda x: x.is_dir() and (not x.is_symlink()), base_dir[0].iterdir()))
            assert len(test_dir) == 1
testdata = [('mypath', True), ('/mypath1', False), ('./mypath1', True), ('../mypath3', False), ('../../mypath4', False), ('mypath5/..', False), ('mypath6/../mypath6', True), ('mypath7/../mypath7/..', False)]

@pytest.mark.parametrize('basename, is_ok', testdata)
def test_mktemp(pytester: Pytester, basename: str, is_ok: bool) -> None:
    if False:
        while True:
            i = 10
    mytemp = pytester.mkdir('mytemp')
    p = pytester.makepyfile("\n        def test_abs_path(tmp_path_factory):\n            tmp_path_factory.mktemp('{}', numbered=False)\n        ".format(basename))
    result = pytester.runpytest(p, '--basetemp=%s' % mytemp)
    if is_ok:
        assert result.ret == 0
        assert mytemp.joinpath(basename).exists()
    else:
        assert result.ret == 1
        result.stdout.fnmatch_lines('*ValueError*')

def test_tmp_path_always_is_realpath(pytester: Pytester, monkeypatch) -> None:
    if False:
        for i in range(10):
            print('nop')
    realtemp = pytester.mkdir('myrealtemp')
    linktemp = pytester.path.joinpath('symlinktemp')
    attempt_symlink_to(linktemp, str(realtemp))
    monkeypatch.setenv('PYTEST_DEBUG_TEMPROOT', str(linktemp))
    pytester.makepyfile('\n        def test_1(tmp_path):\n            assert tmp_path.resolve() == tmp_path\n    ')
    reprec = pytester.inline_run()
    reprec.assertoutcome(passed=1)

def test_tmp_path_too_long_on_parametrization(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('\n        import pytest\n        @pytest.mark.parametrize("arg", ["1"*1000])\n        def test_some(arg, tmp_path):\n            tmp_path.joinpath("hello").touch()\n    ')
    reprec = pytester.inline_run()
    reprec.assertoutcome(passed=1)

def test_tmp_path_factory(pytester: Pytester) -> None:
    if False:
        print('Hello World!')
    pytester.makepyfile("\n        import pytest\n        @pytest.fixture(scope='session')\n        def session_dir(tmp_path_factory):\n            return tmp_path_factory.mktemp('data', numbered=False)\n        def test_some(session_dir):\n            assert session_dir.is_dir()\n    ")
    reprec = pytester.inline_run()
    reprec.assertoutcome(passed=1)

def test_tmp_path_fallback_tox_env(pytester: Pytester, monkeypatch) -> None:
    if False:
        i = 10
        return i + 15
    'Test that tmp_path works even if environment variables required by getpass\n    module are missing (#1010).\n    '
    monkeypatch.delenv('USER', raising=False)
    monkeypatch.delenv('USERNAME', raising=False)
    pytester.makepyfile('\n        def test_some(tmp_path):\n            assert tmp_path.is_dir()\n    ')
    reprec = pytester.inline_run()
    reprec.assertoutcome(passed=1)

@pytest.fixture
def break_getuser(monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr('os.getuid', lambda : -1)
    for envvar in ('LOGNAME', 'USER', 'LNAME', 'USERNAME'):
        monkeypatch.delenv(envvar, raising=False)

@pytest.mark.usefixtures('break_getuser')
@pytest.mark.skipif(sys.platform.startswith('win'), reason='no os.getuid on windows')
def test_tmp_path_fallback_uid_not_found(pytester: Pytester) -> None:
    if False:
        i = 10
        return i + 15
    "Test that tmp_path works even if the current process's user id does not\n    correspond to a valid user.\n    "
    pytester.makepyfile('\n        def test_some(tmp_path):\n            assert tmp_path.is_dir()\n    ')
    reprec = pytester.inline_run()
    reprec.assertoutcome(passed=1)

@pytest.mark.usefixtures('break_getuser')
@pytest.mark.skipif(sys.platform.startswith('win'), reason='no os.getuid on windows')
def test_get_user_uid_not_found():
    if False:
        for i in range(10):
            print('nop')
    "Test that get_user() function works even if the current process's\n    user id does not correspond to a valid user (e.g. running pytest in a\n    Docker container with 'docker run -u'.\n    "
    assert get_user() is None

@pytest.mark.skipif(not sys.platform.startswith('win'), reason='win only')
def test_get_user(monkeypatch):
    if False:
        while True:
            i = 10
    'Test that get_user() function works even if environment variables\n    required by getpass module are missing from the environment on Windows\n    (#1010).\n    '
    monkeypatch.delenv('USER', raising=False)
    monkeypatch.delenv('USERNAME', raising=False)
    assert get_user() is None

class TestNumberedDir:
    PREFIX = 'fun-'

    def test_make(self, tmp_path):
        if False:
            print('Hello World!')
        for i in range(10):
            d = make_numbered_dir(root=tmp_path, prefix=self.PREFIX)
            assert d.name.startswith(self.PREFIX)
            assert d.name.endswith(str(i))
        symlink = tmp_path.joinpath(self.PREFIX + 'current')
        if symlink.exists():
            assert symlink.is_symlink()
            assert symlink.resolve() == d.resolve()

    def test_cleanup_lock_create(self, tmp_path):
        if False:
            print('Hello World!')
        d = tmp_path.joinpath('test')
        d.mkdir()
        lockfile = create_cleanup_lock(d)
        with pytest.raises(OSError, match='cannot create lockfile in .*'):
            create_cleanup_lock(d)
        lockfile.unlink()

    def test_lock_register_cleanup_removal(self, tmp_path: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        lock = create_cleanup_lock(tmp_path)
        registry: List[Callable[..., None]] = []
        register_cleanup_lock_removal(lock, register=registry.append)
        (cleanup_func,) = registry
        assert lock.is_file()
        cleanup_func(original_pid='intentionally_different')
        assert lock.is_file()
        cleanup_func()
        assert not lock.exists()
        cleanup_func()
        assert not lock.exists()

    def _do_cleanup(self, tmp_path: Path, keep: int=2) -> None:
        if False:
            print('Hello World!')
        self.test_make(tmp_path)
        cleanup_numbered_dir(root=tmp_path, prefix=self.PREFIX, keep=keep, consider_lock_dead_if_created_before=0)

    def test_cleanup_keep(self, tmp_path):
        if False:
            while True:
                i = 10
        self._do_cleanup(tmp_path)
        (a, b) = (x for x in tmp_path.iterdir() if not x.is_symlink())
        print(a, b)

    def test_cleanup_keep_0(self, tmp_path: Path):
        if False:
            i = 10
            return i + 15
        self._do_cleanup(tmp_path, 0)
        dir_num = len(list(tmp_path.iterdir()))
        assert dir_num == 0

    def test_cleanup_locked(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        p = make_numbered_dir(root=tmp_path, prefix=self.PREFIX)
        create_cleanup_lock(p)
        assert not pathlib.ensure_deletable(p, consider_lock_dead_if_created_before=p.stat().st_mtime - 1)
        assert pathlib.ensure_deletable(p, consider_lock_dead_if_created_before=p.stat().st_mtime + 1)

    def test_cleanup_ignores_symlink(self, tmp_path):
        if False:
            return 10
        the_symlink = tmp_path / (self.PREFIX + 'current')
        attempt_symlink_to(the_symlink, tmp_path / (self.PREFIX + '5'))
        self._do_cleanup(tmp_path)

    def test_removal_accepts_lock(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        folder = make_numbered_dir(root=tmp_path, prefix=self.PREFIX)
        create_cleanup_lock(folder)
        maybe_delete_a_numbered_dir(folder)
        assert folder.is_dir()

class TestRmRf:

    def test_rm_rf(self, tmp_path):
        if False:
            while True:
                i = 10
        adir = tmp_path / 'adir'
        adir.mkdir()
        rm_rf(adir)
        assert not adir.exists()
        adir.mkdir()
        afile = adir / 'afile'
        afile.write_bytes(b'aa')
        rm_rf(adir)
        assert not adir.exists()

    def test_rm_rf_with_read_only_file(self, tmp_path):
        if False:
            while True:
                i = 10
        'Ensure rm_rf can remove directories with read-only files in them (#5524)'
        fn = tmp_path / 'dir/foo.txt'
        fn.parent.mkdir()
        fn.touch()
        self.chmod_r(fn)
        rm_rf(fn.parent)
        assert not fn.parent.is_dir()

    def chmod_r(self, path):
        if False:
            return 10
        mode = os.stat(str(path)).st_mode
        os.chmod(str(path), mode & ~stat.S_IWRITE)

    def test_rm_rf_with_read_only_directory(self, tmp_path):
        if False:
            for i in range(10):
                print('nop')
        'Ensure rm_rf can remove read-only directories (#5524)'
        adir = tmp_path / 'dir'
        adir.mkdir()
        (adir / 'foo.txt').touch()
        self.chmod_r(adir)
        rm_rf(adir)
        assert not adir.is_dir()

    def test_on_rm_rf_error(self, tmp_path: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        adir = tmp_path / 'dir'
        adir.mkdir()
        fn = adir / 'foo.txt'
        fn.touch()
        self.chmod_r(fn)
        with pytest.warns(pytest.PytestWarning):
            exc_info1 = (RuntimeError, RuntimeError(), None)
            on_rm_rf_error(os.unlink, str(fn), exc_info1, start_path=tmp_path)
            assert fn.is_file()
        exc_info2 = (FileNotFoundError, FileNotFoundError(), None)
        assert not on_rm_rf_error(None, str(fn), exc_info2, start_path=tmp_path)
        with pytest.warns(pytest.PytestWarning, match="^\\(rm_rf\\) unknown function None when removing .*foo.txt:\\n<class 'PermissionError'>: "):
            exc_info3 = (PermissionError, PermissionError(), None)
            on_rm_rf_error(None, str(fn), exc_info3, start_path=tmp_path)
            assert fn.is_file()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.warns(None) as warninfo:
                exc_info4 = PermissionError()
                on_rm_rf_error(os.open, str(fn), exc_info4, start_path=tmp_path)
                assert fn.is_file()
            assert not [x.message for x in warninfo]
        exc_info5 = PermissionError()
        on_rm_rf_error(os.unlink, str(fn), exc_info5, start_path=tmp_path)
        assert not fn.is_file()

def attempt_symlink_to(path, to_path):
    if False:
        print('Hello World!')
    'Try to make a symlink from "path" to "to_path", skipping in case this platform\n    does not support it or we don\'t have sufficient privileges (common on Windows).'
    try:
        Path(path).symlink_to(Path(to_path))
    except OSError:
        pytest.skip('could not create symbolic link')

def test_basetemp_with_read_only_files(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    'Integration test for #5524'
    pytester.makepyfile("\n        import os\n        import stat\n\n        def test(tmp_path):\n            fn = tmp_path / 'foo.txt'\n            fn.write_text('hello', encoding='utf-8')\n            mode = os.stat(str(fn)).st_mode\n            os.chmod(str(fn), mode & ~stat.S_IREAD)\n    ")
    result = pytester.runpytest('--basetemp=tmp')
    assert result.ret == 0
    result = pytester.runpytest('--basetemp=tmp')
    assert result.ret == 0

def test_tmp_path_factory_handles_invalid_dir_characters(tmp_path_factory: TempPathFactory, monkeypatch: MonkeyPatch) -> None:
    if False:
        for i in range(10):
            print('nop')
    monkeypatch.setattr('getpass.getuser', lambda : 'os/<:*?;>agnostic')
    monkeypatch.setattr(tmp_path_factory, '_basetemp', None)
    monkeypatch.setattr(tmp_path_factory, '_given_basetemp', None)
    p = tmp_path_factory.getbasetemp()
    assert 'pytest-of-unknown' in str(p)

@pytest.mark.skipif(not hasattr(os, 'getuid'), reason='checks unix permissions')
def test_tmp_path_factory_create_directory_with_safe_permissions(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Verify that pytest creates directories under /tmp with private permissions.'
    monkeypatch.setenv('PYTEST_DEBUG_TEMPROOT', str(tmp_path))
    tmp_factory = TempPathFactory(None, 3, 'all', lambda *args: None, _ispytest=True)
    basetemp = tmp_factory.getbasetemp()
    assert basetemp.stat().st_mode & 63 == 0
    assert basetemp.parent.stat().st_mode & 63 == 0

@pytest.mark.skipif(not hasattr(os, 'getuid'), reason='checks unix permissions')
def test_tmp_path_factory_fixes_up_world_readable_permissions(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    if False:
        return 10
    "Verify that if a /tmp/pytest-of-foo directory already exists with\n    world-readable permissions, it is fixed.\n\n    pytest used to mkdir with such permissions, that's why we fix it up.\n    "
    monkeypatch.setenv('PYTEST_DEBUG_TEMPROOT', str(tmp_path))
    tmp_factory = TempPathFactory(None, 3, 'all', lambda *args: None, _ispytest=True)
    basetemp = tmp_factory.getbasetemp()
    os.chmod(basetemp.parent, 511)
    assert basetemp.parent.stat().st_mode & 63 != 0
    tmp_factory = TempPathFactory(None, 3, 'all', lambda *args: None, _ispytest=True)
    basetemp = tmp_factory.getbasetemp()
    assert basetemp.parent.stat().st_mode & 63 == 0