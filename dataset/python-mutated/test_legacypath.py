from pathlib import Path
import pytest
from _pytest.compat import LEGACY_PATH
from _pytest.fixtures import TopRequest
from _pytest.legacypath import TempdirFactory
from _pytest.legacypath import Testdir

def test_item_fspath(pytester: pytest.Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makepyfile('def test_func(): pass')
    (items, hookrec) = pytester.inline_genitems()
    assert len(items) == 1
    (item,) = items
    (items2, hookrec) = pytester.inline_genitems(item.nodeid)
    (item2,) = items2
    assert item2.name == item.name
    assert item2.fspath == item.fspath
    assert item2.path == item.path

def test_testdir_testtmproot(testdir: Testdir) -> None:
    if False:
        i = 10
        return i + 15
    'Check test_tmproot is a py.path attribute for backward compatibility.'
    assert testdir.test_tmproot.check(dir=1)

def test_testdir_makefile_dot_prefixes_extension_silently(testdir: Testdir) -> None:
    if False:
        for i in range(10):
            print('nop')
    'For backwards compat #8192'
    p1 = testdir.makefile('foo.bar', '')
    assert '.foo.bar' in str(p1)

def test_testdir_makefile_ext_none_raises_type_error(testdir: Testdir) -> None:
    if False:
        while True:
            i = 10
    'For backwards compat #8192'
    with pytest.raises(TypeError):
        testdir.makefile(None, '')

def test_testdir_makefile_ext_empty_string_makes_file(testdir: Testdir) -> None:
    if False:
        print('Hello World!')
    'For backwards compat #8192'
    p1 = testdir.makefile('', '')
    assert 'test_testdir_makefile' in str(p1)

def attempt_symlink_to(path: str, to_path: str) -> None:
    if False:
        while True:
            i = 10
    'Try to make a symlink from "path" to "to_path", skipping in case this platform\n    does not support it or we don\'t have sufficient privileges (common on Windows).'
    try:
        Path(path).symlink_to(Path(to_path))
    except OSError:
        pytest.skip('could not create symbolic link')

def test_tmpdir_factory(tmpdir_factory: TempdirFactory, tmp_path_factory: pytest.TempPathFactory) -> None:
    if False:
        print('Hello World!')
    assert str(tmpdir_factory.getbasetemp()) == str(tmp_path_factory.getbasetemp())
    dir = tmpdir_factory.mktemp('foo')
    assert dir.exists()

def test_tmpdir_equals_tmp_path(tmpdir: LEGACY_PATH, tmp_path: Path) -> None:
    if False:
        print('Hello World!')
    assert Path(tmpdir) == tmp_path

def test_tmpdir_always_is_realpath(pytester: pytest.Pytester) -> None:
    if False:
        print('Hello World!')
    realtemp = pytester.mkdir('myrealtemp')
    linktemp = pytester.path.joinpath('symlinktemp')
    attempt_symlink_to(str(linktemp), str(realtemp))
    p = pytester.makepyfile('\n        def test_1(tmpdir):\n            import os\n            assert os.path.realpath(str(tmpdir)) == str(tmpdir)\n    ')
    result = pytester.runpytest('-s', p, '--basetemp=%s/bt' % linktemp)
    assert not result.ret

def test_cache_makedir(cache: pytest.Cache) -> None:
    if False:
        for i in range(10):
            print('nop')
    dir = cache.makedir('foo')
    assert dir.exists()
    dir.remove()

def test_fixturerequest_getmodulepath(pytester: pytest.Pytester) -> None:
    if False:
        for i in range(10):
            print('nop')
    modcol = pytester.getmodulecol('def test_somefunc(): pass')
    (item,) = pytester.genitems([modcol])
    assert isinstance(item, pytest.Function)
    req = TopRequest(item, _ispytest=True)
    assert req.path == modcol.path
    assert req.fspath == modcol.fspath

class TestFixtureRequestSessionScoped:

    @pytest.fixture(scope='session')
    def session_request(self, request):
        if False:
            print('Hello World!')
        return request

    def test_session_scoped_unavailable_attributes(self, session_request):
        if False:
            while True:
                i = 10
        with pytest.raises(AttributeError, match='path not available in session-scoped context'):
            session_request.fspath

@pytest.mark.parametrize('config_type', ['ini', 'pyproject'])
def test_addini_paths(pytester: pytest.Pytester, config_type: str) -> None:
    if False:
        print('Hello World!')
    pytester.makeconftest('\n        def pytest_addoption(parser):\n            parser.addini("paths", "my new ini value", type="pathlist")\n            parser.addini("abc", "abc value")\n    ')
    if config_type == 'ini':
        inipath = pytester.makeini('\n            [pytest]\n            paths=hello world/sub.py\n        ')
    elif config_type == 'pyproject':
        inipath = pytester.makepyprojecttoml('\n            [tool.pytest.ini_options]\n            paths=["hello", "world/sub.py"]\n        ')
    config = pytester.parseconfig()
    values = config.getini('paths')
    assert len(values) == 2
    assert values[0] == inipath.parent.joinpath('hello')
    assert values[1] == inipath.parent.joinpath('world/sub.py')
    pytest.raises(ValueError, config.getini, 'other')

def test_override_ini_paths(pytester: pytest.Pytester) -> None:
    if False:
        while True:
            i = 10
    pytester.makeconftest('\n        def pytest_addoption(parser):\n            parser.addini("paths", "my new ini value", type="pathlist")')
    pytester.makeini('\n        [pytest]\n        paths=blah.py')
    pytester.makepyfile('\n        def test_overriden(pytestconfig):\n            config_paths = pytestconfig.getini("paths")\n            print(config_paths)\n            for cpf in config_paths:\n                print(\'\\nuser_path:%s\' % cpf.basename)\n        ')
    result = pytester.runpytest('--override-ini', 'paths=foo/bar1.py foo/bar2.py', '-s')
    result.stdout.fnmatch_lines(['user_path:bar1.py', 'user_path:bar2.py'])

def test_inifile_from_cmdline_main_hook(pytester: pytest.Pytester) -> None:
    if False:
        i = 10
        return i + 15
    'Ensure Config.inifile is available during pytest_cmdline_main (#9396).'
    p = pytester.makeini('\n        [pytest]\n        ')
    pytester.makeconftest('\n        def pytest_cmdline_main(config):\n            print("pytest_cmdline_main inifile =", config.inifile)\n        ')
    result = pytester.runpytest_subprocess('-s')
    result.stdout.fnmatch_lines(f'*pytest_cmdline_main inifile = {p}')