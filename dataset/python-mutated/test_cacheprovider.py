import os
import shutil
from pathlib import Path
from typing import Generator
from typing import List
import pytest
from _pytest.config import ExitCode
from _pytest.monkeypatch import MonkeyPatch
from _pytest.pytester import Pytester
from _pytest.tmpdir import TempPathFactory
pytest_plugins = ('pytester',)

class TestNewAPI:

    def test_config_cache_mkdir(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makeini('[pytest]')
        config = pytester.parseconfigure()
        assert config.cache is not None
        with pytest.raises(ValueError):
            config.cache.mkdir('key/name')
        p = config.cache.mkdir('name')
        assert p.is_dir()

    def test_config_cache_dataerror(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makeini('[pytest]')
        config = pytester.parseconfigure()
        assert config.cache is not None
        cache = config.cache
        pytest.raises(TypeError, lambda : cache.set('key/name', cache))
        config.cache.set('key/name', 0)
        config.cache._getvaluepath('key/name').write_bytes(b'123invalid')
        val = config.cache.get('key/name', -2)
        assert val == -2

    @pytest.mark.filterwarnings('ignore:could not create cache path')
    def test_cache_writefail_cachfile_silent(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makeini('[pytest]')
        pytester.path.joinpath('.pytest_cache').write_text('gone wrong', encoding='utf-8')
        config = pytester.parseconfigure()
        cache = config.cache
        assert cache is not None
        cache.set('test/broken', [])

    @pytest.fixture
    def unwritable_cache_dir(self, pytester: Pytester) -> Generator[Path, None, None]:
        if False:
            return 10
        cache_dir = pytester.path.joinpath('.pytest_cache')
        cache_dir.mkdir()
        mode = cache_dir.stat().st_mode
        cache_dir.chmod(0)
        if os.access(cache_dir, os.W_OK):
            pytest.skip('Failed to make cache dir unwritable')
        yield cache_dir
        cache_dir.chmod(mode)

    @pytest.mark.filterwarnings('ignore:could not create cache path:pytest.PytestWarning')
    def test_cache_writefail_permissions(self, unwritable_cache_dir: Path, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makeini('[pytest]')
        config = pytester.parseconfigure()
        cache = config.cache
        assert cache is not None
        cache.set('test/broken', [])

    @pytest.mark.filterwarnings('default')
    def test_cache_failure_warns(self, pytester: Pytester, monkeypatch: MonkeyPatch, unwritable_cache_dir: Path) -> None:
        if False:
            print('Hello World!')
        monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
        pytester.makepyfile('def test_error(): raise Exception')
        result = pytester.runpytest()
        assert result.ret == 1
        result.stdout.fnmatch_lines(['*= warnings summary =*', '*/cacheprovider.py:*', f'  */cacheprovider.py:*: PytestCacheWarning: could not create cache path {unwritable_cache_dir}/v/cache/nodeids: *', '    config.cache.set("cache/nodeids", sorted(self.cached_nodeids))', '*1 failed, 3 warnings in*'])

    def test_config_cache(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makeconftest('\n            def pytest_configure(config):\n                # see that we get cache information early on\n                assert hasattr(config, "cache")\n        ')
        pytester.makepyfile('\n            def test_session(pytestconfig):\n                assert hasattr(pytestconfig, "cache")\n        ')
        result = pytester.runpytest()
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*1 passed*'])

    def test_cachefuncarg(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            import pytest\n            def test_cachefuncarg(cache):\n                val = cache.get("some/thing", None)\n                assert val is None\n                cache.set("some/thing", [1])\n                pytest.raises(TypeError, lambda: cache.get("some/thing"))\n                val = cache.get("some/thing", [])\n                assert val == [1]\n        ')
        result = pytester.runpytest()
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*1 passed*'])

    def test_custom_rel_cache_dir(self, pytester: Pytester) -> None:
        if False:
            return 10
        rel_cache_dir = os.path.join('custom_cache_dir', 'subdir')
        pytester.makeini('\n            [pytest]\n            cache_dir = {cache_dir}\n        '.format(cache_dir=rel_cache_dir))
        pytester.makepyfile(test_errored='def test_error():\n    assert False')
        pytester.runpytest()
        assert pytester.path.joinpath(rel_cache_dir).is_dir()

    def test_custom_abs_cache_dir(self, pytester: Pytester, tmp_path_factory: TempPathFactory) -> None:
        if False:
            for i in range(10):
                print('nop')
        tmp = tmp_path_factory.mktemp('tmp')
        abs_cache_dir = tmp / 'custom_cache_dir'
        pytester.makeini('\n            [pytest]\n            cache_dir = {cache_dir}\n        '.format(cache_dir=abs_cache_dir))
        pytester.makepyfile(test_errored='def test_error():\n    assert False')
        pytester.runpytest()
        assert abs_cache_dir.is_dir()

    def test_custom_cache_dir_with_env_var(self, pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
        if False:
            while True:
                i = 10
        monkeypatch.setenv('env_var', 'custom_cache_dir')
        pytester.makeini('\n            [pytest]\n            cache_dir = {cache_dir}\n        '.format(cache_dir='$env_var'))
        pytester.makepyfile(test_errored='def test_error():\n    assert False')
        pytester.runpytest()
        assert pytester.path.joinpath('custom_cache_dir').is_dir()

@pytest.mark.parametrize('env', ((), ('TOX_ENV_DIR', '/tox_env_dir')))
def test_cache_reportheader(env, pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
    if False:
        return 10
    pytester.makepyfile('def test_foo(): pass')
    if env:
        monkeypatch.setenv(*env)
        expected = os.path.join(env[1], '.pytest_cache')
    else:
        monkeypatch.delenv('TOX_ENV_DIR', raising=False)
        expected = '.pytest_cache'
    result = pytester.runpytest('-v')
    result.stdout.fnmatch_lines(['cachedir: %s' % expected])

def test_cache_reportheader_external_abspath(pytester: Pytester, tmp_path_factory: TempPathFactory) -> None:
    if False:
        i = 10
        return i + 15
    external_cache = tmp_path_factory.mktemp('test_cache_reportheader_external_abspath_abs')
    pytester.makepyfile('def test_hello(): pass')
    pytester.makeini('\n    [pytest]\n    cache_dir = {abscache}\n    '.format(abscache=external_cache))
    result = pytester.runpytest('-v')
    result.stdout.fnmatch_lines([f'cachedir: {external_cache}'])

def test_cache_show(pytester: Pytester) -> None:
    if False:
        return 10
    result = pytester.runpytest('--cache-show')
    assert result.ret == 0
    result.stdout.fnmatch_lines(['*cache is empty*'])
    pytester.makeconftest('\n        def pytest_configure(config):\n            config.cache.set("my/name", [1,2,3])\n            config.cache.set("my/hello", "world")\n            config.cache.set("other/some", {1:2})\n            dp = config.cache.mkdir("mydb")\n            dp.joinpath("hello").touch()\n            dp.joinpath("world").touch()\n    ')
    result = pytester.runpytest()
    assert result.ret == 5
    result = pytester.runpytest('--cache-show')
    result.stdout.fnmatch_lines(['*cachedir:*', "*- cache values for '[*]' -*", 'cache/nodeids contains:', 'my/name contains:', '  [1, 2, 3]', 'other/some contains:', "  {*'1': 2}", "*- cache directories for '[*]' -*", '*mydb/hello*length 0*', '*mydb/world*length 0*'])
    assert result.ret == 0
    result = pytester.runpytest('--cache-show', '*/hello')
    result.stdout.fnmatch_lines(['*cachedir:*', "*- cache values for '[*]/hello' -*", 'my/hello contains:', "  *'world'", "*- cache directories for '[*]/hello' -*", 'd/mydb/hello*length 0*'])
    stdout = result.stdout.str()
    assert 'other/some' not in stdout
    assert 'd/mydb/world' not in stdout
    assert result.ret == 0

class TestLastFailed:

    def test_lastfailed_usecase(self, pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
        if False:
            print('Hello World!')
        monkeypatch.setattr('sys.dont_write_bytecode', True)
        p = pytester.makepyfile('\n            def test_1(): assert 0\n            def test_2(): assert 0\n            def test_3(): assert 1\n            ')
        result = pytester.runpytest(str(p))
        result.stdout.fnmatch_lines(['*2 failed*'])
        p = pytester.makepyfile('\n            def test_1(): assert 1\n            def test_2(): assert 1\n            def test_3(): assert 0\n            ')
        result = pytester.runpytest(str(p), '--lf')
        result.stdout.fnmatch_lines(['collected 3 items / 1 deselected / 2 selected', 'run-last-failure: rerun previous 2 failures', '*= 2 passed, 1 deselected in *'])
        result = pytester.runpytest(str(p), '--lf')
        result.stdout.fnmatch_lines(['collected 3 items', 'run-last-failure: no previously failed tests, not deselecting items.', '*1 failed*2 passed*'])
        pytester.path.joinpath('.pytest_cache', '.git').mkdir(parents=True)
        result = pytester.runpytest(str(p), '--lf', '--cache-clear')
        result.stdout.fnmatch_lines(['*1 failed*2 passed*'])
        assert pytester.path.joinpath('.pytest_cache', 'README.md').is_file()
        assert pytester.path.joinpath('.pytest_cache', '.git').is_dir()
        if os.path.isdir('.pytest_cache'):
            shutil.rmtree('.pytest_cache')
        result = pytester.runpytest('--lf', '--cache-clear')
        result.stdout.fnmatch_lines(['*1 failed*2 passed*'])

    def test_failedfirst_order(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile(test_a='def test_always_passes(): pass', test_b='def test_always_fails(): assert 0')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['test_a.py*', 'test_b.py*'])
        result = pytester.runpytest('--ff')
        result.stdout.fnmatch_lines(['collected 2 items', 'run-last-failure: rerun previous 1 failure first', 'test_b.py*', 'test_a.py*'])

    def test_lastfailed_failedfirst_order(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile(test_a='def test_always_passes(): assert 1', test_b='def test_always_fails(): assert 0')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['test_a.py*', 'test_b.py*'])
        result = pytester.runpytest('--lf', '--ff')
        result.stdout.fnmatch_lines(['test_b.py*'])
        result.stdout.no_fnmatch_line('*test_a.py*')

    def test_lastfailed_difference_invocations(self, pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
        if False:
            for i in range(10):
                print('nop')
        monkeypatch.setattr('sys.dont_write_bytecode', True)
        pytester.makepyfile(test_a='\n                def test_a1(): assert 0\n                def test_a2(): assert 1\n            ', test_b='def test_b1(): assert 0')
        p = pytester.path.joinpath('test_a.py')
        p2 = pytester.path.joinpath('test_b.py')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*2 failed*'])
        result = pytester.runpytest('--lf', p2)
        result.stdout.fnmatch_lines(['*1 failed*'])
        pytester.makepyfile(test_b='def test_b1(): assert 1')
        result = pytester.runpytest('--lf', p2)
        result.stdout.fnmatch_lines(['*1 passed*'])
        result = pytester.runpytest('--lf', p)
        result.stdout.fnmatch_lines(['collected 2 items / 1 deselected / 1 selected', 'run-last-failure: rerun previous 1 failure', '*= 1 failed, 1 deselected in *'])

    def test_lastfailed_usecase_splice(self, pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
        if False:
            for i in range(10):
                print('nop')
        monkeypatch.setattr('sys.dont_write_bytecode', True)
        pytester.makepyfile('def test_1(): assert 0', test_something='def test_2(): assert 0')
        p2 = pytester.path.joinpath('test_something.py')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*2 failed*'])
        result = pytester.runpytest('--lf', p2)
        result.stdout.fnmatch_lines(['*1 failed*'])
        result = pytester.runpytest('--lf')
        result.stdout.fnmatch_lines(['*2 failed*'])

    def test_lastfailed_xpass(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.inline_runsource('\n            import pytest\n            @pytest.mark.xfail\n            def test_hello():\n                assert 1\n        ')
        config = pytester.parseconfigure()
        assert config.cache is not None
        lastfailed = config.cache.get('cache/lastfailed', -1)
        assert lastfailed == -1

    def test_non_serializable_parametrize(self, pytester: Pytester) -> None:
        if False:
            return 10
        "Test that failed parametrized tests with unmarshable parameters\n        don't break pytest-cache.\n        "
        pytester.makepyfile("\n            import pytest\n\n            @pytest.mark.parametrize('val', [\n                b'\\xac\\x10\\x02G',\n            ])\n            def test_fail(val):\n                assert False\n        ")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*1 failed in*'])

    @pytest.mark.parametrize('parent', ('session', 'package'))
    def test_terminal_report_lastfailed(self, pytester: Pytester, parent: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        if parent == 'package':
            pytester.makepyfile(__init__='')
        test_a = pytester.makepyfile(test_a='\n            def test_a1(): pass\n            def test_a2(): pass\n        ')
        test_b = pytester.makepyfile(test_b='\n            def test_b1(): assert 0\n            def test_b2(): assert 0\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['collected 4 items', '*2 failed, 2 passed in*'])
        result = pytester.runpytest('--lf')
        result.stdout.fnmatch_lines(['collected 2 items', 'run-last-failure: rerun previous 2 failures (skipped 1 file)', '*2 failed in*'])
        result = pytester.runpytest(test_a, '--lf')
        result.stdout.fnmatch_lines(['collected 2 items', 'run-last-failure: 2 known failures not in selected tests', '*2 passed in*'])
        result = pytester.runpytest(test_b, '--lf')
        result.stdout.fnmatch_lines(['collected 2 items', 'run-last-failure: rerun previous 2 failures', '*2 failed in*'])
        result = pytester.runpytest('test_b.py::test_b1', '--lf')
        result.stdout.fnmatch_lines(['collected 1 item', 'run-last-failure: rerun previous 1 failure', '*1 failed in*'])

    def test_terminal_report_failedfirst(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile(test_a='\n            def test_a1(): assert 0\n            def test_a2(): pass\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['collected 2 items', '*1 failed, 1 passed in*'])
        result = pytester.runpytest('--ff')
        result.stdout.fnmatch_lines(['collected 2 items', 'run-last-failure: rerun previous 1 failure first', '*1 failed, 1 passed in*'])

    def test_lastfailed_collectfailure(self, pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
        if False:
            return 10
        pytester.makepyfile(test_maybe="\n            import os\n            env = os.environ\n            if '1' == env['FAILIMPORT']:\n                raise ImportError('fail')\n            def test_hello():\n                assert '0' == env['FAILTEST']\n        ")

        def rlf(fail_import, fail_run):
            if False:
                for i in range(10):
                    print('nop')
            monkeypatch.setenv('FAILIMPORT', str(fail_import))
            monkeypatch.setenv('FAILTEST', str(fail_run))
            pytester.runpytest('-q')
            config = pytester.parseconfigure()
            assert config.cache is not None
            lastfailed = config.cache.get('cache/lastfailed', -1)
            return lastfailed
        lastfailed = rlf(fail_import=0, fail_run=0)
        assert lastfailed == -1
        lastfailed = rlf(fail_import=1, fail_run=0)
        assert list(lastfailed) == ['test_maybe.py']
        lastfailed = rlf(fail_import=0, fail_run=1)
        assert list(lastfailed) == ['test_maybe.py::test_hello']

    def test_lastfailed_failure_subset(self, pytester: Pytester, monkeypatch: MonkeyPatch) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile(test_maybe="\n            import os\n            env = os.environ\n            if '1' == env['FAILIMPORT']:\n                raise ImportError('fail')\n            def test_hello():\n                assert '0' == env['FAILTEST']\n        ")
        pytester.makepyfile(test_maybe2="\n            import os\n            env = os.environ\n            if '1' == env['FAILIMPORT']:\n                raise ImportError('fail')\n\n            def test_hello():\n                assert '0' == env['FAILTEST']\n\n            def test_pass():\n                pass\n        ")

        def rlf(fail_import, fail_run, args=()):
            if False:
                return 10
            monkeypatch.setenv('FAILIMPORT', str(fail_import))
            monkeypatch.setenv('FAILTEST', str(fail_run))
            result = pytester.runpytest('-q', '--lf', *args)
            config = pytester.parseconfigure()
            assert config.cache is not None
            lastfailed = config.cache.get('cache/lastfailed', -1)
            return (result, lastfailed)
        (result, lastfailed) = rlf(fail_import=0, fail_run=0)
        assert lastfailed == -1
        result.stdout.fnmatch_lines(['*3 passed*'])
        (result, lastfailed) = rlf(fail_import=1, fail_run=0)
        assert sorted(list(lastfailed)) == ['test_maybe.py', 'test_maybe2.py']
        (result, lastfailed) = rlf(fail_import=0, fail_run=0, args=('test_maybe2.py',))
        assert list(lastfailed) == ['test_maybe.py']
        (result, lastfailed) = rlf(fail_import=0, fail_run=0, args=('test_maybe2.py',))
        assert list(lastfailed) == ['test_maybe.py']
        result.stdout.fnmatch_lines(['*2 passed*'])

    def test_lastfailed_creates_cache_when_needed(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile(test_empty='')
        pytester.runpytest('-q', '--lf')
        assert not os.path.exists('.pytest_cache/v/cache/lastfailed')
        pytester.makepyfile(test_successful='def test_success():\n    assert True')
        pytester.runpytest('-q', '--lf')
        assert not os.path.exists('.pytest_cache/v/cache/lastfailed')
        pytester.makepyfile(test_errored='def test_error():\n    assert False')
        pytester.runpytest('-q', '--lf')
        assert os.path.exists('.pytest_cache/v/cache/lastfailed')

    def test_xfail_not_considered_failure(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile('\n            import pytest\n            @pytest.mark.xfail\n            def test(): assert 0\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*1 xfailed*'])
        assert self.get_cached_last_failed(pytester) == []

    def test_xfail_strict_considered_failure(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            import pytest\n            @pytest.mark.xfail(strict=True)\n            def test(): pass\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*1 failed*'])
        assert self.get_cached_last_failed(pytester) == ['test_xfail_strict_considered_failure.py::test']

    @pytest.mark.parametrize('mark', ['mark.xfail', 'mark.skip'])
    def test_failed_changed_to_xfail_or_skip(self, pytester: Pytester, mark: str) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('\n            import pytest\n            def test(): assert 0\n        ')
        result = pytester.runpytest()
        assert self.get_cached_last_failed(pytester) == ['test_failed_changed_to_xfail_or_skip.py::test']
        assert result.ret == 1
        pytester.makepyfile('\n            import pytest\n            @pytest.{mark}\n            def test(): assert 0\n        '.format(mark=mark))
        result = pytester.runpytest()
        assert result.ret == 0
        assert self.get_cached_last_failed(pytester) == []
        assert result.ret == 0

    @pytest.mark.parametrize('quiet', [True, False])
    @pytest.mark.parametrize('opt', ['--ff', '--lf'])
    def test_lf_and_ff_prints_no_needless_message(self, quiet: bool, opt: str, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('def test(): assert 0')
        args = [opt]
        if quiet:
            args.append('-q')
        result = pytester.runpytest(*args)
        result.stdout.no_fnmatch_line('*run all*')
        result = pytester.runpytest(*args)
        if quiet:
            result.stdout.no_fnmatch_line('*run all*')
        else:
            assert 'rerun previous' in result.stdout.str()

    def get_cached_last_failed(self, pytester: Pytester) -> List[str]:
        if False:
            print('Hello World!')
        config = pytester.parseconfigure()
        assert config.cache is not None
        return sorted(config.cache.get('cache/lastfailed', {}))

    def test_cache_cumulative(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        'Test workflow where user fixes errors gradually file by file using --lf.'
        test_bar = pytester.makepyfile(test_bar='\n            def test_bar_1(): pass\n            def test_bar_2(): assert 0\n        ')
        test_foo = pytester.makepyfile(test_foo='\n            def test_foo_3(): pass\n            def test_foo_4(): assert 0\n        ')
        pytester.runpytest()
        assert self.get_cached_last_failed(pytester) == ['test_bar.py::test_bar_2', 'test_foo.py::test_foo_4']
        pytester.makepyfile(test_bar='\n            def test_bar_1(): pass\n            def test_bar_2(): pass\n        ')
        result = pytester.runpytest(test_bar)
        result.stdout.fnmatch_lines(['*2 passed*'])
        assert self.get_cached_last_failed(pytester) == ['test_foo.py::test_foo_4']
        result = pytester.runpytest('--last-failed')
        result.stdout.fnmatch_lines(['collected 1 item', 'run-last-failure: rerun previous 1 failure (skipped 1 file)', '*= 1 failed in *'])
        assert self.get_cached_last_failed(pytester) == ['test_foo.py::test_foo_4']
        test_foo = pytester.makepyfile(test_foo='\n            def test_foo_3(): pass\n            def test_foo_4(): pass\n        ')
        result = pytester.runpytest(test_foo, '--last-failed')
        result.stdout.fnmatch_lines(['collected 2 items / 1 deselected / 1 selected', 'run-last-failure: rerun previous 1 failure', '*= 1 passed, 1 deselected in *'])
        assert self.get_cached_last_failed(pytester) == []
        result = pytester.runpytest('--last-failed')
        result.stdout.fnmatch_lines(['*4 passed*'])
        assert self.get_cached_last_failed(pytester) == []

    def test_lastfailed_no_failures_behavior_all_passed(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('\n            def test_1(): pass\n            def test_2(): pass\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*2 passed*'])
        result = pytester.runpytest('--lf')
        result.stdout.fnmatch_lines(['*2 passed*'])
        result = pytester.runpytest('--lf', '--lfnf', 'all')
        result.stdout.fnmatch_lines(['*2 passed*'])
        pytester.makeconftest('\n            deselected = []\n\n            def pytest_deselected(items):\n                global deselected\n                deselected = items\n\n            def pytest_sessionfinish():\n                print("\\ndeselected={}".format(len(deselected)))\n        ')
        result = pytester.runpytest('--lf', '--lfnf', 'none')
        result.stdout.fnmatch_lines(['collected 2 items / 2 deselected / 0 selected', 'run-last-failure: no previously failed tests, deselecting all items.', 'deselected=2', '* 2 deselected in *'])
        assert result.ret == ExitCode.NO_TESTS_COLLECTED

    def test_lastfailed_no_failures_behavior_empty_cache(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        pytester.makepyfile('\n            def test_1(): pass\n            def test_2(): assert 0\n        ')
        result = pytester.runpytest('--lf', '--cache-clear')
        result.stdout.fnmatch_lines(['*1 failed*1 passed*'])
        result = pytester.runpytest('--lf', '--cache-clear', '--lfnf', 'all')
        result.stdout.fnmatch_lines(['*1 failed*1 passed*'])
        result = pytester.runpytest('--lf', '--cache-clear', '--lfnf', 'none')
        result.stdout.fnmatch_lines(['*2 desel*'])

    def test_lastfailed_skip_collection(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        '\n        Test --lf behavior regarding skipping collection of files that are not marked as\n        failed in the cache (#5172).\n        '
        pytester.makepyfile(**{'pkg1/test_1.py': "\n                import pytest\n\n                @pytest.mark.parametrize('i', range(3))\n                def test_1(i): pass\n            ", 'pkg2/test_2.py': "\n                import pytest\n\n                @pytest.mark.parametrize('i', range(5))\n                def test_1(i):\n                    assert i not in (1, 3)\n            "})
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['collected 8 items', '*2 failed*6 passed*'])
        result = pytester.runpytest('--lf')
        result.stdout.fnmatch_lines(['collected 2 items', 'run-last-failure: rerun previous 2 failures (skipped 1 file)', '*= 2 failed in *'])
        pytester.makepyfile(**{'pkg1/test_3.py': '\n                def test_3(): pass\n            '})
        result = pytester.runpytest('--lf')
        result.stdout.fnmatch_lines(['collected 2 items', 'run-last-failure: rerun previous 2 failures (skipped 2 files)', '*= 2 failed in *'])

    def test_lastfailed_skip_collection_with_nesting(self, pytester: Pytester) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Check that file skipping works even when the file with failures is\n        nested at a different level of the collection tree.'
        pytester.makepyfile(**{'test_1.py': '\n                    def test_1(): pass\n                ', 'pkg/__init__.py': '', 'pkg/test_2.py': '\n                    def test_2(): assert False\n                '})
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['collected 2 items', '*1 failed*1 passed*'])
        result = pytester.runpytest('--lf')
        result.stdout.fnmatch_lines(['collected 1 item', 'run-last-failure: rerun previous 1 failure (skipped 1 file)', '*= 1 failed in *'])

    def test_lastfailed_with_known_failures_not_being_selected(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile(**{'pkg1/test_1.py': 'def test_1(): assert 0', 'pkg1/test_2.py': 'def test_2(): pass'})
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['collected 2 items', '* 1 failed, 1 passed in *'])
        Path('pkg1/test_1.py').unlink()
        result = pytester.runpytest('--lf')
        result.stdout.fnmatch_lines(['collected 1 item', 'run-last-failure: 1 known failures not in selected tests', '* 1 passed in *'])
        pytester.makepyfile(**{'pkg1/test_1.py': 'def test_1(): assert 0'})
        result = pytester.runpytest('--lf')
        result.stdout.fnmatch_lines(['collected 1 item', 'run-last-failure: rerun previous 1 failure (skipped 1 file)', '* 1 failed in *'])
        pytester.makepyfile(**{'pkg1/test_1.py': 'def test_renamed(): assert 0'})
        result = pytester.runpytest('--lf', '-rf')
        result.stdout.fnmatch_lines(['collected 2 items', 'run-last-failure: 1 known failures not in selected tests', 'pkg1/test_1.py F *', 'pkg1/test_2.py . *', 'FAILED pkg1/test_1.py::test_renamed - assert 0', '* 1 failed, 1 passed in *'])
        result = pytester.runpytest('--lf', '--co')
        result.stdout.fnmatch_lines(['collected 1 item', 'run-last-failure: rerun previous 1 failure (skipped 1 file)', '', '<Module pkg1/test_1.py>', '  <Function test_renamed>'])

    def test_lastfailed_args_with_deselected(self, pytester: Pytester) -> None:
        if False:
            return 10
        'Test regression with --lf running into NoMatch error.\n\n        This was caused by it not collecting (non-failed) nodes given as\n        arguments.\n        '
        pytester.makepyfile(**{'pkg1/test_1.py': '\n                    def test_pass(): pass\n                    def test_fail(): assert 0\n                '})
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['collected 2 items', '* 1 failed, 1 passed in *'])
        assert result.ret == 1
        result = pytester.runpytest('pkg1/test_1.py::test_pass', '--lf', '--co')
        assert result.ret == 0
        result.stdout.fnmatch_lines(['*collected 1 item', 'run-last-failure: 1 known failures not in selected tests', '', '<Module pkg1/test_1.py>', '  <Function test_pass>'], consecutive=True)
        result = pytester.runpytest('pkg1/test_1.py::test_pass', 'pkg1/test_1.py::test_fail', '--lf', '--co')
        assert result.ret == 0
        result.stdout.fnmatch_lines(['collected 2 items / 1 deselected / 1 selected', 'run-last-failure: rerun previous 1 failure', '', '<Module pkg1/test_1.py>', '  <Function test_fail>', '*= 1/2 tests collected (1 deselected) in *'])

    def test_lastfailed_with_class_items(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        'Test regression with --lf deselecting whole classes.'
        pytester.makepyfile(**{'pkg1/test_1.py': '\n                    class TestFoo:\n                        def test_pass(self): pass\n                        def test_fail(self): assert 0\n\n                    def test_other(): assert 0\n                '})
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['collected 3 items', '* 2 failed, 1 passed in *'])
        assert result.ret == 1
        result = pytester.runpytest('--lf', '--co')
        assert result.ret == 0
        result.stdout.fnmatch_lines(['collected 3 items / 1 deselected / 2 selected', 'run-last-failure: rerun previous 2 failures', '', '<Module pkg1/test_1.py>', '  <Class TestFoo>', '    <Function test_fail>', '  <Function test_other>', '', '*= 2/3 tests collected (1 deselected) in *'], consecutive=True)

    def test_lastfailed_with_all_filtered(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile(**{'pkg1/test_1.py': '\n                    def test_fail(): assert 0\n                    def test_pass(): pass\n                '})
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['collected 2 items', '* 1 failed, 1 passed in *'])
        assert result.ret == 1
        pytester.makepyfile(**{'pkg1/test_1.py': '\n                    def test_pass(): pass\n                '})
        result = pytester.runpytest('--lf', '--co')
        result.stdout.fnmatch_lines(['collected 1 item', 'run-last-failure: 1 known failures not in selected tests', '', '<Module pkg1/test_1.py>', '  <Function test_pass>', '', '*= 1 test collected in*'], consecutive=True)
        assert result.ret == 0

    def test_packages(self, pytester: Pytester) -> None:
        if False:
            return 10
        "Regression test for #7758.\n\n        The particular issue here was that Package nodes were included in the\n        filtering, being themselves Modules for the __init__.py, even if they\n        had failed Modules in them.\n\n        The tests includes a test in an __init__.py file just to make sure the\n        fix doesn't somehow regress that, it is not critical for the issue.\n        "
        pytester.makepyfile(**{'__init__.py': '', 'a/__init__.py': 'def test_a_init(): assert False', 'a/test_one.py': 'def test_1(): assert False', 'b/__init__.py': '', 'b/test_two.py': 'def test_2(): assert False'})
        pytester.makeini('\n            [pytest]\n            python_files = *.py\n            ')
        result = pytester.runpytest()
        result.assert_outcomes(failed=3)
        result = pytester.runpytest('--lf')
        result.assert_outcomes(failed=3)

    def test_non_python_file_skipped(self, pytester: Pytester, dummy_yaml_custom_test: None) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile(**{'test_bad.py': 'def test_bad(): assert False'})
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['collected 2 items', '* 1 failed, 1 passed in *'])
        result = pytester.runpytest('--lf')
        result.stdout.fnmatch_lines(['collected 1 item', 'run-last-failure: rerun previous 1 failure (skipped 1 file)', '* 1 failed in *'])

class TestNewFirst:

    def test_newfirst_usecase(self, pytester: Pytester) -> None:
        if False:
            return 10
        pytester.makepyfile(**{'test_1/test_1.py': '\n                def test_1(): assert 1\n            ', 'test_2/test_2.py': '\n                def test_1(): assert 1\n            '})
        p1 = pytester.path.joinpath('test_1/test_1.py')
        os.utime(p1, ns=(p1.stat().st_atime_ns, int(1000000000.0)))
        result = pytester.runpytest('-v')
        result.stdout.fnmatch_lines(['*test_1/test_1.py::test_1 PASSED*', '*test_2/test_2.py::test_1 PASSED*'])
        result = pytester.runpytest('-v', '--nf')
        result.stdout.fnmatch_lines(['*test_2/test_2.py::test_1 PASSED*', '*test_1/test_1.py::test_1 PASSED*'])
        p1.write_text('def test_1(): assert 1\ndef test_2(): assert 1\n', encoding='utf-8')
        os.utime(p1, ns=(p1.stat().st_atime_ns, int(1000000000.0)))
        result = pytester.runpytest('--nf', '--collect-only', '-q')
        result.stdout.fnmatch_lines(['test_1/test_1.py::test_2', 'test_2/test_2.py::test_1', 'test_1/test_1.py::test_1'])
        pytester.makepyfile(myplugin='\n            def pytest_collection_modifyitems(items):\n                items[:] = sorted(items, key=lambda item: item.nodeid)\n                print("new_items:", [x.nodeid for x in items])\n            ')
        pytester.syspathinsert()
        result = pytester.runpytest('--nf', '-p', 'myplugin', '--collect-only', '-q')
        result.stdout.fnmatch_lines(['new_items: *test_1.py*test_1.py*test_2.py*', 'test_1/test_1.py::test_2', 'test_2/test_2.py::test_1', 'test_1/test_1.py::test_1'])

    def test_newfirst_parametrize(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile(**{'test_1/test_1.py': "\n                import pytest\n                @pytest.mark.parametrize('num', [1, 2])\n                def test_1(num): assert num\n            ", 'test_2/test_2.py': "\n                import pytest\n                @pytest.mark.parametrize('num', [1, 2])\n                def test_1(num): assert num\n            "})
        p1 = pytester.path.joinpath('test_1/test_1.py')
        os.utime(p1, ns=(p1.stat().st_atime_ns, int(1000000000.0)))
        result = pytester.runpytest('-v')
        result.stdout.fnmatch_lines(['*test_1/test_1.py::test_1[1*', '*test_1/test_1.py::test_1[2*', '*test_2/test_2.py::test_1[1*', '*test_2/test_2.py::test_1[2*'])
        result = pytester.runpytest('-v', '--nf')
        result.stdout.fnmatch_lines(['*test_2/test_2.py::test_1[1*', '*test_2/test_2.py::test_1[2*', '*test_1/test_1.py::test_1[1*', '*test_1/test_1.py::test_1[2*'])
        p1.write_text("import pytest\n@pytest.mark.parametrize('num', [1, 2, 3])\ndef test_1(num): assert num\n", encoding='utf-8')
        os.utime(p1, ns=(p1.stat().st_atime_ns, int(1000000000.0)))
        result = pytester.runpytest('-v', '--nf', 'test_2/test_2.py')
        result.stdout.fnmatch_lines(['*test_2/test_2.py::test_1[1*', '*test_2/test_2.py::test_1[2*'])
        result = pytester.runpytest('-v', '--nf')
        result.stdout.fnmatch_lines(['*test_1/test_1.py::test_1[3*', '*test_2/test_2.py::test_1[1*', '*test_2/test_2.py::test_1[2*', '*test_1/test_1.py::test_1[1*', '*test_1/test_1.py::test_1[2*'])

class TestReadme:

    def check_readme(self, pytester: Pytester) -> bool:
        if False:
            for i in range(10):
                print('nop')
        config = pytester.parseconfigure()
        assert config.cache is not None
        readme = config.cache._cachedir.joinpath('README.md')
        return readme.is_file()

    def test_readme_passed(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile('def test_always_passes(): pass')
        pytester.runpytest()
        assert self.check_readme(pytester) is True

    def test_readme_failed(self, pytester: Pytester) -> None:
        if False:
            print('Hello World!')
        pytester.makepyfile('def test_always_fails(): assert 0')
        pytester.runpytest()
        assert self.check_readme(pytester) is True

def test_gitignore(pytester: Pytester) -> None:
    if False:
        return 10
    'Ensure we automatically create .gitignore file in the pytest_cache directory (#3286).'
    from _pytest.cacheprovider import Cache
    config = pytester.parseconfig()
    cache = Cache.for_config(config, _ispytest=True)
    cache.set('foo', 'bar')
    msg = '# Created by pytest automatically.\n*\n'
    gitignore_path = cache._cachedir.joinpath('.gitignore')
    assert gitignore_path.read_text(encoding='UTF-8') == msg
    gitignore_path.write_text('custom', encoding='utf-8')
    cache.set('something', 'else')
    assert gitignore_path.read_text(encoding='UTF-8') == 'custom'

def test_preserve_keys_order(pytester: Pytester) -> None:
    if False:
        return 10
    'Ensure keys order is preserved when saving dicts (#9205).'
    from _pytest.cacheprovider import Cache
    config = pytester.parseconfig()
    cache = Cache.for_config(config, _ispytest=True)
    cache.set('foo', {'z': 1, 'b': 2, 'a': 3, 'd': 10})
    read_back = cache.get('foo', None)
    assert list(read_back.items()) == [('z', 1), ('b', 2), ('a', 3), ('d', 10)]

def test_does_not_create_boilerplate_in_existing_dirs(pytester: Pytester) -> None:
    if False:
        while True:
            i = 10
    from _pytest.cacheprovider import Cache
    pytester.makeini('\n        [pytest]\n        cache_dir = .\n        ')
    config = pytester.parseconfig()
    cache = Cache.for_config(config, _ispytest=True)
    cache.set('foo', 'bar')
    assert os.path.isdir('v')
    assert not os.path.exists('.gitignore')
    assert not os.path.exists('README.md')

def test_cachedir_tag(pytester: Pytester) -> None:
    if False:
        return 10
    'Ensure we automatically create CACHEDIR.TAG file in the pytest_cache directory (#4278).'
    from _pytest.cacheprovider import Cache
    from _pytest.cacheprovider import CACHEDIR_TAG_CONTENT
    config = pytester.parseconfig()
    cache = Cache.for_config(config, _ispytest=True)
    cache.set('foo', 'bar')
    cachedir_tag_path = cache._cachedir.joinpath('CACHEDIR.TAG')
    assert cachedir_tag_path.read_bytes() == CACHEDIR_TAG_CONTENT

def test_clioption_with_cacheshow_and_help(pytester: Pytester) -> None:
    if False:
        return 10
    result = pytester.runpytest('--cache-show', '--help')
    assert result.ret == 0