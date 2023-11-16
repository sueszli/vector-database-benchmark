"""Implementation of the cache provider."""
import dataclasses
import json
import os
from pathlib import Path
from typing import Dict
from typing import final
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Union
from .pathlib import resolve_from_str
from .pathlib import rm_rf
from .reports import CollectReport
from _pytest import nodes
from _pytest._io import TerminalWriter
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.nodes import File
from _pytest.python import Package
from _pytest.reports import TestReport
README_CONTENT = "# pytest cache directory #\n\nThis directory contains data from the pytest's cache plugin,\nwhich provides the `--lf` and `--ff` options, as well as the `cache` fixture.\n\n**Do not** commit this to version control.\n\nSee [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.\n"
CACHEDIR_TAG_CONTENT = b'Signature: 8a477f597d28d172789f06886806bc55\n# This file is a cache directory tag created by pytest.\n# For information about cache directory tags, see:\n#\thttps://bford.info/cachedir/spec.html\n'

@final
@dataclasses.dataclass
class Cache:
    """Instance of the `cache` fixture."""
    _cachedir: Path = dataclasses.field(repr=False)
    _config: Config = dataclasses.field(repr=False)
    _CACHE_PREFIX_DIRS = 'd'
    _CACHE_PREFIX_VALUES = 'v'

    def __init__(self, cachedir: Path, config: Config, *, _ispytest: bool=False) -> None:
        if False:
            return 10
        check_ispytest(_ispytest)
        self._cachedir = cachedir
        self._config = config

    @classmethod
    def for_config(cls, config: Config, *, _ispytest: bool=False) -> 'Cache':
        if False:
            print('Hello World!')
        'Create the Cache instance for a Config.\n\n        :meta private:\n        '
        check_ispytest(_ispytest)
        cachedir = cls.cache_dir_from_config(config, _ispytest=True)
        if config.getoption('cacheclear') and cachedir.is_dir():
            cls.clear_cache(cachedir, _ispytest=True)
        return cls(cachedir, config, _ispytest=True)

    @classmethod
    def clear_cache(cls, cachedir: Path, _ispytest: bool=False) -> None:
        if False:
            return 10
        'Clear the sub-directories used to hold cached directories and values.\n\n        :meta private:\n        '
        check_ispytest(_ispytest)
        for prefix in (cls._CACHE_PREFIX_DIRS, cls._CACHE_PREFIX_VALUES):
            d = cachedir / prefix
            if d.is_dir():
                rm_rf(d)

    @staticmethod
    def cache_dir_from_config(config: Config, *, _ispytest: bool=False) -> Path:
        if False:
            return 10
        'Get the path to the cache directory for a Config.\n\n        :meta private:\n        '
        check_ispytest(_ispytest)
        return resolve_from_str(config.getini('cache_dir'), config.rootpath)

    def warn(self, fmt: str, *, _ispytest: bool=False, **args: object) -> None:
        if False:
            while True:
                i = 10
        'Issue a cache warning.\n\n        :meta private:\n        '
        check_ispytest(_ispytest)
        import warnings
        from _pytest.warning_types import PytestCacheWarning
        warnings.warn(PytestCacheWarning(fmt.format(**args) if args else fmt), self._config.hook, stacklevel=3)

    def mkdir(self, name: str) -> Path:
        if False:
            print('Hello World!')
        'Return a directory path object with the given name.\n\n        If the directory does not yet exist, it will be created. You can use\n        it to manage files to e.g. store/retrieve database dumps across test\n        sessions.\n\n        .. versionadded:: 7.0\n\n        :param name:\n            Must be a string not containing a ``/`` separator.\n            Make sure the name contains your plugin or application\n            identifiers to prevent clashes with other cache users.\n        '
        path = Path(name)
        if len(path.parts) > 1:
            raise ValueError('name is not allowed to contain path separators')
        res = self._cachedir.joinpath(self._CACHE_PREFIX_DIRS, path)
        res.mkdir(exist_ok=True, parents=True)
        return res

    def _getvaluepath(self, key: str) -> Path:
        if False:
            return 10
        return self._cachedir.joinpath(self._CACHE_PREFIX_VALUES, Path(key))

    def get(self, key: str, default):
        if False:
            print('Hello World!')
        'Return the cached value for the given key.\n\n        If no value was yet cached or the value cannot be read, the specified\n        default is returned.\n\n        :param key:\n            Must be a ``/`` separated value. Usually the first\n            name is the name of your plugin or your application.\n        :param default:\n            The value to return in case of a cache-miss or invalid cache value.\n        '
        path = self._getvaluepath(key)
        try:
            with path.open('r', encoding='UTF-8') as f:
                return json.load(f)
        except (ValueError, OSError):
            return default

    def set(self, key: str, value: object) -> None:
        if False:
            return 10
        'Save value for the given key.\n\n        :param key:\n            Must be a ``/`` separated value. Usually the first\n            name is the name of your plugin or your application.\n        :param value:\n            Must be of any combination of basic python types,\n            including nested types like lists of dictionaries.\n        '
        path = self._getvaluepath(key)
        try:
            if path.parent.is_dir():
                cache_dir_exists_already = True
            else:
                cache_dir_exists_already = self._cachedir.exists()
                path.parent.mkdir(exist_ok=True, parents=True)
        except OSError as exc:
            self.warn(f'could not create cache path {path}: {exc}', _ispytest=True)
            return
        if not cache_dir_exists_already:
            self._ensure_supporting_files()
        data = json.dumps(value, ensure_ascii=False, indent=2)
        try:
            f = path.open('w', encoding='UTF-8')
        except OSError as exc:
            self.warn(f'cache could not write path {path}: {exc}', _ispytest=True)
        else:
            with f:
                f.write(data)

    def _ensure_supporting_files(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Create supporting files in the cache dir that are not really part of the cache.'
        readme_path = self._cachedir / 'README.md'
        readme_path.write_text(README_CONTENT, encoding='UTF-8')
        gitignore_path = self._cachedir.joinpath('.gitignore')
        msg = '# Created by pytest automatically.\n*\n'
        gitignore_path.write_text(msg, encoding='UTF-8')
        cachedir_tag_path = self._cachedir.joinpath('CACHEDIR.TAG')
        cachedir_tag_path.write_bytes(CACHEDIR_TAG_CONTENT)

class LFPluginCollWrapper:

    def __init__(self, lfplugin: 'LFPlugin') -> None:
        if False:
            return 10
        self.lfplugin = lfplugin
        self._collected_at_least_one_failure = False

    @hookimpl(wrapper=True)
    def pytest_make_collect_report(self, collector: nodes.Collector) -> Generator[None, CollectReport, CollectReport]:
        if False:
            print('Hello World!')
        res = (yield)
        if isinstance(collector, (Session, Package)):
            lf_paths = self.lfplugin._last_failed_paths

            def sort_key(node: Union[nodes.Item, nodes.Collector]) -> bool:
                if False:
                    for i in range(10):
                        print('nop')
                return node.path in lf_paths
            res.result = sorted(res.result, key=sort_key, reverse=True)
        elif isinstance(collector, File):
            if collector.path in self.lfplugin._last_failed_paths:
                result = res.result
                lastfailed = self.lfplugin.lastfailed
                if not self._collected_at_least_one_failure:
                    if not any((x.nodeid in lastfailed for x in result)):
                        return res
                    self.lfplugin.config.pluginmanager.register(LFPluginCollSkipfiles(self.lfplugin), 'lfplugin-collskip')
                    self._collected_at_least_one_failure = True
                session = collector.session
                result[:] = [x for x in result if x.nodeid in lastfailed or session.isinitpath(x.path) or isinstance(x, nodes.Collector)]
        return res

class LFPluginCollSkipfiles:

    def __init__(self, lfplugin: 'LFPlugin') -> None:
        if False:
            i = 10
            return i + 15
        self.lfplugin = lfplugin

    @hookimpl
    def pytest_make_collect_report(self, collector: nodes.Collector) -> Optional[CollectReport]:
        if False:
            return 10
        if isinstance(collector, File):
            if collector.path not in self.lfplugin._last_failed_paths:
                self.lfplugin._skipped_files += 1
                return CollectReport(collector.nodeid, 'passed', longrepr=None, result=[])
        return None

class LFPlugin:
    """Plugin which implements the --lf (run last-failing) option."""

    def __init__(self, config: Config) -> None:
        if False:
            while True:
                i = 10
        self.config = config
        active_keys = ('lf', 'failedfirst')
        self.active = any((config.getoption(key) for key in active_keys))
        assert config.cache
        self.lastfailed: Dict[str, bool] = config.cache.get('cache/lastfailed', {})
        self._previously_failed_count: Optional[int] = None
        self._report_status: Optional[str] = None
        self._skipped_files = 0
        if config.getoption('lf'):
            self._last_failed_paths = self.get_last_failed_paths()
            config.pluginmanager.register(LFPluginCollWrapper(self), 'lfplugin-collwrapper')

    def get_last_failed_paths(self) -> Set[Path]:
        if False:
            for i in range(10):
                print('nop')
        'Return a set with all Paths of the previously failed nodeids and\n        their parents.'
        rootpath = self.config.rootpath
        result = set()
        for nodeid in self.lastfailed:
            path = rootpath / nodeid.split('::')[0]
            result.add(path)
            result.update(path.parents)
        return {x for x in result if x.exists()}

    def pytest_report_collectionfinish(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        if self.active and self.config.getoption('verbose') >= 0:
            return 'run-last-failure: %s' % self._report_status
        return None

    def pytest_runtest_logreport(self, report: TestReport) -> None:
        if False:
            for i in range(10):
                print('nop')
        if report.when == 'call' and report.passed or report.skipped:
            self.lastfailed.pop(report.nodeid, None)
        elif report.failed:
            self.lastfailed[report.nodeid] = True

    def pytest_collectreport(self, report: CollectReport) -> None:
        if False:
            for i in range(10):
                print('nop')
        passed = report.outcome in ('passed', 'skipped')
        if passed:
            if report.nodeid in self.lastfailed:
                self.lastfailed.pop(report.nodeid)
                self.lastfailed.update(((item.nodeid, True) for item in report.result))
        else:
            self.lastfailed[report.nodeid] = True

    @hookimpl(wrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, config: Config, items: List[nodes.Item]) -> Generator[None, None, None]:
        if False:
            return 10
        res = (yield)
        if not self.active:
            return res
        if self.lastfailed:
            previously_failed = []
            previously_passed = []
            for item in items:
                if item.nodeid in self.lastfailed:
                    previously_failed.append(item)
                else:
                    previously_passed.append(item)
            self._previously_failed_count = len(previously_failed)
            if not previously_failed:
                self._report_status = '%d known failures not in selected tests' % (len(self.lastfailed),)
            else:
                if self.config.getoption('lf'):
                    items[:] = previously_failed
                    config.hook.pytest_deselected(items=previously_passed)
                else:
                    items[:] = previously_failed + previously_passed
                noun = 'failure' if self._previously_failed_count == 1 else 'failures'
                suffix = ' first' if self.config.getoption('failedfirst') else ''
                self._report_status = 'rerun previous {count} {noun}{suffix}'.format(count=self._previously_failed_count, suffix=suffix, noun=noun)
            if self._skipped_files > 0:
                files_noun = 'file' if self._skipped_files == 1 else 'files'
                self._report_status += ' (skipped {files} {files_noun})'.format(files=self._skipped_files, files_noun=files_noun)
        else:
            self._report_status = 'no previously failed tests, '
            if self.config.getoption('last_failed_no_failures') == 'none':
                self._report_status += 'deselecting all items.'
                config.hook.pytest_deselected(items=items[:])
                items[:] = []
            else:
                self._report_status += 'not deselecting items.'
        return res

    def pytest_sessionfinish(self, session: Session) -> None:
        if False:
            i = 10
            return i + 15
        config = self.config
        if config.getoption('cacheshow') or hasattr(config, 'workerinput'):
            return
        assert config.cache is not None
        saved_lastfailed = config.cache.get('cache/lastfailed', {})
        if saved_lastfailed != self.lastfailed:
            config.cache.set('cache/lastfailed', self.lastfailed)

class NFPlugin:
    """Plugin which implements the --nf (run new-first) option."""

    def __init__(self, config: Config) -> None:
        if False:
            return 10
        self.config = config
        self.active = config.option.newfirst
        assert config.cache is not None
        self.cached_nodeids = set(config.cache.get('cache/nodeids', []))

    @hookimpl(wrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, items: List[nodes.Item]) -> Generator[None, None, None]:
        if False:
            while True:
                i = 10
        res = (yield)
        if self.active:
            new_items: Dict[str, nodes.Item] = {}
            other_items: Dict[str, nodes.Item] = {}
            for item in items:
                if item.nodeid not in self.cached_nodeids:
                    new_items[item.nodeid] = item
                else:
                    other_items[item.nodeid] = item
            items[:] = self._get_increasing_order(new_items.values()) + self._get_increasing_order(other_items.values())
            self.cached_nodeids.update(new_items)
        else:
            self.cached_nodeids.update((item.nodeid for item in items))
        return res

    def _get_increasing_order(self, items: Iterable[nodes.Item]) -> List[nodes.Item]:
        if False:
            return 10
        return sorted(items, key=lambda item: item.path.stat().st_mtime, reverse=True)

    def pytest_sessionfinish(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        config = self.config
        if config.getoption('cacheshow') or hasattr(config, 'workerinput'):
            return
        if config.getoption('collectonly'):
            return
        assert config.cache is not None
        config.cache.set('cache/nodeids', sorted(self.cached_nodeids))

def pytest_addoption(parser: Parser) -> None:
    if False:
        return 10
    group = parser.getgroup('general')
    group.addoption('--lf', '--last-failed', action='store_true', dest='lf', help='Rerun only the tests that failed at the last run (or all if none failed)')
    group.addoption('--ff', '--failed-first', action='store_true', dest='failedfirst', help='Run all tests, but run the last failures first. This may re-order tests and thus lead to repeated fixture setup/teardown.')
    group.addoption('--nf', '--new-first', action='store_true', dest='newfirst', help='Run tests from new files first, then the rest of the tests sorted by file mtime')
    group.addoption('--cache-show', action='append', nargs='?', dest='cacheshow', help="Show cache contents, don't perform collection or tests. Optional argument: glob (default: '*').")
    group.addoption('--cache-clear', action='store_true', dest='cacheclear', help='Remove all cache contents at start of test run')
    cache_dir_default = '.pytest_cache'
    if 'TOX_ENV_DIR' in os.environ:
        cache_dir_default = os.path.join(os.environ['TOX_ENV_DIR'], cache_dir_default)
    parser.addini('cache_dir', default=cache_dir_default, help='Cache directory path')
    group.addoption('--lfnf', '--last-failed-no-failures', action='store', dest='last_failed_no_failures', choices=('all', 'none'), default='all', help='With ``--lf``, determines whether to execute tests when there are no previously (known) failures or when no cached ``lastfailed`` data was found. ``all`` (the default) runs the full test suite again. ``none`` just emits a message about no known failures and exits successfully.')

def pytest_cmdline_main(config: Config) -> Optional[Union[int, ExitCode]]:
    if False:
        return 10
    if config.option.cacheshow and (not config.option.help):
        from _pytest.main import wrap_session
        return wrap_session(config, cacheshow)
    return None

@hookimpl(tryfirst=True)
def pytest_configure(config: Config) -> None:
    if False:
        while True:
            i = 10
    config.cache = Cache.for_config(config, _ispytest=True)
    config.pluginmanager.register(LFPlugin(config), 'lfplugin')
    config.pluginmanager.register(NFPlugin(config), 'nfplugin')

@fixture
def cache(request: FixtureRequest) -> Cache:
    if False:
        return 10
    'Return a cache object that can persist state between testing sessions.\n\n    cache.get(key, default)\n    cache.set(key, value)\n\n    Keys must be ``/`` separated strings, where the first part is usually the\n    name of your plugin or application to avoid clashes with other cache users.\n\n    Values can be any object handled by the json stdlib module.\n    '
    assert request.config.cache is not None
    return request.config.cache

def pytest_report_header(config: Config) -> Optional[str]:
    if False:
        return 10
    'Display cachedir with --cache-show and if non-default.'
    if config.option.verbose > 0 or config.getini('cache_dir') != '.pytest_cache':
        assert config.cache is not None
        cachedir = config.cache._cachedir
        try:
            displaypath = cachedir.relative_to(config.rootpath)
        except ValueError:
            displaypath = cachedir
        return f'cachedir: {displaypath}'
    return None

def cacheshow(config: Config, session: Session) -> int:
    if False:
        for i in range(10):
            print('nop')
    from pprint import pformat
    assert config.cache is not None
    tw = TerminalWriter()
    tw.line('cachedir: ' + str(config.cache._cachedir))
    if not config.cache._cachedir.is_dir():
        tw.line('cache is empty')
        return 0
    glob = config.option.cacheshow[0]
    if glob is None:
        glob = '*'
    dummy = object()
    basedir = config.cache._cachedir
    vdir = basedir / Cache._CACHE_PREFIX_VALUES
    tw.sep('-', 'cache values for %r' % glob)
    for valpath in sorted((x for x in vdir.rglob(glob) if x.is_file())):
        key = str(valpath.relative_to(vdir))
        val = config.cache.get(key, dummy)
        if val is dummy:
            tw.line('%s contains unreadable content, will be ignored' % key)
        else:
            tw.line('%s contains:' % key)
            for line in pformat(val).splitlines():
                tw.line('  ' + line)
    ddir = basedir / Cache._CACHE_PREFIX_DIRS
    if ddir.is_dir():
        contents = sorted(ddir.rglob(glob))
        tw.sep('-', 'cache directories for %r' % glob)
        for p in contents:
            if p.is_file():
                key = str(p.relative_to(basedir))
                tw.line(f'{key} is a file of length {p.stat().st_size:d}')
    return 0