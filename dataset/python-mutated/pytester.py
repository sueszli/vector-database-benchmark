"""(Disabled by default) support for testing pytest and pytest plugins.

PYTEST_DONT_REWRITE
"""
import collections.abc
import contextlib
import gc
import importlib
import locale
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from fnmatch import fnmatch
from io import StringIO
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import IO
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from weakref import WeakKeyDictionary
from iniconfig import IniConfig
from iniconfig import SectionWrapper
from _pytest import timing
from _pytest._code import Source
from _pytest.capture import _get_multicapture
from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import main
from _pytest.config import PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import make_numbered_dir
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.tmpdir import TempPathFactory
from _pytest.warning_types import PytestWarning
if TYPE_CHECKING:
    import pexpect
pytest_plugins = ['pytester_assertions']
IGNORE_PAM = ['/var/lib/sss/mc/passwd']

def pytest_addoption(parser: Parser) -> None:
    if False:
        print('Hello World!')
    parser.addoption('--lsof', action='store_true', dest='lsof', default=False, help='Run FD checks if lsof is available')
    parser.addoption('--runpytest', default='inprocess', dest='runpytest', choices=('inprocess', 'subprocess'), help="Run pytest sub runs in tests using an 'inprocess' or 'subprocess' (python -m main) method")
    parser.addini('pytester_example_dir', help='Directory to take the pytester example files from')

def pytest_configure(config: Config) -> None:
    if False:
        for i in range(10):
            print('nop')
    if config.getvalue('lsof'):
        checker = LsofFdLeakChecker()
        if checker.matching_platform():
            config.pluginmanager.register(checker)
    config.addinivalue_line('markers', 'pytester_example_path(*path_segments): join the given path segments to `pytester_example_dir` for this test.')

class LsofFdLeakChecker:

    def get_open_files(self) -> List[Tuple[str, str]]:
        if False:
            i = 10
            return i + 15
        out = subprocess.run(('lsof', '-Ffn0', '-p', str(os.getpid())), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, text=True, encoding=locale.getpreferredencoding(False)).stdout

        def isopen(line: str) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return line.startswith('f') and ('deleted' not in line and 'mem' not in line and ('txt' not in line) and ('cwd' not in line))
        open_files = []
        for line in out.split('\n'):
            if isopen(line):
                fields = line.split('\x00')
                fd = fields[0][1:]
                filename = fields[1][1:]
                if filename in IGNORE_PAM:
                    continue
                if filename.startswith('/'):
                    open_files.append((fd, filename))
        return open_files

    def matching_platform(self) -> bool:
        if False:
            while True:
                i = 10
        try:
            subprocess.run(('lsof', '-v'), check=True)
        except (OSError, subprocess.CalledProcessError):
            return False
        else:
            return True

    @hookimpl(wrapper=True, tryfirst=True)
    def pytest_runtest_protocol(self, item: Item) -> Generator[None, object, object]:
        if False:
            for i in range(10):
                print('nop')
        lines1 = self.get_open_files()
        try:
            return (yield)
        finally:
            if hasattr(sys, 'pypy_version_info'):
                gc.collect()
            lines2 = self.get_open_files()
            new_fds = {t[0] for t in lines2} - {t[0] for t in lines1}
            leaked_files = [t for t in lines2 if t[0] in new_fds]
            if leaked_files:
                error = ['***** %s FD leakage detected' % len(leaked_files), *(str(f) for f in leaked_files), '*** Before:', *(str(f) for f in lines1), '*** After:', *(str(f) for f in lines2), '***** %s FD leakage detected' % len(leaked_files), '*** function %s:%s: %s ' % item.location, 'See issue #2366']
                item.warn(PytestWarning('\n'.join(error)))

@fixture
def _pytest(request: FixtureRequest) -> 'PytestArg':
    if False:
        while True:
            i = 10
    'Return a helper which offers a gethookrecorder(hook) method which\n    returns a HookRecorder instance which helps to make assertions about called\n    hooks.'
    return PytestArg(request)

class PytestArg:

    def __init__(self, request: FixtureRequest) -> None:
        if False:
            while True:
                i = 10
        self._request = request

    def gethookrecorder(self, hook) -> 'HookRecorder':
        if False:
            for i in range(10):
                print('nop')
        hookrecorder = HookRecorder(hook._pm)
        self._request.addfinalizer(hookrecorder.finish_recording)
        return hookrecorder

def get_public_names(values: Iterable[str]) -> List[str]:
    if False:
        while True:
            i = 10
    'Only return names from iterator values without a leading underscore.'
    return [x for x in values if x[0] != '_']

@final
class RecordedHookCall:
    """A recorded call to a hook.

    The arguments to the hook call are set as attributes.
    For example:

    .. code-block:: python

        calls = hook_recorder.getcalls("pytest_runtest_setup")
        # Suppose pytest_runtest_setup was called once with `item=an_item`.
        assert calls[0].item is an_item
    """

    def __init__(self, name: str, kwargs) -> None:
        if False:
            return 10
        self.__dict__.update(kwargs)
        self._name = name

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        d = self.__dict__.copy()
        del d['_name']
        return f'<RecordedHookCall {self._name!r}(**{d!r})>'
    if TYPE_CHECKING:

        def __getattr__(self, key: str):
            if False:
                for i in range(10):
                    print('nop')
            ...

@final
class HookRecorder:
    """Record all hooks called in a plugin manager.

    Hook recorders are created by :class:`Pytester`.

    This wraps all the hook calls in the plugin manager, recording each call
    before propagating the normal calls.
    """

    def __init__(self, pluginmanager: PytestPluginManager, *, _ispytest: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        check_ispytest(_ispytest)
        self._pluginmanager = pluginmanager
        self.calls: List[RecordedHookCall] = []
        self.ret: Optional[Union[int, ExitCode]] = None

        def before(hook_name: str, hook_impls, kwargs) -> None:
            if False:
                return 10
            self.calls.append(RecordedHookCall(hook_name, kwargs))

        def after(outcome, hook_name: str, hook_impls, kwargs) -> None:
            if False:
                i = 10
                return i + 15
            pass
        self._undo_wrapping = pluginmanager.add_hookcall_monitoring(before, after)

    def finish_recording(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._undo_wrapping()

    def getcalls(self, names: Union[str, Iterable[str]]) -> List[RecordedHookCall]:
        if False:
            for i in range(10):
                print('nop')
        'Get all recorded calls to hooks with the given names (or name).'
        if isinstance(names, str):
            names = names.split()
        return [call for call in self.calls if call._name in names]

    def assert_contains(self, entries: Sequence[Tuple[str, str]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        __tracebackhide__ = True
        i = 0
        entries = list(entries)
        backlocals = sys._getframe(1).f_locals
        while entries:
            (name, check) = entries.pop(0)
            for (ind, call) in enumerate(self.calls[i:]):
                if call._name == name:
                    print('NAMEMATCH', name, call)
                    if eval(check, backlocals, call.__dict__):
                        print('CHECKERMATCH', repr(check), '->', call)
                    else:
                        print('NOCHECKERMATCH', repr(check), '-', call)
                        continue
                    i += ind + 1
                    break
                print('NONAMEMATCH', name, 'with', call)
            else:
                fail(f'could not find {name!r} check {check!r}')

    def popcall(self, name: str) -> RecordedHookCall:
        if False:
            i = 10
            return i + 15
        __tracebackhide__ = True
        for (i, call) in enumerate(self.calls):
            if call._name == name:
                del self.calls[i]
                return call
        lines = [f'could not find call {name!r}, in:']
        lines.extend(['  %s' % x for x in self.calls])
        fail('\n'.join(lines))

    def getcall(self, name: str) -> RecordedHookCall:
        if False:
            for i in range(10):
                print('nop')
        values = self.getcalls(name)
        assert len(values) == 1, (name, values)
        return values[0]

    @overload
    def getreports(self, names: "Literal['pytest_collectreport']") -> Sequence[CollectReport]:
        if False:
            return 10
        ...

    @overload
    def getreports(self, names: "Literal['pytest_runtest_logreport']") -> Sequence[TestReport]:
        if False:
            return 10
        ...

    @overload
    def getreports(self, names: Union[str, Iterable[str]]=('pytest_collectreport', 'pytest_runtest_logreport')) -> Sequence[Union[CollectReport, TestReport]]:
        if False:
            i = 10
            return i + 15
        ...

    def getreports(self, names: Union[str, Iterable[str]]=('pytest_collectreport', 'pytest_runtest_logreport')) -> Sequence[Union[CollectReport, TestReport]]:
        if False:
            for i in range(10):
                print('nop')
        return [x.report for x in self.getcalls(names)]

    def matchreport(self, inamepart: str='', names: Union[str, Iterable[str]]=('pytest_runtest_logreport', 'pytest_collectreport'), when: Optional[str]=None) -> Union[CollectReport, TestReport]:
        if False:
            print('Hello World!')
        'Return a testreport whose dotted import path matches.'
        values = []
        for rep in self.getreports(names=names):
            if not when and rep.when != 'call' and rep.passed:
                continue
            if when and rep.when != when:
                continue
            if not inamepart or inamepart in rep.nodeid.split('::'):
                values.append(rep)
        if not values:
            raise ValueError('could not find test report matching %r: no test reports at all!' % (inamepart,))
        if len(values) > 1:
            raise ValueError('found 2 or more testreports matching {!r}: {}'.format(inamepart, values))
        return values[0]

    @overload
    def getfailures(self, names: "Literal['pytest_collectreport']") -> Sequence[CollectReport]:
        if False:
            while True:
                i = 10
        ...

    @overload
    def getfailures(self, names: "Literal['pytest_runtest_logreport']") -> Sequence[TestReport]:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def getfailures(self, names: Union[str, Iterable[str]]=('pytest_collectreport', 'pytest_runtest_logreport')) -> Sequence[Union[CollectReport, TestReport]]:
        if False:
            return 10
        ...

    def getfailures(self, names: Union[str, Iterable[str]]=('pytest_collectreport', 'pytest_runtest_logreport')) -> Sequence[Union[CollectReport, TestReport]]:
        if False:
            i = 10
            return i + 15
        return [rep for rep in self.getreports(names) if rep.failed]

    def getfailedcollections(self) -> Sequence[CollectReport]:
        if False:
            for i in range(10):
                print('nop')
        return self.getfailures('pytest_collectreport')

    def listoutcomes(self) -> Tuple[Sequence[TestReport], Sequence[Union[CollectReport, TestReport]], Sequence[Union[CollectReport, TestReport]]]:
        if False:
            for i in range(10):
                print('nop')
        passed = []
        skipped = []
        failed = []
        for rep in self.getreports(('pytest_collectreport', 'pytest_runtest_logreport')):
            if rep.passed:
                if rep.when == 'call':
                    assert isinstance(rep, TestReport)
                    passed.append(rep)
            elif rep.skipped:
                skipped.append(rep)
            else:
                assert rep.failed, f'Unexpected outcome: {rep!r}'
                failed.append(rep)
        return (passed, skipped, failed)

    def countoutcomes(self) -> List[int]:
        if False:
            for i in range(10):
                print('nop')
        return [len(x) for x in self.listoutcomes()]

    def assertoutcome(self, passed: int=0, skipped: int=0, failed: int=0) -> None:
        if False:
            print('Hello World!')
        __tracebackhide__ = True
        from _pytest.pytester_assertions import assertoutcome
        outcomes = self.listoutcomes()
        assertoutcome(outcomes, passed=passed, skipped=skipped, failed=failed)

    def clear(self) -> None:
        if False:
            while True:
                i = 10
        self.calls[:] = []

@fixture
def linecomp() -> 'LineComp':
    if False:
        return 10
    'A :class: `LineComp` instance for checking that an input linearly\n    contains a sequence of strings.'
    return LineComp()

@fixture(name='LineMatcher')
def LineMatcher_fixture(request: FixtureRequest) -> Type['LineMatcher']:
    if False:
        for i in range(10):
            print('nop')
    'A reference to the :class: `LineMatcher`.\n\n    This is instantiable with a list of lines (without their trailing newlines).\n    This is useful for testing large texts, such as the output of commands.\n    '
    return LineMatcher

@fixture
def pytester(request: FixtureRequest, tmp_path_factory: TempPathFactory, monkeypatch: MonkeyPatch) -> 'Pytester':
    if False:
        for i in range(10):
            print('nop')
    '\n    Facilities to write tests/configuration files, execute pytest in isolation, and match\n    against expected output, perfect for black-box testing of pytest plugins.\n\n    It attempts to isolate the test run from external factors as much as possible, modifying\n    the current working directory to ``path`` and environment variables during initialization.\n\n    It is particularly useful for testing plugins. It is similar to the :fixture:`tmp_path`\n    fixture but provides methods which aid in testing pytest itself.\n    '
    return Pytester(request, tmp_path_factory, monkeypatch, _ispytest=True)

@fixture
def _sys_snapshot() -> Generator[None, None, None]:
    if False:
        for i in range(10):
            print('nop')
    snappaths = SysPathsSnapshot()
    snapmods = SysModulesSnapshot()
    yield
    snapmods.restore()
    snappaths.restore()

@fixture
def _config_for_test() -> Generator[Config, None, None]:
    if False:
        while True:
            i = 10
    from _pytest.config import get_config
    config = get_config()
    yield config
    config._ensure_unconfigure()
rex_session_duration = re.compile('\\d+\\.\\d\\ds')
rex_outcome = re.compile('(\\d+) (\\w+)')

@final
class RunResult:
    """The result of running a command from :class:`~pytest.Pytester`."""

    def __init__(self, ret: Union[int, ExitCode], outlines: List[str], errlines: List[str], duration: float) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            self.ret: Union[int, ExitCode] = ExitCode(ret)
            'The return value.'
        except ValueError:
            self.ret = ret
        self.outlines = outlines
        'List of lines captured from stdout.'
        self.errlines = errlines
        'List of lines captured from stderr.'
        self.stdout = LineMatcher(outlines)
        ':class:`~pytest.LineMatcher` of stdout.\n\n        Use e.g. :func:`str(stdout) <pytest.LineMatcher.__str__()>` to reconstruct stdout, or the commonly used\n        :func:`stdout.fnmatch_lines() <pytest.LineMatcher.fnmatch_lines()>` method.\n        '
        self.stderr = LineMatcher(errlines)
        ':class:`~pytest.LineMatcher` of stderr.'
        self.duration = duration
        'Duration in seconds.'

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return '<RunResult ret=%s len(stdout.lines)=%d len(stderr.lines)=%d duration=%.2fs>' % (self.ret, len(self.stdout.lines), len(self.stderr.lines), self.duration)

    def parseoutcomes(self) -> Dict[str, int]:
        if False:
            while True:
                i = 10
        'Return a dictionary of outcome noun -> count from parsing the terminal\n        output that the test process produced.\n\n        The returned nouns will always be in plural form::\n\n            ======= 1 failed, 1 passed, 1 warning, 1 error in 0.13s ====\n\n        Will return ``{"failed": 1, "passed": 1, "warnings": 1, "errors": 1}``.\n        '
        return self.parse_summary_nouns(self.outlines)

    @classmethod
    def parse_summary_nouns(cls, lines) -> Dict[str, int]:
        if False:
            return 10
        'Extract the nouns from a pytest terminal summary line.\n\n        It always returns the plural noun for consistency::\n\n            ======= 1 failed, 1 passed, 1 warning, 1 error in 0.13s ====\n\n        Will return ``{"failed": 1, "passed": 1, "warnings": 1, "errors": 1}``.\n        '
        for line in reversed(lines):
            if rex_session_duration.search(line):
                outcomes = rex_outcome.findall(line)
                ret = {noun: int(count) for (count, noun) in outcomes}
                break
        else:
            raise ValueError('Pytest terminal summary report not found')
        to_plural = {'warning': 'warnings', 'error': 'errors'}
        return {to_plural.get(k, k): v for (k, v) in ret.items()}

    def assert_outcomes(self, passed: int=0, skipped: int=0, failed: int=0, errors: int=0, xpassed: int=0, xfailed: int=0, warnings: Optional[int]=None, deselected: Optional[int]=None) -> None:
        if False:
            return 10
        "\n        Assert that the specified outcomes appear with the respective\n        numbers (0 means it didn't occur) in the text output from a test run.\n\n        ``warnings`` and ``deselected`` are only checked if not None.\n        "
        __tracebackhide__ = True
        from _pytest.pytester_assertions import assert_outcomes
        outcomes = self.parseoutcomes()
        assert_outcomes(outcomes, passed=passed, skipped=skipped, failed=failed, errors=errors, xpassed=xpassed, xfailed=xfailed, warnings=warnings, deselected=deselected)

class SysModulesSnapshot:

    def __init__(self, preserve: Optional[Callable[[str], bool]]=None) -> None:
        if False:
            while True:
                i = 10
        self.__preserve = preserve
        self.__saved = dict(sys.modules)

    def restore(self) -> None:
        if False:
            print('Hello World!')
        if self.__preserve:
            self.__saved.update(((k, m) for (k, m) in sys.modules.items() if self.__preserve(k)))
        sys.modules.clear()
        sys.modules.update(self.__saved)

class SysPathsSnapshot:

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.__saved = (list(sys.path), list(sys.meta_path))

    def restore(self) -> None:
        if False:
            i = 10
            return i + 15
        (sys.path[:], sys.meta_path[:]) = self.__saved

@final
class Pytester:
    """
    Facilities to write tests/configuration files, execute pytest in isolation, and match
    against expected output, perfect for black-box testing of pytest plugins.

    It attempts to isolate the test run from external factors as much as possible, modifying
    the current working directory to :attr:`path` and environment variables during initialization.
    """
    __test__ = False
    CLOSE_STDIN: 'Final' = NOTSET

    class TimeoutExpired(Exception):
        pass

    def __init__(self, request: FixtureRequest, tmp_path_factory: TempPathFactory, monkeypatch: MonkeyPatch, *, _ispytest: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        check_ispytest(_ispytest)
        self._request = request
        self._mod_collections: WeakKeyDictionary[Collector, List[Union[Item, Collector]]] = WeakKeyDictionary()
        if request.function:
            name: str = request.function.__name__
        else:
            name = request.node.name
        self._name = name
        self._path: Path = tmp_path_factory.mktemp(name, numbered=True)
        self.plugins: List[Union[str, _PluggyPlugin]] = []
        self._sys_path_snapshot = SysPathsSnapshot()
        self._sys_modules_snapshot = self.__take_sys_modules_snapshot()
        self._request.addfinalizer(self._finalize)
        self._method = self._request.config.getoption('--runpytest')
        self._test_tmproot = tmp_path_factory.mktemp(f'tmp-{name}', numbered=True)
        self._monkeypatch = mp = monkeypatch
        self.chdir()
        mp.setenv('PYTEST_DEBUG_TEMPROOT', str(self._test_tmproot))
        mp.delenv('TOX_ENV_DIR', raising=False)
        mp.delenv('PYTEST_ADDOPTS', raising=False)
        tmphome = str(self.path)
        mp.setenv('HOME', tmphome)
        mp.setenv('USERPROFILE', tmphome)
        mp.setenv('PY_COLORS', '0')

    @property
    def path(self) -> Path:
        if False:
            print('Hello World!')
        'Temporary directory path used to create files/run tests from, etc.'
        return self._path

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<Pytester {self.path!r}>'

    def _finalize(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Clean up global state artifacts.\n\n        Some methods modify the global interpreter state and this tries to\n        clean this up. It does not remove the temporary directory however so\n        it can be looked at after the test run has finished.\n        '
        self._sys_modules_snapshot.restore()
        self._sys_path_snapshot.restore()

    def __take_sys_modules_snapshot(self) -> SysModulesSnapshot:
        if False:
            return 10

        def preserve_module(name):
            if False:
                return 10
            return name.startswith(('zope', 'readline'))
        return SysModulesSnapshot(preserve=preserve_module)

    def make_hook_recorder(self, pluginmanager: PytestPluginManager) -> HookRecorder:
        if False:
            print('Hello World!')
        'Create a new :class:`HookRecorder` for a :class:`PytestPluginManager`.'
        pluginmanager.reprec = reprec = HookRecorder(pluginmanager, _ispytest=True)
        self._request.addfinalizer(reprec.finish_recording)
        return reprec

    def chdir(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Cd into the temporary directory.\n\n        This is done automatically upon instantiation.\n        '
        self._monkeypatch.chdir(self.path)

    def _makefile(self, ext: str, lines: Sequence[Union[Any, bytes]], files: Dict[str, str], encoding: str='utf-8') -> Path:
        if False:
            for i in range(10):
                print('nop')
        items = list(files.items())
        if ext and (not ext.startswith('.')):
            raise ValueError(f'pytester.makefile expects a file extension, try .{ext} instead of {ext}')

        def to_text(s: Union[Any, bytes]) -> str:
            if False:
                while True:
                    i = 10
            return s.decode(encoding) if isinstance(s, bytes) else str(s)
        if lines:
            source = '\n'.join((to_text(x) for x in lines))
            basename = self._name
            items.insert(0, (basename, source))
        ret = None
        for (basename, value) in items:
            p = self.path.joinpath(basename).with_suffix(ext)
            p.parent.mkdir(parents=True, exist_ok=True)
            source_ = Source(value)
            source = '\n'.join((to_text(line) for line in source_.lines))
            p.write_text(source.strip(), encoding=encoding)
            if ret is None:
                ret = p
        assert ret is not None
        return ret

    def makefile(self, ext: str, *args: str, **kwargs: str) -> Path:
        if False:
            return 10
        'Create new text file(s) in the test directory.\n\n        :param ext:\n            The extension the file(s) should use, including the dot, e.g. `.py`.\n        :param args:\n            All args are treated as strings and joined using newlines.\n            The result is written as contents to the file.  The name of the\n            file is based on the test function requesting this fixture.\n        :param kwargs:\n            Each keyword is the name of a file, while the value of it will\n            be written as contents of the file.\n        :returns:\n            The first created file.\n\n        Examples:\n\n        .. code-block:: python\n\n            pytester.makefile(".txt", "line1", "line2")\n\n            pytester.makefile(".ini", pytest="[pytest]\\naddopts=-rs\\n")\n\n        To create binary files, use :meth:`pathlib.Path.write_bytes` directly:\n\n        .. code-block:: python\n\n            filename = pytester.path.joinpath("foo.bin")\n            filename.write_bytes(b"...")\n        '
        return self._makefile(ext, args, kwargs)

    def makeconftest(self, source: str) -> Path:
        if False:
            return 10
        'Write a conftest.py file.\n\n        :param source: The contents.\n        :returns: The conftest.py file.\n        '
        return self.makepyfile(conftest=source)

    def makeini(self, source: str) -> Path:
        if False:
            return 10
        'Write a tox.ini file.\n\n        :param source: The contents.\n        :returns: The tox.ini file.\n        '
        return self.makefile('.ini', tox=source)

    def getinicfg(self, source: str) -> SectionWrapper:
        if False:
            i = 10
            return i + 15
        'Return the pytest section from the tox.ini config file.'
        p = self.makeini(source)
        return IniConfig(str(p))['pytest']

    def makepyprojecttoml(self, source: str) -> Path:
        if False:
            return 10
        'Write a pyproject.toml file.\n\n        :param source: The contents.\n        :returns: The pyproject.ini file.\n\n        .. versionadded:: 6.0\n        '
        return self.makefile('.toml', pyproject=source)

    def makepyfile(self, *args, **kwargs) -> Path:
        if False:
            while True:
                i = 10
        'Shortcut for .makefile() with a .py extension.\n\n        Defaults to the test name with a \'.py\' extension, e.g test_foobar.py, overwriting\n        existing files.\n\n        Examples:\n\n        .. code-block:: python\n\n            def test_something(pytester):\n                # Initial file is created test_something.py.\n                pytester.makepyfile("foobar")\n                # To create multiple files, pass kwargs accordingly.\n                pytester.makepyfile(custom="foobar")\n                # At this point, both \'test_something.py\' & \'custom.py\' exist in the test directory.\n\n        '
        return self._makefile('.py', args, kwargs)

    def maketxtfile(self, *args, **kwargs) -> Path:
        if False:
            return 10
        'Shortcut for .makefile() with a .txt extension.\n\n        Defaults to the test name with a \'.txt\' extension, e.g test_foobar.txt, overwriting\n        existing files.\n\n        Examples:\n\n        .. code-block:: python\n\n            def test_something(pytester):\n                # Initial file is created test_something.txt.\n                pytester.maketxtfile("foobar")\n                # To create multiple files, pass kwargs accordingly.\n                pytester.maketxtfile(custom="foobar")\n                # At this point, both \'test_something.txt\' & \'custom.txt\' exist in the test directory.\n\n        '
        return self._makefile('.txt', args, kwargs)

    def syspathinsert(self, path: Optional[Union[str, 'os.PathLike[str]']]=None) -> None:
        if False:
            print('Hello World!')
        'Prepend a directory to sys.path, defaults to :attr:`path`.\n\n        This is undone automatically when this object dies at the end of each\n        test.\n\n        :param path:\n            The path.\n        '
        if path is None:
            path = self.path
        self._monkeypatch.syspath_prepend(str(path))

    def mkdir(self, name: Union[str, 'os.PathLike[str]']) -> Path:
        if False:
            print('Hello World!')
        'Create a new (sub)directory.\n\n        :param name:\n            The name of the directory, relative to the pytester path.\n        :returns:\n            The created directory.\n        '
        p = self.path / name
        p.mkdir()
        return p

    def mkpydir(self, name: Union[str, 'os.PathLike[str]']) -> Path:
        if False:
            i = 10
            return i + 15
        'Create a new python package.\n\n        This creates a (sub)directory with an empty ``__init__.py`` file so it\n        gets recognised as a Python package.\n        '
        p = self.path / name
        p.mkdir()
        p.joinpath('__init__.py').touch()
        return p

    def copy_example(self, name: Optional[str]=None) -> Path:
        if False:
            while True:
                i = 10
        "Copy file from project's directory into the testdir.\n\n        :param name:\n            The name of the file to copy.\n        :return:\n            Path to the copied directory (inside ``self.path``).\n        "
        example_dir_ = self._request.config.getini('pytester_example_dir')
        if example_dir_ is None:
            raise ValueError("pytester_example_dir is unset, can't copy examples")
        example_dir: Path = self._request.config.rootpath / example_dir_
        for extra_element in self._request.node.iter_markers('pytester_example_path'):
            assert extra_element.args
            example_dir = example_dir.joinpath(*extra_element.args)
        if name is None:
            func_name = self._name
            maybe_dir = example_dir / func_name
            maybe_file = example_dir / (func_name + '.py')
            if maybe_dir.is_dir():
                example_path = maybe_dir
            elif maybe_file.is_file():
                example_path = maybe_file
            else:
                raise LookupError(f"{func_name} can't be found as module or package in {example_dir}")
        else:
            example_path = example_dir.joinpath(name)
        if example_path.is_dir() and (not example_path.joinpath('__init__.py').is_file()):
            shutil.copytree(example_path, self.path, symlinks=True, dirs_exist_ok=True)
            return self.path
        elif example_path.is_file():
            result = self.path.joinpath(example_path.name)
            shutil.copy(example_path, result)
            return result
        else:
            raise LookupError(f'example "{example_path}" is not found as a file or directory')

    def getnode(self, config: Config, arg: Union[str, 'os.PathLike[str]']) -> Union[Collector, Item]:
        if False:
            for i in range(10):
                print('nop')
        'Get the collection node of a file.\n\n        :param config:\n           A pytest config.\n           See :py:meth:`parseconfig` and :py:meth:`parseconfigure` for creating it.\n        :param arg:\n            Path to the file.\n        :returns:\n            The node.\n        '
        session = Session.from_config(config)
        assert '::' not in str(arg)
        p = Path(os.path.abspath(arg))
        config.hook.pytest_sessionstart(session=session)
        res = session.perform_collect([str(p)], genitems=False)[0]
        config.hook.pytest_sessionfinish(session=session, exitstatus=ExitCode.OK)
        return res

    def getpathnode(self, path: Union[str, 'os.PathLike[str]']) -> Union[Collector, Item]:
        if False:
            i = 10
            return i + 15
        'Return the collection node of a file.\n\n        This is like :py:meth:`getnode` but uses :py:meth:`parseconfigure` to\n        create the (configured) pytest Config instance.\n\n        :param path:\n            Path to the file.\n        :returns:\n            The node.\n        '
        path = Path(path)
        config = self.parseconfigure(path)
        session = Session.from_config(config)
        x = bestrelpath(session.path, path)
        config.hook.pytest_sessionstart(session=session)
        res = session.perform_collect([x], genitems=False)[0]
        config.hook.pytest_sessionfinish(session=session, exitstatus=ExitCode.OK)
        return res

    def genitems(self, colitems: Sequence[Union[Item, Collector]]) -> List[Item]:
        if False:
            i = 10
            return i + 15
        'Generate all test items from a collection node.\n\n        This recurses into the collection node and returns a list of all the\n        test items contained within.\n\n        :param colitems:\n            The collection nodes.\n        :returns:\n            The collected items.\n        '
        session = colitems[0].session
        result: List[Item] = []
        for colitem in colitems:
            result.extend(session.genitems(colitem))
        return result

    def runitem(self, source: str) -> Any:
        if False:
            print('Hello World!')
        'Run the "test_func" Item.\n\n        The calling test instance (class containing the test method) must\n        provide a ``.getrunner()`` method which should return a runner which\n        can run the test protocol for a single item, e.g.\n        :py:func:`_pytest.runner.runtestprotocol`.\n        '
        item = self.getitem(source)
        testclassinstance = self._request.instance
        runner = testclassinstance.getrunner()
        return runner(item)

    def inline_runsource(self, source: str, *cmdlineargs) -> HookRecorder:
        if False:
            print('Hello World!')
        'Run a test module in process using ``pytest.main()``.\n\n        This run writes "source" into a temporary file and runs\n        ``pytest.main()`` on it, returning a :py:class:`HookRecorder` instance\n        for the result.\n\n        :param source: The source code of the test module.\n        :param cmdlineargs: Any extra command line arguments to use.\n        '
        p = self.makepyfile(source)
        values = list(cmdlineargs) + [p]
        return self.inline_run(*values)

    def inline_genitems(self, *args) -> Tuple[List[Item], HookRecorder]:
        if False:
            while True:
                i = 10
        "Run ``pytest.main(['--collect-only'])`` in-process.\n\n        Runs the :py:func:`pytest.main` function to run all of pytest inside\n        the test process itself like :py:meth:`inline_run`, but returns a\n        tuple of the collected items and a :py:class:`HookRecorder` instance.\n        "
        rec = self.inline_run('--collect-only', *args)
        items = [x.item for x in rec.getcalls('pytest_itemcollected')]
        return (items, rec)

    def inline_run(self, *args: Union[str, 'os.PathLike[str]'], plugins=(), no_reraise_ctrlc: bool=False) -> HookRecorder:
        if False:
            for i in range(10):
                print('nop')
        'Run ``pytest.main()`` in-process, returning a HookRecorder.\n\n        Runs the :py:func:`pytest.main` function to run all of pytest inside\n        the test process itself.  This means it can return a\n        :py:class:`HookRecorder` instance which gives more detailed results\n        from that run than can be done by matching stdout/stderr from\n        :py:meth:`runpytest`.\n\n        :param args:\n            Command line arguments to pass to :py:func:`pytest.main`.\n        :param plugins:\n            Extra plugin instances the ``pytest.main()`` instance should use.\n        :param no_reraise_ctrlc:\n            Typically we reraise keyboard interrupts from the child run. If\n            True, the KeyboardInterrupt exception is captured.\n        '
        importlib.invalidate_caches()
        plugins = list(plugins)
        finalizers = []
        try:
            finalizers.append(self.__take_sys_modules_snapshot().restore)
            finalizers.append(SysPathsSnapshot().restore)
            rec = []

            class Collect:

                def pytest_configure(x, config: Config) -> None:
                    if False:
                        print('Hello World!')
                    rec.append(self.make_hook_recorder(config.pluginmanager))
            plugins.append(Collect())
            ret = main([str(x) for x in args], plugins=plugins)
            if len(rec) == 1:
                reprec = rec.pop()
            else:

                class reprec:
                    pass
            reprec.ret = ret
            if ret == ExitCode.INTERRUPTED and (not no_reraise_ctrlc):
                calls = reprec.getcalls('pytest_keyboard_interrupt')
                if calls and calls[-1].excinfo.type == KeyboardInterrupt:
                    raise KeyboardInterrupt()
            return reprec
        finally:
            for finalizer in finalizers:
                finalizer()

    def runpytest_inprocess(self, *args: Union[str, 'os.PathLike[str]'], **kwargs: Any) -> RunResult:
        if False:
            for i in range(10):
                print('nop')
        'Return result of running pytest in-process, providing a similar\n        interface to what self.runpytest() provides.'
        syspathinsert = kwargs.pop('syspathinsert', False)
        if syspathinsert:
            self.syspathinsert()
        now = timing.time()
        capture = _get_multicapture('sys')
        capture.start_capturing()
        try:
            try:
                reprec = self.inline_run(*args, **kwargs)
            except SystemExit as e:
                ret = e.args[0]
                try:
                    ret = ExitCode(e.args[0])
                except ValueError:
                    pass

                class reprec:
                    ret = ret
            except Exception:
                traceback.print_exc()

                class reprec:
                    ret = ExitCode(3)
        finally:
            (out, err) = capture.readouterr()
            capture.stop_capturing()
            sys.stdout.write(out)
            sys.stderr.write(err)
        assert reprec.ret is not None
        res = RunResult(reprec.ret, out.splitlines(), err.splitlines(), timing.time() - now)
        res.reprec = reprec
        return res

    def runpytest(self, *args: Union[str, 'os.PathLike[str]'], **kwargs: Any) -> RunResult:
        if False:
            for i in range(10):
                print('nop')
        'Run pytest inline or in a subprocess, depending on the command line\n        option "--runpytest" and return a :py:class:`~pytest.RunResult`.'
        new_args = self._ensure_basetemp(args)
        if self._method == 'inprocess':
            return self.runpytest_inprocess(*new_args, **kwargs)
        elif self._method == 'subprocess':
            return self.runpytest_subprocess(*new_args, **kwargs)
        raise RuntimeError(f'Unrecognized runpytest option: {self._method}')

    def _ensure_basetemp(self, args: Sequence[Union[str, 'os.PathLike[str]']]) -> List[Union[str, 'os.PathLike[str]']]:
        if False:
            while True:
                i = 10
        new_args = list(args)
        for x in new_args:
            if str(x).startswith('--basetemp'):
                break
        else:
            new_args.append('--basetemp=%s' % self.path.parent.joinpath('basetemp'))
        return new_args

    def parseconfig(self, *args: Union[str, 'os.PathLike[str]']) -> Config:
        if False:
            return 10
        'Return a new pytest :class:`pytest.Config` instance from given\n        commandline args.\n\n        This invokes the pytest bootstrapping code in _pytest.config to create a\n        new :py:class:`pytest.PytestPluginManager` and call the\n        :hook:`pytest_cmdline_parse` hook to create a new :class:`pytest.Config`\n        instance.\n\n        If :attr:`plugins` has been populated they should be plugin modules\n        to be registered with the plugin manager.\n        '
        import _pytest.config
        new_args = self._ensure_basetemp(args)
        new_args = [str(x) for x in new_args]
        config = _pytest.config._prepareconfig(new_args, self.plugins)
        self._request.addfinalizer(config._ensure_unconfigure)
        return config

    def parseconfigure(self, *args: Union[str, 'os.PathLike[str]']) -> Config:
        if False:
            i = 10
            return i + 15
        'Return a new pytest configured Config instance.\n\n        Returns a new :py:class:`pytest.Config` instance like\n        :py:meth:`parseconfig`, but also calls the :hook:`pytest_configure`\n        hook.\n        '
        config = self.parseconfig(*args)
        config._do_configure()
        return config

    def getitem(self, source: Union[str, 'os.PathLike[str]'], funcname: str='test_func') -> Item:
        if False:
            for i in range(10):
                print('nop')
        "Return the test item for a test function.\n\n        Writes the source to a python file and runs pytest's collection on\n        the resulting module, returning the test item for the requested\n        function name.\n\n        :param source:\n            The module source.\n        :param funcname:\n            The name of the test function for which to return a test item.\n        :returns:\n            The test item.\n        "
        items = self.getitems(source)
        for item in items:
            if item.name == funcname:
                return item
        assert 0, '{!r} item not found in module:\n{}\nitems: {}'.format(funcname, source, items)

    def getitems(self, source: Union[str, 'os.PathLike[str]']) -> List[Item]:
        if False:
            print('Hello World!')
        "Return all test items collected from the module.\n\n        Writes the source to a Python file and runs pytest's collection on\n        the resulting module, returning all test items contained within.\n        "
        modcol = self.getmodulecol(source)
        return self.genitems([modcol])

    def getmodulecol(self, source: Union[str, 'os.PathLike[str]'], configargs=(), *, withinit: bool=False):
        if False:
            while True:
                i = 10
        'Return the module collection node for ``source``.\n\n        Writes ``source`` to a file using :py:meth:`makepyfile` and then\n        runs the pytest collection on it, returning the collection node for the\n        test module.\n\n        :param source:\n            The source code of the module to collect.\n\n        :param configargs:\n            Any extra arguments to pass to :py:meth:`parseconfigure`.\n\n        :param withinit:\n            Whether to also write an ``__init__.py`` file to the same\n            directory to ensure it is a package.\n        '
        if isinstance(source, os.PathLike):
            path = self.path.joinpath(source)
            assert not withinit, 'not supported for paths'
        else:
            kw = {self._name: str(source)}
            path = self.makepyfile(**kw)
        if withinit:
            self.makepyfile(__init__='#')
        self.config = config = self.parseconfigure(path, *configargs)
        return self.getnode(config, path)

    def collect_by_name(self, modcol: Collector, name: str) -> Optional[Union[Item, Collector]]:
        if False:
            for i in range(10):
                print('nop')
        'Return the collection node for name from the module collection.\n\n        Searches a module collection node for a collection node matching the\n        given name.\n\n        :param modcol: A module collection node; see :py:meth:`getmodulecol`.\n        :param name: The name of the node to return.\n        '
        if modcol not in self._mod_collections:
            self._mod_collections[modcol] = list(modcol.collect())
        for colitem in self._mod_collections[modcol]:
            if colitem.name == name:
                return colitem
        return None

    def popen(self, cmdargs: Sequence[Union[str, 'os.PathLike[str]']], stdout: Union[int, TextIO]=subprocess.PIPE, stderr: Union[int, TextIO]=subprocess.PIPE, stdin: Union[NotSetType, bytes, IO[Any], int]=CLOSE_STDIN, **kw):
        if False:
            i = 10
            return i + 15
        'Invoke :py:class:`subprocess.Popen`.\n\n        Calls :py:class:`subprocess.Popen` making sure the current working\n        directory is in ``PYTHONPATH``.\n\n        You probably want to use :py:meth:`run` instead.\n        '
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [os.getcwd(), env.get('PYTHONPATH', '')]))
        kw['env'] = env
        if stdin is self.CLOSE_STDIN:
            kw['stdin'] = subprocess.PIPE
        elif isinstance(stdin, bytes):
            kw['stdin'] = subprocess.PIPE
        else:
            kw['stdin'] = stdin
        popen = subprocess.Popen(cmdargs, stdout=stdout, stderr=stderr, **kw)
        if stdin is self.CLOSE_STDIN:
            assert popen.stdin is not None
            popen.stdin.close()
        elif isinstance(stdin, bytes):
            assert popen.stdin is not None
            popen.stdin.write(stdin)
        return popen

    def run(self, *cmdargs: Union[str, 'os.PathLike[str]'], timeout: Optional[float]=None, stdin: Union[NotSetType, bytes, IO[Any], int]=CLOSE_STDIN) -> RunResult:
        if False:
            i = 10
            return i + 15
        'Run a command with arguments.\n\n        Run a process using :py:class:`subprocess.Popen` saving the stdout and\n        stderr.\n\n        :param cmdargs:\n            The sequence of arguments to pass to :py:class:`subprocess.Popen`,\n            with path-like objects being converted to :py:class:`str`\n            automatically.\n        :param timeout:\n            The period in seconds after which to timeout and raise\n            :py:class:`Pytester.TimeoutExpired`.\n        :param stdin:\n            Optional standard input.\n\n            - If it is :py:attr:`CLOSE_STDIN` (Default), then this method calls\n              :py:class:`subprocess.Popen` with ``stdin=subprocess.PIPE``, and\n              the standard input is closed immediately after the new command is\n              started.\n\n            - If it is of type :py:class:`bytes`, these bytes are sent to the\n              standard input of the command.\n\n            - Otherwise, it is passed through to :py:class:`subprocess.Popen`.\n              For further information in this case, consult the document of the\n              ``stdin`` parameter in :py:class:`subprocess.Popen`.\n        :returns:\n            The result.\n        '
        __tracebackhide__ = True
        cmdargs = tuple((os.fspath(arg) for arg in cmdargs))
        p1 = self.path.joinpath('stdout')
        p2 = self.path.joinpath('stderr')
        print('running:', *cmdargs)
        print('     in:', Path.cwd())
        with p1.open('w', encoding='utf8') as f1, p2.open('w', encoding='utf8') as f2:
            now = timing.time()
            popen = self.popen(cmdargs, stdin=stdin, stdout=f1, stderr=f2, close_fds=sys.platform != 'win32')
            if popen.stdin is not None:
                popen.stdin.close()

            def handle_timeout() -> None:
                if False:
                    print('Hello World!')
                __tracebackhide__ = True
                timeout_message = '{seconds} second timeout expired running: {command}'.format(seconds=timeout, command=cmdargs)
                popen.kill()
                popen.wait()
                raise self.TimeoutExpired(timeout_message)
            if timeout is None:
                ret = popen.wait()
            else:
                try:
                    ret = popen.wait(timeout)
                except subprocess.TimeoutExpired:
                    handle_timeout()
        with p1.open(encoding='utf8') as f1, p2.open(encoding='utf8') as f2:
            out = f1.read().splitlines()
            err = f2.read().splitlines()
        self._dump_lines(out, sys.stdout)
        self._dump_lines(err, sys.stderr)
        with contextlib.suppress(ValueError):
            ret = ExitCode(ret)
        return RunResult(ret, out, err, timing.time() - now)

    def _dump_lines(self, lines, fp):
        if False:
            print('Hello World!')
        try:
            for line in lines:
                print(line, file=fp)
        except UnicodeEncodeError:
            print(f"couldn't print to {fp} because of encoding")

    def _getpytestargs(self) -> Tuple[str, ...]:
        if False:
            i = 10
            return i + 15
        return (sys.executable, '-mpytest')

    def runpython(self, script: 'os.PathLike[str]') -> RunResult:
        if False:
            i = 10
            return i + 15
        'Run a python script using sys.executable as interpreter.'
        return self.run(sys.executable, script)

    def runpython_c(self, command: str) -> RunResult:
        if False:
            while True:
                i = 10
        'Run ``python -c "command"``.'
        return self.run(sys.executable, '-c', command)

    def runpytest_subprocess(self, *args: Union[str, 'os.PathLike[str]'], timeout: Optional[float]=None) -> RunResult:
        if False:
            return 10
        'Run pytest as a subprocess with given arguments.\n\n        Any plugins added to the :py:attr:`plugins` list will be added using the\n        ``-p`` command line option.  Additionally ``--basetemp`` is used to put\n        any temporary files and directories in a numbered directory prefixed\n        with "runpytest-" to not conflict with the normal numbered pytest\n        location for temporary files and directories.\n\n        :param args:\n            The sequence of arguments to pass to the pytest subprocess.\n        :param timeout:\n            The period in seconds after which to timeout and raise\n            :py:class:`Pytester.TimeoutExpired`.\n        :returns:\n            The result.\n        '
        __tracebackhide__ = True
        p = make_numbered_dir(root=self.path, prefix='runpytest-', mode=448)
        args = ('--basetemp=%s' % p,) + args
        plugins = [x for x in self.plugins if isinstance(x, str)]
        if plugins:
            args = ('-p', plugins[0]) + args
        args = self._getpytestargs() + args
        return self.run(*args, timeout=timeout)

    def spawn_pytest(self, string: str, expect_timeout: float=10.0) -> 'pexpect.spawn':
        if False:
            print('Hello World!')
        'Run pytest using pexpect.\n\n        This makes sure to use the right pytest and sets up the temporary\n        directory locations.\n\n        The pexpect child is returned.\n        '
        basetemp = self.path / 'temp-pexpect'
        basetemp.mkdir(mode=448)
        invoke = ' '.join(map(str, self._getpytestargs()))
        cmd = f'{invoke} --basetemp={basetemp} {string}'
        return self.spawn(cmd, expect_timeout=expect_timeout)

    def spawn(self, cmd: str, expect_timeout: float=10.0) -> 'pexpect.spawn':
        if False:
            i = 10
            return i + 15
        'Run a command using pexpect.\n\n        The pexpect child is returned.\n        '
        pexpect = importorskip('pexpect', '3.0')
        if hasattr(sys, 'pypy_version_info') and '64' in platform.machine():
            skip('pypy-64 bit not supported')
        if not hasattr(pexpect, 'spawn'):
            skip('pexpect.spawn not available')
        logfile = self.path.joinpath('spawn.out').open('wb')
        child = pexpect.spawn(cmd, logfile=logfile, timeout=expect_timeout)
        self._request.addfinalizer(logfile.close)
        return child

class LineComp:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.stringio = StringIO()
        ':class:`python:io.StringIO()` instance used for input.'

    def assert_contains_lines(self, lines2: Sequence[str]) -> None:
        if False:
            while True:
                i = 10
        "Assert that ``lines2`` are contained (linearly) in :attr:`stringio`'s value.\n\n        Lines are matched using :func:`LineMatcher.fnmatch_lines <pytest.LineMatcher.fnmatch_lines>`.\n        "
        __tracebackhide__ = True
        val = self.stringio.getvalue()
        self.stringio.truncate(0)
        self.stringio.seek(0)
        lines1 = val.split('\n')
        LineMatcher(lines1).fnmatch_lines(lines2)

class LineMatcher:
    """Flexible matching of text.

    This is a convenience class to test large texts like the output of
    commands.

    The constructor takes a list of lines without their trailing newlines, i.e.
    ``text.splitlines()``.
    """

    def __init__(self, lines: List[str]) -> None:
        if False:
            i = 10
            return i + 15
        self.lines = lines
        self._log_output: List[str] = []

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the entire original text.\n\n        .. versionadded:: 6.2\n            You can use :meth:`str` in older versions.\n        '
        return '\n'.join(self.lines)

    def _getlines(self, lines2: Union[str, Sequence[str], Source]) -> Sequence[str]:
        if False:
            while True:
                i = 10
        if isinstance(lines2, str):
            lines2 = Source(lines2)
        if isinstance(lines2, Source):
            lines2 = lines2.strip().lines
        return lines2

    def fnmatch_lines_random(self, lines2: Sequence[str]) -> None:
        if False:
            print('Hello World!')
        'Check lines exist in the output in any order (using :func:`python:fnmatch.fnmatch`).'
        __tracebackhide__ = True
        self._match_lines_random(lines2, fnmatch)

    def re_match_lines_random(self, lines2: Sequence[str]) -> None:
        if False:
            i = 10
            return i + 15
        'Check lines exist in the output in any order (using :func:`python:re.match`).'
        __tracebackhide__ = True
        self._match_lines_random(lines2, lambda name, pat: bool(re.match(pat, name)))

    def _match_lines_random(self, lines2: Sequence[str], match_func: Callable[[str, str], bool]) -> None:
        if False:
            return 10
        __tracebackhide__ = True
        lines2 = self._getlines(lines2)
        for line in lines2:
            for x in self.lines:
                if line == x or match_func(x, line):
                    self._log('matched: ', repr(line))
                    break
            else:
                msg = 'line %r not found in output' % line
                self._log(msg)
                self._fail(msg)

    def get_lines_after(self, fnline: str) -> Sequence[str]:
        if False:
            return 10
        'Return all lines following the given line in the text.\n\n        The given line can contain glob wildcards.\n        '
        for (i, line) in enumerate(self.lines):
            if fnline == line or fnmatch(line, fnline):
                return self.lines[i + 1:]
        raise ValueError('line %r not found in output' % fnline)

    def _log(self, *args) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._log_output.append(' '.join((str(x) for x in args)))

    @property
    def _log_text(self) -> str:
        if False:
            while True:
                i = 10
        return '\n'.join(self._log_output)

    def fnmatch_lines(self, lines2: Sequence[str], *, consecutive: bool=False) -> None:
        if False:
            return 10
        'Check lines exist in the output (using :func:`python:fnmatch.fnmatch`).\n\n        The argument is a list of lines which have to match and can use glob\n        wildcards.  If they do not match a pytest.fail() is called.  The\n        matches and non-matches are also shown as part of the error message.\n\n        :param lines2: String patterns to match.\n        :param consecutive: Match lines consecutively?\n        '
        __tracebackhide__ = True
        self._match_lines(lines2, fnmatch, 'fnmatch', consecutive=consecutive)

    def re_match_lines(self, lines2: Sequence[str], *, consecutive: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Check lines exist in the output (using :func:`python:re.match`).\n\n        The argument is a list of lines which have to match using ``re.match``.\n        If they do not match a pytest.fail() is called.\n\n        The matches and non-matches are also shown as part of the error message.\n\n        :param lines2: string patterns to match.\n        :param consecutive: match lines consecutively?\n        '
        __tracebackhide__ = True
        self._match_lines(lines2, lambda name, pat: bool(re.match(pat, name)), 're.match', consecutive=consecutive)

    def _match_lines(self, lines2: Sequence[str], match_func: Callable[[str, str], bool], match_nickname: str, *, consecutive: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Underlying implementation of ``fnmatch_lines`` and ``re_match_lines``.\n\n        :param Sequence[str] lines2:\n            List of string patterns to match. The actual format depends on\n            ``match_func``.\n        :param match_func:\n            A callable ``match_func(line, pattern)`` where line is the\n            captured line from stdout/stderr and pattern is the matching\n            pattern.\n        :param str match_nickname:\n            The nickname for the match function that will be logged to stdout\n            when a match occurs.\n        :param consecutive:\n            Match lines consecutively?\n        '
        if not isinstance(lines2, collections.abc.Sequence):
            raise TypeError(f'invalid type for lines2: {type(lines2).__name__}')
        lines2 = self._getlines(lines2)
        lines1 = self.lines[:]
        extralines = []
        __tracebackhide__ = True
        wnick = len(match_nickname) + 1
        started = False
        for line in lines2:
            nomatchprinted = False
            while lines1:
                nextline = lines1.pop(0)
                if line == nextline:
                    self._log('exact match:', repr(line))
                    started = True
                    break
                elif match_func(nextline, line):
                    self._log('%s:' % match_nickname, repr(line))
                    self._log('{:>{width}}'.format('with:', width=wnick), repr(nextline))
                    started = True
                    break
                else:
                    if consecutive and started:
                        msg = f'no consecutive match: {line!r}'
                        self._log(msg)
                        self._log('{:>{width}}'.format('with:', width=wnick), repr(nextline))
                        self._fail(msg)
                    if not nomatchprinted:
                        self._log('{:>{width}}'.format('nomatch:', width=wnick), repr(line))
                        nomatchprinted = True
                    self._log('{:>{width}}'.format('and:', width=wnick), repr(nextline))
                extralines.append(nextline)
            else:
                msg = f'remains unmatched: {line!r}'
                self._log(msg)
                self._fail(msg)
        self._log_output = []

    def no_fnmatch_line(self, pat: str) -> None:
        if False:
            while True:
                i = 10
        'Ensure captured lines do not match the given pattern, using ``fnmatch.fnmatch``.\n\n        :param str pat: The pattern to match lines.\n        '
        __tracebackhide__ = True
        self._no_match_line(pat, fnmatch, 'fnmatch')

    def no_re_match_line(self, pat: str) -> None:
        if False:
            return 10
        'Ensure captured lines do not match the given pattern, using ``re.match``.\n\n        :param str pat: The regular expression to match lines.\n        '
        __tracebackhide__ = True
        self._no_match_line(pat, lambda name, pat: bool(re.match(pat, name)), 're.match')

    def _no_match_line(self, pat: str, match_func: Callable[[str, str], bool], match_nickname: str) -> None:
        if False:
            print('Hello World!')
        'Ensure captured lines does not have a the given pattern, using ``fnmatch.fnmatch``.\n\n        :param str pat: The pattern to match lines.\n        '
        __tracebackhide__ = True
        nomatch_printed = False
        wnick = len(match_nickname) + 1
        for line in self.lines:
            if match_func(line, pat):
                msg = f'{match_nickname}: {pat!r}'
                self._log(msg)
                self._log('{:>{width}}'.format('with:', width=wnick), repr(line))
                self._fail(msg)
            else:
                if not nomatch_printed:
                    self._log('{:>{width}}'.format('nomatch:', width=wnick), repr(pat))
                    nomatch_printed = True
                self._log('{:>{width}}'.format('and:', width=wnick), repr(line))
        self._log_output = []

    def _fail(self, msg: str) -> None:
        if False:
            while True:
                i = 10
        __tracebackhide__ = True
        log_text = self._log_text
        self._log_output = []
        fail(log_text)

    def str(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return the entire original text.'
        return str(self)