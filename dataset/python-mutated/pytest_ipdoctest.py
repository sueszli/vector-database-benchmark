"""Discover and run ipdoctests in modules and test files."""
import builtins
import bdb
import inspect
import os
import platform
import sys
import traceback
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import pytest
from _pytest import outcomes
from _pytest._code.code import ExceptionInfo
from _pytest._code.code import ReprFileLocation
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import safe_getattr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.outcomes import OutcomeException
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import import_path
from _pytest.python_api import approx
from _pytest.warning_types import PytestWarning
if TYPE_CHECKING:
    import doctest
DOCTEST_REPORT_CHOICE_NONE = 'none'
DOCTEST_REPORT_CHOICE_CDIFF = 'cdiff'
DOCTEST_REPORT_CHOICE_NDIFF = 'ndiff'
DOCTEST_REPORT_CHOICE_UDIFF = 'udiff'
DOCTEST_REPORT_CHOICE_ONLY_FIRST_FAILURE = 'only_first_failure'
DOCTEST_REPORT_CHOICES = (DOCTEST_REPORT_CHOICE_NONE, DOCTEST_REPORT_CHOICE_CDIFF, DOCTEST_REPORT_CHOICE_NDIFF, DOCTEST_REPORT_CHOICE_UDIFF, DOCTEST_REPORT_CHOICE_ONLY_FIRST_FAILURE)
RUNNER_CLASS = None
CHECKER_CLASS: Optional[Type['IPDoctestOutputChecker']] = None

def pytest_addoption(parser: Parser) -> None:
    if False:
        while True:
            i = 10
    parser.addini('ipdoctest_optionflags', 'option flags for ipdoctests', type='args', default=['ELLIPSIS'])
    parser.addini('ipdoctest_encoding', 'encoding used for ipdoctest files', default='utf-8')
    group = parser.getgroup('collect')
    group.addoption('--ipdoctest-modules', action='store_true', default=False, help='run ipdoctests in all .py modules', dest='ipdoctestmodules')
    group.addoption('--ipdoctest-report', type=str.lower, default='udiff', help='choose another output format for diffs on ipdoctest failure', choices=DOCTEST_REPORT_CHOICES, dest='ipdoctestreport')
    group.addoption('--ipdoctest-glob', action='append', default=[], metavar='pat', help='ipdoctests file matching pattern, default: test*.txt', dest='ipdoctestglob')
    group.addoption('--ipdoctest-ignore-import-errors', action='store_true', default=False, help='ignore ipdoctest ImportErrors', dest='ipdoctest_ignore_import_errors')
    group.addoption('--ipdoctest-continue-on-failure', action='store_true', default=False, help='for a given ipdoctest, continue to run after the first failure', dest='ipdoctest_continue_on_failure')

def pytest_unconfigure() -> None:
    if False:
        for i in range(10):
            print('nop')
    global RUNNER_CLASS
    RUNNER_CLASS = None

def pytest_collect_file(file_path: Path, parent: Collector) -> Optional[Union['IPDoctestModule', 'IPDoctestTextfile']]:
    if False:
        for i in range(10):
            print('nop')
    config = parent.config
    if file_path.suffix == '.py':
        if config.option.ipdoctestmodules and (not any((_is_setup_py(file_path), _is_main_py(file_path)))):
            mod: IPDoctestModule = IPDoctestModule.from_parent(parent, path=file_path)
            return mod
    elif _is_ipdoctest(config, file_path, parent):
        txt: IPDoctestTextfile = IPDoctestTextfile.from_parent(parent, path=file_path)
        return txt
    return None
if int(pytest.__version__.split('.')[0]) < 7:
    _collect_file = pytest_collect_file

    def pytest_collect_file(path, parent: Collector) -> Optional[Union['IPDoctestModule', 'IPDoctestTextfile']]:
        if False:
            return 10
        return _collect_file(Path(path), parent)
    _import_path = import_path

    def import_path(path, root):
        if False:
            for i in range(10):
                print('nop')
        import py.path
        return _import_path(py.path.local(path))

def _is_setup_py(path: Path) -> bool:
    if False:
        while True:
            i = 10
    if path.name != 'setup.py':
        return False
    contents = path.read_bytes()
    return b'setuptools' in contents or b'distutils' in contents

def _is_ipdoctest(config: Config, path: Path, parent: Collector) -> bool:
    if False:
        while True:
            i = 10
    if path.suffix in ('.txt', '.rst') and parent.session.isinitpath(path):
        return True
    globs = config.getoption('ipdoctestglob') or ['test*.txt']
    return any((fnmatch_ex(glob, path) for glob in globs))

def _is_main_py(path: Path) -> bool:
    if False:
        return 10
    return path.name == '__main__.py'

class ReprFailDoctest(TerminalRepr):

    def __init__(self, reprlocation_lines: Sequence[Tuple[ReprFileLocation, Sequence[str]]]) -> None:
        if False:
            return 10
        self.reprlocation_lines = reprlocation_lines

    def toterminal(self, tw: TerminalWriter) -> None:
        if False:
            while True:
                i = 10
        for (reprlocation, lines) in self.reprlocation_lines:
            for line in lines:
                tw.line(line)
            reprlocation.toterminal(tw)

class MultipleDoctestFailures(Exception):

    def __init__(self, failures: Sequence['doctest.DocTestFailure']) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.failures = failures

def _init_runner_class() -> Type['IPDocTestRunner']:
    if False:
        print('Hello World!')
    import doctest
    from .ipdoctest import IPDocTestRunner

    class PytestDoctestRunner(IPDocTestRunner):
        """Runner to collect failures.

        Note that the out variable in this case is a list instead of a
        stdout-like object.
        """

        def __init__(self, checker: Optional['IPDoctestOutputChecker']=None, verbose: Optional[bool]=None, optionflags: int=0, continue_on_failure: bool=True) -> None:
            if False:
                while True:
                    i = 10
            super().__init__(checker=checker, verbose=verbose, optionflags=optionflags)
            self.continue_on_failure = continue_on_failure

        def report_failure(self, out, test: 'doctest.DocTest', example: 'doctest.Example', got: str) -> None:
            if False:
                i = 10
                return i + 15
            failure = doctest.DocTestFailure(test, example, got)
            if self.continue_on_failure:
                out.append(failure)
            else:
                raise failure

        def report_unexpected_exception(self, out, test: 'doctest.DocTest', example: 'doctest.Example', exc_info: Tuple[Type[BaseException], BaseException, types.TracebackType]) -> None:
            if False:
                return 10
            if isinstance(exc_info[1], OutcomeException):
                raise exc_info[1]
            if isinstance(exc_info[1], bdb.BdbQuit):
                outcomes.exit('Quitting debugger')
            failure = doctest.UnexpectedException(test, example, exc_info)
            if self.continue_on_failure:
                out.append(failure)
            else:
                raise failure
    return PytestDoctestRunner

def _get_runner(checker: Optional['IPDoctestOutputChecker']=None, verbose: Optional[bool]=None, optionflags: int=0, continue_on_failure: bool=True) -> 'IPDocTestRunner':
    if False:
        print('Hello World!')
    global RUNNER_CLASS
    if RUNNER_CLASS is None:
        RUNNER_CLASS = _init_runner_class()
    return RUNNER_CLASS(checker=checker, verbose=verbose, optionflags=optionflags, continue_on_failure=continue_on_failure)

class IPDoctestItem(pytest.Item):

    def __init__(self, name: str, parent: 'Union[IPDoctestTextfile, IPDoctestModule]', runner: Optional['IPDocTestRunner']=None, dtest: Optional['doctest.DocTest']=None) -> None:
        if False:
            return 10
        super().__init__(name, parent)
        self.runner = runner
        self.dtest = dtest
        self.obj = None
        self.fixture_request: Optional[FixtureRequest] = None

    @classmethod
    def from_parent(cls, parent: 'Union[IPDoctestTextfile, IPDoctestModule]', *, name: str, runner: 'IPDocTestRunner', dtest: 'doctest.DocTest'):
        if False:
            return 10
        'The public named constructor.'
        return super().from_parent(name=name, parent=parent, runner=runner, dtest=dtest)

    def setup(self) -> None:
        if False:
            while True:
                i = 10
        if self.dtest is not None:
            self.fixture_request = _setup_fixtures(self)
            globs = dict(getfixture=self.fixture_request.getfixturevalue)
            for (name, value) in self.fixture_request.getfixturevalue('ipdoctest_namespace').items():
                globs[name] = value
            self.dtest.globs.update(globs)
            from .ipdoctest import IPExample
            if isinstance(self.dtest.examples[0], IPExample):
                self._user_ns_orig = {}
                self._user_ns_orig.update(_ip.user_ns)
                _ip.user_ns.update(self.dtest.globs)
                _ip.user_ns.pop('_', None)
                _ip.user_ns['__builtins__'] = builtins
                self.dtest.globs = _ip.user_ns

    def teardown(self) -> None:
        if False:
            while True:
                i = 10
        from .ipdoctest import IPExample
        if isinstance(self.dtest.examples[0], IPExample):
            self.dtest.globs = {}
            _ip.user_ns.clear()
            _ip.user_ns.update(self._user_ns_orig)
            del self._user_ns_orig
        self.dtest.globs.clear()

    def runtest(self) -> None:
        if False:
            print('Hello World!')
        assert self.dtest is not None
        assert self.runner is not None
        _check_all_skipped(self.dtest)
        self._disable_output_capturing_for_darwin()
        failures: List['doctest.DocTestFailure'] = []
        had_underscore_value = hasattr(builtins, '_')
        underscore_original_value = getattr(builtins, '_', None)
        curdir = os.getcwd()
        os.chdir(self.fspath.dirname)
        try:
            self.runner.run(self.dtest, out=failures, clear_globs=False)
        finally:
            os.chdir(curdir)
            if had_underscore_value:
                setattr(builtins, '_', underscore_original_value)
            elif hasattr(builtins, '_'):
                delattr(builtins, '_')
        if failures:
            raise MultipleDoctestFailures(failures)

    def _disable_output_capturing_for_darwin(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Disable output capturing. Otherwise, stdout is lost to ipdoctest (pytest#985).'
        if platform.system() != 'Darwin':
            return
        capman = self.config.pluginmanager.getplugin('capturemanager')
        if capman:
            capman.suspend_global_capture(in_=True)
            (out, err) = capman.read_global_capture()
            sys.stdout.write(out)
            sys.stderr.write(err)

    def repr_failure(self, excinfo: ExceptionInfo[BaseException]) -> Union[str, TerminalRepr]:
        if False:
            print('Hello World!')
        import doctest
        failures: Optional[Sequence[Union[doctest.DocTestFailure, doctest.UnexpectedException]]] = None
        if isinstance(excinfo.value, (doctest.DocTestFailure, doctest.UnexpectedException)):
            failures = [excinfo.value]
        elif isinstance(excinfo.value, MultipleDoctestFailures):
            failures = excinfo.value.failures
        if failures is None:
            return super().repr_failure(excinfo)
        reprlocation_lines = []
        for failure in failures:
            example = failure.example
            test = failure.test
            filename = test.filename
            if test.lineno is None:
                lineno = None
            else:
                lineno = test.lineno + example.lineno + 1
            message = type(failure).__name__
            reprlocation = ReprFileLocation(filename, lineno, message)
            checker = _get_checker()
            report_choice = _get_report_choice(self.config.getoption('ipdoctestreport'))
            if lineno is not None:
                assert failure.test.docstring is not None
                lines = failure.test.docstring.splitlines(False)
                assert test.lineno is not None
                lines = ['%03d %s' % (i + test.lineno + 1, x) for (i, x) in enumerate(lines)]
                lines = lines[max(example.lineno - 9, 0):example.lineno + 1]
            else:
                lines = ['EXAMPLE LOCATION UNKNOWN, not showing all tests of that example']
                indent = '>>>'
                for line in example.source.splitlines():
                    lines.append(f'??? {indent} {line}')
                    indent = '...'
            if isinstance(failure, doctest.DocTestFailure):
                lines += checker.output_difference(example, failure.got, report_choice).split('\n')
            else:
                inner_excinfo = ExceptionInfo.from_exc_info(failure.exc_info)
                lines += ['UNEXPECTED EXCEPTION: %s' % repr(inner_excinfo.value)]
                lines += [x.strip('\n') for x in traceback.format_exception(*failure.exc_info)]
            reprlocation_lines.append((reprlocation, lines))
        return ReprFailDoctest(reprlocation_lines)

    def reportinfo(self) -> Tuple[Union['os.PathLike[str]', str], Optional[int], str]:
        if False:
            i = 10
            return i + 15
        assert self.dtest is not None
        return (self.path, self.dtest.lineno, '[ipdoctest] %s' % self.name)
    if int(pytest.__version__.split('.')[0]) < 7:

        @property
        def path(self) -> Path:
            if False:
                while True:
                    i = 10
            return Path(self.fspath)

def _get_flag_lookup() -> Dict[str, int]:
    if False:
        return 10
    import doctest
    return dict(DONT_ACCEPT_TRUE_FOR_1=doctest.DONT_ACCEPT_TRUE_FOR_1, DONT_ACCEPT_BLANKLINE=doctest.DONT_ACCEPT_BLANKLINE, NORMALIZE_WHITESPACE=doctest.NORMALIZE_WHITESPACE, ELLIPSIS=doctest.ELLIPSIS, IGNORE_EXCEPTION_DETAIL=doctest.IGNORE_EXCEPTION_DETAIL, COMPARISON_FLAGS=doctest.COMPARISON_FLAGS, ALLOW_UNICODE=_get_allow_unicode_flag(), ALLOW_BYTES=_get_allow_bytes_flag(), NUMBER=_get_number_flag())

def get_optionflags(parent):
    if False:
        while True:
            i = 10
    optionflags_str = parent.config.getini('ipdoctest_optionflags')
    flag_lookup_table = _get_flag_lookup()
    flag_acc = 0
    for flag in optionflags_str:
        flag_acc |= flag_lookup_table[flag]
    return flag_acc

def _get_continue_on_failure(config):
    if False:
        print('Hello World!')
    continue_on_failure = config.getvalue('ipdoctest_continue_on_failure')
    if continue_on_failure:
        if config.getvalue('usepdb'):
            continue_on_failure = False
    return continue_on_failure

class IPDoctestTextfile(pytest.Module):
    obj = None

    def collect(self) -> Iterable[IPDoctestItem]:
        if False:
            i = 10
            return i + 15
        import doctest
        from .ipdoctest import IPDocTestParser
        encoding = self.config.getini('ipdoctest_encoding')
        text = self.path.read_text(encoding)
        filename = str(self.path)
        name = self.path.name
        globs = {'__name__': '__main__'}
        optionflags = get_optionflags(self)
        runner = _get_runner(verbose=False, optionflags=optionflags, checker=_get_checker(), continue_on_failure=_get_continue_on_failure(self.config))
        parser = IPDocTestParser()
        test = parser.get_doctest(text, globs, name, filename, 0)
        if test.examples:
            yield IPDoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)
    if int(pytest.__version__.split('.')[0]) < 7:

        @property
        def path(self) -> Path:
            if False:
                i = 10
                return i + 15
            return Path(self.fspath)

        @classmethod
        def from_parent(cls, parent, *, fspath=None, path: Optional[Path]=None, **kw):
            if False:
                for i in range(10):
                    print('nop')
            if path is not None:
                import py.path
                fspath = py.path.local(path)
            return super().from_parent(parent=parent, fspath=fspath, **kw)

def _check_all_skipped(test: 'doctest.DocTest') -> None:
    if False:
        i = 10
        return i + 15
    'Raise pytest.skip() if all examples in the given DocTest have the SKIP\n    option set.'
    import doctest
    all_skipped = all((x.options.get(doctest.SKIP, False) for x in test.examples))
    if all_skipped:
        pytest.skip('all docstests skipped by +SKIP option')

def _is_mocked(obj: object) -> bool:
    if False:
        while True:
            i = 10
    'Return if an object is possibly a mock object by checking the\n    existence of a highly improbable attribute.'
    return safe_getattr(obj, 'pytest_mock_example_attribute_that_shouldnt_exist', None) is not None

@contextmanager
def _patch_unwrap_mock_aware() -> Generator[None, None, None]:
    if False:
        i = 10
        return i + 15
    "Context manager which replaces ``inspect.unwrap`` with a version\n    that's aware of mock objects and doesn't recurse into them."
    real_unwrap = inspect.unwrap

    def _mock_aware_unwrap(func: Callable[..., Any], *, stop: Optional[Callable[[Any], Any]]=None) -> Any:
        if False:
            while True:
                i = 10
        try:
            if stop is None or stop is _is_mocked:
                return real_unwrap(func, stop=_is_mocked)
            _stop = stop
            return real_unwrap(func, stop=lambda obj: _is_mocked(obj) or _stop(func))
        except Exception as e:
            warnings.warn("Got %r when unwrapping %r.  This is usually caused by a violation of Python's object protocol; see e.g. https://github.com/pytest-dev/pytest/issues/5080" % (e, func), PytestWarning)
            raise
    inspect.unwrap = _mock_aware_unwrap
    try:
        yield
    finally:
        inspect.unwrap = real_unwrap

class IPDoctestModule(pytest.Module):

    def collect(self) -> Iterable[IPDoctestItem]:
        if False:
            for i in range(10):
                print('nop')
        import doctest
        from .ipdoctest import DocTestFinder, IPDocTestParser

        class MockAwareDocTestFinder(DocTestFinder):
            """A hackish ipdoctest finder that overrides stdlib internals to fix a stdlib bug.

            https://github.com/pytest-dev/pytest/issues/3456
            https://bugs.python.org/issue25532
            """

            def _find_lineno(self, obj, source_lines):
                if False:
                    return 10
                'Doctest code does not take into account `@property`, this\n                is a hackish way to fix it. https://bugs.python.org/issue17446\n\n                Wrapped Doctests will need to be unwrapped so the correct\n                line number is returned. This will be reported upstream. #8796\n                '
                if isinstance(obj, property):
                    obj = getattr(obj, 'fget', obj)
                if hasattr(obj, '__wrapped__'):
                    obj = inspect.unwrap(obj)
                return super()._find_lineno(obj, source_lines)

            def _find(self, tests, obj, name, module, source_lines, globs, seen) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                if _is_mocked(obj):
                    return
                with _patch_unwrap_mock_aware():
                    super()._find(tests, obj, name, module, source_lines, globs, seen)
        if self.path.name == 'conftest.py':
            if int(pytest.__version__.split('.')[0]) < 7:
                module = self.config.pluginmanager._importconftest(self.path, self.config.getoption('importmode'))
            else:
                module = self.config.pluginmanager._importconftest(self.path, self.config.getoption('importmode'), rootpath=self.config.rootpath)
        else:
            try:
                module = import_path(self.path, root=self.config.rootpath)
            except ImportError:
                if self.config.getvalue('ipdoctest_ignore_import_errors'):
                    pytest.skip('unable to import module %r' % self.path)
                else:
                    raise
        finder = MockAwareDocTestFinder(parser=IPDocTestParser())
        optionflags = get_optionflags(self)
        runner = _get_runner(verbose=False, optionflags=optionflags, checker=_get_checker(), continue_on_failure=_get_continue_on_failure(self.config))
        for test in finder.find(module, module.__name__):
            if test.examples:
                yield IPDoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)
    if int(pytest.__version__.split('.')[0]) < 7:

        @property
        def path(self) -> Path:
            if False:
                print('Hello World!')
            return Path(self.fspath)

        @classmethod
        def from_parent(cls, parent, *, fspath=None, path: Optional[Path]=None, **kw):
            if False:
                i = 10
                return i + 15
            if path is not None:
                import py.path
                fspath = py.path.local(path)
            return super().from_parent(parent=parent, fspath=fspath, **kw)

def _setup_fixtures(doctest_item: IPDoctestItem) -> FixtureRequest:
    if False:
        for i in range(10):
            print('nop')
    'Used by IPDoctestTextfile and IPDoctestItem to setup fixture information.'

    def func() -> None:
        if False:
            print('Hello World!')
        pass
    doctest_item.funcargs = {}
    fm = doctest_item.session._fixturemanager
    doctest_item._fixtureinfo = fm.getfixtureinfo(node=doctest_item, func=func, cls=None, funcargs=False)
    fixture_request = FixtureRequest(doctest_item, _ispytest=True)
    fixture_request._fillfixtures()
    return fixture_request

def _init_checker_class() -> Type['IPDoctestOutputChecker']:
    if False:
        return 10
    import doctest
    import re
    from .ipdoctest import IPDoctestOutputChecker

    class LiteralsOutputChecker(IPDoctestOutputChecker):
        _unicode_literal_re = re.compile('(\\W|^)[uU]([rR]?[\\\'\\"])', re.UNICODE)
        _bytes_literal_re = re.compile('(\\W|^)[bB]([rR]?[\\\'\\"])', re.UNICODE)
        _number_re = re.compile('\n            (?P<number>\n              (?P<mantissa>\n                (?P<integer1> [+-]?\\d*)\\.(?P<fraction>\\d+)\n                |\n                (?P<integer2> [+-]?\\d+)\\.\n              )\n              (?:\n                [Ee]\n                (?P<exponent1> [+-]?\\d+)\n              )?\n              |\n              (?P<integer3> [+-]?\\d+)\n              (?:\n                [Ee]\n                (?P<exponent2> [+-]?\\d+)\n              )\n            )\n            ', re.VERBOSE)

        def check_output(self, want: str, got: str, optionflags: int) -> bool:
            if False:
                print('Hello World!')
            if super().check_output(want, got, optionflags):
                return True
            allow_unicode = optionflags & _get_allow_unicode_flag()
            allow_bytes = optionflags & _get_allow_bytes_flag()
            allow_number = optionflags & _get_number_flag()
            if not allow_unicode and (not allow_bytes) and (not allow_number):
                return False

            def remove_prefixes(regex: Pattern[str], txt: str) -> str:
                if False:
                    i = 10
                    return i + 15
                return re.sub(regex, '\\1\\2', txt)
            if allow_unicode:
                want = remove_prefixes(self._unicode_literal_re, want)
                got = remove_prefixes(self._unicode_literal_re, got)
            if allow_bytes:
                want = remove_prefixes(self._bytes_literal_re, want)
                got = remove_prefixes(self._bytes_literal_re, got)
            if allow_number:
                got = self._remove_unwanted_precision(want, got)
            return super().check_output(want, got, optionflags)

        def _remove_unwanted_precision(self, want: str, got: str) -> str:
            if False:
                return 10
            wants = list(self._number_re.finditer(want))
            gots = list(self._number_re.finditer(got))
            if len(wants) != len(gots):
                return got
            offset = 0
            for (w, g) in zip(wants, gots):
                fraction: Optional[str] = w.group('fraction')
                exponent: Optional[str] = w.group('exponent1')
                if exponent is None:
                    exponent = w.group('exponent2')
                precision = 0 if fraction is None else len(fraction)
                if exponent is not None:
                    precision -= int(exponent)
                if float(w.group()) == approx(float(g.group()), abs=10 ** (-precision)):
                    got = got[:g.start() + offset] + w.group() + got[g.end() + offset:]
                    offset += w.end() - w.start() - (g.end() - g.start())
            return got
    return LiteralsOutputChecker

def _get_checker() -> 'IPDoctestOutputChecker':
    if False:
        i = 10
        return i + 15
    'Return a IPDoctestOutputChecker subclass that supports some\n    additional options:\n\n    * ALLOW_UNICODE and ALLOW_BYTES options to ignore u\'\' and b\'\'\n      prefixes (respectively) in string literals. Useful when the same\n      ipdoctest should run in Python 2 and Python 3.\n\n    * NUMBER to ignore floating-point differences smaller than the\n      precision of the literal number in the ipdoctest.\n\n    An inner class is used to avoid importing "ipdoctest" at the module\n    level.\n    '
    global CHECKER_CLASS
    if CHECKER_CLASS is None:
        CHECKER_CLASS = _init_checker_class()
    return CHECKER_CLASS()

def _get_allow_unicode_flag() -> int:
    if False:
        print('Hello World!')
    'Register and return the ALLOW_UNICODE flag.'
    import doctest
    return doctest.register_optionflag('ALLOW_UNICODE')

def _get_allow_bytes_flag() -> int:
    if False:
        i = 10
        return i + 15
    'Register and return the ALLOW_BYTES flag.'
    import doctest
    return doctest.register_optionflag('ALLOW_BYTES')

def _get_number_flag() -> int:
    if False:
        for i in range(10):
            print('nop')
    'Register and return the NUMBER flag.'
    import doctest
    return doctest.register_optionflag('NUMBER')

def _get_report_choice(key: str) -> int:
    if False:
        return 10
    'Return the actual `ipdoctest` module flag value.\n\n    We want to do it as late as possible to avoid importing `ipdoctest` and all\n    its dependencies when parsing options, as it adds overhead and breaks tests.\n    '
    import doctest
    return {DOCTEST_REPORT_CHOICE_UDIFF: doctest.REPORT_UDIFF, DOCTEST_REPORT_CHOICE_CDIFF: doctest.REPORT_CDIFF, DOCTEST_REPORT_CHOICE_NDIFF: doctest.REPORT_NDIFF, DOCTEST_REPORT_CHOICE_ONLY_FIRST_FAILURE: doctest.REPORT_ONLY_FIRST_FAILURE, DOCTEST_REPORT_CHOICE_NONE: 0}[key]

@pytest.fixture(scope='session')
def ipdoctest_namespace() -> Dict[str, Any]:
    if False:
        print('Hello World!')
    'Fixture that returns a :py:class:`dict` that will be injected into the\n    namespace of ipdoctests.'
    return dict()