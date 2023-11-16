import base64
import contextlib
import enum
import hashlib
import inspect
import io
import os
import re
import shutil
import sys
from collections import Counter, OrderedDict
from typing import Callable, List, Optional, Tuple, Type, TypeVar, Union
import llnl.util.filesystem as fs
import llnl.util.tty as tty
from llnl.string import plural
from llnl.util.lang import nullcontext
from llnl.util.tty.color import colorize
import spack.error
import spack.paths
import spack.util.spack_json as sjson
from spack.installer import InstallError
from spack.spec import Spec
from spack.util.prefix import Prefix
TestFailureType = Tuple[BaseException, str]
test_suite_filename = 'test_suite.lock'
results_filename = 'results.txt'
spack_install_test_log = 'install-time-test-log.txt'
ListOrStringType = Union[str, List[str]]
LogType = Union['tty.log.nixlog', 'tty.log.winlog']
Pb = TypeVar('Pb', bound='spack.package_base.PackageBase')
PackageObjectOrClass = Union[Pb, Type[Pb]]

class TestStatus(enum.Enum):
    """Names of different stand-alone test states."""
    NO_TESTS = -1
    SKIPPED = 0
    FAILED = 1
    PASSED = 2

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return f'{self.name}'

    def lower(self):
        if False:
            i = 10
            return i + 15
        name = f'{self.name}'
        return name.lower()

def get_escaped_text_output(filename: str) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Retrieve and escape the expected text output from the file\n\n    Args:\n        filename: path to the file\n\n    Returns:\n        escaped text lines read from the file\n    '
    with open(filename) as f:
        expected = f.read()
    return [re.escape(ln) for ln in expected.split('\n')]

def get_test_stage_dir():
    if False:
        print('Hello World!')
    'Retrieves the ``config:test_stage`` path to the configured test stage\n    root directory\n\n    Returns:\n        str: absolute path to the configured test stage root or, if none,\n            the default test stage path\n    '
    return spack.util.path.canonicalize_path(spack.config.get('config:test_stage', spack.paths.default_test_path))

def cache_extra_test_sources(pkg: Pb, srcs: ListOrStringType):
    if False:
        return 10
    'Copy relative source paths to the corresponding install test subdir\n\n    This routine is intended as an optional install test setup helper for\n    grabbing source files/directories during the installation process and\n    copying them to the installation test subdirectory for subsequent use\n    during install testing.\n\n    Args:\n        pkg: package being tested\n        srcs: relative path for file(s) and or subdirectory(ies) located in\n            the staged source path that are to be copied to the corresponding\n            location(s) under the install testing directory.\n\n    Raises:\n        spack.installer.InstallError: if any of the source paths are absolute\n            or do not exist\n            under the build stage\n    '
    errors = []
    paths = [srcs] if isinstance(srcs, str) else srcs
    for path in paths:
        pre = f"Source path ('{path}')"
        src_path = os.path.join(pkg.stage.source_path, path)
        dest_path = os.path.join(install_test_root(pkg), path)
        if os.path.isabs(path):
            errors.append(f'{pre} must be relative to the build stage directory.')
            continue
        if os.path.isdir(src_path):
            fs.install_tree(src_path, dest_path)
        elif os.path.exists(src_path):
            fs.mkdirp(os.path.dirname(dest_path))
            fs.copy(src_path, dest_path)
        else:
            errors.append(f'{pre} for the copy does not exist')
    if errors:
        raise InstallError('\n'.join(errors), pkg=pkg)

def check_outputs(expected: Union[list, set, str], actual: str):
    if False:
        while True:
            i = 10
    'Ensure the expected outputs are contained in the actual outputs.\n\n    Args:\n        expected: expected raw output string(s)\n        actual: actual output string\n\n    Raises:\n        RuntimeError: the expected output is not found in the actual output\n    '
    expected = expected if isinstance(expected, (list, set)) else [expected]
    errors = []
    for check in expected:
        if not re.search(check, actual):
            errors.append(f"Expected '{check}' in output '{actual}'")
    if errors:
        raise RuntimeError('\n  '.join(errors))

def find_required_file(root: str, filename: str, expected: int=1, recursive: bool=True) -> ListOrStringType:
    if False:
        print('Hello World!')
    'Find the required file(s) under the root directory.\n\n    Args:\n       root: root directory for the search\n       filename: name of the file being located\n       expected: expected number of files to be found under the directory\n           (default is 1)\n       recursive: ``True`` if subdirectories are to be recursively searched,\n           else ``False`` (default is ``True``)\n\n    Returns: the path(s), relative to root, to the required file(s)\n\n    Raises:\n        Exception: SkipTest when number of files detected does not match expected\n    '
    paths = fs.find(root, filename, recursive=recursive)
    num_paths = len(paths)
    if num_paths != expected:
        files = ': {}'.format(', '.join(paths)) if num_paths else ''
        raise SkipTest('Expected {} of {} under {} but {} found{}'.format(plural(expected, 'copy', 'copies'), filename, root, plural(num_paths, 'copy', 'copies'), files))
    return paths[0] if expected == 1 else paths

def install_test_root(pkg: Pb):
    if False:
        for i in range(10):
            print('nop')
    'The install test root directory.\n\n    Args:\n        pkg: package being tested\n    '
    return os.path.join(pkg.metadata_dir, 'test')

def print_message(logger: LogType, msg: str, verbose: bool=False):
    if False:
        return 10
    'Print the message to the log, optionally echoing.\n\n    Args:\n        logger: instance of the output logger (e.g. nixlog or winlog)\n        msg: message being output\n        verbose: ``True`` displays verbose output, ``False`` suppresses\n            it (``False`` is default)\n    '
    if verbose:
        with logger.force_echo():
            tty.info(msg, format='g')
    else:
        tty.info(msg, format='g')

def overall_status(current_status: 'TestStatus', substatuses: List['TestStatus']) -> 'TestStatus':
    if False:
        print('Hello World!')
    'Determine the overall status based on the current and associated sub status values.\n\n    Args:\n        current_status: current overall status, assumed to default to PASSED\n        substatuses: status of each test part or overall status of each test spec\n    Returns:\n        test status encompassing the main test and all subtests\n    '
    if current_status in [TestStatus.SKIPPED, TestStatus.NO_TESTS, TestStatus.FAILED]:
        return current_status
    skipped = 0
    for status in substatuses:
        if status == TestStatus.FAILED:
            return status
        elif status == TestStatus.SKIPPED:
            skipped += 1
    if skipped and skipped == len(substatuses):
        return TestStatus.SKIPPED
    return current_status

class PackageTest:
    """The class that manages stand-alone (post-install) package tests."""

    def __init__(self, pkg: Pb):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            pkg: package being tested\n\n        Raises:\n            ValueError: if the package is not concrete\n        '
        if not pkg.spec.concrete:
            raise ValueError('Stand-alone tests require a concrete package')
        self.counts: 'Counter' = Counter()
        self.pkg = pkg
        self.test_failures: List[TestFailureType] = []
        self.test_parts: OrderedDict[str, 'TestStatus'] = OrderedDict()
        self.test_log_file: str
        self.pkg_id: str
        if pkg.test_suite:
            self.test_log_file = pkg.test_suite.log_file_for_spec(pkg.spec)
            self.tested_file = pkg.test_suite.tested_file_for_spec(pkg.spec)
            self.pkg_id = pkg.test_suite.test_pkg_id(pkg.spec)
        else:
            pkg.test_suite = TestSuite([pkg.spec])
            self.test_log_file = fs.join_path(pkg.stage.path, spack_install_test_log)
            self.pkg_id = pkg.spec.format('{name}-{version}-{hash:7}')
        self._logger = None

    @property
    def logger(self) -> Optional[LogType]:
        if False:
            print('Hello World!')
        'The current logger or, if none, sets to one.'
        if not self._logger:
            self._logger = tty.log.log_output(self.test_log_file)
        return self._logger

    @contextlib.contextmanager
    def test_logger(self, verbose: bool=False, externals: bool=False):
        if False:
            print('Hello World!')
        'Context manager for setting up the test logger\n\n        Args:\n            verbose: Display verbose output, including echoing to stdout,\n                otherwise suppress it\n            externals: ``True`` for performing tests if external package,\n                ``False`` to skip them\n        '
        fs.touch(self.test_log_file)
        fs.set_install_permissions(self.test_log_file)
        with tty.log.log_output(self.test_log_file, verbose) as self._logger:
            with self.logger.force_echo():
                tty.msg('Testing package ' + colorize('@*g{' + self.pkg_id + '}'))
            old_debug = tty.is_debug()
            tty.set_debug(True)
            try:
                yield self.logger
            finally:
                tty.set_debug(old_debug)

    @property
    def archived_install_test_log(self) -> str:
        if False:
            i = 10
            return i + 15
        return fs.join_path(self.pkg.metadata_dir, spack_install_test_log)

    def archive_install_test_log(self, dest_dir: str):
        if False:
            return 10
        if os.path.exists(self.test_log_file):
            fs.install(self.test_log_file, self.archived_install_test_log)

    def add_failure(self, exception: Exception, msg: str):
        if False:
            while True:
                i = 10
        'Add the failure details to the current list.'
        self.test_failures.append((exception, msg))

    def status(self, name: str, status: 'TestStatus', msg: Optional[str]=None):
        if False:
            while True:
                i = 10
        'Track and print the test status for the test part name.'
        part_name = f'{self.pkg.__class__.__name__}::{name}'
        extra = '' if msg is None else f': {msg}'
        substatuses = []
        for (pname, substatus) in self.test_parts.items():
            if pname != part_name and pname.startswith(part_name):
                substatuses.append(substatus)
        if substatuses:
            status = overall_status(status, substatuses)
        print(f'{status}: {part_name}{extra}')
        self.test_parts[part_name] = status
        self.counts[status] += 1

    def phase_tests(self, builder: spack.builder.Builder, phase_name: str, method_names: List[str]):
        if False:
            i = 10
            return i + 15
        "Execute the builder's package phase-time tests.\n\n        Args:\n            builder: builder for package being tested\n            phase_name: the name of the build-time phase (e.g., ``build``, ``install``)\n            method_names: phase-specific callback method names\n        "
        verbose = tty.is_verbose()
        fail_fast = spack.config.get('config:fail_fast', False)
        with self.test_logger(verbose=verbose, externals=False) as logger:
            print_message(logger, f'Running {phase_name}-time tests', verbose)
            builder.pkg.test_suite.current_test_spec = builder.pkg.spec
            builder.pkg.test_suite.current_base_spec = builder.pkg.spec
            have_tests = any((name.startswith('test') for name in method_names))
            if have_tests:
                copy_test_files(builder.pkg, builder.pkg.spec)
            for name in method_names:
                try:
                    fn = getattr(builder.pkg, name, getattr(builder, name))
                    msg = f'RUN-TESTS: {phase_name}-time tests [{name}]'
                    print_message(logger, msg, verbose)
                    fn()
                except AttributeError as e:
                    msg = f'RUN-TESTS: method not implemented [{name}]'
                    print_message(logger, msg, verbose)
                    self.add_failure(e, msg)
                    if fail_fast:
                        break
            if have_tests:
                print_message(logger, 'Completed testing', verbose)
            if self.test_failures:
                raise TestFailure(self.test_failures)

    def stand_alone_tests(self, kwargs):
        if False:
            return 10
        "Run the package's stand-alone tests.\n\n        Args:\n            kwargs (dict): arguments to be used by the test process\n        "
        import spack.build_environment
        spack.build_environment.start_build_process(self.pkg, test_process, kwargs)

    def parts(self) -> int:
        if False:
            return 10
        'The total number of (checked) test parts.'
        try:
            total = self.counts.total()
        except AttributeError:
            nums = [n for (_, n) in self.counts.items()]
            total = sum(nums)
        return total

    def print_log_path(self):
        if False:
            print('Hello World!')
        'Print the test log file path.'
        log = self.archived_install_test_log
        if not os.path.isfile(log):
            log = self.test_log_file
            if not (log and os.path.isfile(log)):
                tty.debug('There is no test log file (staged or installed)')
                return
        print(f'\nSee test results at:\n  {log}')

    def ran_tests(self) -> bool:
        if False:
            return 10
        '``True`` if ran tests, ``False`` otherwise.'
        return self.parts() > self.counts[TestStatus.NO_TESTS]

    def summarize(self):
        if False:
            return 10
        'Collect test results summary lines for this spec.'
        lines = []
        lines.append('{:=^80}'.format(f' SUMMARY: {self.pkg_id} '))
        for (name, status) in self.test_parts.items():
            msg = f'{name} .. {status}'
            lines.append(msg)
        summary = [f'{n} {s.lower()}' for (s, n) in self.counts.items() if n > 0]
        totals = ' {} of {} parts '.format(', '.join(summary), self.parts())
        lines.append(f'{totals:=^80}')
        return lines

    def write_tested_status(self):
        if False:
            for i in range(10):
                print('nop')
        'Write the overall status to the tested file.\n\n        If there any test part failures, then the tests failed. If all test\n        parts are skipped, then the tests were skipped. If any tests passed\n        then the tests passed; otherwise, there were not tests executed.\n        '
        status = TestStatus.NO_TESTS
        if self.counts[TestStatus.FAILED] > 0:
            status = TestStatus.FAILED
        else:
            skipped = self.counts[TestStatus.SKIPPED]
            if skipped and self.parts() == skipped:
                status = TestStatus.SKIPPED
            elif self.counts[TestStatus.PASSED] > 0:
                status = TestStatus.PASSED
        with open(self.tested_file, 'w') as f:
            f.write(f'{status.value}\n')

@contextlib.contextmanager
def test_part(pkg: Pb, test_name: str, purpose: str, work_dir: str='.', verbose: bool=False):
    if False:
        print('Hello World!')
    wdir = '.' if work_dir is None else work_dir
    tester = pkg.tester
    assert test_name and test_name.startswith('test'), f"Test name must start with 'test' but {test_name} was provided"
    if test_name == 'test':
        tty.warn("{}: the 'test' method is deprecated. Convert stand-alone test(s) to methods with names starting 'test_'.".format(pkg.name))
    title = 'test: {}: {}'.format(test_name, purpose or 'unspecified purpose')
    with fs.working_dir(wdir, create=True):
        try:
            context = tester.logger.force_echo if verbose else nullcontext
            with context():
                tty.info(title, format='g')
                yield
            tester.status(test_name, TestStatus.PASSED)
        except SkipTest as e:
            tester.status(test_name, TestStatus.SKIPPED, str(e))
        except (AssertionError, BaseException) as e:
            (exc_type, _, tb) = sys.exc_info()
            tester.status(test_name, TestStatus.FAILED, str(e))
            import traceback
            stack = traceback.extract_stack()[:-1]
            for (i, entry) in enumerate(stack):
                (filename, lineno, function, text) = entry
                if spack.repo.is_package_file(filename):
                    with open(filename) as f:
                        lines = f.readlines()
                    new_lineno = lineno - 2
                    text = lines[new_lineno]
                    if isinstance(entry, tuple):
                        new_entry = (filename, new_lineno, function, text)
                        stack[i] = new_entry
                    elif isinstance(entry, list):
                        stack[i][1] = new_lineno
            out = traceback.format_list(stack)
            for line in out:
                print(line.rstrip('\n'))
            if exc_type is spack.util.executable.ProcessError or exc_type is TypeError:
                iostr = io.StringIO()
                spack.build_environment.write_log_summary(iostr, 'test', tester.test_log_file, last=1)
                m = iostr.getvalue()
            else:
                m = '\n'.join(spack.build_environment.get_package_context(tb))
            exc = e
            if spack.config.get('config:fail_fast', False):
                raise TestFailure([(exc, m)])
            else:
                tester.add_failure(exc, m)

def copy_test_files(pkg: Pb, test_spec: spack.spec.Spec):
    if False:
        while True:
            i = 10
    "Copy the spec's cached and custom test files to the test stage directory.\n\n    Args:\n        pkg: package being tested\n        test_spec: spec being tested, where the spec may be virtual\n\n    Raises:\n        TestSuiteError: package must be part of an active test suite\n    "
    if pkg is None or pkg.test_suite is None:
        base = 'Cannot copy test files'
        msg = f'{base} without a package' if pkg is None else f'{pkg.name}: {base}: test suite is missing'
        raise TestSuiteError(msg)
    if test_spec.concrete:
        cache_source = install_test_root(test_spec.package)
        cache_dir = pkg.test_suite.current_test_cache_dir
        if os.path.isdir(cache_source) and (not os.path.exists(cache_dir)):
            fs.install_tree(cache_source, cache_dir)
    try:
        pkg_cls = test_spec.package_class
    except spack.repo.UnknownPackageError:
        tty.debug(f'{test_spec.name}: skipping test data copy since no package class found')
        return
    data_source = Prefix(pkg_cls.package_dir).test
    data_dir = pkg.test_suite.current_test_data_dir
    if os.path.isdir(data_source) and (not os.path.exists(data_dir)):
        shutil.copytree(data_source, data_dir)

def test_function_names(pkg: PackageObjectOrClass, add_virtuals: bool=False) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Grab the names of all non-empty test functions.\n\n    Args:\n        pkg: package or package class of interest\n        add_virtuals: ``True`` adds test methods of provided package\n            virtual, ``False`` only returns test functions of the package\n\n    Returns:\n        names of non-empty test functions\n\n    Raises:\n        ValueError: occurs if pkg is not a package class\n    '
    fns = test_functions(pkg, add_virtuals)
    return [f'{cls_name}.{fn.__name__}' for (cls_name, fn) in fns]

def test_functions(pkg: PackageObjectOrClass, add_virtuals: bool=False) -> List[Tuple[str, Callable]]:
    if False:
        return 10
    "Grab all non-empty test functions.\n\n    Args:\n        pkg: package or package class of interest\n        add_virtuals: ``True`` adds test methods of provided package\n            virtual, ``False`` only returns test functions of the package\n\n    Returns:\n        list of non-empty test functions' (name, function)\n\n    Raises:\n        ValueError: occurs if pkg is not a package class\n    "
    instance = isinstance(pkg, spack.package_base.PackageBase)
    if not (instance or issubclass(pkg, spack.package_base.PackageBase)):
        raise ValueError(f'Expected a package (class), not {pkg} ({type(pkg)})')
    pkg_cls = pkg.__class__ if instance else pkg
    classes = [pkg_cls]
    if add_virtuals:
        vpkgs = virtuals(pkg)
        for vname in vpkgs:
            try:
                classes.append(Spec(vname).package_class)
            except spack.repo.UnknownPackageError:
                tty.debug(f'{vname}: virtual does not appear to have a package file')

    def skip(line):
        if False:
            i = 10
            return i + 15
        ln = line.strip()
        return ln.startswith('#') or ('warn' in ln and 'deprecated' in ln)
    doc_regex = '\\s+("""[\\w\\s\\(\\)\\-\\,\\;\\:]+""")'
    tests = []
    for clss in classes:
        methods = inspect.getmembers(clss, predicate=lambda x: inspect.isfunction(x))
        for (name, test_fn) in methods:
            if not name.startswith('test'):
                continue
            source = re.sub(doc_regex, '', inspect.getsource(test_fn)).splitlines()[1:]
            lines = [ln.strip() for ln in source if not skip(ln)]
            if not lines:
                continue
            tests.append((clss.__name__, test_fn))
    return tests

def process_test_parts(pkg: Pb, test_specs: List[spack.spec.Spec], verbose: bool=False):
    if False:
        print('Hello World!')
    'Process test parts associated with the package.\n\n    Args:\n        pkg: package being tested\n        test_specs: list of test specs\n        verbose: Display verbose output (suppress by default)\n\n    Raises:\n        TestSuiteError: package must be part of an active test suite\n    '
    if pkg is None or pkg.test_suite is None:
        base = 'Cannot process tests'
        msg = f'{base} without a package' if pkg is None else f'{pkg.name}: {base}: test suite is missing'
        raise TestSuiteError(msg)
    test_suite = pkg.test_suite
    tester = pkg.tester
    try:
        work_dir = test_suite.test_dir_for_spec(pkg.spec)
        for spec in test_specs:
            test_suite.current_test_spec = spec
            try:
                tests = test_functions(spec.package_class)
            except spack.repo.UnknownPackageError:
                continue
            if len(tests) == 0:
                tester.status(spec.name, TestStatus.NO_TESTS)
                continue
            copy_test_files(pkg, spec)
            for (_, test_fn) in tests:
                with test_part(pkg, test_fn.__name__, purpose=getattr(test_fn, '__doc__'), work_dir=work_dir, verbose=verbose):
                    test_fn(pkg)
        if tester.test_failures:
            raise TestFailure(tester.test_failures)
    finally:
        if tester.ran_tests():
            tester.write_tested_status()
            tty.msg('Completed testing')
            lines = tester.summarize()
            tty.msg('\n{}'.format('\n'.join(lines)))
            if tester.test_failures:
                tty.msg(f'\n\nSee test results at:\n  {tester.test_log_file}')
        else:
            tty.msg('No tests to run')

def test_process(pkg: Pb, kwargs):
    if False:
        i = 10
        return i + 15
    verbose = kwargs.get('verbose', True)
    externals = kwargs.get('externals', False)
    with pkg.tester.test_logger(verbose, externals) as logger:
        if pkg.spec.external and (not externals):
            print_message(logger, 'Skipped tests for external package', verbose)
            pkg.tester.status(pkg.spec.name, TestStatus.SKIPPED)
            return
        if not pkg.spec.installed:
            print_message(logger, 'Skipped not installed package', verbose)
            pkg.tester.status(pkg.spec.name, TestStatus.SKIPPED)
            return
        v_names = virtuals(pkg)
        test_specs = [pkg.spec] + [spack.spec.Spec(v_name) for v_name in sorted(v_names)]
        process_test_parts(pkg, test_specs, verbose)

def virtuals(pkg):
    if False:
        i = 10
        return i + 15
    'Return a list of unique virtuals for the package.\n\n    Args:\n        pkg: package of interest\n\n    Returns: names of unique virtual packages\n    '
    v_names = list({vspec.name for vspec in pkg.virtuals_provided})
    c_names = ('gcc', 'intel', 'intel-parallel-studio', 'pgi')
    if pkg.name in c_names:
        v_names.extend(['c', 'cxx', 'fortran'])
    if pkg.spec.satisfies('llvm+clang'):
        v_names.extend(['c', 'cxx'])
    return v_names

def get_all_test_suites():
    if False:
        i = 10
        return i + 15
    'Retrieves all validly staged TestSuites\n\n    Returns:\n        list: a list of TestSuite objects, which may be empty if there are none\n    '
    stage_root = get_test_stage_dir()
    if not os.path.isdir(stage_root):
        return []

    def valid_stage(d):
        if False:
            print('Hello World!')
        dirpath = os.path.join(stage_root, d)
        return os.path.isdir(dirpath) and test_suite_filename in os.listdir(dirpath)
    candidates = [os.path.join(stage_root, d, test_suite_filename) for d in os.listdir(stage_root) if valid_stage(d)]
    test_suites = [TestSuite.from_file(c) for c in candidates]
    return test_suites

def get_named_test_suites(name):
    if False:
        print('Hello World!')
    'Retrieves test suites with the provided name.\n\n    Returns:\n        list: a list of matching TestSuite instances, which may be empty if none\n\n    Raises:\n        Exception: TestSuiteNameError if no name is provided\n    '
    if not name:
        raise TestSuiteNameError('Test suite name is required.')
    test_suites = get_all_test_suites()
    return [ts for ts in test_suites if ts.name == name]

def get_test_suite(name: str) -> Optional['TestSuite']:
    if False:
        i = 10
        return i + 15
    'Ensure there is only one matching test suite with the provided name.\n\n    Returns:\n        the name if one matching test suite, else None\n\n    Raises:\n        TestSuiteNameError: If there are more than one matching TestSuites\n    '
    suites = get_named_test_suites(name)
    if len(suites) > 1:
        raise TestSuiteNameError(f"Too many suites named '{name}'. May shadow hash.")
    if not suites:
        return None
    return suites[0]

def write_test_suite_file(suite):
    if False:
        i = 10
        return i + 15
    'Write the test suite to its (JSON) lock file.'
    with open(suite.stage.join(test_suite_filename), 'w') as f:
        sjson.dump(suite.to_dict(), stream=f)

def write_test_summary(counts: 'Counter'):
    if False:
        print('Hello World!')
    'Write summary of the totals for each relevant status category.\n\n    Args:\n        counts: counts of the occurrences of relevant test status types\n    '
    summary = [f'{n} {s.lower()}' for (s, n) in counts.items() if n > 0]
    try:
        total = counts.total()
    except AttributeError:
        nums = [n for (_, n) in counts.items()]
        total = sum(nums)
    if total:
        print('{:=^80}'.format(' {} of {} '.format(', '.join(summary), plural(total, 'spec'))))

class TestSuite:
    """The class that manages specs for ``spack test run`` execution."""

    def __init__(self, specs, alias=None):
        if False:
            return 10
        self.specs = [spec.copy() for spec in specs]
        self.current_test_spec = None
        self.current_base_spec = None
        self.alias = alias
        self._hash = None
        self._stage = None
        self.counts: 'Counter' = Counter()

    @property
    def name(self):
        if False:
            print('Hello World!')
        'The name (alias or, if none, hash) of the test suite.'
        return self.alias if self.alias else self.content_hash

    @property
    def content_hash(self):
        if False:
            i = 10
            return i + 15
        'The hash used to uniquely identify the test suite.'
        if not self._hash:
            json_text = sjson.dump(self.to_dict())
            sha = hashlib.sha1(json_text.encode('utf-8'))
            b32_hash = base64.b32encode(sha.digest()).lower()
            b32_hash = b32_hash.decode('utf-8')
            self._hash = b32_hash
        return self._hash

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        self.write_reproducibility_data()
        remove_directory = kwargs.get('remove_directory', True)
        dirty = kwargs.get('dirty', False)
        fail_first = kwargs.get('fail_first', False)
        externals = kwargs.get('externals', False)
        for spec in self.specs:
            try:
                if spec.package.test_suite:
                    raise TestSuiteSpecError('Package {} cannot be run in two test suites at once'.format(spec.package.name))
                spec.package.test_suite = self
                self.current_base_spec = spec
                self.current_test_spec = spec
                test_dir = self.test_dir_for_spec(spec)
                if os.path.exists(test_dir):
                    shutil.rmtree(test_dir)
                fs.mkdirp(test_dir)
                spec.package.do_test(dirty=dirty, externals=externals)
                if remove_directory:
                    shutil.rmtree(test_dir)
                status = self.test_status(spec, externals)
                self.counts[status] += 1
                self.write_test_result(spec, status)
            except SkipTest:
                status = TestStatus.SKIPPED
                self.counts[status] += 1
                self.write_test_result(spec, TestStatus.SKIPPED)
            except BaseException as exc:
                status = TestStatus.FAILED
                self.counts[status] += 1
                tty.debug(f'Test failure: {str(exc)}')
                if isinstance(exc, (SyntaxError, TestSuiteSpecError)):
                    self.ensure_stage()
                    msg = f'Testing package {self.test_pkg_id(spec)}\n{str(exc)}'
                    _add_msg_to_file(self.log_file_for_spec(spec), msg)
                msg = f'Test failure: {str(exc)}'
                _add_msg_to_file(self.log_file_for_spec(spec), msg)
                self.write_test_result(spec, TestStatus.FAILED)
                if fail_first:
                    break
            finally:
                spec.package.test_suite = None
                self.current_test_spec = None
                self.current_base_spec = None
        write_test_summary(self.counts)
        if self.counts[TestStatus.FAILED]:
            for spec in self.specs:
                print('\nSee {} test results at:\n  {}'.format(spec.format('{name}-{version}-{hash:7}'), self.log_file_for_spec(spec)))
        failures = self.counts[TestStatus.FAILED]
        if failures:
            raise TestSuiteFailure(failures)

    def test_status(self, spec: spack.spec.Spec, externals: bool) -> Optional[TestStatus]:
        if False:
            while True:
                i = 10
        "Determine the overall test results status for the spec.\n\n        Args:\n            spec: instance of the spec under test\n            externals: ``True`` if externals are to be tested, else ``False``\n\n        Returns:\n            the spec's test status if available or ``None``\n        "
        tests_status_file = self.tested_file_for_spec(spec)
        if not os.path.exists(tests_status_file):
            self.ensure_stage()
            if spec.external and (not externals):
                status = TestStatus.SKIPPED
            elif not spec.installed:
                status = TestStatus.SKIPPED
            else:
                status = TestStatus.NO_TESTS
            return status
        with open(tests_status_file, 'r') as f:
            value = f.read().strip('\n')
            return TestStatus(int(value)) if value else TestStatus.NO_TESTS

    def ensure_stage(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure the test suite stage directory exists.'
        if not os.path.exists(self.stage):
            fs.mkdirp(self.stage)

    @property
    def stage(self):
        if False:
            while True:
                i = 10
        "The root test suite stage directory.\n\n        Returns:\n            str: the spec's test stage directory path\n        "
        if not self._stage:
            self._stage = Prefix(fs.join_path(get_test_stage_dir(), self.content_hash))
        return self._stage

    @stage.setter
    def stage(self, value):
        if False:
            while True:
                i = 10
        'Set the value of a non-default stage directory.'
        self._stage = value if isinstance(value, Prefix) else Prefix(value)

    @property
    def results_file(self):
        if False:
            while True:
                i = 10
        'The path to the results summary file.'
        return self.stage.join(results_filename)

    @classmethod
    def test_pkg_id(cls, spec):
        if False:
            print('Hello World!')
        'The standard install test package identifier.\n\n        Args:\n            spec: instance of the spec under test\n\n        Returns:\n            str: the install test package identifier\n        '
        return spec.format_path('{name}-{version}-{hash:7}')

    @classmethod
    def test_log_name(cls, spec):
        if False:
            i = 10
            return i + 15
        "The standard log filename for a spec.\n\n        Args:\n            spec (spack.spec.Spec): instance of the spec under test\n\n        Returns:\n            str: the spec's log filename\n        "
        return '%s-test-out.txt' % cls.test_pkg_id(spec)

    def log_file_for_spec(self, spec):
        if False:
            for i in range(10):
                print('nop')
        "The test log file path for the provided spec.\n\n        Args:\n            spec (spack.spec.Spec): instance of the spec under test\n\n        Returns:\n            str: the path to the spec's log file\n        "
        return self.stage.join(self.test_log_name(spec))

    def test_dir_for_spec(self, spec):
        if False:
            print('Hello World!')
        "The path to the test stage directory for the provided spec.\n\n        Args:\n            spec (spack.spec.Spec): instance of the spec under test\n\n        Returns:\n            str: the spec's test stage directory path\n        "
        return Prefix(self.stage.join(self.test_pkg_id(spec)))

    @classmethod
    def tested_file_name(cls, spec):
        if False:
            i = 10
            return i + 15
        "The standard test status filename for the spec.\n\n        Args:\n            spec (spack.spec.Spec): instance of the spec under test\n\n        Returns:\n            str: the spec's test status filename\n        "
        return '%s-tested.txt' % cls.test_pkg_id(spec)

    def tested_file_for_spec(self, spec):
        if False:
            while True:
                i = 10
        "The test status file path for the spec.\n\n        Args:\n            spec (spack.spec.Spec): instance of the spec under test\n\n        Returns:\n            str: the spec's test status file path\n        "
        return fs.join_path(self.stage, self.tested_file_name(spec))

    @property
    def current_test_cache_dir(self):
        if False:
            while True:
                i = 10
        "Path to the test stage directory where the current spec's cached\n        build-time files were automatically copied.\n\n        Returns:\n            str: path to the current spec's staged, cached build-time files.\n\n        Raises:\n            TestSuiteSpecError: If there is no spec being tested\n        "
        if not (self.current_test_spec and self.current_base_spec):
            raise TestSuiteSpecError('Unknown test cache directory: no specs being tested')
        test_spec = self.current_test_spec
        base_spec = self.current_base_spec
        return self.test_dir_for_spec(base_spec).cache.join(test_spec.name)

    @property
    def current_test_data_dir(self):
        if False:
            while True:
                i = 10
        "Path to the test stage directory where the current spec's custom\n        package (data) files were automatically copied.\n\n        Returns:\n            str: path to the current spec's staged, custom package (data) files\n\n        Raises:\n            TestSuiteSpecError: If there is no spec being tested\n        "
        if not (self.current_test_spec and self.current_base_spec):
            raise TestSuiteSpecError('Unknown test data directory: no specs being tested')
        test_spec = self.current_test_spec
        base_spec = self.current_base_spec
        return self.test_dir_for_spec(base_spec).data.join(test_spec.name)

    def write_test_result(self, spec, result):
        if False:
            return 10
        "Write the spec's test result to the test suite results file.\n\n        Args:\n            spec (spack.spec.Spec): instance of the spec under test\n            result (str): result from the spec's test execution (e.g, PASSED)\n        "
        msg = f'{self.test_pkg_id(spec)} {result}'
        _add_msg_to_file(self.results_file, msg)

    def write_reproducibility_data(self):
        if False:
            while True:
                i = 10
        for spec in self.specs:
            repo_cache_path = self.stage.repo.join(spec.name)
            spack.repo.PATH.dump_provenance(spec, repo_cache_path)
            for vspec in spec.package.virtuals_provided:
                repo_cache_path = self.stage.repo.join(vspec.name)
                if not os.path.exists(repo_cache_path):
                    try:
                        spack.repo.PATH.dump_provenance(vspec, repo_cache_path)
                    except spack.repo.UnknownPackageError:
                        pass
        write_test_suite_file(self)

    def to_dict(self):
        if False:
            i = 10
            return i + 15
        "Build a dictionary for the test suite.\n\n        Returns:\n            dict: The dictionary contains entries for up to two keys:\n\n                specs: list of the test suite's specs in dictionary form\n                alias: the alias, or name, given to the test suite if provided\n        "
        specs = [s.to_dict() for s in self.specs]
        d = {'specs': specs}
        if self.alias:
            d['alias'] = self.alias
        return d

    @staticmethod
    def from_dict(d):
        if False:
            while True:
                i = 10
        "Instantiates a TestSuite based on a dictionary specs and an\n        optional alias:\n\n            specs: list of the test suite's specs in dictionary form\n            alias: the test suite alias\n\n        Returns:\n            TestSuite: Instance created from the specs\n        "
        specs = [Spec.from_dict(spec_dict) for spec_dict in d['specs']]
        alias = d.get('alias', None)
        return TestSuite(specs, alias)

    @staticmethod
    def from_file(filename):
        if False:
            while True:
                i = 10
        'Instantiate a TestSuite using the specs and optional alias\n        provided in the given file.\n\n        Args:\n            filename (str): The path to the JSON file containing the test\n                suite specs and optional alias.\n\n        Raises:\n            BaseException: sjson.SpackJSONError if problem parsing the file\n        '
        try:
            with open(filename) as f:
                data = sjson.load(f)
                test_suite = TestSuite.from_dict(data)
                content_hash = os.path.basename(os.path.dirname(filename))
                test_suite._hash = content_hash
                return test_suite
        except Exception as e:
            raise sjson.SpackJSONError('error parsing JSON TestSuite:', e)

def _add_msg_to_file(filename, msg):
    if False:
        while True:
            i = 10
    'Append the message to the specified file.\n\n    Args:\n        filename (str): path to the file\n        msg (str): message to be appended to the file\n    '
    with open(filename, 'a+') as f:
        f.write(f'{msg}\n')

class SkipTest(Exception):
    """Raised when a test (part) is being skipped."""

class TestFailure(spack.error.SpackError):
    """Raised when package tests have failed for an installation."""

    def __init__(self, failures: List[TestFailureType]):
        if False:
            while True:
                i = 10
        num = len(failures)
        msg = '{} failed.\n'.format(plural(num, 'test'))
        for (failure, message) in failures:
            msg += '\n\n%s\n' % str(failure)
            msg += '\n%s\n' % message
        super().__init__(msg)

class TestSuiteError(spack.error.SpackError):
    """Raised when there is an error with the test suite."""

class TestSuiteFailure(spack.error.SpackError):
    """Raised when one or more tests in a suite have failed."""

    def __init__(self, num_failures):
        if False:
            i = 10
            return i + 15
        msg = '%d test(s) in the suite failed.\n' % num_failures
        super().__init__(msg)

class TestSuiteSpecError(spack.error.SpackError):
    """Raised when there is an issue associated with the spec being tested."""

class TestSuiteNameError(spack.error.SpackError):
    """Raised when there is an issue with the naming of the test suite."""