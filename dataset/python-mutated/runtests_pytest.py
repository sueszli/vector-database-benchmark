"""Backwards compatible functions for running tests from SymPy using pytest.

SymPy historically had its own testing framework that aimed to:
- be compatible with pytest;
- operate similarly (or identically) to pytest;
- not require any external dependencies;
- have all the functionality in one file only;
- have no magic, just import the test file and execute the test functions; and
- be portable.

To reduce the maintence burden of developing an independent testing framework
and to leverage the benefits of existing Python testing infrastructure, SymPy
now uses pytest (and various of its plugins) to run the test suite.

To maintain backwards compatibility with the legacy testing interface of SymPy,
which implemented functions that allowed users to run the tests on their
installed version of SymPy, the functions in this module are implemented to
match the existing API while thinly wrapping pytest.

These two key functions are `test` and `doctest`.

"""
import functools
import importlib.util
import os
import pathlib
import re
from fnmatch import fnmatch
from typing import List, Optional, Tuple
try:
    import pytest
except ImportError:

    class NoPytestError(Exception):
        """Raise when an internal test helper function is called with pytest."""

    class pytest:
        """Shadow to support pytest features when pytest can't be imported."""

        @staticmethod
        def main(*args, **kwargs):
            if False:
                return 10
            msg = 'pytest must be installed to run tests via this function'
            raise NoPytestError(msg)
from sympy.testing.runtests import test as test_sympy
TESTPATHS_DEFAULT = (pathlib.Path('sympy'), pathlib.Path('doc', 'src'))
BLACKLIST_DEFAULT = ('sympy/integrals/rubi/rubi_tests/tests',)

class PytestPluginManager:
    """Module names for pytest plugins used by SymPy."""
    PYTEST: str = 'pytest'
    RANDOMLY: str = 'pytest_randomly'
    SPLIT: str = 'pytest_split'
    TIMEOUT: str = 'pytest_timeout'
    XDIST: str = 'xdist'

    @functools.cached_property
    def has_pytest(self) -> bool:
        if False:
            while True:
                i = 10
        return bool(importlib.util.find_spec(self.PYTEST))

    @functools.cached_property
    def has_randomly(self) -> bool:
        if False:
            while True:
                i = 10
        return bool(importlib.util.find_spec(self.RANDOMLY))

    @functools.cached_property
    def has_split(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return bool(importlib.util.find_spec(self.SPLIT))

    @functools.cached_property
    def has_timeout(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return bool(importlib.util.find_spec(self.TIMEOUT))

    @functools.cached_property
    def has_xdist(self) -> bool:
        if False:
            print('Hello World!')
        return bool(importlib.util.find_spec(self.XDIST))
split_pattern = re.compile('([1-9][0-9]*)/([1-9][0-9]*)')

@functools.lru_cache
def sympy_dir() -> pathlib.Path:
    if False:
        print('Hello World!')
    'Returns the root SymPy directory.'
    return pathlib.Path(__file__).parents[2]

def update_args_with_rootdir(args: List[str]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Adds `--rootdir` and path to the args `list` passed to `pytest.main`.\n\n    This is required to ensure that pytest is able to find the SymPy tests in\n    instances where it gets confused determining the root directory, e.g. when\n    running with Pyodide (e.g. `bin/test_pyodide.mjs`).\n\n    '
    args.extend(['--rootdir', str(sympy_dir())])
    return args

def update_args_with_paths(paths: List[str], keywords: Optional[Tuple[str]], args: List[str]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Appends valid paths and flags to the args `list` passed to `pytest.main`.\n\n    The are three different types of "path" that a user may pass to the `paths`\n    positional arguments, all of which need to be handled slightly differently:\n\n    1. Nothing is passed\n        The paths to the `testpaths` defined in `pytest.ini` need to be appended\n        to the arguments list.\n    2. Full, valid paths are passed\n        These paths need to be validated but can then be directly appended to\n        the arguments list.\n    3. Partial paths are passed.\n        The `testpaths` defined in `pytest.ini` need to be recursed and any\n        matches be appended to the arguments list.\n\n    '

    def find_paths_matching_partial(partial_paths):
        if False:
            return 10
        partial_path_file_patterns = []
        for partial_path in partial_paths:
            if len(partial_path) >= 4:
                has_test_prefix = partial_path[:4] == 'test'
                has_py_suffix = partial_path[-3:] == '.py'
            elif len(partial_path) >= 3:
                has_test_prefix = False
                has_py_suffix = partial_path[-3:] == '.py'
            else:
                has_test_prefix = False
                has_py_suffix = False
            if has_test_prefix and has_py_suffix:
                partial_path_file_patterns.append(partial_path)
            elif has_test_prefix:
                partial_path_file_patterns.append(f'{partial_path}*.py')
            elif has_py_suffix:
                partial_path_file_patterns.append(f'test*{partial_path}')
            else:
                partial_path_file_patterns.append(f'test*{partial_path}*.py')
        matches = []
        for testpath in valid_testpaths_default:
            for (path, dirs, files) in os.walk(testpath, topdown=True):
                zipped = zip(partial_paths, partial_path_file_patterns)
                for (partial_path, partial_path_file) in zipped:
                    if fnmatch(path, f'*{partial_path}*'):
                        matches.append(str(pathlib.Path(path)))
                        dirs[:] = []
                    else:
                        for file in files:
                            if fnmatch(file, partial_path_file):
                                matches.append(str(pathlib.Path(path, file)))
        return matches

    def is_tests_file(filepath: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        path = pathlib.Path(filepath)
        if not path.is_file():
            return False
        if not path.parts[-1].startswith('test_'):
            return False
        if not path.suffix == '.py':
            return False
        return True

    def find_tests_matching_keywords(keywords, filepath):
        if False:
            while True:
                i = 10
        matches = []
        with open(filepath, encoding='utf-8') as tests_file:
            source = tests_file.read()
            for line in source.splitlines():
                if line.lstrip().startswith('def '):
                    for kw in keywords:
                        if line.lower().find(kw.lower()) != -1:
                            test_name = line.split(' ')[1].split('(')[0]
                            full_test_path = filepath + '::' + test_name
                            matches.append(full_test_path)
        return matches
    valid_testpaths_default = []
    for testpath in TESTPATHS_DEFAULT:
        absolute_testpath = pathlib.Path(sympy_dir(), testpath)
        if absolute_testpath.exists():
            valid_testpaths_default.append(str(absolute_testpath))
    candidate_paths = []
    if paths:
        full_paths = []
        partial_paths = []
        for path in paths:
            if pathlib.Path(path).exists():
                full_paths.append(str(pathlib.Path(sympy_dir(), path)))
            else:
                partial_paths.append(path)
        matched_paths = find_paths_matching_partial(partial_paths)
        candidate_paths.extend(full_paths)
        candidate_paths.extend(matched_paths)
    else:
        candidate_paths.extend(valid_testpaths_default)
    if keywords is not None and keywords != ():
        matches = []
        for path in candidate_paths:
            if is_tests_file(path):
                test_matches = find_tests_matching_keywords(keywords, path)
                matches.extend(test_matches)
            else:
                for (root, dirnames, filenames) in os.walk(path):
                    for filename in filenames:
                        absolute_filepath = str(pathlib.Path(root, filename))
                        if is_tests_file(absolute_filepath):
                            test_matches = find_tests_matching_keywords(keywords, absolute_filepath)
                            matches.extend(test_matches)
        args.extend(matches)
    else:
        args.extend(candidate_paths)
    return args

def make_absolute_path(partial_path: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Convert a partial path to an absolute path.\n\n    A path such a `sympy/core` might be needed. However, absolute paths should\n    be used in the arguments to pytest in all cases as it avoids errors that\n    arise from nonexistent paths.\n\n    This function assumes that partial_paths will be passed in such that they\n    begin with the explicit `sympy` directory, i.e. `sympy/...`.\n\n    '

    def is_valid_partial_path(partial_path: str) -> bool:
        if False:
            print('Hello World!')
        'Assumption that partial paths are defined from the `sympy` root.'
        return pathlib.Path(partial_path).parts[0] == 'sympy'
    if not is_valid_partial_path(partial_path):
        msg = f'Partial path {dir(partial_path)} is invalid, partial paths are expected to be defined with the `sympy` directory as the root.'
        raise ValueError(msg)
    absolute_path = str(pathlib.Path(sympy_dir(), partial_path))
    return absolute_path

def test(*paths, subprocess=True, rerun=0, **kwargs):
    if False:
        print('Hello World!')
    'Interface to run tests via pytest compatible with SymPy\'s test runner.\n\n    Explanation\n    ===========\n\n    Note that a `pytest.ExitCode`, which is an `enum`, is returned. This is\n    different to the legacy SymPy test runner which would return a `bool`. If\n    all tests sucessfully pass the `pytest.ExitCode.OK` with value `0` is\n    returned, whereas the legacy SymPy test runner would return `True`. In any\n    other scenario, a non-zero `enum` value is returned, whereas the legacy\n    SymPy test runner would return `False`. Users need to, therefore, be careful\n    if treating the pytest exit codes as booleans because\n    `bool(pytest.ExitCode.OK)` evaluates to `False`, the opposite of legacy\n    behaviour.\n\n    Examples\n    ========\n\n    >>> import sympy  # doctest: +SKIP\n\n    Run one file:\n\n    >>> sympy.test(\'sympy/core/tests/test_basic.py\')  # doctest: +SKIP\n    >>> sympy.test(\'_basic\')  # doctest: +SKIP\n\n    Run all tests in sympy/functions/ and some particular file:\n\n    >>> sympy.test("sympy/core/tests/test_basic.py",\n    ...            "sympy/functions")  # doctest: +SKIP\n\n    Run all tests in sympy/core and sympy/utilities:\n\n    >>> sympy.test("/core", "/util")  # doctest: +SKIP\n\n    Run specific test from a file:\n\n    >>> sympy.test("sympy/core/tests/test_basic.py",\n    ...            kw="test_equality")  # doctest: +SKIP\n\n    Run specific test from any file:\n\n    >>> sympy.test(kw="subs")  # doctest: +SKIP\n\n    Run the tests using the legacy SymPy runner:\n\n    >>> sympy.test(use_sympy_runner=True)  # doctest: +SKIP\n\n    Note that this option is slated for deprecation in the near future and is\n    only currently provided to ensure users have an alternative option while the\n    pytest-based runner receives real-world testing.\n\n    Parameters\n    ==========\n    paths : first n positional arguments of strings\n        Paths, both partial and absolute, describing which subset(s) of the test\n        suite are to be run.\n    subprocess : bool, default is True\n        Legacy option, is currently ignored.\n    rerun : int, default is 0\n        Legacy option, is ignored.\n    use_sympy_runner : bool or None, default is None\n        Temporary option to invoke the legacy SymPy test runner instead of\n        `pytest.main`. Will be removed in the near future.\n    verbose : bool, default is False\n        Sets the verbosity of the pytest output. Using `True` will add the\n        `--verbose` option to the pytest call.\n    tb : str, \'auto\', \'long\', \'short\', \'line\', \'native\', or \'no\'\n        Sets the traceback print mode of pytest using the `--tb` option.\n    kw : str\n        Only run tests which match the given substring expression. An expression\n        is a Python evaluatable expression where all names are substring-matched\n        against test names and their parent classes. Example: -k \'test_method or\n        test_other\' matches all test functions and classes whose name contains\n        \'test_method\' or \'test_other\', while -k \'not test_method\' matches those\n        that don\'t contain \'test_method\' in their names. -k \'not test_method and\n        not test_other\' will eliminate the matches. Additionally keywords are\n        matched to classes and functions containing extra names in their\n        \'extra_keyword_matches\' set, as well as functions which have names\n        assigned directly to them. The matching is case-insensitive.\n    pdb : bool, default is False\n        Start the interactive Python debugger on errors or `KeyboardInterrupt`.\n    colors : bool, default is True\n        Color terminal output.\n    force_colors : bool, default is False\n        Legacy option, is ignored.\n    sort : bool, default is True\n        Run the tests in sorted order. pytest uses a sorted test order by\n        default. Requires pytest-randomly.\n    seed : int\n        Seed to use for random number generation. Requires pytest-randomly.\n    timeout : int, default is 0\n        Timeout in seconds before dumping the stacks. 0 means no timeout.\n        Requires pytest-timeout.\n    fail_on_timeout : bool, default is False\n        Legacy option, is currently ignored.\n    slow : bool, default is False\n        Run the subset of tests marked as `slow`.\n    enhance_asserts : bool, default is False\n        Legacy option, is currently ignored.\n    split : string in form `<SPLIT>/<GROUPS>` or None, default is None\n        Used to split the tests up. As an example, if `split=\'2/3\' is used then\n        only the middle third of tests are run. Requires pytest-split.\n    time_balance : bool, default is True\n        Legacy option, is currently ignored.\n    blacklist : iterable of test paths as strings, default is BLACKLIST_DEFAULT\n        Blacklisted test paths are ignored using the `--ignore` option. Paths\n        may be partial or absolute. If partial then they are matched against\n        all paths in the pytest tests path.\n    parallel : bool, default is False\n        Parallelize the test running using pytest-xdist. If `True` then pytest\n        will automatically detect the number of CPU cores available and use them\n        all. Requires pytest-xdist.\n    store_durations : bool, False\n        Store test durations into the file `.test_durations`. The is used by\n        `pytest-split` to help determine more even splits when more than one\n        test group is being used. Requires pytest-split.\n\n    '
    if kwargs.get('use_sympy_runner', False):
        kwargs.pop('parallel', False)
        kwargs.pop('store_durations', False)
        kwargs.pop('use_sympy_runner', True)
        if kwargs.get('slow') is None:
            kwargs['slow'] = False
        return test_sympy(*paths, subprocess=True, rerun=0, **kwargs)
    pytest_plugin_manager = PytestPluginManager()
    if not pytest_plugin_manager.has_pytest:
        pytest.main()
    args = []
    args = update_args_with_rootdir(args)
    if kwargs.get('verbose', False):
        args.append('--verbose')
    if (tb := kwargs.get('tb')):
        args.extend(['--tb', tb])
    if kwargs.get('pdb'):
        args.append('--pdb')
    if not kwargs.get('colors', True):
        args.extend(['--color', 'no'])
    if (seed := kwargs.get('seed')):
        if not pytest_plugin_manager.has_randomly:
            msg = '`pytest-randomly` plugin required to control random seed.'
            raise ModuleNotFoundError(msg)
        args.extend(['--randomly-seed', str(seed)])
    if kwargs.get('sort', True) and pytest_plugin_manager.has_randomly:
        args.append('--randomly-dont-reorganize')
    elif not kwargs.get('sort', True) and (not pytest_plugin_manager.has_randomly):
        msg = '`pytest-randomly` plugin required to randomize test order.'
        raise ModuleNotFoundError(msg)
    if (timeout := kwargs.get('timeout', None)):
        if not pytest_plugin_manager.has_timeout:
            msg = '`pytest-timeout` plugin required to apply timeout to tests.'
            raise ModuleNotFoundError(msg)
        args.extend(['--timeout', str(int(timeout))])
    if (slow := (kwargs.get('slow', None) is not None)):
        if slow:
            args.extend(['-m', 'slow'])
        else:
            args.extend(['-m', 'not slow'])
    if (split := kwargs.get('split')) is not None:
        if not pytest_plugin_manager.has_split:
            msg = '`pytest-split` plugin required to run tests as groups.'
            raise ModuleNotFoundError(msg)
        match = split_pattern.match(split)
        if not match:
            msg = 'split must be a string of the form a/b where a and b are positive nonzero ints'
            raise ValueError(msg)
        (group, splits) = map(str, match.groups())
        args.extend(['--group', group, '--splits', splits])
        if group > splits:
            msg = f'cannot have a group number {group} with only {splits} splits'
            raise ValueError(msg)
    if (blacklist := kwargs.get('blacklist', BLACKLIST_DEFAULT)):
        for path in blacklist:
            args.extend(['--ignore', make_absolute_path(path)])
    if kwargs.get('parallel', False):
        if not pytest_plugin_manager.has_xdist:
            msg = '`pytest-xdist` plugin required to run tests in parallel.'
            raise ModuleNotFoundError(msg)
        args.extend(['-n', 'auto'])
    if kwargs.get('store_durations', False):
        if not pytest_plugin_manager.has_split:
            msg = '`pytest-split` plugin required to store test durations.'
            raise ModuleNotFoundError(msg)
        args.append('--store-durations')
    if (keywords := kwargs.get('kw')) is not None:
        keywords = tuple((str(kw) for kw in keywords))
    else:
        keywords = ()
    args = update_args_with_paths(paths, keywords, args)
    exit_code = pytest.main(args)
    return exit_code

def doctest():
    if False:
        while True:
            i = 10
    "Interface to run doctests via pytest compatible with SymPy's test runner.\n    "
    raise NotImplementedError