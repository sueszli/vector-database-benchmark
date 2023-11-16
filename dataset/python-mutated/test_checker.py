"""Integration tests for the checker submodule."""
from __future__ import annotations
import importlib.metadata
import sys
from unittest import mock
import pytest
from flake8 import checker
from flake8.plugins import finder
from flake8.processor import FileProcessor
PHYSICAL_LINE = '# Physical line content'
EXPECTED_REPORT = (1, 1, 'T000 Expected Message')
EXPECTED_REPORT_PHYSICAL_LINE = (1, 'T000 Expected Message')
EXPECTED_RESULT_PHYSICAL_LINE = ('T000', 0, 1, 'Expected Message', None)

class PluginClass:
    """Simple file plugin class yielding the expected report."""

    def __init__(self, tree):
        if False:
            print('Hello World!')
        'Construct a dummy object to provide mandatory parameter.'
        pass

    def run(self):
        if False:
            while True:
                i = 10
        'Run class yielding one element containing the expected report.'
        yield (EXPECTED_REPORT + (type(self),))

def plugin_func_gen(tree):
    if False:
        return 10
    'Yield the expected report.'
    yield (EXPECTED_REPORT + (type(plugin_func_gen),))

def plugin_func_list(tree):
    if False:
        for i in range(10):
            print('nop')
    'Return a list of expected reports.'
    return [EXPECTED_REPORT + (type(plugin_func_list),)]

def plugin_func_physical_ret(physical_line):
    if False:
        i = 10
        return i + 15
    'Expect report from a physical_line. Single return.'
    return EXPECTED_REPORT_PHYSICAL_LINE

def plugin_func_physical_none(physical_line):
    if False:
        print('Hello World!')
    'Expect report from a physical_line. No results.'
    return None

def plugin_func_physical_list_single(physical_line):
    if False:
        i = 10
        return i + 15
    'Expect report from a physical_line. List of single result.'
    return [EXPECTED_REPORT_PHYSICAL_LINE]

def plugin_func_physical_list_multiple(physical_line):
    if False:
        i = 10
        return i + 15
    'Expect report from a physical_line. List of multiple results.'
    return [EXPECTED_REPORT_PHYSICAL_LINE] * 2

def plugin_func_physical_gen_single(physical_line):
    if False:
        while True:
            i = 10
    'Expect report from a physical_line. Generator of single result.'
    yield EXPECTED_REPORT_PHYSICAL_LINE

def plugin_func_physical_gen_multiple(physical_line):
    if False:
        return 10
    'Expect report from a physical_line. Generator of multiple results.'
    for _ in range(3):
        yield EXPECTED_REPORT_PHYSICAL_LINE

def plugin_func_out_of_bounds(logical_line):
    if False:
        return 10
    'This produces an error out of bounds.'
    yield (10000, 'L100 test')

def mock_file_checker_with_plugin(plugin_target):
    if False:
        while True:
            i = 10
    'Get a mock FileChecker class with plugin_target registered.\n\n    Useful as a starting point for mocking reports/results.\n    '
    to_load = [finder.Plugin('flake-package', '9001', importlib.metadata.EntryPoint('Q', f'{plugin_target.__module__}:{plugin_target.__name__}', 'flake8.extension'))]
    opts = finder.PluginOptions.blank()
    plugins = finder.load_plugins(to_load, opts)
    with mock.patch('flake8.processor.FileProcessor.read_lines', return_value=['Line 1']):
        file_checker = checker.FileChecker(filename='-', plugins=plugins.checkers, options=mock.MagicMock())
    return file_checker

@pytest.mark.parametrize('plugin_target', [PluginClass, plugin_func_gen, plugin_func_list])
def test_handle_file_plugins(plugin_target):
    if False:
        i = 10
        return i + 15
    'Test the FileChecker class handling different file plugin types.'
    file_checker = mock_file_checker_with_plugin(plugin_target)
    file_checker.processor.build_ast = lambda : True
    report = mock.Mock()
    file_checker.report = report
    file_checker.run_ast_checks()
    report.assert_called_once_with(error_code=None, line_number=EXPECTED_REPORT[0], column=EXPECTED_REPORT[1], text=EXPECTED_REPORT[2])

@pytest.mark.parametrize('plugin_target,len_results', [(plugin_func_physical_ret, 1), (plugin_func_physical_none, 0), (plugin_func_physical_list_single, 1), (plugin_func_physical_list_multiple, 2), (plugin_func_physical_gen_single, 1), (plugin_func_physical_gen_multiple, 3)])
def test_line_check_results(plugin_target, len_results):
    if False:
        for i in range(10):
            print('nop')
    'Test the FileChecker class handling results from line checks.'
    file_checker = mock_file_checker_with_plugin(plugin_target)
    file_checker.run_physical_checks(PHYSICAL_LINE)
    expected = [EXPECTED_RESULT_PHYSICAL_LINE] * len_results
    assert file_checker.results == expected

def test_logical_line_offset_out_of_bounds():
    if False:
        print('Hello World!')
    'Ensure that logical line offsets that are out of bounds do not crash.'
    file_checker = mock_file_checker_with_plugin(plugin_func_out_of_bounds)
    logical_ret = ('', 'print("xxxxxxxxxxx")', [(0, (1, 0)), (5, (1, 5)), (6, (1, 6)), (19, (1, 19)), (20, (1, 20))])
    with mock.patch.object(FileProcessor, 'build_logical_line', return_value=logical_ret):
        file_checker.run_logical_checks()
        assert file_checker.results == [('L100', 0, 0, 'test', None)]
PLACEHOLDER_CODE = 'some_line = "of" * code'

@pytest.mark.parametrize('results, expected_order', [([], []), ([('A101', 1, 1, 'placeholder error', PLACEHOLDER_CODE), ('A101', 2, 1, 'placeholder error', PLACEHOLDER_CODE)], [0, 1]), ([('A101', 2, 1, 'placeholder error', PLACEHOLDER_CODE), ('A101', 1, 1, 'placeholder error', PLACEHOLDER_CODE)], [1, 0]), ([('A101', 1, 2, 'placeholder error', PLACEHOLDER_CODE), ('A101', 1, 1, 'placeholder error', PLACEHOLDER_CODE), ('A101', 2, 1, 'placeholder error', PLACEHOLDER_CODE)], [1, 0, 2]), ([('A101', 2, 1, 'placeholder error', PLACEHOLDER_CODE), ('A101', 1, 1, 'placeholder error', PLACEHOLDER_CODE), ('A101', 1, 2, 'placeholder error', PLACEHOLDER_CODE)], [1, 2, 0]), ([('A101', 1, 2, 'placeholder error', PLACEHOLDER_CODE), ('A101', 2, 2, 'placeholder error', PLACEHOLDER_CODE), ('A101', 2, 1, 'placeholder error', PLACEHOLDER_CODE)], [0, 2, 1]), ([('A101', 1, 3, 'placeholder error', PLACEHOLDER_CODE), ('A101', 2, 2, 'placeholder error', PLACEHOLDER_CODE), ('A101', 3, 1, 'placeholder error', PLACEHOLDER_CODE)], [0, 1, 2]), ([('A101', 1, 1, 'placeholder error', PLACEHOLDER_CODE), ('A101', 1, 3, 'placeholder error', PLACEHOLDER_CODE), ('A101', 2, 2, 'placeholder error', PLACEHOLDER_CODE)], [0, 1, 2]), ([('A101', 1, 1, 'placeholder error', PLACEHOLDER_CODE), ('A101', 2, 1, 'charlie error', PLACEHOLDER_CODE)], [0, 1])])
def test_report_order(results, expected_order):
    if False:
        i = 10
        return i + 15
    '\n    Test in which order the results will be reported.\n\n    It gets a list of reports from the file checkers and verifies that the\n    result will be ordered independent from the original report.\n    '

    def count_side_effect(name, sorted_results):
        if False:
            return 10
        'Side effect for the result handler to tell all are reported.'
        return len(sorted_results)
    expected_results = [results[index] for index in expected_order]
    style_guide = mock.MagicMock(spec=['options', 'processing_file'])
    manager = checker.Manager(style_guide, finder.Checkers([], [], []), [])
    manager.results = [('placeholder', results, {})]
    handler = mock.Mock(side_effect=count_side_effect)
    with mock.patch.object(manager, '_handle_results', handler):
        assert manager.report() == (len(results), len(results))
        handler.assert_called_once_with('placeholder', expected_results)

def test_acquire_when_multiprocessing_pool_can_initialize():
    if False:
        while True:
            i = 10
    'Verify successful importing of hardware semaphore support.\n\n    Mock the behaviour of a platform that has a hardware sem_open\n    implementation, and then attempt to initialize a multiprocessing\n    Pool object.\n\n    This simulates the behaviour on most common platforms.\n    '
    with mock.patch('multiprocessing.Pool') as pool:
        result = checker._try_initialize_processpool(2, [])
    pool.assert_called_once_with(2, checker._mp_init, initargs=([],))
    assert result is pool.return_value

def test_acquire_when_multiprocessing_pool_can_not_initialize():
    if False:
        print('Hello World!')
    'Verify unsuccessful importing of hardware semaphore support.\n\n    Mock the behaviour of a platform that has not got a hardware sem_open\n    implementation, and then attempt to initialize a multiprocessing\n    Pool object.\n\n    This scenario will occur on platforms such as Termux and on some\n    more exotic devices.\n\n    https://github.com/python/cpython/blob/4e02981de0952f54bf87967f8e10d169d6946b40/Lib/multiprocessing/synchronize.py#L30-L33\n    '
    with mock.patch('multiprocessing.Pool', side_effect=ImportError) as pool:
        result = checker._try_initialize_processpool(2, [])
    pool.assert_called_once_with(2, checker._mp_init, initargs=([],))
    assert result is None

def test_handling_syntaxerrors_across_pythons():
    if False:
        return 10
    'Verify we properly handle exception argument tuples.\n\n    Python 3.10 added more information to the SyntaxError parse token tuple.\n    We need to handle that correctly to avoid crashing.\n    https://github.com/PyCQA/flake8/issues/1372\n    '
    if sys.version_info < (3, 10):
        err = SyntaxError('invalid syntax', ('<unknown>', 2, 5, 'bad python:\n'))
        expected = (2, 4)
    else:
        err = SyntaxError('invalid syntax', ('<unknown>', 2, 1, 'bad python:\n', 2, 11))
        expected = (2, 1)
    file_checker = checker.FileChecker(filename='-', plugins=finder.Checkers([], [], []), options=mock.MagicMock())
    actual = file_checker._extract_syntax_information(err)
    assert actual == expected