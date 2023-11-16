"""Unit tests for scripts/pre_commit_linter.py."""
from __future__ import annotations
import multiprocessing
import os
import subprocess
import sys
from core.tests import test_utils
from typing import List, Optional
from . import pre_commit_linter
from .. import concurrent_task_utils
from .. import install_third_party_libs
LINTER_TESTS_DIR = os.path.join(os.getcwd(), 'scripts', 'linters', 'test_files')
PYLINTRC_FILEPATH = os.path.join(os.getcwd(), '.pylintrc')
VALID_HTML_FILEPATH = os.path.join(LINTER_TESTS_DIR, 'valid.html')
VALID_CSS_FILEPATH = os.path.join(LINTER_TESTS_DIR, 'valid.css')
INVALID_CSS_FILEPATH = os.path.join(LINTER_TESTS_DIR, 'invalid.css')
VALID_JS_FILEPATH = os.path.join(LINTER_TESTS_DIR, 'valid.js')
VALID_TS_FILEPATH = os.path.join(LINTER_TESTS_DIR, 'valid.ts')
VALID_PY_FILEPATH = os.path.join(LINTER_TESTS_DIR, 'valid.py')

def mock_exit(unused_status: int) -> None:
    if False:
        while True:
            i = 10
    'Mock for sys.exit.'
    pass

def mock_install_third_party_libs_main() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Mock for install_third_party_libs.'
    return

def all_checks_passed(linter_stdout: List[str]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Helper function to check if all checks have passed.\n\n    Args:\n        linter_stdout: list(str). List of output messages from\n            pre_commit_linter.\n\n    Returns:\n        bool. Whether all checks have passed or not.\n    '
    return 'All Linter Checks Passed.' in linter_stdout[-1]

class PreCommitLinterTests(test_utils.LinterTestBase):
    """Tests for methods in pre_commit_linter module."""

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.sys_swap = self.swap(sys, 'exit', mock_exit)
        self.install_swap = self.swap_with_checks(install_third_party_libs, 'main', mock_install_third_party_libs_main)

    def test_main_with_no_files(self) -> None:
        if False:
            print('Hello World!')

        def mock_get_all_filepaths(unused_path: str, unused_files: List[str], unused_shard: str, namespace: multiprocessing.managers.Namespace) -> List[str]:
            if False:
                for i in range(10):
                    print('nop')
            return []
        all_filepath_swap = self.swap(pre_commit_linter, '_get_all_filepaths', mock_get_all_filepaths)
        with self.print_swap, self.sys_swap:
            with self.install_swap:
                with all_filepath_swap:
                    pre_commit_linter.main()
        self.assert_same_list_elements(['No files to check'], self.linter_stdout)

    def test_main_with_no_args(self) -> None:
        if False:
            return 10

        def mock_get_changed_filepaths() -> List[str]:
            if False:
                return 10
            return []
        get_changed_filepaths_swap = self.swap(pre_commit_linter, '_get_changed_filepaths', mock_get_changed_filepaths)
        with self.print_swap, self.sys_swap:
            with self.install_swap:
                with get_changed_filepaths_swap:
                    pre_commit_linter.main()
        self.assert_same_list_elements(['No files to check'], self.linter_stdout)

    def test_main_with_non_other_shard(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        mock_shards = {'1': ['a/', 'b/']}

        def mock_get_filepaths_from_path(path: str, namespace: multiprocessing.managers.Namespace) -> List[str]:
            if False:
                return 10
            if path == mock_shards['1'][0]:
                return [VALID_PY_FILEPATH]
            return []
        shards_swap = self.swap(pre_commit_linter, 'SHARDS', mock_shards)
        get_filenames_from_path_swap = self.swap_with_checks(pre_commit_linter, '_get_filepaths_from_path', mock_get_filepaths_from_path, expected_args=[(prefix,) for prefix in mock_shards['1']])
        with self.print_swap, self.sys_swap, shards_swap:
            with self.install_swap:
                with get_filenames_from_path_swap:
                    pre_commit_linter.main(args=['--shard', '1'])
        self.assertFalse(all_checks_passed(self.linter_stdout))

    def test_main_with_invalid_shards(self) -> None:
        if False:
            print('Hello World!')

        def mock_get_filepaths_from_path(unused_path: str, namespace: multiprocessing.managers.Namespace) -> List[str]:
            if False:
                while True:
                    i = 10
            return ['mock_file', 'mock_file']

        def mock_install_third_party_main() -> None:
            if False:
                print('Hello World!')
            raise AssertionError('Third party libs should not be installed.')
        mock_shards = {'1': ['a/']}
        shards_swap = self.swap(pre_commit_linter, 'SHARDS', mock_shards)
        get_filenames_from_path_swap = self.swap_with_checks(pre_commit_linter, '_get_filepaths_from_path', mock_get_filepaths_from_path, expected_args=[(prefix,) for prefix in mock_shards['1']])
        install_swap = self.swap(install_third_party_libs, 'main', mock_install_third_party_main)
        with self.print_swap, self.sys_swap, install_swap, shards_swap:
            with get_filenames_from_path_swap:
                with self.assertRaisesRegex(RuntimeError, 'mock_file in multiple shards'):
                    pre_commit_linter.main(args=['--shard', '1'])

    def test_main_with_other_shard(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def mock_get_filepaths_from_path(path: str, namespace: multiprocessing.managers.Namespace) -> List[str]:
            if False:
                return 10
            if os.path.abspath(path) == os.getcwd():
                return [VALID_PY_FILEPATH, 'nonexistent_file']
            elif path == 'core/templates/':
                return ['nonexistent_file']
            else:
                return []
        mock_shards = {'1': ['a/'], 'other': ['b/']}
        shards_swap = self.swap(pre_commit_linter, 'SHARDS', mock_shards)
        filenames_from_path_expected_args = [(os.getcwd(),)] + [(prefix,) for prefix in mock_shards['1']]
        get_filenames_from_path_swap = self.swap_with_checks(pre_commit_linter, '_get_filepaths_from_path', mock_get_filepaths_from_path, expected_args=filenames_from_path_expected_args)
        with self.print_swap, self.sys_swap, shards_swap:
            with self.install_swap:
                with get_filenames_from_path_swap:
                    pre_commit_linter.main(args=['--shard', pre_commit_linter.OTHER_SHARD_NAME])
        self.assertFalse(all_checks_passed(self.linter_stdout))

    def test_main_with_files_arg(self) -> None:
        if False:
            while True:
                i = 10
        with self.print_swap, self.sys_swap:
            with self.install_swap:
                pre_commit_linter.main(args=['--files=%s' % PYLINTRC_FILEPATH])
        self.assertTrue(all_checks_passed(self.linter_stdout))

    def test_main_with_error_message(self) -> None:
        if False:
            while True:
                i = 10
        all_errors_swap = self.swap(concurrent_task_utils, 'ALL_ERRORS', ['This is an error.'])
        with self.print_swap, self.sys_swap:
            with self.install_swap, all_errors_swap:
                pre_commit_linter.main(args=['--path=%s' % VALID_PY_FILEPATH])
        self.assert_same_list_elements(['This is an error.'], self.linter_stdout)

    def test_main_with_path_arg(self) -> None:
        if False:
            print('Hello World!')
        with self.print_swap, self.sys_swap:
            with self.install_swap:
                pre_commit_linter.main(args=['--path=%s' % INVALID_CSS_FILEPATH])
        self.assertFalse(all_checks_passed(self.linter_stdout))
        self.assert_same_list_elements(['19:16', 'Unexpected whitespace before ":"'], self.linter_stdout)

    def test_main_with_invalid_filepath_with_path_arg(self) -> None:
        if False:
            return 10
        with self.print_swap, self.assertRaisesRegex(SystemExit, '1'):
            pre_commit_linter.main(args=['--path=invalid_file.py'])
        self.assert_same_list_elements(['Could not locate file or directory'], self.linter_stdout)

    def test_main_with_invalid_filepath_with_file_arg(self) -> None:
        if False:
            print('Hello World!')
        with self.print_swap, self.assertRaisesRegex(SystemExit, '1'):
            pre_commit_linter.main(args=['--files=invalid_file.py'])
        self.assert_same_list_elements(['The following file(s) do not exist'], self.linter_stdout)

    def test_path_arg_with_directory_name(self) -> None:
        if False:
            print('Hello World!')

        def mock_get_all_files_in_directory(unused_input_path: str, unused_excluded_glob_patterns: List[str]) -> List[str]:
            if False:
                while True:
                    i = 10
            return [VALID_PY_FILEPATH]
        get_all_files_swap = self.swap(pre_commit_linter, '_get_all_files_in_directory', mock_get_all_files_in_directory)
        with self.print_swap, self.sys_swap:
            with self.install_swap:
                with get_all_files_swap:
                    pre_commit_linter.main(args=['--path=scripts/linters/'])
        self.assertFalse(all_checks_passed(self.linter_stdout))

    def test_main_with_only_check_file_extensions_arg(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.print_swap, self.sys_swap:
            with self.install_swap:
                pre_commit_linter.main(args=['--path=%s' % VALID_TS_FILEPATH, '--only-check-file-extensions=ts'])
        self.assertFalse(all_checks_passed(self.linter_stdout))

    def test_main_with_only_check_file_extensions_arg_with_js_ts_options(self) -> None:
        if False:
            i = 10
            return i + 15
        with self.print_swap, self.assertRaisesRegex(SystemExit, '1'):
            pre_commit_linter.main(args=['--path=%s' % VALID_TS_FILEPATH, '--only-check-file-extensions', 'ts', 'js'])
        self.assert_same_list_elements(['Please use only one of "js" or "ts", as we do not have separate linters for JS and TS files. If both these options are used together, then the JS/TS linter will be run twice.'], self.linter_stdout)

    def test_get_all_files_in_directory(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.print_swap, self.sys_swap:
            with self.install_swap:
                pre_commit_linter.main(args=['--path=scripts/linters/', '--only-check-file-extensions=ts'])

    def test_html_file(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with self.print_swap, self.sys_swap, self.install_swap:
            pre_commit_linter.main(args=['--path=%s' % VALID_HTML_FILEPATH])
        self.assert_same_list_elements(['All Linter Checks Passed.'], self.linter_stdout)

    def test_get_changed_filepaths(self) -> None:
        if False:
            i = 10
            return i + 15

        def mock_check_output(unused_list: List[str]) -> Optional[str]:
            if False:
                while True:
                    i = 10
            return ''
        subprocess_swap = self.swap(subprocess, 'check_output', mock_check_output)
        with self.print_swap, self.sys_swap:
            with self.install_swap, subprocess_swap:
                pre_commit_linter.main()