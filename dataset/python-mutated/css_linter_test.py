"""Unit tests for scripts/linters/css_linter.py."""
from __future__ import annotations
import os
import subprocess
from core.tests import test_utils
from scripts import scripts_test_utils
from typing import Final, List
from . import css_linter
LINTER_TESTS_DIR: Final = os.path.join(os.getcwd(), 'scripts', 'linters', 'test_files')
VALID_CSS_FILEPATH: Final = os.path.join(LINTER_TESTS_DIR, 'valid.css')
INVALID_CSS_FILEPATH: Final = os.path.join(LINTER_TESTS_DIR, 'invalid.css')

class ThirdPartyCSSLintChecksManagerTests(test_utils.LinterTestBase):
    """Tests for ThirdPartyCSSLintChecksManager class."""

    def test_all_filepaths_with_success(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        filepaths = [VALID_CSS_FILEPATH, INVALID_CSS_FILEPATH]
        third_party_linter = css_linter.ThirdPartyCSSLintChecksManager(filepaths)
        returned_filepaths = third_party_linter.all_filepaths
        self.assertEqual(returned_filepaths, filepaths)

    def test_perform_all_lint_checks_with_invalid_file(self) -> None:
        if False:
            print('Hello World!')
        third_party_linter = css_linter.ThirdPartyCSSLintChecksManager([INVALID_CSS_FILEPATH])
        lint_task_report = third_party_linter.lint_css_files()
        self.assert_same_list_elements(['19:16', 'Unexpected whitespace before ":"'], lint_task_report.get_report())
        self.assertEqual('Stylelint', lint_task_report.name)
        self.assertTrue(lint_task_report.failed)

    def test_perform_all_lint_checks_with_invalid_stylelint_path(self) -> None:
        if False:
            print('Hello World!')

        def mock_join(*unused_args: str) -> str:
            if False:
                for i in range(10):
                    print('nop')
            return 'node_modules/stylelint/bin/stylelinter.js'
        join_swap = self.swap(os.path, 'join', mock_join)
        third_party_linter = css_linter.ThirdPartyCSSLintChecksManager([INVALID_CSS_FILEPATH])
        with self.print_swap, join_swap, self.assertRaisesRegex(Exception, 'ERROR    Please run start.py first to install node-eslint or node-stylelint and its dependencies.'):
            third_party_linter.perform_all_lint_checks()

    def test_perform_all_lint_checks_with_stderr(self) -> None:
        if False:
            print('Hello World!')

        def mock_popen(unused_commands: List[str], stdout: int, stderr: int) -> scripts_test_utils.PopenStub:
            if False:
                while True:
                    i = 10
            return scripts_test_utils.PopenStub(stdout=b'True', stderr=b'True')
        popen_swap = self.swap_with_checks(subprocess, 'Popen', mock_popen)
        third_party_linter = css_linter.ThirdPartyCSSLintChecksManager([VALID_CSS_FILEPATH])
        with self.print_swap, popen_swap, self.assertRaisesRegex(Exception, 'True'):
            third_party_linter.perform_all_lint_checks()

    def test_perform_all_lint_checks_with_no_files(self) -> None:
        if False:
            i = 10
            return i + 15
        third_party_linter = css_linter.ThirdPartyCSSLintChecksManager([])
        lint_task_report = third_party_linter.perform_all_lint_checks()
        self.assertEqual('There are no HTML or CSS files to lint.', lint_task_report[0].get_report()[0])
        self.assertEqual('CSS lint', lint_task_report[0].name)
        self.assertFalse(lint_task_report[0].failed)

    def test_perform_all_lint_checks_with_valid_file(self) -> None:
        if False:
            i = 10
            return i + 15
        third_party_linter = css_linter.ThirdPartyCSSLintChecksManager([VALID_CSS_FILEPATH])
        lint_task_report = third_party_linter.perform_all_lint_checks()
        self.assertTrue(isinstance(lint_task_report, list))

    def test_get_linters(self) -> None:
        if False:
            while True:
                i = 10
        (custom_linter, third_party_linter) = css_linter.get_linters([VALID_CSS_FILEPATH, INVALID_CSS_FILEPATH])
        self.assertEqual(custom_linter, None)
        self.assertTrue(isinstance(third_party_linter, css_linter.ThirdPartyCSSLintChecksManager))