"""Lint checks for python files."""
from __future__ import annotations
import os
import subprocess
from typing import Final, List, Tuple
from . import linter_utils
from .. import common
from .. import concurrent_task_utils
STYLELINT_CONFIG: Final = os.path.join('.stylelintrc')

class ThirdPartyCSSLintChecksManager(linter_utils.BaseLinter):
    """Manages all the third party Python linting functions."""

    def __init__(self, files_to_lint: List[str]) -> None:
        if False:
            while True:
                i = 10
        'Constructs a ThirdPartyCSSLintChecksManager object.\n\n        Args:\n            files_to_lint: list(str). A list of filepaths to lint.\n        '
        super().__init__()
        self.files_to_lint = files_to_lint

    @property
    def all_filepaths(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Return all filepaths.'
        return self.files_to_lint

    @staticmethod
    def _get_trimmed_error_output(css_lint_output: str) -> str:
        if False:
            return 10
        'Remove extra bits from stylelint error messages.\n\n        Args:\n            css_lint_output: str. Output returned by the css linter.\n\n        Returns:\n            str. A string with the trimmed error messages.\n        '
        return '%s\n' % css_lint_output

    def lint_css_files(self) -> concurrent_task_utils.TaskResult:
        if False:
            print('Hello World!')
        'Prints a list of lint errors in the given list of CSS files.\n\n        Returns:\n            TaskResult. A TaskResult object representing the result of the lint\n            check.\n\n        Raises:\n            Exception. The start.py file not executed.\n        '
        node_path = os.path.join(common.NODE_PATH, 'bin', 'node')
        stylelint_path = os.path.join('node_modules', 'stylelint', 'bin', 'stylelint.js')
        if not os.path.exists(stylelint_path):
            raise Exception('ERROR    Please run start.py first to install node-eslint or node-stylelint and its dependencies.')
        failed = False
        stripped_error_messages = []
        full_error_messages = []
        name = 'Stylelint'
        stylelint_cmd_args = [node_path, stylelint_path, '--config=' + STYLELINT_CONFIG]
        proc_args = stylelint_cmd_args + self.all_filepaths
        proc = subprocess.Popen(proc_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (encoded_linter_stdout, encoded_linter_stderr) = proc.communicate()
        linter_stdout = encoded_linter_stdout.decode('utf-8')
        linter_stderr = encoded_linter_stderr.decode('utf-8')
        if linter_stderr:
            raise Exception(linter_stderr)
        if linter_stdout:
            full_error_messages.append(linter_stdout)
            stripped_error_messages.append(self._get_trimmed_error_output(linter_stdout))
            failed = True
        return concurrent_task_utils.TaskResult(name, failed, stripped_error_messages, full_error_messages)

    def perform_all_lint_checks(self) -> List[concurrent_task_utils.TaskResult]:
        if False:
            i = 10
            return i + 15
        'Perform all the lint checks and returns the messages returned by all\n        the checks.\n\n        Returns:\n            list(TaskResult). A list of TaskResult objects representing the\n            results of the lint checks.\n        '
        if not self.all_filepaths:
            return [concurrent_task_utils.TaskResult('CSS lint', False, [], ['There are no HTML or CSS files to lint.'])]
        return [self.lint_css_files()]

def get_linters(files_to_lint: List[str]) -> Tuple[None, ThirdPartyCSSLintChecksManager]:
    if False:
        print('Hello World!')
    'Creates ThirdPartyCSSLintChecksManager and returns it.\n\n    Args:\n        files_to_lint: list(str). A list of filepaths to lint.\n\n    Returns:\n        tuple(None, ThirdPartyCSSLintChecksManager). A 2-tuple of custom and\n        third_party linter objects.\n    '
    return (None, ThirdPartyCSSLintChecksManager(files_to_lint))