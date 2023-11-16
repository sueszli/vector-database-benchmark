"""Unit tests for scripts/run_presubmit_checks.py."""
from __future__ import annotations
import builtins
import os
import subprocess
from core.tests import test_utils
from scripts import run_backend_tests
from scripts import run_frontend_tests
from scripts import run_presubmit_checks
from scripts.linters import pre_commit_linter

class RunPresubmitChecksTests(test_utils.GenericTestBase):
    """Unit tests for scripts/run_presubmit_checks.py."""

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.print_arr: list[str] = []

        def mock_print(msg: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.print_arr.append(msg)
        self.print_swap = self.swap(builtins, 'print', mock_print)
        current_dir = os.path.abspath(os.getcwd())
        self.changed_frontend_file = os.path.join(current_dir, 'core', 'templates', 'testFile.ts')
        self.current_branch = 'test-branch'
        self.cmd_to_check_current_branch = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
        self.cmd_to_match_current_branch_with_remote = ['git', 'ls-remote', '--heads', 'origin', self.current_branch, '|', 'wc', '-l']
        self.scripts_called = {'run_frontend_tests': False, 'run_backend_tests': False, 'pre_commit_linter': False}

        def mock_frontend_tests(args: list[str]) -> None:
            if False:
                print('Hello World!')
            self.scripts_called['run_frontend_tests'] = True

        def mock_backend_tests(args: list[str]) -> None:
            if False:
                i = 10
                return i + 15
            self.scripts_called['run_backend_tests'] = True

        def mock_pre_commit_linter(args: list[str]) -> None:
            if False:
                print('Hello World!')
            self.scripts_called['pre_commit_linter'] = True
        self.swap_frontend_tests = self.swap_with_checks(run_frontend_tests, 'main', mock_frontend_tests, expected_kwargs=[{'args': ['--run_minified_tests']}])
        self.swap_backend_tests = self.swap_with_checks(run_backend_tests, 'main', mock_backend_tests, expected_kwargs=[{'args': []}])
        self.swap_pre_commit_linter = self.swap_with_checks(pre_commit_linter, 'main', mock_pre_commit_linter, expected_kwargs=[{'args': []}])

    def test_run_presubmit_checks_when_branch_is_specified(self) -> None:
        if False:
            i = 10
            return i + 15
        specified_branch = 'develop'
        cmd_to_get_all_changed_file = ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM', specified_branch]

        def mock_check_output(cmd: list[str], encoding: str='utf-8') -> str:
            if False:
                return 10
            if cmd == self.cmd_to_check_current_branch:
                return self.current_branch
            elif cmd == self.cmd_to_match_current_branch_with_remote:
                return '1'
            elif cmd == cmd_to_get_all_changed_file:
                return self.changed_frontend_file
            else:
                raise Exception('Invalid cmd passed: %s' % cmd)
        swap_check_output = self.swap(subprocess, 'check_output', mock_check_output)
        with self.print_swap, swap_check_output, self.swap_pre_commit_linter:
            with self.swap_backend_tests, self.swap_frontend_tests:
                run_presubmit_checks.main(args=['-b', specified_branch])
        for script in self.scripts_called:
            self.assertTrue(script)
        self.assertIn('Linting passed.', self.print_arr)
        self.assertIn('Comparing the current branch with %s' % specified_branch, self.print_arr)
        self.assertIn('Frontend tests passed.', self.print_arr)
        self.assertIn('Backend tests passed.', self.print_arr)

    def test_run_presubmit_checks_when_current_branch_exists_on_remote_origin(self) -> None:
        if False:
            i = 10
            return i + 15
        cmd_to_get_all_changed_file = ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM', 'origin/%s' % self.current_branch]

        def mock_check_output(cmd: list[str], encoding: str='utf-8') -> str:
            if False:
                for i in range(10):
                    print('nop')
            if cmd == self.cmd_to_check_current_branch:
                return self.current_branch
            elif cmd == self.cmd_to_match_current_branch_with_remote:
                return '1'
            elif cmd == cmd_to_get_all_changed_file:
                return self.changed_frontend_file
            else:
                raise Exception('Invalid cmd passed: %s' % cmd)
        swap_check_output = self.swap(subprocess, 'check_output', mock_check_output)
        with self.print_swap, swap_check_output, self.swap_pre_commit_linter:
            with self.swap_backend_tests, self.swap_frontend_tests:
                run_presubmit_checks.main(args=[])
        for script in self.scripts_called:
            self.assertTrue(script)
        self.assertIn('Linting passed.', self.print_arr)
        self.assertIn('Comparing the current branch with origin/%s' % self.current_branch, self.print_arr)
        self.assertIn('Frontend tests passed.', self.print_arr)
        self.assertIn('Backend tests passed.', self.print_arr)

    def test_frontend_tests_are_not_run_when_no_frontend_files_are_changed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        cmd_to_get_all_changed_file = ['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM', 'develop']

        def mock_check_output(cmd: list[str], encoding: str='utf-8') -> str:
            if False:
                print('Hello World!')
            if cmd == self.cmd_to_check_current_branch:
                return self.current_branch
            elif cmd == self.cmd_to_match_current_branch_with_remote:
                return '0'
            elif cmd == cmd_to_get_all_changed_file:
                return ''
            else:
                raise Exception('Invalid cmd passed: %s' % cmd)
        swap_check_output = self.swap(subprocess, 'check_output', mock_check_output)
        with self.print_swap, swap_check_output, self.swap_pre_commit_linter:
            with self.swap_backend_tests:
                run_presubmit_checks.main(args=[])
        self.assertFalse(self.scripts_called['run_frontend_tests'])
        self.assertTrue(self.scripts_called['pre_commit_linter'])
        self.assertTrue(self.scripts_called['run_backend_tests'])
        self.assertIn('Linting passed.', self.print_arr)
        self.assertIn('Comparing the current branch with develop', self.print_arr)
        self.assertIn('Backend tests passed.', self.print_arr)
        self.assertNotIn('Frontend tests passed.', self.print_arr)