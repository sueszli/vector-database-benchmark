"""Unit tests for scripts/check_overall_backend_test_coverage.py."""
from __future__ import annotations
import builtins
import os
import subprocess
import sys
from core.tests import test_utils
from scripts import check_overall_backend_test_coverage
from scripts import common

class CheckOverallBackendTestCoverageTests(test_utils.GenericTestBase):
    """Unit tests for scripts/check_overall_backend_test_coverage.py."""

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.print_arr: list[str] = []

        def mock_print(msg: str, end: str='\n') -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.print_arr.append(msg)
        self.print_swap = self.swap(builtins, 'print', mock_print)
        self.env = os.environ.copy()
        self.cmd = [sys.executable, '-m', 'coverage', 'report', '--omit="%s*","third_party/*","/usr/share/*"' % common.OPPIA_TOOLS_DIR, '--show-missing']

    def test_no_data_in_coverage_report_throws_error(self) -> None:
        if False:
            return 10

        class MockProcess:
            returncode = 0
            stdout = 'No data to report.'
            stderr = 'None'

        def mock_subprocess_run(*args: str, **kwargs: str) -> MockProcess:
            if False:
                i = 10
                return i + 15
            return MockProcess()
        swap_subprocess_run = self.swap_with_checks(subprocess, 'run', mock_subprocess_run, expected_args=((self.cmd,),), expected_kwargs=[{'capture_output': True, 'encoding': 'utf-8', 'env': self.env, 'check': False}])
        with swap_subprocess_run, self.assertRaisesRegex(RuntimeError, 'Run backend tests before running this script. ' + '\nOUTPUT: No data to report.\nERROR: None'):
            check_overall_backend_test_coverage.main()

    def test_failure_to_execute_coverage_command_throws_error(self) -> None:
        if False:
            i = 10
            return i + 15

        class MockProcess:
            returncode = 1
            stdout = 'Some error occured.'
            stderr = 'Some error.'

        def mock_subprocess_run(*args: str, **kwargs: str) -> MockProcess:
            if False:
                while True:
                    i = 10
            return MockProcess()
        swap_subprocess_run = self.swap_with_checks(subprocess, 'run', mock_subprocess_run, expected_args=((self.cmd,),), expected_kwargs=[{'capture_output': True, 'encoding': 'utf-8', 'env': self.env, 'check': False}])
        with swap_subprocess_run, self.assertRaisesRegex(RuntimeError, 'Failed to calculate coverage because subprocess failed. ' + '\nOUTPUT: Some error occured.\nERROR: Some error.'):
            check_overall_backend_test_coverage.main()

    def test_error_in_parsing_coverage_report_throws_error(self) -> None:
        if False:
            i = 10
            return i + 15

        class MockProcess:
            returncode = 0
            stdout = 'TOTALL     40571  10682  13759   1161   70% '

        def mock_subprocess_run(*args: str, **kwargs: str) -> MockProcess:
            if False:
                return 10
            return MockProcess()
        swap_subprocess_run = self.swap_with_checks(subprocess, 'run', mock_subprocess_run, expected_args=((self.cmd,),), expected_kwargs=[{'capture_output': True, 'encoding': 'utf-8', 'env': self.env, 'check': False}])
        with swap_subprocess_run, self.assertRaisesRegex(RuntimeError, 'Error in parsing coverage report.'):
            check_overall_backend_test_coverage.main()

    def test_overall_backend_coverage_checks_failed(self) -> None:
        if False:
            return 10

        class MockProcess:
            returncode = 0
            stdout = 'TOTAL     40571  10682  13759   1161   70% '

        def mock_subprocess_run(*args: str, **kwargs: str) -> MockProcess:
            if False:
                print('Hello World!')
            return MockProcess()
        swap_subprocess_run = self.swap_with_checks(subprocess, 'run', mock_subprocess_run, expected_args=((self.cmd,),), expected_kwargs=[{'capture_output': True, 'encoding': 'utf-8', 'env': self.env, 'check': False}])
        swap_sys_exit = self.swap_with_checks(sys, 'exit', lambda _: None, expected_args=((1,),))
        with self.print_swap, swap_sys_exit, swap_subprocess_run:
            check_overall_backend_test_coverage.main()
        self.assertIn('Backend overall line coverage checks failed.', self.print_arr)

    def test_overall_backend_coverage_checks_passed(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        class MockProcess:
            returncode = 0
            stdout = 'TOTAL     40571  0  13759   0   100% '

        def mock_subprocess_run(*args: str, **kwargs: str) -> MockProcess:
            if False:
                i = 10
                return i + 15
            return MockProcess()
        swap_subprocess_run = self.swap_with_checks(subprocess, 'run', mock_subprocess_run, expected_args=((self.cmd,),), expected_kwargs=[{'capture_output': True, 'encoding': 'utf-8', 'env': self.env, 'check': False}])
        with self.print_swap, swap_subprocess_run:
            check_overall_backend_test_coverage.main()
        self.assertIn('Backend overall line coverage checks passed.', self.print_arr)