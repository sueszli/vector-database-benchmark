import os
import sys
import unittest
from coalib import coala, coala_ci
from coala_utils.ContextManagers import prepare_file
from tests.TestUtilities import bear_test_module, execute_coala

class coalaCITest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.old_argv = sys.argv

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        sys.argv = self.old_argv

    def test_log(self, debug=False):
        if False:
            i = 10
            return i + 15
        (retval, stdout, stderr) = execute_coala(coala_ci.main, 'coala-ci', '--help', debug=debug)
        self.assertIn('usage: coala', stdout)
        self.assertIn('Use of `coala-ci` executable is deprecated', stderr)
        self.assertEqual(retval, 0, 'coala must return zero when successful')

    def test_log_debug(self):
        if False:
            while True:
                i = 10
        self.test_log(debug=True)

    def test_nonexistent(self, debug=False):
        if False:
            return 10
        (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '--non-interactive', '-c', 'nonex', 'test')
        self.assertFalse(stdout)
        self.assertRegex(stderr, ".*\\[ERROR\\].*Requested coafile '.*' does not exist")
        self.assertNotEqual(retval, 0, 'coala must return nonzero when errors occured')
        (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '-c', 'nonex', '--show-bears', '--filter-by-language', 'Python')
        self.assertNotIn(stderr, "Requested coafile '.coafile' does not exist")
        (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '-c', 'nonex', '--show-bears')
        self.assertIn(stderr, "Requested coafile '.coafile' does not exist")

    def test_nonexistent_debug(self):
        if False:
            print('Hello World!')
        self.test_nonexistent(debug=True)

    def test_find_no_issues(self, debug=False):
        if False:
            print('Hello World!')
        with bear_test_module():
            with prepare_file(['#include <a>'], None) as (lines, filename):
                (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '--non-interactive', '-c', os.devnull, '-f', filename, '-b', 'SpaceConsistencyTestBear', '--settings', 'use_spaces=True', debug=debug)
                self.assertEqual('Executing section cli...\n', stdout)
                if not debug:
                    self.assertFalse(stderr)
                else:
                    self.assertTrue(stderr)
                self.assertEqual(retval, 0, 'coala must return zero when successful')

    def test_section_ordering(self, debug=False):
        if False:
            for i in range(10):
                print('nop')
        with bear_test_module():
            with prepare_file(['#include <a>'], None) as (lines, filename):
                (retval, stdout, stderr) = execute_coala(coala.main, 'coala', 'b', 'a', '--non-interactive', '-S', 'a.bears=SpaceConsistencyTestBear', f'a.files={filename}', 'a.use_spaces=True', 'b.bears=SpaceConsistencyTestBear', f'b.files={filename}', 'b.use_spaces=True', '-c', os.devnull, debug=debug)
                stdout_list = stdout.splitlines(True)
                self.assertEqual('Executing section b...\n', stdout_list[0])
                self.assertEqual('Executing section a...\n', stdout_list[1])
                (retval, stdout, stderr) = execute_coala(coala.main, 'coala', 'a', 'b', '--non-interactive', '-S', 'a.bears=SpaceConsistencyTestBear', f'a.files={filename}', 'a.use_spaces=True', 'b.bears=SpaceConsistencyTestBear', f'b.files={filename}', 'b.use_spaces=True', '-c', os.devnull, debug=debug)
                stdout_list = stdout.splitlines(True)
                self.assertEqual('Executing section a...\n', stdout_list[0])
                self.assertEqual('Executing section b...\n', stdout_list[1])

    def test_find_no_issues_debug(self):
        if False:
            while True:
                i = 10
        self.test_find_no_issues(debug=True)

    def test_find_issues(self, debug=False):
        if False:
            for i in range(10):
                print('nop')
        with bear_test_module():
            with prepare_file(['#fixme'], None) as (lines, filename):
                (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '--non-interactive', '-c', os.devnull, '-b', 'LineCountTestBear', '-f', filename, debug=debug)
                self.assertIn('This file has 1 lines.', stdout, 'The output should report count as 1 lines')
                self.assertIn('This result has no patch attached.', stderr)
                self.assertNotEqual(retval, 0, 'coala must return nonzero when errors occured')

    def test_find_issues_debug(self):
        if False:
            while True:
                i = 10
        self.test_find_issues(debug=True)

    def test_show_patch(self, debug=False):
        if False:
            return 10
        with bear_test_module():
            with prepare_file(['\t#include <a>'], None) as (lines, filename):
                (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '--non-interactive', '-c', os.devnull, '-f', filename, '-b', 'SpaceConsistencyTestBear', '--settings', 'use_spaces=True', debug=debug)
                self.assertIn('Line contains ', stdout)
                self.assertIn("Applied 'ShowPatchAction'", stderr)
                self.assertEqual(retval, 5, 'coala must return exitcode 5 when it autofixes the code.')

    def test_show_patch_debug(self):
        if False:
            print('Hello World!')
        self.test_show_patch(debug=True)

    def test_fail_acquire_settings(self, debug=False):
        if False:
            i = 10
            return i + 15
        with bear_test_module():
            (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '--non-interactive', '-b', 'SpaceConsistencyTestBear', '-c', os.devnull, debug=debug)
            self.assertFalse(stdout)
            self.assertIn('During execution, we found that some', stderr)
            self.assertNotEqual(retval, 0, 'coala was expected to return non-zero')

    def test_additional_parameters_settings(self, debug=False):
        if False:
            for i in range(10):
                print('nop')
        with bear_test_module():
            with prepare_file(['\t#include <a>'], None) as (lines, filename):
                (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '--non-interactive', '-S', 'name.bears=SpaceConsistencyTestBear', f'name.files={filename}', 'name.enabled=False', '-c', os.devnull, debug=debug)
                self.assertEqual('Executing section cli...\n', stdout)
                self.assertNotIn('During execution, we found that some required settings were not provided.', stderr)
                self.assertEqual(retval, 0, 'coala was expected to return zero')

    def test_fail_acquire_settings_debug(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(AssertionError, 'During execution, we found that some required settings were not provided.'):
            self.test_fail_acquire_settings(debug=True)

    def test_limit_files_affirmative(self):
        if False:
            i = 10
            return i + 15
        sample_text = '\t#include <a>'
        with open('match.cpp', 'w') as match:
            with open('noMatch.cpp', 'w') as no_match:
                match.write(sample_text)
                no_match.write(sample_text)
        with bear_test_module():
            (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '--non-interactive', '-c', os.devnull, '--limit-files', 'match*', '-f', 'match.cpp', 'noMatch.cpp', '-b', 'SpaceConsistencyTestBear', '--settings', 'use_spaces=True')
            os.remove('match.cpp')
            os.remove('noMatch.cpp')
            self.assertIn('match.cpp', stdout)
            self.assertNotIn('noMatch.cpp', stdout)
            self.assertIn("Applied 'ShowPatchAction'", stderr)
            self.assertEqual(retval, 5, 'coala must return exitcode 5 when it autofixes the code.')

    def test_limit_files_negative(self):
        if False:
            while True:
                i = 10
        with bear_test_module():
            with prepare_file(['\t#include <a>'], None) as (lines, filename):
                (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '--non-interactive', '-c', os.devnull, '--limit-files', 'some_pattern', '-f', filename, '-b', 'SpaceConsistencyTestBear', '--settings', 'use_spaces=True')
                self.assertEqual('Executing section cli...\n', stdout)
                self.assertFalse(stderr)
                self.assertEqual(retval, 0, 'coala must return zero when successful')

    def test_bear_dirs(self):
        if False:
            for i in range(10):
                print('nop')
        with prepare_file(['random_text'], None) as (lines, filename):
            (retval, stdout, stderr) = execute_coala(coala.main, 'coala', '--non-interactive', '-c', os.devnull, '--bear-dirs', 'tests/test_bears', '-b', 'LineCountTestBear', '-f', filename)
            self.assertIn('This file has 1 lines.', stdout)
            self.assertIn('This result has no patch attached.', stderr)
            self.assertNotEqual(retval, 0, 'coala must return nonzero when errors occured')