"""Unit tests for scripts/third_party_size_check.py"""
from __future__ import annotations
import builtins
import os
import shutil
import sys
from core import utils
from core.tests import test_utils
from scripts import third_party_size_check

class ThirdPartySizeCheckTests(test_utils.GenericTestBase):
    """Unit tests for scripts/third_party_size_check.py"""

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.print_arr: list[str] = []

        def mock_print(msg: str) -> None:
            if False:
                return 10
            self.print_arr.append(msg)
        self.print_swap = self.swap(builtins, 'print', mock_print)
        if os.path.isdir(os.path.join(os.getcwd(), 'dummy_dir')):
            shutil.rmtree('dummy_dir')
        skip_files_list = 'random_file.py\n# This is a comment\nnew_file.py'
        os.mkdir('dummy_dir', mode=511)
        os.mkdir('dummy_dir/dummy_dir2', mode=511)
        with open('dummy_dir/file1.py', 'w', encoding='utf-8') as f:
            f.write(skip_files_list)
        with open('dummy_dir/file2.py', 'w', encoding='utf-8') as f:
            f.write('Text message')
        with open('dummy_dir/random_file3.py', 'w', encoding='utf-8') as f:
            f.write('Text message')
        with open('dummy_dir/dummy_dir2/file4.py', 'w', encoding='utf-8') as f:
            f.write('Text message')
        os.symlink('/dummy_dir/file1.py', 'dummy_dir/symlink_file.py')
        self.dummy_file_object = open('dummy_dir/file1.py', 'r', encoding='utf-8')

    def tearDown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        self.dummy_file_object.close()
        shutil.rmtree('dummy_dir')

    def test_get_skip_files_list(self) -> None:
        if False:
            while True:
                i = 10
        swap_open = self.swap_with_checks(utils, 'open_file', lambda *unused_args, **unused_kwargs: self.dummy_file_object, expected_args=(('.gcloudignore', 'r'),))
        with swap_open:
            returned_list = third_party_size_check.get_skip_files_list()
        skipped_files = ['random_file.py', 'new_file.py']
        expected_skipped_files_list = [os.path.join(os.getcwd(), file) for file in skipped_files]
        self.assertEqual(returned_list, expected_skipped_files_list)

    def test_get_skip_files_list_throws_error(self) -> None:
        if False:
            while True:
                i = 10
        err = IOError('XYZ error.')
        print_swap = self.swap_with_checks(builtins, 'print', lambda _: None, expected_args=((err,),))

        def mock_open_file(*unused_args: str) -> None:
            if False:
                i = 10
                return i + 15
            raise err
        swap_open = self.swap_with_checks(utils, 'open_file', mock_open_file, expected_args=(('.gcloudignore', 'r'),))
        swap_sys_exit = self.swap_with_checks(sys, 'exit', lambda _: None, expected_args=((1,),))
        with swap_open, swap_sys_exit, print_swap:
            third_party_size_check.get_skip_files_list()

    def test_check_size_in_dir(self) -> None:
        if False:
            i = 10
            return i + 15

        def mock_get_skip_files_list() -> list[str]:
            if False:
                return 10
            return []
        swap_get_skip_files_list = self.swap(third_party_size_check, 'get_skip_files_list', mock_get_skip_files_list)
        swap_third_party_dir = self.swap(third_party_size_check, 'THIRD_PARTY_PATH', os.path.join(os.getcwd(), 'dummy_dir'))
        with swap_third_party_dir, swap_get_skip_files_list, self.print_swap:
            third_party_size_check.check_third_party_size()
        self.assertIn('    Number of files in third-party folder: 4', self.print_arr)

    def test_check_size_in_dir_ignores_files_to_be_skipped(self) -> None:
        if False:
            while True:
                i = 10

        def mock_get_skip_files_list() -> list[str]:
            if False:
                i = 10
                return i + 15
            return [os.path.join(os.getcwd(), 'dummy_dir', 'file1.py'), os.path.join(os.getcwd(), 'dummy_dir', 'random*.py')]
        swap_get_skip_files_list = self.swap(third_party_size_check, 'get_skip_files_list', mock_get_skip_files_list)
        swap_third_party_dir = self.swap(third_party_size_check, 'THIRD_PARTY_PATH', os.path.join(os.getcwd(), 'dummy_dir'))
        with swap_third_party_dir, swap_get_skip_files_list, self.print_swap:
            third_party_size_check.check_third_party_size()
        self.assertIn('    Number of files in third-party folder: 2', self.print_arr)

    def test_check_third_party_size_pass(self) -> None:
        if False:
            print('Hello World!')
        swap_check_size_in_dir = self.swap(third_party_size_check, '_check_size_in_dir', lambda *unused_args: 100)
        with self.print_swap, swap_check_size_in_dir:
            third_party_size_check.check_third_party_size()
        self.assertIn('    The size of third-party folder is within the limits.', self.print_arr)

    def test_check_third_party_size_fail(self) -> None:
        if False:
            i = 10
            return i + 15
        swap_check_size_in_dir = self.swap(third_party_size_check, '_check_size_in_dir', lambda *unused_args: third_party_size_check.THIRD_PARTY_SIZE_LIMIT + 1)
        swap_sys_exit = self.swap(sys, 'exit', lambda _: None)
        with self.print_swap, swap_check_size_in_dir, swap_sys_exit:
            third_party_size_check.check_third_party_size()
        self.assertIn('    ERROR: The third-party folder size exceeded the %d files limit.' % third_party_size_check.THIRD_PARTY_SIZE_LIMIT, self.print_arr)