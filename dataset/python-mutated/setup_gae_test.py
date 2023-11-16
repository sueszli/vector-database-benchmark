"""Unit tests for scripts/setup_gae.py."""
from __future__ import annotations
import builtins
import os
import tarfile
from core.tests import test_utils
from typing import List, Tuple
from . import common
from . import setup_gae
RELEASE_TEST_DIR = os.path.join('core', 'tests', 'release_sources', '')
MOCK_TMP_UNZIP_PATH = os.path.join(RELEASE_TEST_DIR, 'tmp_unzip.zip')
MOCK_TMP_UNTAR_PATH = os.path.join(RELEASE_TEST_DIR, 'tmp_unzip.tar.gz')

class SetupGaeTests(test_utils.GenericTestBase):
    """Test the methods for setup gae script."""

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.check_function_calls = {'walk_is_called': False, 'remove_is_called': False, 'makedirs_is_called': False, 'url_retrieve_is_called': False}
        self.expected_check_function_calls = {'walk_is_called': True, 'remove_is_called': True, 'makedirs_is_called': True, 'url_retrieve_is_called': True}
        self.raise_error = False

        def mock_walk(unused_path: str) -> List[Tuple[str, List[str], List[str]]]:
            if False:
                for i in range(10):
                    print('nop')
            self.check_function_calls['walk_is_called'] = True
            return []

        def mock_remove(unused_path: str) -> None:
            if False:
                print('Hello World!')
            self.check_function_calls['remove_is_called'] = True

        def mock_makedirs(unused_path: str) -> None:
            if False:
                print('Hello World!')
            self.check_function_calls['makedirs_is_called'] = True
        self.print_arr: List[str] = []

        def mock_print(msg: str) -> None:
            if False:
                while True:
                    i = 10
            self.print_arr.append(msg)

        def mock_url_retrieve(unused_url: str, filename: str) -> None:
            if False:
                i = 10
                return i + 15
            self.check_function_calls['url_retrieve_is_called'] = True
            if self.raise_error:
                raise Exception
        self.walk_swap = self.swap(os, 'walk', mock_walk)
        self.remove_swap = self.swap(os, 'remove', mock_remove)
        self.makedirs_swap = self.swap(os, 'makedirs', mock_makedirs)
        self.print_swap = self.swap(builtins, 'print', mock_print)
        self.url_retrieve_swap = self.swap(common, 'url_retrieve', mock_url_retrieve)

    def test_main_with_no_installs_required(self) -> None:
        if False:
            while True:
                i = 10
        check_file_removals = {'root/file1.js': False, 'root/file2.pyc': False}
        expected_check_file_removals = {'root/file1.js': False, 'root/file2.pyc': True}

        def mock_walk(unused_path: str) -> List[Tuple[str, List[str], List[str]]]:
            if False:
                while True:
                    i = 10
            return [('root', ['dir1'], ['file1.js', 'file2.pyc'])]

        def mock_remove(path: str) -> None:
            if False:
                i = 10
                return i + 15
            check_file_removals[path] = True

        def mock_exists(unused_path: str) -> bool:
            if False:
                print('Hello World!')
            return True
        walk_swap = self.swap(os, 'walk', mock_walk)
        remove_swap = self.swap(os, 'remove', mock_remove)
        exists_swap = self.swap(os.path, 'exists', mock_exists)
        with walk_swap, remove_swap, exists_swap:
            setup_gae.main(args=[])
        self.assertEqual(check_file_removals, expected_check_file_removals)

    def test_gcloud_install_without_errors(self) -> None:
        if False:
            i = 10
            return i + 15
        self.check_function_calls['open_is_called'] = False
        self.check_function_calls['extractall_is_called'] = False
        self.check_function_calls['close_is_called'] = False
        self.expected_check_function_calls['open_is_called'] = True
        self.expected_check_function_calls['extractall_is_called'] = True
        self.expected_check_function_calls['close_is_called'] = True

        def mock_exists(path: str) -> bool:
            if False:
                return 10
            if path == common.GOOGLE_CLOUD_SDK_HOME:
                return False
            return True
        temp_file = tarfile.open(name=MOCK_TMP_UNTAR_PATH)

        def mock_open(name: str) -> tarfile.TarFile:
            if False:
                print('Hello World!')
            self.check_function_calls['open_is_called'] = True
            return temp_file

        def mock_extractall(unused_self: str, path: str) -> None:
            if False:
                return 10
            self.check_function_calls['extractall_is_called'] = True

        def mock_close(unused_self: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.check_function_calls['close_is_called'] = True
        exists_swap = self.swap(os.path, 'exists', mock_exists)
        open_swap = self.swap(tarfile, 'open', mock_open)
        extractall_swap = self.swap(tarfile.TarFile, 'extractall', mock_extractall)
        close_swap = self.swap(tarfile.TarFile, 'close', mock_close)
        with self.walk_swap, self.remove_swap, self.makedirs_swap:
            with self.print_swap, self.url_retrieve_swap, exists_swap:
                with open_swap, extractall_swap, close_swap:
                    setup_gae.main(args=[])
        self.assertEqual(self.check_function_calls, self.expected_check_function_calls)
        self.assertTrue('Download complete. Installing Google Cloud SDK...' in self.print_arr)

    def test_gcloud_install_with_errors(self) -> None:
        if False:
            while True:
                i = 10
        self.expected_check_function_calls['remove_is_called'] = False
        self.raise_error = True

        def mock_exists(path: str) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            if path == common.GOOGLE_CLOUD_SDK_HOME:
                return False
            return True
        exists_swap = self.swap(os.path, 'exists', mock_exists)
        with self.walk_swap, self.remove_swap, self.makedirs_swap:
            with self.print_swap, self.url_retrieve_swap, exists_swap:
                with self.assertRaisesRegex(Exception, 'Error downloading Google Cloud SDK.'):
                    setup_gae.main(args=[])
        self.assertEqual(self.check_function_calls, self.expected_check_function_calls)
        self.assertTrue('Error downloading Google Cloud SDK. Exiting.' in self.print_arr)