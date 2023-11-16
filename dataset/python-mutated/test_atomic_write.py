import errno
import os
import unittest
import pytest
from trashcli.fs import atomic_write, open_for_write_in_exclusive_and_create_mode, read_file
from ..support.my_path import MyPath

@pytest.mark.slow
class Test_atomic_write(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.temp_dir = MyPath.make_temp_dir()

    def test_the_second_open_should_fail(self):
        if False:
            return 10
        path = self.temp_dir / 'a'
        file_handle = open_for_write_in_exclusive_and_create_mode(path)
        try:
            open_for_write_in_exclusive_and_create_mode(path)
            self.fail()
        except OSError as e:
            assert e.errno == errno.EEXIST
        os.close(file_handle)

    def test_short_filename(self):
        if False:
            while True:
                i = 10
        path = self.temp_dir / 'a'
        atomic_write(path, b'contents')
        assert 'contents' == read_file(path)

    def test_too_long_filename(self):
        if False:
            return 10
        path = self.temp_dir / ('a' * 2000)
        try:
            atomic_write(path, b'contents')
            self.fail()
        except OSError as e:
            assert e.errno == errno.ENAMETOOLONG

    def test_filename_already_taken(self):
        if False:
            i = 10
            return i + 15
        atomic_write(self.temp_dir / 'a', b'contents')
        try:
            atomic_write(self.temp_dir / 'a', b'contents')
            self.fail()
        except OSError as e:
            assert e.errno == errno.EEXIST

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.temp_dir.clean_up()