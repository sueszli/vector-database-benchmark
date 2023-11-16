import unittest
from unittest.mock import MagicMock, patch
import pytest
from ..iorw import HDFSHandler

class MockHadoopFileSystem(MagicMock):

    def get_file_info(self, path):
        if False:
            while True:
                i = 10
        return [MockFileInfo('test1.ipynb'), MockFileInfo('test2.ipynb')]

    def open_input_stream(self, path):
        if False:
            while True:
                i = 10
        return MockHadoopFile()

    def open_output_stream(self, path):
        if False:
            i = 10
            return i + 15
        return MockHadoopFile()

class MockHadoopFile:

    def __init__(self):
        if False:
            return 10
        self._content = b'Content of notebook'

    def __enter__(self, *args):
        if False:
            return 10
        return self

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        pass

    def read(self):
        if False:
            while True:
                i = 10
        return self._content

    def write(self, new_content):
        if False:
            while True:
                i = 10
        self._content = new_content
        return 1

class MockFileInfo:

    def __init__(self, path):
        if False:
            return 10
        self.path = path

@pytest.mark.skip(reason='No valid dep package for python 3.12 yet')
@patch('papermill.iorw.HadoopFileSystem', side_effect=MockHadoopFileSystem())
class HDFSTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.hdfs_handler = HDFSHandler()

    def test_hdfs_listdir(self, mock_hdfs_filesystem):
        if False:
            for i in range(10):
                print('nop')
        client = self.hdfs_handler._get_client()
        self.assertEqual(self.hdfs_handler.listdir('hdfs:///Projects/'), ['test1.ipynb', 'test2.ipynb'])
        self.assertIs(client, self.hdfs_handler._get_client())

    def test_hdfs_read(self, mock_hdfs_filesystem):
        if False:
            while True:
                i = 10
        client = self.hdfs_handler._get_client()
        self.assertEqual(self.hdfs_handler.read('hdfs:///Projects/test1.ipynb'), b'Content of notebook')
        self.assertIs(client, self.hdfs_handler._get_client())

    def test_hdfs_write(self, mock_hdfs_filesystem):
        if False:
            return 10
        client = self.hdfs_handler._get_client()
        self.assertEqual(self.hdfs_handler.write('hdfs:///Projects/test1.ipynb', b'New content'), 1)
        self.assertIs(client, self.hdfs_handler._get_client())