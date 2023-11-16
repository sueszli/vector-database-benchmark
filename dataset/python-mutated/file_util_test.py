import errno
import os
import unittest
from unittest.mock import MagicMock, mock_open, patch
import pytest
from streamlit import file_util, util
FILENAME = '/some/cache/file'
mock_get_path = MagicMock(return_value=FILENAME)

class FileUtilTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.patch1 = patch('streamlit.file_util.os.stat')
        self.os_stat = self.patch1.start()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.patch1.stop()

    @patch('streamlit.file_util.get_streamlit_file_path', mock_get_path)
    @patch('streamlit.file_util.open', mock_open(read_data='data'))
    def test_streamlit_read(self):
        if False:
            i = 10
            return i + 15
        'Test streamlitfile_util.streamlit_read.'
        with file_util.streamlit_read(FILENAME) as input:
            data = input.read()
        self.assertEqual('data', data)

    @patch('streamlit.file_util.get_streamlit_file_path', mock_get_path)
    @patch('streamlit.file_util.open', mock_open(read_data=b'\xaa\xbb'))
    def test_streamlit_read_binary(self):
        if False:
            for i in range(10):
                print('nop')
        'Test streamlitfile_util.streamlit_read.'
        with file_util.streamlit_read(FILENAME, binary=True) as input:
            data = input.read()
        self.assertEqual(b'\xaa\xbb', data)

    @patch('streamlit.file_util.get_streamlit_file_path', mock_get_path)
    @patch('streamlit.file_util.open', mock_open(read_data='data'))
    def test_streamlit_read_zero_bytes(self):
        if False:
            return 10
        'Test streamlitfile_util.streamlit_read.'
        self.os_stat.return_value.st_size = 0
        with pytest.raises(util.Error) as e:
            with file_util.streamlit_read(FILENAME) as input:
                input.read()
        self.assertEqual(str(e.value), 'Read zero byte file: "/some/cache/file"')

    @patch('streamlit.file_util.get_streamlit_file_path', mock_get_path)
    def test_streamlit_write(self):
        if False:
            while True:
                i = 10
        'Test streamlitfile_util.streamlit_write.'
        dirname = os.path.dirname(file_util.get_streamlit_file_path(FILENAME))
        with patch('streamlit.file_util.open', mock_open()) as open, patch('streamlit.util.os.makedirs') as makedirs, file_util.streamlit_write(FILENAME) as output:
            output.write('some data')
            open().write.assert_called_once_with('some data')
            makedirs.assert_called_once_with(dirname, exist_ok=True)

    @patch('streamlit.file_util.get_streamlit_file_path', mock_get_path)
    @patch('streamlit.env_util.IS_DARWIN', True)
    def test_streamlit_write_exception(self):
        if False:
            while True:
                i = 10
        'Test streamlitfile_util.streamlit_write.'
        with patch('streamlit.file_util.open', mock_open()) as p, patch('streamlit.util.os.makedirs'):
            p.side_effect = OSError(errno.EINVAL, '[Errno 22] Invalid argument')
            with pytest.raises(util.Error) as e, file_util.streamlit_write(FILENAME) as output:
                output.write('some data')
            error_msg = 'Unable to write file: /some/cache/file\nPython is limited to files below 2GB on OSX. See https://bugs.python.org/issue24658'
            self.assertEqual(str(e.value), error_msg)

    def test_get_project_streamlit_file_path(self):
        if False:
            for i in range(10):
                print('nop')
        expected = os.path.join(os.getcwd(), file_util.CONFIG_FOLDER_NAME, 'some/random/file')
        self.assertEqual(expected, file_util.get_project_streamlit_file_path('some/random/file'))
        self.assertEqual(expected, file_util.get_project_streamlit_file_path('some', 'random', 'file'))

    def test_get_app_static_dir(self):
        if False:
            return 10
        self.assertEqual(file_util.get_app_static_dir('/some_path/to/app/myapp.py'), '/some_path/to/app/static')

    @patch('os.path.getsize', MagicMock(return_value=42))
    @patch('os.walk', MagicMock(return_value=[('dir1', [], ['file1', 'file2', 'file3']), ('dir2', [], ['file4', 'file5'])]))
    def test_get_directory_size(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(file_util.get_directory_size('the_dir'), 42 * 5)

class FileIsInFolderTest(unittest.TestCase):

    def test_file_in_folder(self):
        if False:
            while True:
                i = 10
        ret = file_util.file_is_in_folder_glob('/a/b/c/foo.py', '/a/b/c/')
        self.assertTrue(ret)
        ret = file_util.file_is_in_folder_glob('/a/b/c/foo.py', '/a/b/c')
        self.assertTrue(ret)

    def test_file_in_subfolder(self):
        if False:
            for i in range(10):
                print('nop')
        ret = file_util.file_is_in_folder_glob('/a/b/c/foo.py', '/a')
        self.assertTrue(ret)
        ret = file_util.file_is_in_folder_glob('/a/b/c/foo.py', '/a/')
        self.assertTrue(ret)
        ret = file_util.file_is_in_folder_glob('/a/b/c/foo.py', '/a/b')
        self.assertTrue(ret)
        ret = file_util.file_is_in_folder_glob('/a/b/c/foo.py', '/a/b/')
        self.assertTrue(ret)

    def test_file_not_in_folder(self):
        if False:
            i = 10
            return i + 15
        ret = file_util.file_is_in_folder_glob('/a/b/c/foo.py', '/d/e/f/')
        self.assertFalse(ret)
        ret = file_util.file_is_in_folder_glob('/a/b/c/foo.py', '/d/e/f')
        self.assertFalse(ret)

    def test_rel_file_not_in_folder(self):
        if False:
            while True:
                i = 10
        ret = file_util.file_is_in_folder_glob('foo.py', '/d/e/f/')
        self.assertFalse(ret)
        ret = file_util.file_is_in_folder_glob('foo.py', '/d/e/f')
        self.assertFalse(ret)

    def test_file_in_folder_glob(self):
        if False:
            for i in range(10):
                print('nop')
        ret = file_util.file_is_in_folder_glob('/a/b/c/foo.py', '**/c')
        self.assertTrue(ret)

    def test_file_not_in_folder_glob(self):
        if False:
            i = 10
            return i + 15
        ret = file_util.file_is_in_folder_glob('/a/b/c/foo.py', '**/f')
        self.assertFalse(ret)

    def test_rel_file_not_in_folder_glob(self):
        if False:
            print('Hello World!')
        ret = file_util.file_is_in_folder_glob('foo.py', '**/f')
        self.assertFalse(ret)

    def test_rel_file_not_in_folder_glob(self):
        if False:
            for i in range(10):
                print('nop')
        ret = file_util.file_is_in_folder_glob('foo.py', '')
        self.assertTrue(ret)

class FileInPythonPathTest(unittest.TestCase):

    @staticmethod
    def _make_it_absolute(path):
        if False:
            i = 10
            return i + 15
        return os.path.join(os.getcwd(), path)

    def test_no_pythonpath(self):
        if False:
            for i in range(10):
                print('nop')
        with patch('os.environ', {}) as d:
            self.assertFalse(file_util.file_in_pythonpath(self._make_it_absolute('../something/dir1/dir2/module')))

    def test_empty_pythonpath(self):
        if False:
            return 10
        with patch('os.environ', {'PYTHONPATH': ''}):
            self.assertFalse(file_util.file_in_pythonpath(self._make_it_absolute('something/dir1/dir2/module')))

    def test_python_path_relative(self):
        if False:
            for i in range(10):
                print('nop')
        with patch('os.environ', {'PYTHONPATH': 'something'}):
            self.assertTrue(file_util.file_in_pythonpath(self._make_it_absolute('something/dir1/dir2/module')))
            self.assertFalse(file_util.file_in_pythonpath(self._make_it_absolute('something_else/module')))
            self.assertFalse(file_util.file_in_pythonpath(self._make_it_absolute('../something/dir1/dir2/module')))

    def test_python_path_absolute(self):
        if False:
            return 10
        with patch('os.environ', {'PYTHONPATH': self._make_it_absolute('something')}):
            self.assertTrue(file_util.file_in_pythonpath(self._make_it_absolute('something/dir1/dir2/module')))
            self.assertFalse(file_util.file_in_pythonpath(self._make_it_absolute('something_else/module')))
            self.assertFalse(file_util.file_in_pythonpath(self._make_it_absolute('../something/dir1/dir2/module')))

    def test_python_path_mixed(self):
        if False:
            return 10
        with patch('os.environ', {'PYTHONPATH': os.pathsep.join([self._make_it_absolute('something'), 'something'])}):
            self.assertTrue(file_util.file_in_pythonpath(self._make_it_absolute('something/dir1/dir2/module')))
            self.assertFalse(file_util.file_in_pythonpath(self._make_it_absolute('something_else/module')))

    def test_current_directory(self):
        if False:
            print('Hello World!')
        with patch('os.environ', {'PYTHONPATH': '.'}):
            self.assertTrue(file_util.file_in_pythonpath(self._make_it_absolute('something/dir1/dir2/module')))
            self.assertTrue(file_util.file_in_pythonpath(self._make_it_absolute('something_else/module')))
            self.assertFalse(file_util.file_in_pythonpath(self._make_it_absolute('../something_else/module')))