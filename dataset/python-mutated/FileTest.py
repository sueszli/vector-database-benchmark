import os
import unittest
from coalib.io.File import File
TEST_FILE_DIR = os.path.join(os.path.split(__file__)[0], 'file_test_files')

def get_path_components(filepath):
    if False:
        print('Hello World!')
    '\n    Splits the filepath into components to provide a unique\n    test results that passes on all CIs.\n    '
    return os.path.normpath(filepath).split(os.sep)

class FileTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        file_test_dir = TEST_FILE_DIR
        self.test_file = os.path.join(file_test_dir, 'test1.txt')
        self.other_test_file = os.path.join(file_test_dir, 'test2.txt')
        self.uut = File(self.test_file)
        self.other_file = File(self.other_test_file)

    def test_equal(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.uut, File(self.test_file))
        self.assertNotEqual(self.uut, self.other_file)

    def test_iter(self):
        if False:
            while True:
                i = 10
        self.assertEqual(list(self.uut), ['This is a test file.\n'])

    def test_line(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.uut.get_line(0), 'This is a test file.\n')
        with self.assertRaises(IndexError):
            self.uut.get_line(1)

    def test_deprecated_dict_getitem(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.uut[0], 'This is a test file.\n')
        with self.assertRaises(IndexError):
            self.uut[1]

    def test_lines(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.uut.lines, ('This is a test file.\n',))

    def test_raw(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.uut.raw, b'This is a test file.')

    def test_string(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.uut.string, 'This is a test file.')

    def test_timestamp(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.uut.timestamp, os.path.getmtime(self.test_file))

    def test_name(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(get_path_components(self.uut.name)[-4:], ['tests', 'io', 'file_test_files', 'test1.txt'])