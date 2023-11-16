"""Tests for the extract_object_files module."""
import io
import os
import pathlib
from typing import List
from absl.testing import parameterized
from tensorflow.lite.ios import extract_object_files
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test

class ExtractObjectFilesTest(parameterized.TestCase):

    @parameterized.named_parameters(dict(testcase_name='Simple extraction', dirname='simple', object_files=['foo.o', 'bar.o']), dict(testcase_name='Extended filename', dirname='extended_filename', object_files=['short.o', 'long_file_name_with_extended_format.o']), dict(testcase_name='Odd bytes pad handling', dirname='odd_bytes', object_files=['odd.o', 'even.o']), dict(testcase_name='Duplicate object names should be separated out', dirname='duplicate_names', object_files=['foo.o', 'foo_1.o', 'foo_2.o']), dict(testcase_name='Exact same file should not be extracted again', dirname='skip_same_file', object_files=['foo.o']))
    def test_extract_object_files(self, dirname: str, object_files: List[str]):
        if False:
            i = 10
            return i + 15
        dest_dir = self.create_tempdir().full_path
        input_file_relpath = os.path.join('testdata', dirname, 'input.a')
        archive_path = resource_loader.get_path_to_datafile(input_file_relpath)
        with open(archive_path, 'rb') as archive_file:
            extract_object_files.extract_object_files(archive_file, dest_dir)
        self.assertCountEqual(object_files, os.listdir(dest_dir))
        for file in object_files:
            actual = pathlib.Path(os.path.join(dest_dir, file)).read_bytes()
            expected = pathlib.Path(resource_loader.get_path_to_datafile(os.path.join('testdata', dirname, file))).read_bytes()
            self.assertEqual(actual, expected)

    def test_invalid_archive(self):
        if False:
            print('Hello World!')
        with io.BytesIO(b'this is an invalid archive file') as archive_file:
            with self.assertRaises(RuntimeError):
                extract_object_files.extract_object_files(archive_file, self.create_tempdir().full_path)
if __name__ == '__main__':
    test.main()