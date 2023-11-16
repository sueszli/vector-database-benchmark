"""Tests for Bio.File module."""
import os.path
import shutil
import tempfile
import unittest
from io import StringIO
from Bio import bgzf
from Bio import File

class RandomAccess(unittest.TestCase):
    """Random access tests."""

    def test_plain(self):
        if False:
            while True:
                i = 10
        'Test plain text file.'
        with File._open_for_random_access('Quality/example.fastq') as handle:
            self.assertIn('r', handle.mode)
            self.assertIn('b', handle.mode)

    def test_bgzf(self):
        if False:
            for i in range(10):
                print('nop')
        'Test BGZF compressed file.'
        with File._open_for_random_access('Quality/example.fastq.bgz') as handle:
            self.assertIsInstance(handle, bgzf.BgzfReader)

    def test_gzip(self):
        if False:
            i = 10
            return i + 15
        'Test gzip compressed file.'
        self.assertRaises(ValueError, File._open_for_random_access, 'Quality/example.fastq.gz')

class AsHandleTestCase(unittest.TestCase):
    """Tests for as_handle function."""

    def setUp(self):
        if False:
            print('Hello World!')
        'Initialise temporary directory.'
        self.temp_dir = tempfile.mkdtemp(prefix='biopython-test')

    def tearDown(self):
        if False:
            while True:
                i = 10
        'Remove temporary directory.'
        shutil.rmtree(self.temp_dir)

    def _path(self, *args):
        if False:
            print('Hello World!')
        return os.path.join(self.temp_dir, *args)

    def test_handle(self):
        if False:
            for i in range(10):
                print('nop')
        'Test as_handle with a file-like object argument.'
        p = self._path('test_file.fasta')
        with open(p, 'wb') as fp:
            with File.as_handle(fp) as handle:
                self.assertEqual(fp, handle, 'as_handle should return argument when given a file-like object')
                self.assertFalse(handle.closed)
            self.assertFalse(handle.closed, 'Exiting as_handle given a file-like object should not close the file')

    def test_string_path(self):
        if False:
            return 10
        'Test as_handle with a string path argument.'
        p = self._path('test_file.fasta')
        mode = 'wb'
        with File.as_handle(p, mode=mode) as handle:
            self.assertEqual(p, handle.name)
            self.assertEqual(mode, handle.mode)
            self.assertFalse(handle.closed)
        self.assertTrue(handle.closed)

    def test_path_object(self):
        if False:
            i = 10
            return i + 15
        'Test as_handle with a pathlib.Path object.'
        from pathlib import Path
        p = Path(self._path('test_file.fasta'))
        mode = 'wb'
        with File.as_handle(p, mode=mode) as handle:
            self.assertEqual(str(p.absolute()), handle.name)
            self.assertEqual(mode, handle.mode)
            self.assertFalse(handle.closed)
        self.assertTrue(handle.closed)

    def test_custom_path_like_object(self):
        if False:
            i = 10
            return i + 15
        'Test as_handle with a custom path-like object.'

        class CustomPathLike:

            def __init__(self, path):
                if False:
                    while True:
                        i = 10
                self.path = path

            def __fspath__(self):
                if False:
                    print('Hello World!')
                return self.path
        p = CustomPathLike(self._path('test_file.fasta'))
        mode = 'wb'
        with File.as_handle(p, mode=mode) as handle:
            self.assertEqual(p.path, handle.name)
            self.assertEqual(mode, handle.mode)
            self.assertFalse(handle.closed)
        self.assertTrue(handle.closed)

    def test_stringio(self):
        if False:
            for i in range(10):
                print('nop')
        'Testing passing StringIO handles.'
        s = StringIO()
        with File.as_handle(s) as handle:
            self.assertIs(s, handle)

class BaseClassTests(unittest.TestCase):
    """Tests for _IndexedSeqFileProxy base class."""

    def test_instance_exception(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, File._IndexedSeqFileProxy)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)