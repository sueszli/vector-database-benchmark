import os
import shutil
import sys
import tempfile
import unittest
from typing import Optional
from unittest.mock import MagicMock

class TestFileIO(unittest.TestCase):
    _tmpdir: Optional[str] = None
    _tmpfile: Optional[str] = None
    _tmpfile_contents = 'Hello, World'

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        cls._tmpdir = tempfile.mkdtemp()
        with open(os.path.join(cls._tmpdir, 'test.txt'), 'w') as f:
            cls._tmpfile = f.name
            f.write(cls._tmpfile_contents)
            f.flush()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            return 10
        if cls._tmpdir is not None:
            shutil.rmtree(cls._tmpdir)

    def test_file_io(self):
        if False:
            while True:
                i = 10
        from fairseq.file_io import PathManager
        with PathManager.open(os.path.join(self._tmpdir, 'test.txt'), 'r') as f:
            s = f.read()
        self.assertEqual(s, self._tmpfile_contents)

    def test_file_io_oss(self):
        if False:
            print('Hello World!')
        sys.modules['iopath'] = MagicMock()
        from fairseq.file_io import PathManager
        with PathManager.open(os.path.join(self._tmpdir, 'test.txt'), 'r') as f:
            s = f.read()
        self.assertEqual(s, self._tmpfile_contents)

    def test_file_io_async(self):
        if False:
            return 10
        try:
            from fairseq.file_io import PathManager
            _asyncfile = os.path.join(self._tmpdir, 'async.txt')
            f = PathManager.opena(_asyncfile, 'wb')
            f.close()
        finally:
            self.assertTrue(PathManager.async_close())