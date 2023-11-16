"""Tests for the 'hidden' utility."""
import ctypes
import errno
import subprocess
import sys
import tempfile
import unittest
from beets import util
from beets.util import hidden

class HiddenFileTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_osx_hidden(self):
        if False:
            i = 10
            return i + 15
        if not sys.platform == 'darwin':
            self.skipTest('sys.platform is not darwin')
            return
        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                command = ['chflags', 'hidden', f.name]
                subprocess.Popen(command).wait()
            except OSError as e:
                if e.errno == errno.ENOENT:
                    self.skipTest('unable to find chflags')
                else:
                    raise e
            self.assertTrue(hidden.is_hidden(f.name))

    def test_windows_hidden(self):
        if False:
            while True:
                i = 10
        if not sys.platform == 'win32':
            self.skipTest('sys.platform is not windows')
            return
        hidden_mask = 2
        with tempfile.NamedTemporaryFile() as f:
            success = ctypes.windll.kernel32.SetFileAttributesW(f.name, hidden_mask)
            if not success:
                self.skipTest('unable to set file attributes')
            self.assertTrue(hidden.is_hidden(f.name))

    def test_other_hidden(self):
        if False:
            return 10
        if sys.platform == 'darwin' or sys.platform == 'win32':
            self.skipTest('sys.platform is known')
            return
        with tempfile.NamedTemporaryFile(prefix='.tmp') as f:
            fn = util.bytestring_path(f.name)
            self.assertTrue(hidden.is_hidden(fn))

def suite():
    if False:
        while True:
            i = 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')