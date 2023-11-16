"""Tests for the gprof2html script in the Tools directory."""
import os
import sys
import unittest
from unittest import mock
import tempfile
from test.test_tools import skip_if_missing, import_tool
skip_if_missing()

class Gprof2htmlTests(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.gprof = import_tool('gprof2html')
        oldargv = sys.argv

        def fixup():
            if False:
                i = 10
                return i + 15
            sys.argv = oldargv
        self.addCleanup(fixup)
        sys.argv = []

    def test_gprof(self):
        if False:
            print('Hello World!')
        with mock.patch.object(self.gprof, 'webbrowser') as wmock, tempfile.TemporaryDirectory() as tmpdir:
            fn = os.path.join(tmpdir, 'abc')
            open(fn, 'w').close()
            sys.argv = ['gprof2html', fn]
            self.gprof.main()
        self.assertTrue(wmock.open.called)
if __name__ == '__main__':
    unittest.main()