import unittest
from test import support
from test.support import os_helper
import os
import platform
import sys
from os import path
startfile = support.get_attribute(os, 'startfile')

@unittest.skipIf(platform.win32_is_iot(), 'starting files is not supported on Windows IoT Core or nanoserver')
class TestCase(unittest.TestCase):

    def test_nonexisting(self):
        if False:
            return 10
        self.assertRaises(OSError, startfile, 'nonexisting.vbs')

    def test_empty(self):
        if False:
            print('Hello World!')
        with os_helper.change_cwd(path.dirname(sys.executable)):
            empty = path.join(path.dirname(__file__), 'empty.vbs')
            startfile(empty)
            startfile(empty, 'open')
        startfile(empty, cwd=path.dirname(sys.executable))

    def test_python(self):
        if False:
            print('Hello World!')
        (cwd, name) = path.split(sys.executable)
        startfile(name, arguments='-V', cwd=cwd)
        startfile(name, arguments='-V', cwd=cwd, show_cmd=0)
if __name__ == '__main__':
    unittest.main()