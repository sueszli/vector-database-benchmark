import os
import stat
import tempfile
from unittest.mock import Mock
from twisted.trial import unittest
from buildbot.process import remotetransfer

class TestFileWriter(unittest.TestCase):

    def testInit(self):
        if False:
            print('Hello World!')
        mockedExists = Mock(return_value=False)
        self.patch(os.path, 'exists', mockedExists)
        mockedMakedirs = Mock()
        self.patch(os, 'makedirs', mockedMakedirs)
        mockedMkstemp = Mock(return_value=(7, 'tmpname'))
        self.patch(tempfile, 'mkstemp', mockedMkstemp)
        mockedFdopen = Mock()
        self.patch(os, 'fdopen', mockedFdopen)
        destfile = os.path.join('dir', 'file')
        remotetransfer.FileWriter(destfile, 64, stat.S_IRUSR)
        absdir = os.path.dirname(os.path.abspath(os.path.join('dir', 'file')))
        mockedExists.assert_called_once_with(absdir)
        mockedMakedirs.assert_called_once_with(absdir)
        mockedMkstemp.assert_called_once_with(dir=absdir, prefix='buildbot-transfer-')
        mockedFdopen.assert_called_once_with(7, 'wb')

class TestStringFileWriter(unittest.TestCase):

    def testBasic(self):
        if False:
            while True:
                i = 10
        sfw = remotetransfer.StringFileWriter()
        sfw.remote_write(b'bytes')
        sfw.remote_write(' or str')
        self.assertEqual(sfw.buffer, 'bytes or str')