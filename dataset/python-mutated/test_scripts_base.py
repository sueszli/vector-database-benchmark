from __future__ import absolute_import
from __future__ import print_function
import os
import sys
from twisted.trial import unittest
from buildbot_worker.compat import NativeStringIO
from buildbot_worker.scripts import base
from buildbot_worker.test.util import misc

class TestIsWorkerDir(misc.FileIOMixin, misc.StdoutAssertionsMixin, unittest.TestCase):
    """Test buildbot_worker.scripts.base.isWorkerDir()"""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.mocked_stdout = NativeStringIO()
        self.patch(sys, 'stdout', self.mocked_stdout)
        self.tac_file_path = os.path.join('testdir', 'buildbot.tac')

    def assertReadErrorMessage(self, strerror):
        if False:
            print('Hello World!')
        expected_message = "error reading '{0}': {1}\ninvalid worker directory 'testdir'\n".format(self.tac_file_path, strerror)
        self.assertEqual(self.mocked_stdout.getvalue(), expected_message, 'unexpected error message on stdout')

    def test_open_error(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that open() errors are handled.'
        self.setUpOpenError(1, 'open-error', 'dummy')
        self.assertFalse(base.isWorkerDir('testdir'))
        self.assertReadErrorMessage('open-error')
        self.open.assert_called_once_with(self.tac_file_path)

    def test_read_error(self):
        if False:
            while True:
                i = 10
        'Test that read() errors on buildbot.tac file are handled.'
        self.setUpReadError(1, 'read-error', 'dummy')
        self.assertFalse(base.isWorkerDir('testdir'))
        self.assertReadErrorMessage('read-error')
        self.open.assert_called_once_with(self.tac_file_path)

    def test_unexpected_tac_contents(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that unexpected contents in buildbot.tac is handled.'
        self.setUpOpen('dummy-contents')
        self.assertFalse(base.isWorkerDir('testdir'))
        self.assertEqual(self.mocked_stdout.getvalue(), "unexpected content in '{0}'\n".format(self.tac_file_path) + "invalid worker directory 'testdir'\n", 'unexpected error message on stdout')
        self.open.assert_called_once_with(self.tac_file_path)

    def test_workerdir_good(self):
        if False:
            for i in range(10):
                print('nop')
        'Test checking valid worker directory.'
        self.setUpOpen("Application('buildbot-worker')")
        self.assertTrue(base.isWorkerDir('testdir'))
        self.open.assert_called_once_with(self.tac_file_path)