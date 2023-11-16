from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
import sys
import twisted.python.procutils
from twisted.python import runtime
from twisted.trial import unittest
from buildbot_worker.commands import utils

class GetCommand(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.which_results = {}

        def which(arg):
            if False:
                while True:
                    i = 10
            return self.which_results.get(arg, [])
        self.patch(twisted.python.procutils, 'which', which)
        self.patch(utils, 'which', which)

    def set_which_results(self, results):
        if False:
            return 10
        self.which_results = results

    def test_getCommand_empty(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_which_results({'xeyes': []})
        with self.assertRaises(RuntimeError):
            utils.getCommand('xeyes')

    def test_getCommand_single(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_which_results({'xeyes': ['/usr/bin/xeyes']})
        self.assertEqual(utils.getCommand('xeyes'), '/usr/bin/xeyes')

    def test_getCommand_multi(self):
        if False:
            while True:
                i = 10
        self.set_which_results({'xeyes': ['/usr/bin/xeyes', '/usr/X11/bin/xeyes']})
        self.assertEqual(utils.getCommand('xeyes'), '/usr/bin/xeyes')

    def test_getCommand_single_exe(self):
        if False:
            while True:
                i = 10
        self.set_which_results({'xeyes': ['/usr/bin/xeyes'], 'xeyes.exe': ['c:\\program files\\xeyes.exe']})
        self.assertEqual(utils.getCommand('xeyes'), '/usr/bin/xeyes')

    def test_getCommand_multi_exe(self):
        if False:
            for i in range(10):
                print('nop')
        self.set_which_results({'xeyes': ['c:\\program files\\xeyes.com', 'c:\\program files\\xeyes.exe'], 'xeyes.exe': ['c:\\program files\\xeyes.exe']})
        if runtime.platformType == 'win32':
            self.assertEqual(utils.getCommand('xeyes'), 'c:\\program files\\xeyes.exe')
        else:
            self.assertEqual(utils.getCommand('xeyes'), 'c:\\program files\\xeyes.com')

class RmdirRecursive(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.target = 'testdir'
        try:
            if os.path.exists(self.target):
                shutil.rmtree(self.target)
        except Exception:
            e = sys.exc_info()[0]
            raise unittest.SkipTest('could not clean before test: {0}'.format(e))
        os.mkdir(os.path.join(self.target))
        with open(os.path.join(self.target, 'a'), 'w'):
            pass
        os.mkdir(os.path.join(self.target, 'd'))
        with open(os.path.join(self.target, 'd', 'a'), 'w'):
            pass
        os.mkdir(os.path.join(self.target, 'd', 'd'))
        with open(os.path.join(self.target, 'd', 'd', 'a'), 'w'):
            pass

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            if os.path.exists(self.target):
                shutil.rmtree(self.target)
        except Exception:
            print('\n(target directory was not removed by test, and cleanup failed too)\n')
            raise

    def test_rmdirRecursive_easy(self):
        if False:
            i = 10
            return i + 15
        utils.rmdirRecursive(self.target)
        self.assertFalse(os.path.exists(self.target))

    def test_rmdirRecursive_symlink(self):
        if False:
            while True:
                i = 10
        if runtime.platformType == 'win32':
            raise unittest.SkipTest('no symlinks on this platform')
        os.mkdir('noperms')
        with open('noperms/x', 'w'):
            pass
        os.chmod('noperms/x', 0)
        try:
            os.symlink('../noperms', os.path.join(self.target, 'link'))
            utils.rmdirRecursive(self.target)
            self.assertTrue(os.path.exists('noperms'))
        finally:
            os.chmod('noperms/x', 511)
            os.unlink('noperms/x')
            os.rmdir('noperms')
        self.assertFalse(os.path.exists(self.target))