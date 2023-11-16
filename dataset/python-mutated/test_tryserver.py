import os
import sys
from io import StringIO
from twisted.trial import unittest
from buildbot.scripts import tryserver
from buildbot.test.util import dirs

class TestStatusLog(dirs.DirsMixin, unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.newdir = os.path.join('jobdir', 'new')
        self.tmpdir = os.path.join('jobdir', 'tmp')
        self.setUpDirs('jobdir', self.newdir, self.tmpdir)

    def test_trycmd(self):
        if False:
            while True:
                i = 10
        config = {'jobdir': 'jobdir'}
        inputfile = StringIO('this is my try job')
        self.patch(sys, 'stdin', inputfile)
        rc = tryserver.tryserver(config)
        self.assertEqual(rc, 0)
        newfiles = os.listdir(self.newdir)
        tmpfiles = os.listdir(self.tmpdir)
        self.assertEqual((len(newfiles), len(tmpfiles)), (1, 0))
        with open(os.path.join(self.newdir, newfiles[0]), 'rt', encoding='utf-8') as f:
            self.assertEqual(f.read(), 'this is my try job')