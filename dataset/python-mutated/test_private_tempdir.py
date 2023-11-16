import os
import shutil
import tempfile
from twisted.trial import unittest
from buildbot.test.util.decorators import skipUnlessPlatformIs
from buildbot.util.private_tempdir import PrivateTemporaryDirectory

class TestTemporaryDirectory(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tempdir = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(self.tempdir)

    def test_simple(self):
        if False:
            return 10
        with PrivateTemporaryDirectory(dir=self.tempdir) as dir:
            self.assertTrue(os.path.isdir(dir))
        self.assertFalse(os.path.isdir(dir))

    @skipUnlessPlatformIs('posix')
    def test_mode(self):
        if False:
            for i in range(10):
                print('nop')
        with PrivateTemporaryDirectory(dir=self.tempdir, mode=448) as dir:
            self.assertEqual(16832, os.stat(dir).st_mode)

    def test_cleanup(self):
        if False:
            print('Hello World!')
        ctx = PrivateTemporaryDirectory(dir=self.tempdir)
        self.assertTrue(os.path.isdir(ctx.name))
        ctx.cleanup()
        self.assertFalse(os.path.isdir(ctx.name))
        ctx.cleanup()
        ctx.cleanup()