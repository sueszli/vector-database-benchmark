import os
import stat
from twisted.internet import defer
from twisted.python.filepath import FilePath
from twisted.trial import unittest
from buildbot.secrets.providers.file import SecretInAFile
from buildbot.test.util.config import ConfigErrorsMixin
from buildbot.util.misc import writeLocalFile

class TestSecretInFile(ConfigErrorsMixin, unittest.TestCase):

    def createTempDir(self, dirname):
        if False:
            return 10
        tempdir = FilePath(self.mktemp())
        tempdir.createDirectory()
        return tempdir.path

    def createFileTemp(self, tempdir, filename, text='', chmodRights=448):
        if False:
            while True:
                i = 10
        file_path = os.path.join(tempdir, filename)
        writeLocalFile(file_path, text, chmodRights)
        return file_path

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.tmp_dir = self.createTempDir('temp')
        self.filepath = self.createFileTemp(self.tmp_dir, 'tempfile.txt', text='key value\n')
        self.srvfile = SecretInAFile(self.tmp_dir)
        yield self.srvfile.startService()

    @defer.inlineCallbacks
    def tearDown(self):
        if False:
            return 10
        yield self.srvfile.stopService()

    def testCheckConfigSecretInAFileService(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.srvfile.name, 'SecretInAFile')
        self.assertEqual(self.srvfile._dirname, self.tmp_dir)

    def testCheckConfigErrorSecretInAFileService(self):
        if False:
            while True:
                i = 10
        if os.name != 'posix':
            self.skipTest('Permission checks only works on posix systems')
        filepath = self.createFileTemp(self.tmp_dir, 'tempfile2.txt', chmodRights=stat.S_IROTH)
        expctd_msg_error = ' on file tempfile2.txt are too open. It is required that your secret files are NOT accessible by others!'
        with self.assertRaisesConfigError(expctd_msg_error):
            self.srvfile.checkConfig(self.tmp_dir)
        os.remove(filepath)

    @defer.inlineCallbacks
    def testCheckConfigfileExtension(self):
        if False:
            return 10
        filepath = self.createFileTemp(self.tmp_dir, 'tempfile2.ini', text='test suffix', chmodRights=stat.S_IRWXU)
        filepath2 = self.createFileTemp(self.tmp_dir, 'tempfile2.txt', text='some text', chmodRights=stat.S_IRWXU)
        yield self.srvfile.reconfigService(self.tmp_dir, suffixes=['.ini'])
        self.assertEqual(self.srvfile.get('tempfile2'), 'test suffix')
        self.assertEqual(self.srvfile.get('tempfile3'), None)
        os.remove(filepath)
        os.remove(filepath2)

    @defer.inlineCallbacks
    def testReconfigSecretInAFileService(self):
        if False:
            while True:
                i = 10
        otherdir = self.createTempDir('temp2')
        yield self.srvfile.reconfigService(otherdir)
        self.assertEqual(self.srvfile.name, 'SecretInAFile')
        self.assertEqual(self.srvfile._dirname, otherdir)

    def testGetSecretInFile(self):
        if False:
            return 10
        value = self.srvfile.get('tempfile.txt')
        self.assertEqual(value, 'key value')

    @defer.inlineCallbacks
    def testGetSecretInFileSuffixes(self):
        if False:
            while True:
                i = 10
        yield self.srvfile.reconfigService(self.tmp_dir, suffixes=['.txt'])
        value = self.srvfile.get('tempfile')
        self.assertEqual(value, 'key value')

    def testGetSecretInFileNotFound(self):
        if False:
            return 10
        value = self.srvfile.get('tempfile2.txt')
        self.assertEqual(value, None)

    @defer.inlineCallbacks
    def testGetSecretInFileNoStrip(self):
        if False:
            while True:
                i = 10
        yield self.srvfile.reconfigService(self.tmp_dir, strip=False)
        value = self.srvfile.get('tempfile.txt')
        self.assertEqual(value, 'key value\n')