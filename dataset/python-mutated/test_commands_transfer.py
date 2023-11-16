from __future__ import absolute_import
from __future__ import print_function
import io
import os
import re
import shutil
import sys
import tarfile
from twisted.internet import defer
from twisted.internet import reactor
from twisted.python import failure
from twisted.python import runtime
from twisted.trial import unittest
from buildbot_worker.commands import transfer
from buildbot_worker.test.fake.remote import FakeRemote
from buildbot_worker.test.util.command import CommandTestMixin

class FakeMasterMethods(object):

    def __init__(self, add_update):
        if False:
            print('Hello World!')
        self.add_update = add_update
        self.delay_write = False
        self.count_writes = False
        self.keep_data = False
        self.write_out_of_space_at = None
        self.delay_read = False
        self.count_reads = False
        self.unpack_fail = False
        self.written = False
        self.read = False
        self.data = b''

    def remote_write(self, data):
        if False:
            return 10
        if self.write_out_of_space_at is not None:
            self.write_out_of_space_at -= len(data)
            if self.write_out_of_space_at <= 0:
                f = failure.Failure(RuntimeError('out of space'))
                return defer.fail(f)
        if self.count_writes:
            self.add_update('write {0}'.format(len(data)))
        elif not self.written:
            self.add_update('write(s)')
            self.written = True
        if self.keep_data:
            self.data += data
        if self.delay_write:
            d = defer.Deferred()
            reactor.callLater(0.01, d.callback, None)
            return d
        return None

    def remote_read(self, length):
        if False:
            i = 10
            return i + 15
        if self.count_reads:
            self.add_update('read {0}'.format(length))
        elif not self.read:
            self.add_update('read(s)')
            self.read = True
        if not self.data:
            return ''
        (_slice, self.data) = (self.data[:length], self.data[length:])
        if self.delay_read:
            d = defer.Deferred()
            reactor.callLater(0.01, d.callback, _slice)
            return d
        return _slice

    def remote_unpack(self):
        if False:
            for i in range(10):
                print('nop')
        self.add_update('unpack')
        if self.unpack_fail:
            return defer.fail(failure.Failure(RuntimeError('out of space')))
        return None

    def remote_utime(self, accessed_modified):
        if False:
            for i in range(10):
                print('nop')
        self.add_update('utime - {0}'.format(accessed_modified[0]))

    def remote_close(self):
        if False:
            for i in range(10):
                print('nop')
        self.add_update('close')

class TestUploadFile(CommandTestMixin, unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.setUpCommand()
        self.fakemaster = FakeMasterMethods(self.add_update)
        self.datadir = os.path.join(self.basedir, 'workdir')
        if os.path.exists(self.datadir):
            shutil.rmtree(self.datadir)
        os.makedirs(self.datadir)
        self.datafile = os.path.join(self.datadir, 'data')
        with open(self.datafile, mode='wb') as f:
            f.write(b'this is some data\n' * 10)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tearDownCommand()
        if os.path.exists(self.datadir):
            shutil.rmtree(self.datadir)

    @defer.inlineCallbacks
    def test_simple(self):
        if False:
            while True:
                i = 10
        self.fakemaster.count_writes = True
        path = os.path.join(self.basedir, 'workdir', os.path.expanduser('data'))
        self.make_command(transfer.WorkerFileUploadCommand, {'path': path, 'writer': FakeRemote(self.fakemaster), 'maxsize': 1000, 'blocksize': 64, 'keepstamp': False})
        yield self.run_command()
        self.assertUpdates([('header', 'sending {0}\n'.format(self.datafile)), 'write 64', 'write 64', 'write 52', 'close', ('rc', 0)])

    @defer.inlineCallbacks
    def test_truncated(self):
        if False:
            i = 10
            return i + 15
        self.fakemaster.count_writes = True
        path = os.path.join(self.basedir, 'workdir', os.path.expanduser('data'))
        self.make_command(transfer.WorkerFileUploadCommand, {'path': path, 'writer': FakeRemote(self.fakemaster), 'maxsize': 100, 'blocksize': 64, 'keepstamp': False})
        yield self.run_command()
        self.assertUpdates([('header', 'sending {0}\n'.format(self.datafile)), 'write 64', 'write 36', 'close', ('rc', 1), ('stderr', "Maximum filesize reached, truncating file '{0}'".format(self.datafile))])

    @defer.inlineCallbacks
    def test_missing(self):
        if False:
            for i in range(10):
                print('nop')
        path = os.path.join(self.basedir, 'workdir', os.path.expanduser('data-nosuch'))
        self.make_command(transfer.WorkerFileUploadCommand, {'path': path, 'writer': FakeRemote(self.fakemaster), 'maxsize': 100, 'blocksize': 64, 'keepstamp': False})
        yield self.run_command()
        df = self.datafile + '-nosuch'
        self.assertUpdates([('header', 'sending {0}\n'.format(df)), 'close', ('rc', 1), ('stderr', "Cannot open file '{0}' for upload".format(df))])

    @defer.inlineCallbacks
    def test_out_of_space(self):
        if False:
            i = 10
            return i + 15
        self.fakemaster.write_out_of_space_at = 70
        self.fakemaster.count_writes = True
        path = os.path.join(self.basedir, 'workdir', os.path.expanduser('data'))
        self.make_command(transfer.WorkerFileUploadCommand, {'path': path, 'writer': FakeRemote(self.fakemaster), 'maxsize': 1000, 'blocksize': 64, 'keepstamp': False})
        yield self.assertFailure(self.run_command(), RuntimeError)
        self.assertUpdates([('header', 'sending {0}\n'.format(self.datafile)), 'write 64', 'close', ('rc', 1)])

    @defer.inlineCallbacks
    def test_interrupted(self):
        if False:
            for i in range(10):
                print('nop')
        self.fakemaster.delay_write = True
        path = os.path.join(self.basedir, 'workdir', os.path.expanduser('data'))
        self.make_command(transfer.WorkerFileUploadCommand, {'path': path, 'writer': FakeRemote(self.fakemaster), 'maxsize': 100, 'blocksize': 2, 'keepstamp': False})
        d = self.run_command()
        interrupt_d = defer.Deferred()
        reactor.callLater(0.01, interrupt_d.callback, None)

        def do_interrupt(_):
            if False:
                while True:
                    i = 10
            return self.cmd.interrupt()
        interrupt_d.addCallback(do_interrupt)
        yield defer.DeferredList([d, interrupt_d])
        self.assertUpdates([('header', 'sending {0}\n'.format(self.datafile)), 'write(s)', 'close', ('rc', 1)])

    @defer.inlineCallbacks
    def test_timestamp(self):
        if False:
            while True:
                i = 10
        self.fakemaster.count_writes = True
        timestamp = (os.path.getatime(self.datafile), os.path.getmtime(self.datafile))
        path = os.path.join(self.basedir, 'workdir', os.path.expanduser('data'))
        self.make_command(transfer.WorkerFileUploadCommand, {'path': path, 'writer': FakeRemote(self.fakemaster), 'maxsize': 1000, 'blocksize': 64, 'keepstamp': True})
        yield self.run_command()
        self.assertUpdates([('header', 'sending {0}\n'.format(self.datafile)), 'write 64', 'write 64', 'write 52', 'close', 'utime - {0}'.format(timestamp[0]), ('rc', 0)])

class TestWorkerDirectoryUpload(CommandTestMixin, unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.setUpCommand()
        self.fakemaster = FakeMasterMethods(self.add_update)
        self.datadir = os.path.join(self.basedir, 'workdir', 'data')
        if os.path.exists(self.datadir):
            shutil.rmtree(self.datadir)
        os.makedirs(self.datadir)
        with open(os.path.join(self.datadir, 'aa'), mode='wb') as f:
            f.write(b'lots of a' * 100)
        with open(os.path.join(self.datadir, 'bb'), mode='wb') as f:
            f.write(b'and a little b' * 17)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tearDownCommand()
        if os.path.exists(self.datadir):
            shutil.rmtree(self.datadir)

    @defer.inlineCallbacks
    def test_simple(self, compress=None):
        if False:
            while True:
                i = 10
        self.fakemaster.keep_data = True
        path = os.path.join(self.basedir, 'workdir', os.path.expanduser('data'))
        self.make_command(transfer.WorkerDirectoryUploadCommand, {'workdir': 'workdir', 'path': path, 'writer': FakeRemote(self.fakemaster), 'maxsize': None, 'blocksize': 512, 'compress': compress})
        yield self.run_command()
        self.assertUpdates([('header', 'sending {0}\n'.format(self.datadir)), 'write(s)', 'unpack', ('rc', 0)])
        f = io.BytesIO(self.fakemaster.data)
        a = tarfile.open(fileobj=f, name='check.tar', mode='r')
        exp_names = ['.', 'aa', 'bb']
        got_names = [n.rstrip('/') for n in a.getnames()]
        got_names = sorted([n or '.' for n in got_names])
        self.assertEqual(got_names, exp_names, 'expected archive contents')
        a.close()
        f.close()

    def test_simple_bz2(self):
        if False:
            while True:
                i = 10
        return self.test_simple('bz2')

    def test_simple_gz(self):
        if False:
            while True:
                i = 10
        return self.test_simple('gz')
    if sys.version_info[:2] <= (2, 4):
        test_simple_bz2.skip = 'bz2 stream decompression not supported on Python-2.4'

    @defer.inlineCallbacks
    def test_out_of_space_unpack(self):
        if False:
            i = 10
            return i + 15
        self.fakemaster.keep_data = True
        self.fakemaster.unpack_fail = True
        path = os.path.join(self.basedir, 'workdir', os.path.expanduser('data'))
        self.make_command(transfer.WorkerDirectoryUploadCommand, {'path': path, 'workersrc': 'data', 'writer': FakeRemote(self.fakemaster), 'maxsize': None, 'blocksize': 512, 'compress': None})
        yield self.assertFailure(self.run_command(), RuntimeError)
        self.assertUpdates([('header', 'sending {0}\n'.format(self.datadir)), 'write(s)', 'unpack', ('rc', 1)])

class TestWorkerDirectoryUploadNoDir(CommandTestMixin, unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setUpCommand()
        self.fakemaster = FakeMasterMethods(self.add_update)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tearDownCommand()

    @defer.inlineCallbacks
    def test_directory_not_available(self):
        if False:
            for i in range(10):
                print('nop')
        path = os.path.join(self.basedir, 'workdir', os.path.expanduser('data'))
        self.make_command(transfer.WorkerDirectoryUploadCommand, {'path': path, 'workersrc': 'data', 'writer': FakeRemote(self.fakemaster), 'maxsize': None, 'blocksize': 512, 'compress': None})
        yield self.run_command()
        updates = self.get_updates()
        self.assertEqual(updates[0], ('rc', 1))
        self.assertEqual(updates[1][0], 'stderr')
        error_msg = updates[1][1]
        pattern = re.compile('Cannot read directory (.*?) for upload: (.*?)')
        match = pattern.match(error_msg)
        self.assertNotEqual(error_msg, match)

class TestDownloadFile(CommandTestMixin, unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.setUpCommand()
        self.fakemaster = FakeMasterMethods(self.add_update)
        if os.path.exists(self.basedir):
            shutil.rmtree(self.basedir)
        os.makedirs(self.basedir)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.tearDownCommand()
        if os.path.exists(self.basedir):
            shutil.rmtree(self.basedir)

    @defer.inlineCallbacks
    def test_simple(self):
        if False:
            print('Hello World!')
        self.fakemaster.count_reads = True
        self.fakemaster.data = test_data = b'1234' * 13
        assert len(self.fakemaster.data) == 52
        path = os.path.join(self.basedir, os.path.expanduser('data'))
        self.make_command(transfer.WorkerFileDownloadCommand, {'path': path, 'reader': FakeRemote(self.fakemaster), 'maxsize': None, 'blocksize': 32, 'mode': 511})
        yield self.run_command()
        self.assertUpdates(['read 32', 'read 32', 'read 32', 'close', ('rc', 0)])
        datafile = os.path.join(self.basedir, 'data')
        self.assertTrue(os.path.exists(datafile))
        with open(datafile, mode='rb') as f:
            datafileContent = f.read()
        self.assertEqual(datafileContent, test_data)
        if runtime.platformType != 'win32':
            self.assertEqual(os.stat(datafile).st_mode & 511, 511)

    @defer.inlineCallbacks
    def test_mkdir(self):
        if False:
            i = 10
            return i + 15
        self.fakemaster.data = test_data = b'hi'
        path = os.path.join(self.basedir, 'workdir', os.path.expanduser(os.path.join('subdir', 'data')))
        self.make_command(transfer.WorkerFileDownloadCommand, {'path': path, 'reader': FakeRemote(self.fakemaster), 'maxsize': None, 'blocksize': 32, 'mode': 511})
        yield self.run_command()
        self.assertUpdates(['read(s)', 'close', ('rc', 0)])
        datafile = os.path.join(self.basedir, 'workdir', 'subdir', 'data')
        self.assertTrue(os.path.exists(datafile))
        with open(datafile, mode='rb') as f:
            datafileContent = f.read()
        self.assertEqual(datafileContent, test_data)

    @defer.inlineCallbacks
    def test_failure(self):
        if False:
            print('Hello World!')
        self.fakemaster.data = 'hi'
        os.makedirs(os.path.join(self.basedir, 'dir'))
        path = os.path.join(self.basedir, os.path.expanduser('dir'))
        self.make_command(transfer.WorkerFileDownloadCommand, {'path': path, 'reader': FakeRemote(self.fakemaster), 'maxsize': None, 'blocksize': 32, 'mode': 511})
        yield self.run_command()
        self.assertUpdates(['close', ('rc', 1), ('stderr', "Cannot open file '{0}' for download".format(os.path.join(self.basedir, 'dir')))])

    @defer.inlineCallbacks
    def test_truncated(self):
        if False:
            while True:
                i = 10
        self.fakemaster.data = test_data = b'tenchars--' * 10
        path = os.path.join(self.basedir, os.path.expanduser('data'))
        self.make_command(transfer.WorkerFileDownloadCommand, {'path': path, 'reader': FakeRemote(self.fakemaster), 'maxsize': 50, 'blocksize': 32, 'mode': 511})
        yield self.run_command()
        self.assertUpdates(['read(s)', 'close', ('rc', 1), ('stderr', "Maximum filesize reached, truncating file '{0}'".format(os.path.join(self.basedir, 'data')))])
        datafile = os.path.join(self.basedir, 'data')
        self.assertTrue(os.path.exists(datafile))
        with open(datafile, mode='rb') as f:
            data = f.read()
        self.assertEqual(data, test_data[:50])

    @defer.inlineCallbacks
    def test_interrupted(self):
        if False:
            while True:
                i = 10
        self.fakemaster.data = b'tenchars--' * 100
        self.fakemaster.delay_read = True
        path = os.path.join(self.basedir, os.path.expanduser('data'))
        self.make_command(transfer.WorkerFileDownloadCommand, {'path': path, 'reader': FakeRemote(self.fakemaster), 'maxsize': 100, 'blocksize': 2, 'mode': 511})
        d = self.run_command()
        interrupt_d = defer.Deferred()
        reactor.callLater(0.01, interrupt_d.callback, None)

        def do_interrupt(_):
            if False:
                i = 10
                return i + 15
            return self.cmd.interrupt()
        interrupt_d.addCallback(do_interrupt)
        yield defer.DeferredList([d, interrupt_d])
        self.assertUpdates(['read(s)', 'close', ('rc', 1)])