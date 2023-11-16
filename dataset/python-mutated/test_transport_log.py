"""Tests for log+ transport decorator."""
from bzrlib import transport
from bzrlib.tests import TestCaseWithMemoryTransport
from bzrlib.trace import mutter
from bzrlib.transport.log import TransportLogDecorator

class TestTransportLog(TestCaseWithMemoryTransport):

    def test_log_transport(self):
        if False:
            print('Hello World!')
        base_transport = self.get_transport('')
        logging_transport = transport.get_transport('log+' + base_transport.base)
        mutter('where are you?')
        logging_transport.mkdir('subdir')
        log = self.get_log()
        self.assertContainsRe(log, 'mkdir memory\\+\\d+://.*subdir')
        self.assertContainsRe(log, '  --> None')
        self.assertTrue(logging_transport.has('subdir'))
        self.assertTrue(base_transport.has('subdir'))

    def test_log_readv(self):
        if False:
            for i in range(10):
                print('nop')
        base_transport = DummyReadvTransport()
        logging_transport = TransportLogDecorator('log+dummy:///', _decorated=base_transport)
        result = base_transport.readv('foo', [(0, 10)])
        self.assertTrue(getattr(result, 'next'))
        result = logging_transport.readv('foo', [(0, 10)])
        self.assertTrue(getattr(result, 'next'))
        self.assertEqual(list(result), [(0, 'abcdefghij')])

class DummyReadvTransport(object):
    base = 'dummy:///'

    def readv(self, filename, offset_length_pairs):
        if False:
            i = 10
            return i + 15
        yield (0, 'abcdefghij')

    def abspath(self, path):
        if False:
            while True:
                i = 10
        return self.base + path