from unittest import mock
from twisted.internet import defer
from twisted.python import failure
from twisted.trial import unittest
from buildbot.mq import base

class QueueRef(unittest.TestCase):

    def test_success(self):
        if False:
            print('Hello World!')
        cb = mock.Mock(name='cb')
        qref = base.QueueRef(cb)
        qref.invoke('rk', 'd')
        cb.assert_called_with('rk', 'd')

    def test_success_deferred(self):
        if False:
            print('Hello World!')
        cb = mock.Mock(name='cb')
        cb.return_value = defer.succeed(None)
        qref = base.QueueRef(cb)
        qref.invoke('rk', 'd')
        cb.assert_called_with('rk', 'd')

    def test_exception(self):
        if False:
            i = 10
            return i + 15
        cb = mock.Mock(name='cb')
        cb.side_effect = RuntimeError('oh noes!')
        qref = base.QueueRef(cb)
        qref.invoke('rk', 'd')
        cb.assert_called_with('rk', 'd')
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)

    def test_failure(self):
        if False:
            i = 10
            return i + 15
        cb = mock.Mock(name='cb')
        cb.return_value = defer.fail(failure.Failure(RuntimeError('oh noes!')))
        qref = base.QueueRef(cb)
        qref.invoke('rk', 'd')
        cb.assert_called_with('rk', 'd')
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)