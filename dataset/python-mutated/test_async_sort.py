from twisted.internet import defer
from twisted.python import log
from twisted.trial import unittest
from buildbot.test.util.logging import LoggingMixin
from buildbot.util.async_sort import async_sort

class AsyncSort(unittest.TestCase, LoggingMixin):

    def setUp(self) -> None:
        if False:
            return 10
        self.setUpLogging()
        return super().setUp()

    @defer.inlineCallbacks
    def test_sync_call(self):
        if False:
            return 10
        l = ['b', 'c', 'a']
        yield async_sort(l, lambda x: x)
        return self.assertEqual(l, ['a', 'b', 'c'])

    @defer.inlineCallbacks
    def test_async_call(self):
        if False:
            print('Hello World!')
        l = ['b', 'c', 'a']
        yield async_sort(l, defer.succeed)
        self.assertEqual(l, ['a', 'b', 'c'])

    @defer.inlineCallbacks
    def test_async_fail(self):
        if False:
            return 10
        l = ['b', 'c', 'a']
        self.patch(log, 'err', lambda f: None)

        class SortFail(Exception):
            pass
        with self.assertRaises(SortFail):
            yield async_sort(l, lambda x: defer.succeed(x) if x != 'a' else defer.fail(SortFail('ono')))
        self.assertEqual(len(self.flushLoggedErrors(SortFail)), 1)
        self.assertEqual(l, ['b', 'c', 'a'])