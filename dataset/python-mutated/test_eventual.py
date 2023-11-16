from twisted.internet import defer
from twisted.python import log
from twisted.trial import unittest
from buildbot.util import eventual

class Eventually(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        eventual._theSimpleQueue = eventual._SimpleCallQueue()
        self.old_log_err = log.err
        self.results = []

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        log.err = self.old_log_err
        return eventual.flushEventualQueue()

    def cb(self, *args, **kwargs):
        if False:
            print('Hello World!')
        r = args
        if kwargs:
            r = r + (kwargs,)
        self.results.append(r)

    @defer.inlineCallbacks
    def assertResults(self, exp):
        if False:
            return 10
        yield eventual.flushEventualQueue()
        self.assertEqual(self.results, exp)

    def test_eventually_calls(self):
        if False:
            for i in range(10):
                print('nop')
        eventual.eventually(self.cb)
        return self.assertResults([()])

    def test_eventually_args(self):
        if False:
            i = 10
            return i + 15
        eventual.eventually(self.cb, 1, 2, a='a')
        return self.assertResults([(1, 2, {'a': 'a'})])

    def test_eventually_err(self):
        if False:
            i = 10
            return i + 15
        log.err = lambda : self.results.append('err')

        def cb_fails():
            if False:
                while True:
                    i = 10
            raise RuntimeError('should not cause test failure')
        eventual.eventually(cb_fails)
        return self.assertResults(['err'])

    def test_eventually_butNotNow(self):
        if False:
            i = 10
            return i + 15
        eventual.eventually(self.cb, 1)
        self.assertFalse(self.results)
        return self.assertResults([(1,)])

    def test_eventually_order(self):
        if False:
            return 10
        eventual.eventually(self.cb, 1)
        eventual.eventually(self.cb, 2)
        eventual.eventually(self.cb, 3)
        return self.assertResults([(1,), (2,), (3,)])

    def test_flush_waitForChainedEventuallies(self):
        if False:
            return 10

        def chain(n):
            if False:
                for i in range(10):
                    print('nop')
            self.results.append(n)
            if n <= 0:
                return
            eventual.eventually(chain, n - 1)
        chain(3)
        return self.assertResults([3, 2, 1, 0])

    def test_flush_waitForTreeEventuallies(self):
        if False:
            print('Hello World!')

        def tree(n):
            if False:
                return 10
            self.results.append(n)
            if n <= 0:
                return
            eventual.eventually(tree, n - 1)
            eventual.eventually(tree, n - 1)
        tree(2)
        return self.assertResults([2, 1, 1, 0, 0, 0, 0])

    def test_flush_duringTurn(self):
        if False:
            print('Hello World!')
        testd = defer.Deferred()

        def cb():
            if False:
                return 10
            d = eventual.flushEventualQueue()
            d.addCallback(testd.callback)
        eventual.eventually(cb)
        return testd

    def test_fireEventually_call(self):
        if False:
            return 10
        d = eventual.fireEventually(13)
        d.addCallback(self.cb)
        return self.assertResults([(13,)])