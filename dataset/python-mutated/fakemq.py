from twisted.internet import defer
from buildbot.mq import base
from buildbot.test.util import validation
from buildbot.util import deferwaiter
from buildbot.util import service
from buildbot.util import tuplematch

class FakeMQConnector(service.AsyncMultiService, base.MQBase):
    verifyMessages = True

    def __init__(self, testcase):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.testcase = testcase
        self.setup_called = False
        self.productions = []
        self.qrefs = []
        self._deferwaiter = deferwaiter.DeferWaiter()

    @defer.inlineCallbacks
    def stopService(self):
        if False:
            for i in range(10):
                print('nop')
        yield self._deferwaiter.wait()
        yield super().stopService()

    def setup(self):
        if False:
            while True:
                i = 10
        self.setup_called = True
        return defer.succeed(None)

    def produce(self, routingKey, data):
        if False:
            return 10
        self.testcase.assertIsInstance(routingKey, tuple)
        if any((not isinstance(k, str) for k in routingKey)):
            raise AssertionError(f'{routingKey} is not all str')
        self.productions.append((routingKey, data))

    def callConsumer(self, routingKey, msg):
        if False:
            for i in range(10):
                print('nop')
        if self.verifyMessages:
            validation.verifyMessage(self.testcase, routingKey, msg)
        matched = False
        for q in self.qrefs:
            if tuplematch.matchTuple(routingKey, q.filter):
                matched = True
                self._deferwaiter.add(q.callback(routingKey, msg))
        if not matched:
            raise AssertionError('no consumer found')

    def startConsuming(self, callback, filter, persistent_name=None):
        if False:
            print('Hello World!')
        if any((not isinstance(k, str) and k is not None for k in filter)):
            raise AssertionError(f'{filter} is not a filter')
        qref = FakeQueueRef()
        qref.qrefs = self.qrefs
        qref.callback = callback
        qref.filter = filter
        qref.persistent_name = persistent_name
        self.qrefs.append(qref)
        return defer.succeed(qref)

    def clearProductions(self):
        if False:
            for i in range(10):
                print('nop')
        'Clear out the cached productions'
        self.productions = []

    def assertProductions(self, exp, orderMatters=True):
        if False:
            for i in range(10):
                print('nop')
        'Assert that the given messages have been produced, then flush the\n        list of produced messages.\n\n        If C{orderMatters} is false, then the messages are sorted first; use\n        this in cases where the messages must all be produced, but the order is\n        not specified.\n        '
        if orderMatters:
            self.testcase.assertEqual(self.productions, exp)
        else:
            self.testcase.assertEqual(sorted(self.productions), sorted(exp))
        self.productions = []

class FakeQueueRef:

    def stopConsuming(self):
        if False:
            print('Hello World!')
        if self in self.qrefs:
            self.qrefs.remove(self)