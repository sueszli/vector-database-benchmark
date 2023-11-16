"""
Tests for Perspective Broker module.

TODO: update protocol level tests to use new connection API, leaving
only specific tests for old API.
"""
import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest

class Dummy(pb.Viewable):

    def view_doNothing(self, user):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(user, DummyPerspective):
            return 'hello world!'
        else:
            return 'goodbye, cruel world!'

class DummyPerspective(pb.Avatar):
    """
    An L{IPerspective} avatar which will be used in some tests.
    """

    def perspective_getDummyViewPoint(self):
        if False:
            print('Hello World!')
        return Dummy()

@implementer(portal.IRealm)
class DummyRealm:

    def requestAvatar(self, avatarId, mind, *interfaces):
        if False:
            i = 10
            return i + 15
        for iface in interfaces:
            if iface is pb.IPerspective:
                return (iface, DummyPerspective(avatarId), lambda : None)

class IOPump:
    """
    Utility to pump data between clients and servers for protocol testing.

    Perhaps this is a utility worthy of being in protocol.py?
    """

    def __init__(self, client, server, clientIO, serverIO):
        if False:
            print('Hello World!')
        self.client = client
        self.server = server
        self.clientIO = clientIO
        self.serverIO = serverIO

    def flush(self):
        if False:
            print('Hello World!')
        "\n        Pump until there is no more input or output or until L{stop} is called.\n        This does not run any timers, so don't use it with any code that calls\n        reactor.callLater.\n        "
        self._stop = False
        timeout = time.time() + 5
        while not self._stop and self.pump():
            if time.time() > timeout:
                return

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Stop a running L{flush} operation, even if data remains to be\n        transferred.\n        '
        self._stop = True

    def pump(self):
        if False:
            return 10
        '\n        Move data back and forth.\n\n        Returns whether any data was moved.\n        '
        self.clientIO.seek(0)
        self.serverIO.seek(0)
        cData = self.clientIO.read()
        sData = self.serverIO.read()
        self.clientIO.seek(0)
        self.serverIO.seek(0)
        self.clientIO.truncate()
        self.serverIO.truncate()
        self.client.transport._checkProducer()
        self.server.transport._checkProducer()
        for byte in iterbytes(cData):
            self.server.dataReceived(byte)
        for byte in iterbytes(sData):
            self.client.dataReceived(byte)
        if cData or sData:
            return 1
        else:
            return 0

def connectServerAndClient(test, clientFactory, serverFactory):
    if False:
        print('Hello World!')
    '\n    Create a server and a client and connect the two with an\n    L{IOPump}.\n\n    @param test: the test case where the client and server will be\n        used.\n    @type test: L{twisted.trial.unittest.TestCase}\n\n    @param clientFactory: The factory that creates the client object.\n    @type clientFactory: L{twisted.spread.pb.PBClientFactory}\n\n    @param serverFactory: The factory that creates the server object.\n    @type serverFactory: L{twisted.spread.pb.PBServerFactory}\n\n    @return: a 3-tuple of (client, server, pump)\n    @rtype: (L{twisted.spread.pb.Broker}, L{twisted.spread.pb.Broker},\n        L{IOPump})\n    '
    addr = ('127.0.0.1',)
    clientBroker = clientFactory.buildProtocol(addr)
    serverBroker = serverFactory.buildProtocol(addr)
    clientTransport = StringIO()
    serverTransport = StringIO()
    clientBroker.makeConnection(protocol.FileWrapper(clientTransport))
    serverBroker.makeConnection(protocol.FileWrapper(serverTransport))
    pump = IOPump(clientBroker, serverBroker, clientTransport, serverTransport)

    def maybeDisconnect(broker):
        if False:
            for i in range(10):
                print('nop')
        if not broker.disconnected:
            broker.connectionLost(failure.Failure(main.CONNECTION_DONE))

    def disconnectClientFactory():
        if False:
            i = 10
            return i + 15
        clientFactory.clientConnectionLost(connector=None, reason=failure.Failure(main.CONNECTION_DONE))
    test.addCleanup(maybeDisconnect, clientBroker)
    test.addCleanup(maybeDisconnect, serverBroker)
    test.addCleanup(disconnectClientFactory)
    pump.pump()
    return (clientBroker, serverBroker, pump)

class _ReconnectingFakeConnectorState:
    """
    Manages connection notifications for a
    L{_ReconnectingFakeConnector} instance.

    @ivar notifications: pending L{Deferreds} that will fire when the
        L{_ReconnectingFakeConnector}'s connect method is called
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.notifications = deque()

    def notifyOnConnect(self):
        if False:
            return 10
        "\n        Connection notification.\n\n        @return: A L{Deferred} that fires when this instance's\n            L{twisted.internet.interfaces.IConnector.connect} method\n            is called.\n        @rtype: L{Deferred}\n        "
        notifier = Deferred()
        self.notifications.appendleft(notifier)
        return notifier

    def notifyAll(self):
        if False:
            i = 10
            return i + 15
        '\n        Fire all pending notifications.\n        '
        while self.notifications:
            self.notifications.pop().callback(self)

class _ReconnectingFakeConnector(_FakeConnector):
    """
    A fake L{IConnector} that can fire L{Deferred}s when its
    C{connect} method is called.
    """

    def __init__(self, address, state):
        if False:
            return 10
        "\n        @param address: An L{IAddress} provider that represents this\n            connector's destination.\n        @type address: An L{IAddress} provider.\n\n        @param state: The state instance\n        @type state: L{_ReconnectingFakeConnectorState}\n        "
        super().__init__(address)
        self._state = state

    def connect(self):
        if False:
            while True:
                i = 10
        '\n        A C{connect} implementation that calls C{reconnectCallback}\n        '
        super().connect()
        self._state.notifyAll()

def connectedServerAndClient(test, realm=None):
    if False:
        return 10
    '\n    Connect a client and server L{Broker} together with an L{IOPump}\n\n    @param realm: realm to use, defaulting to a L{DummyRealm}\n\n    @returns: a 3-tuple (client, server, pump).\n    '
    realm = realm or DummyRealm()
    checker = checkers.InMemoryUsernamePasswordDatabaseDontUse(guest=b'guest')
    serverFactory = pb.PBServerFactory(portal.Portal(realm, [checker]))
    clientFactory = pb.PBClientFactory()
    return connectServerAndClient(test, clientFactory, serverFactory)

class SimpleRemote(pb.Referenceable):

    def remote_thunk(self, arg):
        if False:
            while True:
                i = 10
        self.arg = arg
        return arg + 1

    def remote_knuth(self, arg):
        if False:
            i = 10
            return i + 15
        raise Exception()

class NestedRemote(pb.Referenceable):

    def remote_getSimple(self):
        if False:
            print('Hello World!')
        return SimpleRemote()

class SimpleCopy(pb.Copyable):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = 1
        self.y = {'Hello': 'World'}
        self.z = ['test']

class SimpleLocalCopy(pb.RemoteCopy):
    pass
pb.setUnjellyableForClass(SimpleCopy, SimpleLocalCopy)

class SimpleFactoryCopy(pb.Copyable):
    """
    @cvar allIDs: hold every created instances of this class.
    @type allIDs: C{dict}
    """
    allIDs: Dict[int, 'SimpleFactoryCopy'] = {}

    def __init__(self, id):
        if False:
            for i in range(10):
                print('nop')
        self.id = id
        SimpleFactoryCopy.allIDs[id] = self

def createFactoryCopy(state):
    if False:
        for i in range(10):
            print('nop')
    '\n    Factory of L{SimpleFactoryCopy}, getting a created instance given the\n    C{id} found in C{state}.\n    '
    stateId = state.get('id', None)
    if stateId is None:
        raise RuntimeError(f"factory copy state has no 'id' member {repr(state)}")
    if stateId not in SimpleFactoryCopy.allIDs:
        raise RuntimeError(f'factory class has no ID: {SimpleFactoryCopy.allIDs}')
    inst = SimpleFactoryCopy.allIDs[stateId]
    if not inst:
        raise RuntimeError('factory method found no object with id')
    return inst
pb.setUnjellyableFactoryForClass(SimpleFactoryCopy, createFactoryCopy)

class NestedCopy(pb.Referenceable):

    def remote_getCopy(self):
        if False:
            i = 10
            return i + 15
        return SimpleCopy()

    def remote_getFactory(self, value):
        if False:
            i = 10
            return i + 15
        return SimpleFactoryCopy(value)

class SimpleCache(pb.Cacheable):

    def __init___(self):
        if False:
            for i in range(10):
                print('nop')
        self.x = 1
        self.y = {'Hello': 'World'}
        self.z = ['test']

class NestedComplicatedCache(pb.Referenceable):

    def __init__(self):
        if False:
            return 10
        self.c = VeryVeryComplicatedCacheable()

    def remote_getCache(self):
        if False:
            while True:
                i = 10
        return self.c

class VeryVeryComplicatedCacheable(pb.Cacheable):

    def __init__(self):
        if False:
            print('Hello World!')
        self.x = 1
        self.y = 2
        self.foo = 3

    def setFoo4(self):
        if False:
            while True:
                i = 10
        self.foo = 4
        self.observer.callRemote('foo', 4)

    def getStateToCacheAndObserveFor(self, perspective, observer):
        if False:
            i = 10
            return i + 15
        self.observer = observer
        return {'x': self.x, 'y': self.y, 'foo': self.foo}

    def stoppedObserving(self, perspective, observer):
        if False:
            print('Hello World!')
        log.msg('stopped observing')
        observer.callRemote('end')
        if observer == self.observer:
            self.observer = None

class RatherBaroqueCache(pb.RemoteCache):

    def observe_foo(self, newFoo):
        if False:
            for i in range(10):
                print('nop')
        self.foo = newFoo

    def observe_end(self):
        if False:
            for i in range(10):
                print('nop')
        log.msg('the end of things')
pb.setUnjellyableForClass(VeryVeryComplicatedCacheable, RatherBaroqueCache)

class SimpleLocalCache(pb.RemoteCache):

    def setCopyableState(self, state):
        if False:
            return 10
        self.__dict__.update(state)

    def checkMethod(self):
        if False:
            i = 10
            return i + 15
        return self.check

    def checkSelf(self):
        if False:
            return 10
        return self

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        return 1
pb.setUnjellyableForClass(SimpleCache, SimpleLocalCache)

class NestedCache(pb.Referenceable):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.x = SimpleCache()

    def remote_getCache(self):
        if False:
            return 10
        return [self.x, self.x]

    def remote_putCache(self, cache):
        if False:
            while True:
                i = 10
        return self.x is cache

class Observable(pb.Referenceable):

    def __init__(self):
        if False:
            print('Hello World!')
        self.observers = []

    def remote_observe(self, obs):
        if False:
            print('Hello World!')
        self.observers.append(obs)

    def remote_unobserve(self, obs):
        if False:
            return 10
        self.observers.remove(obs)

    def notify(self, obj):
        if False:
            i = 10
            return i + 15
        for observer in self.observers:
            observer.callRemote('notify', self, obj)

class DeferredRemote(pb.Referenceable):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.run = 0

    def runMe(self, arg):
        if False:
            for i in range(10):
                print('nop')
        self.run = arg
        return arg + 1

    def dontRunMe(self, arg):
        if False:
            i = 10
            return i + 15
        assert 0, "shouldn't have been run!"

    def remote_doItLater(self):
        if False:
            return 10
        '\n        Return a L{Deferred} to be fired on client side. When fired,\n        C{self.runMe} is called.\n        '
        d = Deferred()
        d.addCallbacks(self.runMe, self.dontRunMe)
        self.d = d
        return d

class Observer(pb.Referenceable):
    notified = 0
    obj = None

    def remote_notify(self, other, obj):
        if False:
            while True:
                i = 10
        self.obj = obj
        self.notified = self.notified + 1
        other.callRemote('unobserve', self)

class NewStyleCopy(pb.Copyable, pb.RemoteCopy):

    def __init__(self, s):
        if False:
            while True:
                i = 10
        self.s = s
pb.setUnjellyableForClass(NewStyleCopy, NewStyleCopy)

class NewStyleCopy2(pb.Copyable, pb.RemoteCopy):
    allocated = 0
    initialized = 0
    value = 1

    def __new__(self):
        if False:
            while True:
                i = 10
        NewStyleCopy2.allocated += 1
        inst = object.__new__(self)
        inst.value = 2
        return inst

    def __init__(self):
        if False:
            while True:
                i = 10
        NewStyleCopy2.initialized += 1
pb.setUnjellyableForClass(NewStyleCopy2, NewStyleCopy2)

class NewStyleCacheCopy(pb.Cacheable, pb.RemoteCache):

    def getStateToCacheAndObserveFor(self, perspective, observer):
        if False:
            i = 10
            return i + 15
        return self.__dict__
pb.setUnjellyableForClass(NewStyleCacheCopy, NewStyleCacheCopy)

class Echoer(pb.Root):

    def remote_echo(self, st):
        if False:
            return 10
        return st

    def remote_echoWithKeywords(self, st, **kw):
        if False:
            i = 10
            return i + 15
        return (st, kw)

class CachedReturner(pb.Root):

    def __init__(self, cache):
        if False:
            for i in range(10):
                print('nop')
        self.cache = cache

    def remote_giveMeCache(self, st):
        if False:
            for i in range(10):
                print('nop')
        return self.cache

class NewStyleTests(unittest.SynchronousTestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        '\n        Create a pb server using L{Echoer} protocol and connect a client to it.\n        '
        self.serverFactory = pb.PBServerFactory(Echoer())
        clientFactory = pb.PBClientFactory()
        (client, self.server, self.pump) = connectServerAndClient(test=self, clientFactory=clientFactory, serverFactory=self.serverFactory)
        self.ref = self.successResultOf(clientFactory.getRootObject())

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        '\n        Close client and server connections, reset values of L{NewStyleCopy2}\n        class variables.\n        '
        NewStyleCopy2.allocated = 0
        NewStyleCopy2.initialized = 0
        NewStyleCopy2.value = 1

    def test_newStyle(self):
        if False:
            return 10
        '\n        Create a new style object, send it over the wire, and check the result.\n        '
        orig = NewStyleCopy('value')
        d = self.ref.callRemote('echo', orig)
        self.pump.flush()

        def cb(res):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIsInstance(res, NewStyleCopy)
            self.assertEqual(res.s, 'value')
            self.assertFalse(res is orig)
        d.addCallback(cb)
        return d

    def test_alloc(self):
        if False:
            i = 10
            return i + 15
        '\n        Send a new style object and check the number of allocations.\n        '
        orig = NewStyleCopy2()
        self.assertEqual(NewStyleCopy2.allocated, 1)
        self.assertEqual(NewStyleCopy2.initialized, 1)
        d = self.ref.callRemote('echo', orig)
        self.pump.flush()

        def cb(res):
            if False:
                print('Hello World!')
            self.assertIsInstance(res, NewStyleCopy2)
            self.assertEqual(res.value, 2)
            self.assertEqual(NewStyleCopy2.allocated, 3)
            self.assertEqual(NewStyleCopy2.initialized, 1)
            self.assertIsNot(res, orig)
        d.addCallback(cb)
        return d

    def test_newStyleWithKeywords(self):
        if False:
            return 10
        '\n        Create a new style object with keywords,\n        send it over the wire, and check the result.\n        '
        orig = NewStyleCopy('value1')
        d = self.ref.callRemote('echoWithKeywords', orig, keyword1='one', keyword2='two')
        self.pump.flush()

        def cb(res):
            if False:
                i = 10
                return i + 15
            self.assertIsInstance(res, tuple)
            self.assertIsInstance(res[0], NewStyleCopy)
            self.assertIsInstance(res[1], dict)
            self.assertEqual(res[0].s, 'value1')
            self.assertIsNot(res[0], orig)
            self.assertEqual(res[1], {'keyword1': 'one', 'keyword2': 'two'})
        d.addCallback(cb)
        return d

class ConnectionNotifyServerFactory(pb.PBServerFactory):
    """
    A server factory which stores the last connection and fires a
    L{Deferred} on connection made. This factory can handle only one
    client connection.

    @ivar protocolInstance: the last protocol instance.
    @type protocolInstance: C{pb.Broker}

    @ivar connectionMade: the deferred fired upon connection.
    @type connectionMade: C{Deferred}
    """
    protocolInstance = None

    def __init__(self, root):
        if False:
            while True:
                i = 10
        '\n        Initialize the factory.\n        '
        pb.PBServerFactory.__init__(self, root)
        self.connectionMade = Deferred()

    def clientConnectionMade(self, protocol):
        if False:
            while True:
                i = 10
        '\n        Store the protocol and fire the connection deferred.\n        '
        self.protocolInstance = protocol
        (d, self.connectionMade) = (self.connectionMade, None)
        if d is not None:
            d.callback(None)

class NewStyleCachedTests(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        Create a pb server using L{CachedReturner} protocol and connect a\n        client to it.\n        '
        self.orig = NewStyleCacheCopy()
        self.orig.s = 'value'
        self.server = reactor.listenTCP(0, ConnectionNotifyServerFactory(CachedReturner(self.orig)))
        clientFactory = pb.PBClientFactory()
        reactor.connectTCP('localhost', self.server.getHost().port, clientFactory)

        def gotRoot(ref):
            if False:
                print('Hello World!')
            self.ref = ref
        d1 = clientFactory.getRootObject().addCallback(gotRoot)
        d2 = self.server.factory.connectionMade
        return gatherResults([d1, d2])

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Close client and server connections.\n        '
        self.server.factory.protocolInstance.transport.loseConnection()
        self.ref.broker.transport.loseConnection()
        return self.server.stopListening()

    def test_newStyleCache(self):
        if False:
            while True:
                i = 10
        '\n        A new-style cacheable object can be retrieved and re-retrieved over a\n        single connection.  The value of an attribute of the cacheable can be\n        accessed on the receiving side.\n        '
        d = self.ref.callRemote('giveMeCache', self.orig)

        def cb(res, again):
            if False:
                print('Hello World!')
            self.assertIsInstance(res, NewStyleCacheCopy)
            self.assertEqual('value', res.s)
            self.assertIsNot(self.orig, res)
            if again:
                self.res = res
                return self.ref.callRemote('giveMeCache', self.orig)
        d.addCallback(cb, True)
        d.addCallback(cb, False)
        return d

class BrokerTests(unittest.TestCase):
    thunkResult = None

    def tearDown(self):
        if False:
            while True:
                i = 10
        try:
            os.unlink('None-None-TESTING.pub')
        except OSError:
            pass

    def thunkErrorBad(self, error):
        if False:
            i = 10
            return i + 15
        self.fail(f'This should cause a return value, not {error}')

    def thunkResultGood(self, result):
        if False:
            return 10
        self.thunkResult = result

    def thunkErrorGood(self, tb):
        if False:
            print('Hello World!')
        pass

    def thunkResultBad(self, result):
        if False:
            i = 10
            return i + 15
        self.fail(f'This should cause an error, not {result}')

    def test_reference(self):
        if False:
            i = 10
            return i + 15
        (c, s, pump) = connectedServerAndClient(test=self)

        class X(pb.Referenceable):

            def remote_catch(self, arg):
                if False:
                    return 10
                self.caught = arg

        class Y(pb.Referenceable):

            def remote_throw(self, a, b):
                if False:
                    return 10
                a.callRemote('catch', b)
        s.setNameForLocal('y', Y())
        y = c.remoteForName('y')
        x = X()
        z = X()
        y.callRemote('throw', x, z)
        pump.pump()
        pump.pump()
        pump.pump()
        self.assertIs(x.caught, z, 'X should have caught Z')
        self.assertEqual(y.remoteMethod('throw'), y.remoteMethod('throw'))

    def test_result(self):
        if False:
            i = 10
            return i + 15
        (c, s, pump) = connectedServerAndClient(test=self)
        for (x, y) in ((c, s), (s, c)):
            foo = SimpleRemote()
            x.setNameForLocal('foo', foo)
            bar = y.remoteForName('foo')
            self.expectedThunkResult = 8
            bar.callRemote('thunk', self.expectedThunkResult - 1).addCallbacks(self.thunkResultGood, self.thunkErrorBad)
            pump.pump()
            pump.pump()
            self.assertEqual(self.thunkResult, self.expectedThunkResult, "result wasn't received.")

    def refcountResult(self, result):
        if False:
            for i in range(10):
                print('nop')
        self.nestedRemote = result

    def test_tooManyRefs(self):
        if False:
            print('Hello World!')
        l = []
        e = []
        (c, s, pump) = connectedServerAndClient(test=self)
        foo = NestedRemote()
        s.setNameForLocal('foo', foo)
        x = c.remoteForName('foo')
        for igno in range(pb.MAX_BROKER_REFS + 10):
            if s.transport.closed or c.transport.closed:
                break
            x.callRemote('getSimple').addCallbacks(l.append, e.append)
            pump.pump()
        expected = pb.MAX_BROKER_REFS - 1
        self.assertTrue(s.transport.closed, 'transport was not closed')
        self.assertEqual(len(l), expected, f'expected {expected} got {len(l)}')

    def test_copy(self):
        if False:
            print('Hello World!')
        (c, s, pump) = connectedServerAndClient(test=self)
        foo = NestedCopy()
        s.setNameForLocal('foo', foo)
        x = c.remoteForName('foo')
        x.callRemote('getCopy').addCallbacks(self.thunkResultGood, self.thunkErrorBad)
        pump.pump()
        pump.pump()
        self.assertEqual(self.thunkResult.x, 1)
        self.assertEqual(self.thunkResult.y['Hello'], 'World')
        self.assertEqual(self.thunkResult.z[0], 'test')

    def test_observe(self):
        if False:
            return 10
        (c, s, pump) = connectedServerAndClient(test=self)
        a = Observable()
        b = Observer()
        s.setNameForLocal('a', a)
        ra = c.remoteForName('a')
        ra.callRemote('observe', b)
        pump.pump()
        a.notify(1)
        pump.pump()
        pump.pump()
        a.notify(10)
        pump.pump()
        pump.pump()
        self.assertIsNotNone(b.obj, "didn't notify")
        self.assertEqual(b.obj, 1, 'notified too much')

    def test_defer(self):
        if False:
            i = 10
            return i + 15
        (c, s, pump) = connectedServerAndClient(test=self)
        d = DeferredRemote()
        s.setNameForLocal('d', d)
        e = c.remoteForName('d')
        pump.pump()
        pump.pump()
        results = []
        e.callRemote('doItLater').addCallback(results.append)
        pump.pump()
        pump.pump()
        self.assertFalse(d.run, 'Deferred method run too early.')
        d.d.callback(5)
        self.assertEqual(d.run, 5, 'Deferred method run too late.')
        pump.pump()
        pump.pump()
        self.assertEqual(results[0], 6, 'Incorrect result.')

    def test_refcount(self):
        if False:
            return 10
        (c, s, pump) = connectedServerAndClient(test=self)
        foo = NestedRemote()
        s.setNameForLocal('foo', foo)
        bar = c.remoteForName('foo')
        bar.callRemote('getSimple').addCallbacks(self.refcountResult, self.thunkErrorBad)
        pump.pump()
        pump.pump()
        rluid = self.nestedRemote.luid
        self.assertIn(rluid, s.localObjects)
        del self.nestedRemote
        if sys.hexversion >= 33554432:
            gc.collect()
        pump.pump()
        pump.pump()
        pump.pump()
        self.assertNotIn(rluid, s.localObjects)

    def test_cache(self):
        if False:
            return 10
        (c, s, pump) = connectedServerAndClient(test=self)
        obj = NestedCache()
        obj2 = NestedComplicatedCache()
        vcc = obj2.c
        s.setNameForLocal('obj', obj)
        s.setNameForLocal('xxx', obj2)
        o2 = c.remoteForName('obj')
        o3 = c.remoteForName('xxx')
        coll = []
        o2.callRemote('getCache').addCallback(coll.append).addErrback(coll.append)
        o2.callRemote('getCache').addCallback(coll.append).addErrback(coll.append)
        complex = []
        o3.callRemote('getCache').addCallback(complex.append)
        o3.callRemote('getCache').addCallback(complex.append)
        pump.flush()
        self.assertEqual(complex[0].x, 1)
        self.assertEqual(complex[0].y, 2)
        self.assertEqual(complex[0].foo, 3)
        vcc.setFoo4()
        pump.flush()
        self.assertEqual(complex[0].foo, 4)
        self.assertEqual(len(coll), 2)
        cp = coll[0][0]
        self.assertIdentical(cp.checkMethod().__self__, cp, 'potential refcounting issue')
        self.assertIdentical(cp.checkSelf(), cp, 'other potential refcounting issue')
        col2 = []
        o2.callRemote('putCache', cp).addCallback(col2.append)
        pump.flush()
        self.assertTrue(col2[0])
        self.assertEqual(o2.remoteMethod('getCache'), o2.remoteMethod('getCache'))
        luid = cp.luid
        baroqueLuid = complex[0].luid
        self.assertIn(luid, s.remotelyCachedObjects, "remote cache doesn't have it")
        del coll
        del cp
        pump.flush()
        del complex
        del col2
        pump.flush()
        if sys.hexversion >= 33554432:
            gc.collect()
        pump.flush()
        self.assertNotIn(luid, s.remotelyCachedObjects, 'Server still had it after GC')
        self.assertNotIn(luid, c.locallyCachedObjects, 'Client still had it after GC')
        self.assertNotIn(baroqueLuid, s.remotelyCachedObjects, 'Server still had complex after GC')
        self.assertNotIn(baroqueLuid, c.locallyCachedObjects, 'Client still had complex after GC')
        self.assertIsNone(vcc.observer, 'observer was not removed')

    def test_publishable(self):
        if False:
            return 10
        try:
            os.unlink('None-None-TESTING.pub')
        except OSError:
            pass
        (c, s, pump) = connectedServerAndClient(test=self)
        foo = GetPublisher()
        s.setNameForLocal('foo', foo)
        bar = c.remoteForName('foo')
        accum = []
        bar.callRemote('getPub').addCallbacks(accum.append, self.thunkErrorBad)
        pump.flush()
        obj = accum.pop()
        self.assertEqual(obj.activateCalled, 1)
        self.assertEqual(obj.isActivated, 1)
        self.assertEqual(obj.yayIGotPublished, 1)
        self.assertEqual(obj._wasCleanWhenLoaded, 0)
        (c, s, pump) = connectedServerAndClient(test=self)
        s.setNameForLocal('foo', foo)
        bar = c.remoteForName('foo')
        bar.callRemote('getPub').addCallbacks(accum.append, self.thunkErrorBad)
        pump.flush()
        obj = accum.pop()
        self.assertEqual(obj._wasCleanWhenLoaded, 1)

    def gotCopy(self, val):
        if False:
            for i in range(10):
                print('nop')
        self.thunkResult = val.id

    def test_factoryCopy(self):
        if False:
            i = 10
            return i + 15
        (c, s, pump) = connectedServerAndClient(test=self)
        ID = 99
        obj = NestedCopy()
        s.setNameForLocal('foo', obj)
        x = c.remoteForName('foo')
        x.callRemote('getFactory', ID).addCallbacks(self.gotCopy, self.thunkResultBad)
        pump.pump()
        pump.pump()
        pump.pump()
        self.assertEqual(self.thunkResult, ID, f'ID not correct on factory object {self.thunkResult}')
bigString = b'helloworld' * 50
callbackArgs = None
callbackKeyword = None

def finishedCallback(*args, **kw):
    if False:
        print('Hello World!')
    global callbackArgs, callbackKeyword
    callbackArgs = args
    callbackKeyword = kw

class Pagerizer(pb.Referenceable):

    def __init__(self, callback, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        (self.callback, self.args, self.kw) = (callback, args, kw)

    def remote_getPages(self, collector):
        if False:
            return 10
        util.StringPager(collector, bigString, 100, self.callback, *self.args, **self.kw)
        self.args = self.kw = None

class FilePagerizer(pb.Referenceable):
    pager = None

    def __init__(self, filename, callback, *args, **kw):
        if False:
            i = 10
            return i + 15
        self.filename = filename
        (self.callback, self.args, self.kw) = (callback, args, kw)

    def remote_getPages(self, collector):
        if False:
            while True:
                i = 10
        self.pager = util.FilePager(collector, open(self.filename, 'rb'), self.callback, *self.args, **self.kw)
        self.args = self.kw = None

class PagingTests(unittest.TestCase):
    """
    Test pb objects sending data by pages.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        Create a file used to test L{util.FilePager}.\n        '
        self.filename = self.mktemp()
        with open(self.filename, 'wb') as f:
            f.write(bigString)

    def test_pagingWithCallback(self):
        if False:
            while True:
                i = 10
        '\n        Test L{util.StringPager}, passing a callback to fire when all pages\n        are sent.\n        '
        (c, s, pump) = connectedServerAndClient(test=self)
        s.setNameForLocal('foo', Pagerizer(finishedCallback, 'hello', value=10))
        x = c.remoteForName('foo')
        l = []
        util.getAllPages(x, 'getPages').addCallback(l.append)
        while not l:
            pump.pump()
        self.assertEqual(b''.join(l[0]), bigString, 'Pages received not equal to pages sent!')
        self.assertEqual(callbackArgs, ('hello',), 'Completed callback not invoked')
        self.assertEqual(callbackKeyword, {'value': 10}, 'Completed callback not invoked')

    def test_pagingWithoutCallback(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test L{util.StringPager} without a callback.\n        '
        (c, s, pump) = connectedServerAndClient(test=self)
        s.setNameForLocal('foo', Pagerizer(None))
        x = c.remoteForName('foo')
        l = []
        util.getAllPages(x, 'getPages').addCallback(l.append)
        while not l:
            pump.pump()
        self.assertEqual(b''.join(l[0]), bigString, 'Pages received not equal to pages sent!')

    def test_emptyFilePaging(self):
        if False:
            while True:
                i = 10
        '\n        Test L{util.FilePager}, sending an empty file.\n        '
        filenameEmpty = self.mktemp()
        open(filenameEmpty, 'w').close()
        (c, s, pump) = connectedServerAndClient(test=self)
        pagerizer = FilePagerizer(filenameEmpty, None)
        s.setNameForLocal('bar', pagerizer)
        x = c.remoteForName('bar')
        l = []
        util.getAllPages(x, 'getPages').addCallback(l.append)
        ttl = 10
        while not l and ttl > 0:
            pump.pump()
            ttl -= 1
        if not ttl:
            self.fail('getAllPages timed out')
        self.assertEqual(b''.join(l[0]), b'', 'Pages received not equal to pages sent!')

    def test_filePagingWithCallback(self):
        if False:
            i = 10
            return i + 15
        "\n        Test L{util.FilePager}, passing a callback to fire when all pages\n        are sent, and verify that the pager doesn't keep chunks in memory.\n        "
        (c, s, pump) = connectedServerAndClient(test=self)
        pagerizer = FilePagerizer(self.filename, finishedCallback, 'frodo', value=9)
        s.setNameForLocal('bar', pagerizer)
        x = c.remoteForName('bar')
        l = []
        util.getAllPages(x, 'getPages').addCallback(l.append)
        while not l:
            pump.pump()
        self.assertEqual(b''.join(l[0]), bigString, 'Pages received not equal to pages sent!')
        self.assertEqual(callbackArgs, ('frodo',), 'Completed callback not invoked')
        self.assertEqual(callbackKeyword, {'value': 9}, 'Completed callback not invoked')
        self.assertEqual(pagerizer.pager.chunks, [])

    def test_filePagingWithoutCallback(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test L{util.FilePager} without a callback.\n        '
        (c, s, pump) = connectedServerAndClient(test=self)
        pagerizer = FilePagerizer(self.filename, None)
        s.setNameForLocal('bar', pagerizer)
        x = c.remoteForName('bar')
        l = []
        util.getAllPages(x, 'getPages').addCallback(l.append)
        while not l:
            pump.pump()
        self.assertEqual(b''.join(l[0]), bigString, 'Pages received not equal to pages sent!')
        self.assertEqual(pagerizer.pager.chunks, [])

class DumbPublishable(publish.Publishable):

    def getStateToPublish(self):
        if False:
            while True:
                i = 10
        return {'yayIGotPublished': 1}

class DumbPub(publish.RemotePublished):

    def activated(self):
        if False:
            i = 10
            return i + 15
        self.activateCalled = 1

class GetPublisher(pb.Referenceable):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.pub = DumbPublishable('TESTING')

    def remote_getPub(self):
        if False:
            print('Hello World!')
        return self.pub
pb.setUnjellyableForClass(DumbPublishable, DumbPub)

class DisconnectionTests(unittest.TestCase):
    """
    Test disconnection callbacks.
    """

    def error(self, *args):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError(f"I shouldn't have been called: {args}")

    def gotDisconnected(self):
        if False:
            while True:
                i = 10
        '\n        Called on broker disconnect.\n        '
        self.gotCallback = 1

    def objectDisconnected(self, o):
        if False:
            return 10
        '\n        Called on RemoteReference disconnect.\n        '
        self.assertEqual(o, self.remoteObject)
        self.objectCallback = 1

    def test_badSerialization(self):
        if False:
            return 10
        (c, s, pump) = connectedServerAndClient(test=self)
        pump.pump()
        s.setNameForLocal('o', BadCopySet())
        g = c.remoteForName('o')
        l = []
        g.callRemote('setBadCopy', BadCopyable()).addErrback(l.append)
        pump.flush()
        self.assertEqual(len(l), 1)

    def test_disconnection(self):
        if False:
            print('Hello World!')
        (c, s, pump) = connectedServerAndClient(test=self)
        pump.pump()
        s.setNameForLocal('o', SimpleRemote())
        r = c.remoteForName('o')
        pump.pump()
        pump.pump()
        pump.pump()
        c.notifyOnDisconnect(self.error)
        self.assertIn(self.error, c.disconnects)
        c.dontNotifyOnDisconnect(self.error)
        self.assertNotIn(self.error, c.disconnects)
        r.notifyOnDisconnect(self.error)
        self.assertIn(r._disconnected, c.disconnects)
        self.assertIn(self.error, r.disconnectCallbacks)
        r.dontNotifyOnDisconnect(self.error)
        self.assertNotIn(r._disconnected, c.disconnects)
        self.assertNotIn(self.error, r.disconnectCallbacks)
        c.notifyOnDisconnect(self.gotDisconnected)
        r.notifyOnDisconnect(self.objectDisconnected)
        self.remoteObject = r
        c.connectionLost(failure.Failure(main.CONNECTION_DONE))
        self.assertTrue(self.gotCallback)
        self.assertTrue(self.objectCallback)

class FreakOut(Exception):
    pass

class BadCopyable(pb.Copyable):

    def getStateToCopyFor(self, p):
        if False:
            i = 10
            return i + 15
        raise FreakOut()

class BadCopySet(pb.Referenceable):

    def remote_setBadCopy(self, bc):
        if False:
            print('Hello World!')
        return None

class LocalRemoteTest(util.LocalAsRemote):
    reportAllTracebacks = 0

    def sync_add1(self, x):
        if False:
            return 10
        return x + 1

    def async_add(self, x=0, y=1):
        if False:
            for i in range(10):
                print('nop')
        return x + y

    def async_fail(self):
        if False:
            return 10
        raise RuntimeError()

@implementer(pb.IPerspective)
class MyPerspective(pb.Avatar):
    """
    @ivar loggedIn: set to C{True} when the avatar is logged in.
    @type loggedIn: C{bool}

    @ivar loggedOut: set to C{True} when the avatar is logged out.
    @type loggedOut: C{bool}
    """
    loggedIn = loggedOut = False

    def __init__(self, avatarId):
        if False:
            i = 10
            return i + 15
        self.avatarId = avatarId

    def perspective_getAvatarId(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the avatar identifier which was used to access this avatar.\n        '
        return self.avatarId

    def perspective_getViewPoint(self):
        if False:
            i = 10
            return i + 15
        return MyView()

    def perspective_add(self, a, b):
        if False:
            return 10
        '\n        Add the given objects and return the result.  This is a method\n        unavailable on L{Echoer}, so it can only be invoked by authenticated\n        users who received their avatar from L{TestRealm}.\n        '
        return a + b

    def logout(self):
        if False:
            print('Hello World!')
        self.loggedOut = True

class TestRealm:
    """
    A realm which repeatedly gives out a single instance of L{MyPerspective}
    for non-anonymous logins and which gives out a new instance of L{Echoer}
    for each anonymous login.

    @ivar lastPerspective: The L{MyPerspective} most recently created and
        returned from C{requestAvatar}.

    @ivar perspectiveFactory: A one-argument callable which will be used to
        create avatars to be returned from C{requestAvatar}.
    """
    perspectiveFactory = MyPerspective
    lastPerspective = None

    def requestAvatar(self, avatarId, mind, interface):
        if False:
            return 10
        '\n        Verify that the mind and interface supplied have the expected values\n        (this should really be done somewhere else, like inside a test method)\n        and return an avatar appropriate for the given identifier.\n        '
        assert interface == pb.IPerspective
        assert mind == 'BRAINS!'
        if avatarId is checkers.ANONYMOUS:
            return (pb.IPerspective, Echoer(), lambda : None)
        else:
            self.lastPerspective = self.perspectiveFactory(avatarId)
            self.lastPerspective.loggedIn = True
            return (pb.IPerspective, self.lastPerspective, self.lastPerspective.logout)

class MyView(pb.Viewable):

    def view_check(self, user):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(user, MyPerspective)

class LeakyRealm(TestRealm):
    """
    A realm which hangs onto a reference to the mind object in its logout
    function.
    """

    def __init__(self, mindEater):
        if False:
            print('Hello World!')
        '\n        Create a L{LeakyRealm}.\n\n        @param mindEater: a callable that will be called with the C{mind}\n        object when it is available\n        '
        self._mindEater = mindEater

    def requestAvatar(self, avatarId, mind, interface):
        if False:
            return 10
        self._mindEater(mind)
        persp = self.perspectiveFactory(avatarId)
        return (pb.IPerspective, persp, lambda : (mind, persp.logout()))

class NewCredLeakTests(unittest.TestCase):
    """
    Tests to try to trigger memory leaks.
    """

    def test_logoutLeak(self):
        if False:
            while True:
                i = 10
        '\n        The server does not leak a reference when the client disconnects\n        suddenly, even if the cred logout function forms a reference cycle with\n        the perspective.\n        '
        self.mindRef = None

        def setMindRef(mind):
            if False:
                return 10
            self.mindRef = weakref.ref(mind)
        (clientBroker, serverBroker, pump) = connectedServerAndClient(test=self, realm=LeakyRealm(setMindRef))
        connectionBroken = []
        root = clientBroker.remoteForName('root')
        d = root.callRemote('login', b'guest')

        def cbResponse(x):
            if False:
                print('Hello World!')
            (challenge, challenger) = x
            mind = SimpleRemote()
            return challenger.callRemote('respond', pb.respond(challenge, b'guest'), mind)
        d.addCallback(cbResponse)

        def connectionLost(_):
            if False:
                print('Hello World!')
            pump.stop()
            connectionBroken.append(1)
            serverBroker.connectionLost(failure.Failure(RuntimeError('boom')))
        d.addCallback(connectionLost)
        pump.flush()
        self.assertEqual(connectionBroken, [1])
        gc.collect()
        self.assertIsNone(self.mindRef())

class NewCredTests(unittest.TestCase):
    """
    Tests related to the L{twisted.cred} support in PB.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        '\n        Create a portal with no checkers and wrap it around a simple test\n        realm.  Set up a PB server on a TCP port which serves perspectives\n        using that portal.\n        '
        self.realm = TestRealm()
        self.portal = portal.Portal(self.realm)
        self.serverFactory = ConnectionNotifyServerFactory(self.portal)
        self.clientFactory = pb.PBClientFactory()

    def establishClientAndServer(self, _ignored=None):
        if False:
            print('Hello World!')
        '\n        Connect a client obtained from C{clientFactory} and a server\n        obtained from the current server factory via an L{IOPump},\n        then assign them to the appropriate instance variables\n\n        @ivar clientFactory: the broker client factory\n        @ivar clientFactory: L{pb.PBClientFactory} instance\n\n        @ivar client: the client broker\n        @type client: L{pb.Broker}\n\n        @ivar server: the server broker\n        @type server: L{pb.Broker}\n\n        @ivar pump: the IOPump connecting the client and server\n        @type pump: L{IOPump}\n\n        @ivar connector: A connector whose connect method recreates\n            the above instance variables\n        @type connector: L{twisted.internet.base.IConnector}\n        '
        (self.client, self.server, self.pump) = connectServerAndClient(self, self.clientFactory, self.serverFactory)
        self.connectorState = _ReconnectingFakeConnectorState()
        self.connector = _ReconnectingFakeConnector(address.IPv4Address('TCP', '127.0.0.1', 4321), self.connectorState)
        self.connectorState.notifyOnConnect().addCallback(self.establishClientAndServer)

    def completeClientLostConnection(self, reason=failure.Failure(main.CONNECTION_DONE)):
        if False:
            while True:
                i = 10
        "\n        Asserts that the client broker's transport was closed and then\n        mimics the event loop by calling the broker's connectionLost\n        callback with C{reason}, followed by C{self.clientFactory}'s\n        C{clientConnectionLost}\n\n        @param reason: (optional) the reason to pass to the client\n            broker's connectionLost callback\n        @type reason: L{Failure}\n        "
        self.assertTrue(self.client.transport.closed)
        self.client.connectionLost(reason)
        self.clientFactory.clientConnectionLost(self.connector, reason)

    def test_getRootObject(self):
        if False:
            i = 10
            return i + 15
        "\n        Assert that L{PBClientFactory.getRootObject}'s Deferred fires with\n        a L{RemoteReference}, and that disconnecting it runs its\n        disconnection callbacks.\n        "
        self.establishClientAndServer()
        rootObjDeferred = self.clientFactory.getRootObject()

        def gotRootObject(rootObj):
            if False:
                print('Hello World!')
            self.assertIsInstance(rootObj, pb.RemoteReference)
            return rootObj

        def disconnect(rootObj):
            if False:
                print('Hello World!')
            disconnectedDeferred = Deferred()
            rootObj.notifyOnDisconnect(disconnectedDeferred.callback)
            self.clientFactory.disconnect()
            self.completeClientLostConnection()
            return disconnectedDeferred
        rootObjDeferred.addCallback(gotRootObject)
        rootObjDeferred.addCallback(disconnect)
        return rootObjDeferred

    def test_deadReferenceError(self):
        if False:
            return 10
        '\n        Test that when a connection is lost, calling a method on a\n        RemoteReference obtained from it raises L{DeadReferenceError}.\n        '
        self.establishClientAndServer()
        rootObjDeferred = self.clientFactory.getRootObject()

        def gotRootObject(rootObj):
            if False:
                return 10
            disconnectedDeferred = Deferred()
            rootObj.notifyOnDisconnect(disconnectedDeferred.callback)

            def lostConnection(ign):
                if False:
                    i = 10
                    return i + 15
                self.assertRaises(pb.DeadReferenceError, rootObj.callRemote, 'method')
            disconnectedDeferred.addCallback(lostConnection)
            self.clientFactory.disconnect()
            self.completeClientLostConnection()
            return disconnectedDeferred
        return rootObjDeferred.addCallback(gotRootObject)

    def test_clientConnectionLost(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that if the L{reconnecting} flag is passed with a True value then\n        a remote call made from a disconnection notification callback gets a\n        result successfully.\n        '

        class ReconnectOnce(pb.PBClientFactory):
            reconnectedAlready = False

            def clientConnectionLost(self, connector, reason):
                if False:
                    print('Hello World!')
                reconnecting = not self.reconnectedAlready
                self.reconnectedAlready = True
                result = pb.PBClientFactory.clientConnectionLost(self, connector, reason, reconnecting)
                if reconnecting:
                    connector.connect()
                return result
        self.clientFactory = ReconnectOnce()
        self.establishClientAndServer()
        rootObjDeferred = self.clientFactory.getRootObject()

        def gotRootObject(rootObj):
            if False:
                return 10
            self.assertIsInstance(rootObj, pb.RemoteReference)
            d = Deferred()
            rootObj.notifyOnDisconnect(d.callback)
            self.clientFactory.disconnect()
            self.completeClientLostConnection()

            def disconnected(ign):
                if False:
                    while True:
                        i = 10
                d = self.clientFactory.getRootObject()

                def gotAnotherRootObject(anotherRootObj):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.assertIsInstance(anotherRootObj, pb.RemoteReference)
                    d = Deferred()
                    anotherRootObj.notifyOnDisconnect(d.callback)
                    self.clientFactory.disconnect()
                    self.completeClientLostConnection()
                    return d
                return d.addCallback(gotAnotherRootObject)
            return d.addCallback(disconnected)
        return rootObjDeferred.addCallback(gotRootObject)

    def test_immediateClose(self):
        if False:
            print('Hello World!')
        "\n        Test that if a Broker loses its connection without receiving any bytes,\n        it doesn't raise any exceptions or log any errors.\n        "
        self.establishClientAndServer()
        serverProto = self.serverFactory.buildProtocol(('127.0.0.1', 12345))
        serverProto.makeConnection(protocol.FileWrapper(StringIO()))
        serverProto.connectionLost(failure.Failure(main.CONNECTION_DONE))

    def test_loginConnectionRefused(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{PBClientFactory.login} returns a L{Deferred} which is errbacked\n        with the L{ConnectionRefusedError} if the underlying connection is\n        refused.\n        '
        clientFactory = pb.PBClientFactory()
        loginDeferred = clientFactory.login(credentials.UsernamePassword(b'foo', b'bar'))
        clientFactory.clientConnectionFailed(None, failure.Failure(ConnectionRefusedError('Test simulated refused connection')))
        return self.assertFailure(loginDeferred, ConnectionRefusedError)

    def test_loginLogout(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that login can be performed with IUsernamePassword credentials and\n        that when the connection is dropped the avatar is logged out.\n        '
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user=b'pass'))
        creds = credentials.UsernamePassword(b'user', b'pass')
        mind = 'BRAINS!'
        loginCompleted = Deferred()
        d = self.clientFactory.login(creds, mind)

        def cbLogin(perspective):
            if False:
                i = 10
                return i + 15
            self.assertTrue(self.realm.lastPerspective.loggedIn)
            self.assertIsInstance(perspective, pb.RemoteReference)
            return loginCompleted

        def cbDisconnect(ignored):
            if False:
                while True:
                    i = 10
            self.clientFactory.disconnect()
            self.completeClientLostConnection()
        d.addCallback(cbLogin)
        d.addCallback(cbDisconnect)

        def cbLogout(ignored):
            if False:
                i = 10
                return i + 15
            self.assertTrue(self.realm.lastPerspective.loggedOut)
        d.addCallback(cbLogout)
        self.establishClientAndServer()
        self.pump.flush()
        gc.collect()
        self.pump.flush()
        loginCompleted.callback(None)
        return d

    def test_logoutAfterDecref(self):
        if False:
            i = 10
            return i + 15
        '\n        If a L{RemoteReference} to an L{IPerspective} avatar is decrefed and\n        there remain no other references to the avatar on the server, the\n        avatar is garbage collected and the logout method called.\n        '
        loggedOut = Deferred()

        class EventPerspective(pb.Avatar):
            """
            An avatar which fires a Deferred when it is logged out.
            """

            def __init__(self, avatarId):
                if False:
                    i = 10
                    return i + 15
                pass

            def logout(self):
                if False:
                    while True:
                        i = 10
                loggedOut.callback(None)
        self.realm.perspectiveFactory = EventPerspective
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(foo=b'bar'))
        d = self.clientFactory.login(credentials.UsernamePassword(b'foo', b'bar'), 'BRAINS!')

        def cbLoggedIn(avatar):
            if False:
                return 10
            return loggedOut
        d.addCallback(cbLoggedIn)

        def cbLoggedOut(ignored):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(self.serverFactory.protocolInstance._localCleanup, {})
        d.addCallback(cbLoggedOut)
        self.establishClientAndServer()
        self.pump.flush()
        gc.collect()
        self.pump.flush()
        return d

    def test_concurrentLogin(self):
        if False:
            print('Hello World!')
        '\n        Two different correct login attempts can be made on the same root\n        object at the same time and produce two different resulting avatars.\n        '
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(foo=b'bar', baz=b'quux'))
        firstLogin = self.clientFactory.login(credentials.UsernamePassword(b'foo', b'bar'), 'BRAINS!')
        secondLogin = self.clientFactory.login(credentials.UsernamePassword(b'baz', b'quux'), 'BRAINS!')
        d = gatherResults([firstLogin, secondLogin])

        def cbLoggedIn(result):
            if False:
                for i in range(10):
                    print('nop')
            (first, second) = result
            return gatherResults([first.callRemote('getAvatarId'), second.callRemote('getAvatarId')])
        d.addCallback(cbLoggedIn)

        def cbAvatarIds(x):
            if False:
                return 10
            (first, second) = x
            self.assertEqual(first, b'foo')
            self.assertEqual(second, b'baz')
        d.addCallback(cbAvatarIds)
        self.establishClientAndServer()
        self.pump.flush()
        return d

    def test_badUsernamePasswordLogin(self):
        if False:
            print('Hello World!')
        '\n        Test that a login attempt with an invalid user or invalid password\n        fails in the appropriate way.\n        '
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user=b'pass'))
        firstLogin = self.clientFactory.login(credentials.UsernamePassword(b'nosuchuser', b'pass'))
        secondLogin = self.clientFactory.login(credentials.UsernamePassword(b'user', b'wrongpass'))
        self.assertFailure(firstLogin, UnauthorizedLogin)
        self.assertFailure(secondLogin, UnauthorizedLogin)
        d = gatherResults([firstLogin, secondLogin])

        def cleanup(ignore):
            if False:
                return 10
            errors = self.flushLoggedErrors(UnauthorizedLogin)
            self.assertEqual(len(errors), 2)
        d.addCallback(cleanup)
        self.establishClientAndServer()
        self.pump.flush()
        return d

    def test_anonymousLogin(self):
        if False:
            i = 10
            return i + 15
        '\n        Verify that a PB server using a portal configured with a checker which\n        allows IAnonymous credentials can be logged into using IAnonymous\n        credentials.\n        '
        self.portal.registerChecker(checkers.AllowAnonymousAccess())
        d = self.clientFactory.login(credentials.Anonymous(), 'BRAINS!')

        def cbLoggedIn(perspective):
            if False:
                while True:
                    i = 10
            return perspective.callRemote('echo', 123)
        d.addCallback(cbLoggedIn)
        d.addCallback(self.assertEqual, 123)
        self.establishClientAndServer()
        self.pump.flush()
        return d

    def test_anonymousLoginNotPermitted(self):
        if False:
            print('Hello World!')
        '\n        Verify that without an anonymous checker set up, anonymous login is\n        rejected.\n        '
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user='pass'))
        d = self.clientFactory.login(credentials.Anonymous(), 'BRAINS!')
        self.assertFailure(d, UnhandledCredentials)

        def cleanup(ignore):
            if False:
                return 10
            errors = self.flushLoggedErrors(UnhandledCredentials)
            self.assertEqual(len(errors), 1)
        d.addCallback(cleanup)
        self.establishClientAndServer()
        self.pump.flush()
        return d

    def test_anonymousLoginWithMultipleCheckers(self):
        if False:
            print('Hello World!')
        '\n        Like L{test_anonymousLogin} but against a portal with a checker for\n        both IAnonymous and IUsernamePassword.\n        '
        self.portal.registerChecker(checkers.AllowAnonymousAccess())
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user=b'pass'))
        d = self.clientFactory.login(credentials.Anonymous(), 'BRAINS!')

        def cbLogin(perspective):
            if False:
                for i in range(10):
                    print('nop')
            return perspective.callRemote('echo', 123)
        d.addCallback(cbLogin)
        d.addCallback(self.assertEqual, 123)
        self.establishClientAndServer()
        self.pump.flush()
        return d

    def test_authenticatedLoginWithMultipleCheckers(self):
        if False:
            i = 10
            return i + 15
        '\n        Like L{test_anonymousLoginWithMultipleCheckers} but check that\n        username/password authentication works.\n        '
        self.portal.registerChecker(checkers.AllowAnonymousAccess())
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user=b'pass'))
        d = self.clientFactory.login(credentials.UsernamePassword(b'user', b'pass'), 'BRAINS!')

        def cbLogin(perspective):
            if False:
                return 10
            return perspective.callRemote('add', 100, 23)
        d.addCallback(cbLogin)
        d.addCallback(self.assertEqual, 123)
        self.establishClientAndServer()
        self.pump.flush()
        return d

    def test_view(self):
        if False:
            i = 10
            return i + 15
        '\n        Verify that a viewpoint can be retrieved after authenticating with\n        cred.\n        '
        self.portal.registerChecker(checkers.InMemoryUsernamePasswordDatabaseDontUse(user=b'pass'))
        d = self.clientFactory.login(credentials.UsernamePassword(b'user', b'pass'), 'BRAINS!')

        def cbLogin(perspective):
            if False:
                for i in range(10):
                    print('nop')
            return perspective.callRemote('getViewPoint')
        d.addCallback(cbLogin)

        def cbView(viewpoint):
            if False:
                return 10
            return viewpoint.callRemote('check')
        d.addCallback(cbView)
        d.addCallback(self.assertTrue)
        self.establishClientAndServer()
        self.pump.flush()
        return d

@implementer(pb.IPerspective)
class NonSubclassingPerspective:

    def __init__(self, avatarId):
        if False:
            return 10
        pass

    def perspectiveMessageReceived(self, broker, message, args, kwargs):
        if False:
            return 10
        args = broker.unserialize(args, self)
        kwargs = broker.unserialize(kwargs, self)
        return broker.serialize((message, args, kwargs))

    def logout(self):
        if False:
            print('Hello World!')
        self.loggedOut = True

class NSPTests(unittest.TestCase):
    """
    Tests for authentication against a realm where the L{IPerspective}
    implementation is not a subclass of L{Avatar}.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.realm = TestRealm()
        self.realm.perspectiveFactory = NonSubclassingPerspective
        self.portal = portal.Portal(self.realm)
        self.checker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
        self.checker.addUser(b'user', b'pass')
        self.portal.registerChecker(self.checker)
        self.factory = WrappingFactory(pb.PBServerFactory(self.portal))
        self.port = reactor.listenTCP(0, self.factory, interface='127.0.0.1')
        self.addCleanup(self.port.stopListening)
        self.portno = self.port.getHost().port

    def test_NSP(self):
        if False:
            i = 10
            return i + 15
        '\n        An L{IPerspective} implementation which does not subclass\n        L{Avatar} can expose remote methods for the client to call.\n        '
        factory = pb.PBClientFactory()
        d = factory.login(credentials.UsernamePassword(b'user', b'pass'), 'BRAINS!')
        reactor.connectTCP('127.0.0.1', self.portno, factory)
        d.addCallback(lambda p: p.callRemote('ANYTHING', 'here', bar='baz'))
        d.addCallback(self.assertEqual, ('ANYTHING', ('here',), {'bar': 'baz'}))

        def cleanup(ignored):
            if False:
                while True:
                    i = 10
            factory.disconnect()
            for p in self.factory.protocols:
                p.transport.loseConnection()
        d.addCallback(cleanup)
        return d

class IForwarded(Interface):
    """
    Interface used for testing L{util.LocalAsyncForwarder}.
    """

    def forwardMe():
        if False:
            while True:
                i = 10
        '\n        Simple synchronous method.\n        '

    def forwardDeferred():
        if False:
            print('Hello World!')
        '\n        Simple asynchronous method.\n        '

@implementer(IForwarded)
class Forwarded:
    """
    Test implementation of L{IForwarded}.

    @ivar forwarded: set if C{forwardMe} is called.
    @type forwarded: C{bool}
    @ivar unforwarded: set if C{dontForwardMe} is called.
    @type unforwarded: C{bool}
    """
    forwarded = False
    unforwarded = False

    def forwardMe(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set a local flag to test afterwards.\n        '
        self.forwarded = True

    def dontForwardMe(self):
        if False:
            while True:
                i = 10
        "\n        Set a local flag to test afterwards. This should not be called as it's\n        not in the interface.\n        "
        self.unforwarded = True

    def forwardDeferred(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Asynchronously return C{True}.\n        '
        return succeed(True)

class SpreadUtilTests(unittest.TestCase):
    """
    Tests for L{twisted.spread.util}.
    """

    def test_sync(self):
        if False:
            while True:
                i = 10
        '\n        Call a synchronous method of a L{util.LocalAsRemote} object and check\n        the result.\n        '
        o = LocalRemoteTest()
        self.assertEqual(o.callRemote('add1', 2), 3)

    def test_async(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Call an asynchronous method of a L{util.LocalAsRemote} object and check\n        the result.\n        '
        o = LocalRemoteTest()
        o = LocalRemoteTest()
        d = o.callRemote('add', 2, y=4)
        self.assertIsInstance(d, Deferred)
        d.addCallback(self.assertEqual, 6)
        return d

    def test_asyncFail(self):
        if False:
            print('Hello World!')
        '\n        Test an asynchronous failure on a remote method call.\n        '
        o = LocalRemoteTest()
        d = o.callRemote('fail')

        def eb(f):
            if False:
                return 10
            self.assertIsInstance(f, failure.Failure)
            f.trap(RuntimeError)
        d.addCallbacks(lambda res: self.fail('supposed to fail'), eb)
        return d

    def test_remoteMethod(self):
        if False:
            while True:
                i = 10
        '\n        Test the C{remoteMethod} facility of L{util.LocalAsRemote}.\n        '
        o = LocalRemoteTest()
        m = o.remoteMethod('add1')
        self.assertEqual(m(3), 4)

    def test_localAsyncForwarder(self):
        if False:
            while True:
                i = 10
        '\n        Test a call to L{util.LocalAsyncForwarder} using L{Forwarded} local\n        object.\n        '
        f = Forwarded()
        lf = util.LocalAsyncForwarder(f, IForwarded)
        lf.callRemote('forwardMe')
        self.assertTrue(f.forwarded)
        lf.callRemote('dontForwardMe')
        self.assertFalse(f.unforwarded)
        rr = lf.callRemote('forwardDeferred')
        l = []
        rr.addCallback(l.append)
        self.assertEqual(l[0], 1)

class PBWithSecurityOptionsTests(unittest.TestCase):
    """
    Test security customization.
    """

    def test_clientDefaultSecurityOptions(self):
        if False:
            i = 10
            return i + 15
        '\n        By default, client broker should use C{jelly.globalSecurity} as\n        security settings.\n        '
        factory = pb.PBClientFactory()
        broker = factory.buildProtocol(None)
        self.assertIs(broker.security, jelly.globalSecurity)

    def test_serverDefaultSecurityOptions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        By default, server broker should use C{jelly.globalSecurity} as\n        security settings.\n        '
        factory = pb.PBServerFactory(Echoer())
        broker = factory.buildProtocol(None)
        self.assertIs(broker.security, jelly.globalSecurity)

    def test_clientSecurityCustomization(self):
        if False:
            return 10
        '\n        Check that the security settings are passed from the client factory to\n        the broker object.\n        '
        security = jelly.SecurityOptions()
        factory = pb.PBClientFactory(security=security)
        broker = factory.buildProtocol(None)
        self.assertIs(broker.security, security)

    def test_serverSecurityCustomization(self):
        if False:
            return 10
        '\n        Check that the security settings are passed from the server factory to\n        the broker object.\n        '
        security = jelly.SecurityOptions()
        factory = pb.PBServerFactory(Echoer(), security=security)
        broker = factory.buildProtocol(None)
        self.assertIs(broker.security, security)