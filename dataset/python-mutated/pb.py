"""
Perspective Broker

"This isn't a professional opinion, but it's probably got enough
internet to kill you." --glyph

Introduction
============

This is a broker for proxies for and copies of objects.  It provides a
translucent interface layer to those proxies.

The protocol is not opaque, because it provides objects which represent the
remote proxies and require no context (server references, IDs) to operate on.

It is not transparent because it does I{not} attempt to make remote objects
behave identically, or even similarly, to local objects.  Method calls are
invoked asynchronously, and specific rules are applied when serializing
arguments.

To get started, begin with L{PBClientFactory} and L{PBServerFactory}.

@author: Glyph Lefkowitz
"""
import random
from hashlib import md5
from zope.interface import Interface, implementer
from twisted.cred.credentials import Anonymous, IAnonymous, ICredentials, IUsernameHashedPassword
from twisted.cred.portal import Portal
from twisted.internet import defer, protocol
from twisted.persisted import styles
from twisted.python import failure, log, reflect
from twisted.python.compat import cmp, comparable
from twisted.python.components import registerAdapter
from twisted.spread import banana
from twisted.spread.flavors import Cacheable, Copyable, IPBRoot, Jellyable, NoSuchMethod, Referenceable, RemoteCache, RemoteCacheObserver, RemoteCopy, Root, Serializable, Viewable, ViewPoint, copyTags, setCopierForClass, setCopierForClassTree, setFactoryForClass, setUnjellyableFactoryForClass, setUnjellyableForClass, setUnjellyableForClassTree
from twisted.spread.interfaces import IJellyable, IUnjellyable
from twisted.spread.jelly import _newInstance, globalSecurity, jelly, unjelly
MAX_BROKER_REFS = 1024
portno = 8787

class ProtocolError(Exception):
    """
    This error is raised when an invalid protocol statement is received.
    """

class DeadReferenceError(ProtocolError):
    """
    This error is raised when a method is called on a dead reference (one whose
    broker has been disconnected).
    """

class Error(Exception):
    """
    This error can be raised to generate known error conditions.

    When a PB callable method (perspective_, remote_, view_) raises
    this error, it indicates that a traceback should not be printed,
    but instead, the string representation of the exception should be
    sent.
    """

class RemoteError(Exception):
    """
    This class is used to wrap a string-ified exception from the remote side to
    be able to reraise it. (Raising string exceptions is no longer possible in
    Python 2.6+)

    The value of this exception will be a str() representation of the remote
    value.

    @ivar remoteType: The full import path of the exception class which was
        raised on the remote end.
    @type remoteType: C{str}

    @ivar remoteTraceback: The remote traceback.
    @type remoteTraceback: C{str}

    @note: It's not possible to include the remoteTraceback if this exception is
        thrown into a generator. It must be accessed as an attribute.
    """

    def __init__(self, remoteType, value, remoteTraceback):
        if False:
            return 10
        Exception.__init__(self, value)
        self.remoteType = remoteType
        self.remoteTraceback = remoteTraceback

@comparable
class RemoteMethod:
    """
    This is a translucent reference to a remote message.
    """

    def __init__(self, obj, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize with a L{RemoteReference} and the name of this message.\n        '
        self.obj = obj
        self.name = name

    def __cmp__(self, other):
        if False:
            return 10
        return cmp((self.obj, self.name), other)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return hash((self.obj, self.name))

    def __call__(self, *args, **kw):
        if False:
            return 10
        '\n        Asynchronously invoke a remote method.\n        '
        return self.obj.broker._sendMessage(b'', self.obj.perspective, self.obj.luid, self.name.encode('utf-8'), args, kw)

class PBConnectionLost(Exception):
    pass

class IPerspective(Interface):
    """
    per*spec*tive, n. : The relationship of aspects of a subject to each
    other and to a whole: 'a perspective of history'; 'a need to view
    the problem in the proper perspective'.

    This is a Perspective Broker-specific wrapper for an avatar. That
    is to say, a PB-published view on to the business logic for the
    system's concept of a 'user'.

    The concept of attached/detached is no longer implemented by the
    framework. The realm is expected to implement such semantics if
    needed.
    """

    def perspectiveMessageReceived(broker, message, args, kwargs):
        if False:
            while True:
                i = 10
        "\n        This method is called when a network message is received.\n\n        @arg broker: The Perspective Broker.\n\n        @type message: str\n        @arg message: The name of the method called by the other end.\n\n        @type args: list in jelly format\n        @arg args: The arguments that were passed by the other end. It\n                   is recommend that you use the `unserialize' method of the\n                   broker to decode this.\n\n        @type kwargs: dict in jelly format\n        @arg kwargs: The keyword arguments that were passed by the\n                     other end.  It is recommended that you use the\n                     `unserialize' method of the broker to decode this.\n\n        @rtype: A jelly list.\n        @return: It is recommended that you use the `serialize' method\n                 of the broker on whatever object you need to return to\n                 generate the return value.\n        "

@implementer(IPerspective)
class Avatar:
    """
    A default IPerspective implementor.

    This class is intended to be subclassed, and a realm should return
    an instance of such a subclass when IPerspective is requested of
    it.

    A peer requesting a perspective will receive only a
    L{RemoteReference} to a pb.Avatar.  When a method is called on
    that L{RemoteReference}, it will translate to a method on the
    remote perspective named 'perspective_methodname'.  (For more
    information on invoking methods on other objects, see
    L{flavors.ViewPoint}.)
    """

    def perspectiveMessageReceived(self, broker, message, args, kw):
        if False:
            print('Hello World!')
        '\n        This method is called when a network message is received.\n\n        This will call::\n\n            self.perspective_%(message)s(*broker.unserialize(args),\n                                         **broker.unserialize(kw))\n\n        to handle the method; subclasses of Avatar are expected to\n        implement methods using this naming convention.\n        '
        args = broker.unserialize(args, self)
        kw = broker.unserialize(kw, self)
        method = getattr(self, 'perspective_%s' % message)
        try:
            state = method(*args, **kw)
        except TypeError:
            log.msg(f"{method} didn't accept {args} and {kw}")
            raise
        return broker.serialize(state, self, method, args, kw)

class AsReferenceable(Referenceable):
    """
    A reference directed towards another object.
    """

    def __init__(self, object, messageType='remote'):
        if False:
            print('Hello World!')
        self.remoteMessageReceived = getattr(object, messageType + 'MessageReceived')

@implementer(IUnjellyable)
@comparable
class RemoteReference(Serializable, styles.Ephemeral):
    """
    A translucent reference to a remote object.

    I may be a reference to a L{flavors.ViewPoint}, a
    L{flavors.Referenceable}, or an L{IPerspective} implementer (e.g.,
    pb.Avatar).  From the client's perspective, it is not possible to
    tell which except by convention.

    I am a "translucent" reference because although no additional
    bookkeeping overhead is given to the application programmer for
    manipulating a reference, return values are asynchronous.

    See also L{twisted.internet.defer}.

    @ivar broker: The broker I am obtained through.
    @type broker: L{Broker}
    """

    def __init__(self, perspective, broker, luid, doRefCount):
        if False:
            while True:
                i = 10
        '(internal) Initialize me with a broker and a locally-unique ID.\n\n        The ID is unique only to the particular Perspective Broker\n        instance.\n        '
        self.luid = luid
        self.broker = broker
        self.doRefCount = doRefCount
        self.perspective = perspective
        self.disconnectCallbacks = []

    def notifyOnDisconnect(self, callback):
        if False:
            print('Hello World!')
        '\n        Register a callback to be called if our broker gets disconnected.\n\n        @param callback: a callable which will be called with one\n                         argument, this instance.\n        '
        assert callable(callback)
        self.disconnectCallbacks.append(callback)
        if len(self.disconnectCallbacks) == 1:
            self.broker.notifyOnDisconnect(self._disconnected)

    def dontNotifyOnDisconnect(self, callback):
        if False:
            i = 10
            return i + 15
        '\n        Remove a callback that was registered with notifyOnDisconnect.\n\n        @param callback: a callable\n        '
        self.disconnectCallbacks.remove(callback)
        if not self.disconnectCallbacks:
            self.broker.dontNotifyOnDisconnect(self._disconnected)

    def _disconnected(self):
        if False:
            while True:
                i = 10
        '\n        Called if we are disconnected and have callbacks registered.\n        '
        for callback in self.disconnectCallbacks:
            callback(self)
        self.disconnectCallbacks = None

    def jellyFor(self, jellier):
        if False:
            print('Hello World!')
        '\n        If I am being sent back to where I came from, serialize as a local backreference.\n        '
        if jellier.invoker:
            assert self.broker == jellier.invoker, "Can't send references to brokers other than their own."
            return (b'local', self.luid)
        else:
            return (b'unpersistable', 'References cannot be serialized')

    def unjellyFor(self, unjellier, unjellyList):
        if False:
            print('Hello World!')
        self.__init__(unjellier.invoker.unserializingPerspective, unjellier.invoker, unjellyList[1], 1)
        return self

    def callRemote(self, _name, *args, **kw):
        if False:
            i = 10
            return i + 15
        '\n        Asynchronously invoke a remote method.\n\n        @type _name: L{str}\n        @param _name:  the name of the remote method to invoke\n        @param args: arguments to serialize for the remote function\n        @param kw:  keyword arguments to serialize for the remote function.\n        @rtype:   L{twisted.internet.defer.Deferred}\n        @returns: a Deferred which will be fired when the result of\n                  this remote call is received.\n        '
        if not isinstance(_name, bytes):
            _name = _name.encode('utf8')
        return self.broker._sendMessage(b'', self.perspective, self.luid, _name, args, kw)

    def remoteMethod(self, key):
        if False:
            print('Hello World!')
        '\n\n        @param key: The key.\n        @return: A L{RemoteMethod} for this key.\n        '
        return RemoteMethod(self, key)

    def __cmp__(self, other):
        if False:
            i = 10
            return i + 15
        '\n\n        @param other: another L{RemoteReference} to compare me to.\n        '
        if isinstance(other, RemoteReference):
            if other.broker == self.broker:
                return cmp(self.luid, other.luid)
        return cmp(self.broker, other)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Hash me.\n        '
        return self.luid

    def __del__(self):
        if False:
            print('Hello World!')
        '\n        Do distributed reference counting on finalization.\n        '
        if self.doRefCount:
            self.broker.sendDecRef(self.luid)
setUnjellyableForClass('remote', RemoteReference)

class Local:
    """
    (internal) A reference to a local object.
    """

    def __init__(self, object, perspective=None):
        if False:
            while True:
                i = 10
        '\n        Initialize.\n        '
        self.object = object
        self.perspective = perspective
        self.refcount = 1

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'<pb.Local {self.object!r} ref:{self.refcount}>'

    def incref(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Increment the reference count.\n\n        @return: the reference count after incrementing\n        '
        self.refcount = self.refcount + 1
        return self.refcount

    def decref(self):
        if False:
            i = 10
            return i + 15
        '\n        Decrement the reference count.\n\n        @return: the reference count after decrementing\n        '
        self.refcount = self.refcount - 1
        return self.refcount

class CopyableFailure(failure.Failure, Copyable):
    """
    A L{flavors.RemoteCopy} and L{flavors.Copyable} version of
    L{twisted.python.failure.Failure} for serialization.
    """
    unsafeTracebacks = 0

    def getStateToCopy(self):
        if False:
            print('Hello World!')
        '\n        Collect state related to the exception which occurred, discarding\n        state which cannot reasonably be serialized.\n        '
        state = self.__dict__.copy()
        state['tb'] = None
        state['frames'] = []
        state['stack'] = []
        state['value'] = str(self.value)
        if isinstance(self.type, bytes):
            state['type'] = self.type
        else:
            state['type'] = reflect.qual(self.type).encode('utf-8')
        if self.unsafeTracebacks:
            state['traceback'] = self.getTraceback()
        else:
            state['traceback'] = 'Traceback unavailable\n'
        return state

class CopiedFailure(RemoteCopy, failure.Failure):
    """
    A L{CopiedFailure} is a L{pb.RemoteCopy} of a L{failure.Failure}
    transferred via PB.

    @ivar type: The full import path of the exception class which was raised on
        the remote end.
    @type type: C{str}

    @ivar value: A str() representation of the remote value.
    @type value: L{CopiedFailure} or C{str}

    @ivar traceback: The remote traceback.
    @type traceback: C{str}
    """

    def printTraceback(self, file=None, elideFrameworkCode=0, detail='default'):
        if False:
            for i in range(10):
                print('nop')
        if file is None:
            file = log.logfile
        failureType = self.type
        if not isinstance(failureType, str):
            failureType = failureType.decode('utf-8')
        file.write('Traceback from remote host -- ')
        file.write(failureType + ': ' + self.value)
        file.write('\n')

    def throwExceptionIntoGenerator(self, g):
        if False:
            for i in range(10):
                print('nop')
        '\n        Throw the original exception into the given generator, preserving\n        traceback information if available. In the case of a L{CopiedFailure}\n        where the exception type is a string, a L{pb.RemoteError} is thrown\n        instead.\n\n        @return: The next value yielded from the generator.\n        @raise StopIteration: If there are no more values in the generator.\n        @raise RemoteError: The wrapped remote exception.\n        '
        return g.throw(RemoteError(self.type, self.value, self.traceback))
    printBriefTraceback = printTraceback
    printDetailedTraceback = printTraceback
setUnjellyableForClass(CopyableFailure, CopiedFailure)

def failure2Copyable(fail, unsafeTracebacks=0):
    if False:
        print('Hello World!')
    f = _newInstance(CopyableFailure, fail.__dict__)
    f.unsafeTracebacks = unsafeTracebacks
    return f

class Broker(banana.Banana):
    """
    I am a broker for objects.
    """
    version = 6
    username = None
    factory = None

    def __init__(self, isClient=1, security=globalSecurity):
        if False:
            i = 10
            return i + 15
        banana.Banana.__init__(self, isClient)
        self.disconnected = 0
        self.disconnects = []
        self.failures = []
        self.connects = []
        self.localObjects = {}
        self.security = security
        self.pageProducers = []
        self.currentRequestID = 0
        self.currentLocalID = 0
        self.unserializingPerspective = None
        self.luids = {}
        self.remotelyCachedObjects = {}
        self.remotelyCachedLUIDs = {}
        self.locallyCachedObjects = {}
        self.waitingForAnswers = {}
        self._localCleanup = {}

    def resumeProducing(self):
        if False:
            return 10
        '\n        Called when the consumer attached to me runs out of buffer.\n        '
        for pageridx in range(len(self.pageProducers) - 1, -1, -1):
            pager = self.pageProducers[pageridx]
            pager.sendNextPage()
            if not pager.stillPaging():
                del self.pageProducers[pageridx]
        if not self.pageProducers:
            self.transport.unregisterProducer()

    def pauseProducing(self):
        if False:
            return 10
        pass

    def stopProducing(self):
        if False:
            i = 10
            return i + 15
        pass

    def registerPageProducer(self, pager):
        if False:
            i = 10
            return i + 15
        self.pageProducers.append(pager)
        if len(self.pageProducers) == 1:
            self.transport.registerProducer(self, 0)

    def expressionReceived(self, sexp):
        if False:
            for i in range(10):
                print('nop')
        "\n        Evaluate an expression as it's received.\n        "
        if isinstance(sexp, list):
            command = sexp[0]
            if not isinstance(command, str):
                command = command.decode('utf8')
            methodName = 'proto_%s' % command
            method = getattr(self, methodName, None)
            if method:
                method(*sexp[1:])
            else:
                self.sendCall(b'didNotUnderstand', command)
        else:
            raise ProtocolError('Non-list expression received.')

    def proto_version(self, vnum):
        if False:
            return 10
        '\n        Protocol message: (version version-number)\n\n        Check to make sure that both ends of the protocol are speaking\n        the same version dialect.\n\n        @param vnum: The version number.\n        '
        if vnum != self.version:
            raise ProtocolError(f'Version Incompatibility: {self.version} {vnum}')

    def sendCall(self, *exp):
        if False:
            while True:
                i = 10
        '\n        Utility method to send an expression to the other side of the connection.\n\n        @param exp: The expression.\n        '
        self.sendEncoded(exp)

    def proto_didNotUnderstand(self, command):
        if False:
            return 10
        "\n        Respond to stock 'C{didNotUnderstand}' message.\n\n        Log the command that was not understood and continue. (Note:\n        this will probably be changed to close the connection or raise\n        an exception in the future.)\n\n        @param command: The command to log.\n        "
        log.msg("Didn't understand command: %r" % command)

    def connectionReady(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize. Called after Banana negotiation is done.\n        '
        self.sendCall(b'version', self.version)
        for notifier in self.connects:
            try:
                notifier()
            except BaseException:
                log.deferr()
        self.connects = None
        self.factory.clientConnectionMade(self)

    def connectionFailed(self):
        if False:
            return 10
        for notifier in self.failures:
            try:
                notifier()
            except BaseException:
                log.deferr()
        self.failures = None
    waitingForAnswers = None

    def connectionLost(self, reason):
        if False:
            for i in range(10):
                print('nop')
        '\n        The connection was lost.\n\n        @param reason: message to put in L{failure.Failure}\n        '
        self.disconnected = 1
        self.luids = None
        if self.waitingForAnswers:
            for d in self.waitingForAnswers.values():
                try:
                    d.errback(failure.Failure(PBConnectionLost(reason)))
                except BaseException:
                    log.deferr()
        for lobj in self.remotelyCachedObjects.values():
            cacheable = lobj.object
            perspective = lobj.perspective
            try:
                cacheable.stoppedObserving(perspective, RemoteCacheObserver(self, cacheable, perspective))
            except BaseException:
                log.deferr()
        for notifier in self.disconnects[:]:
            try:
                notifier()
            except BaseException:
                log.deferr()
        self.disconnects = None
        self.waitingForAnswers = None
        self.localSecurity = None
        self.remoteSecurity = None
        self.remotelyCachedObjects = None
        self.remotelyCachedLUIDs = None
        self.locallyCachedObjects = None
        self.localObjects = None

    def notifyOnDisconnect(self, notifier):
        if False:
            i = 10
            return i + 15
        '\n\n        @param notifier: callback to call when the Broker disconnects.\n        '
        assert callable(notifier)
        self.disconnects.append(notifier)

    def notifyOnFail(self, notifier):
        if False:
            return 10
        '\n\n        @param notifier: callback to call if the Broker fails to connect.\n        '
        assert callable(notifier)
        self.failures.append(notifier)

    def notifyOnConnect(self, notifier):
        if False:
            print('Hello World!')
        '\n\n        @param notifier: callback to call when the Broker connects.\n        '
        assert callable(notifier)
        if self.connects is None:
            try:
                notifier()
            except BaseException:
                log.err()
        else:
            self.connects.append(notifier)

    def dontNotifyOnDisconnect(self, notifier):
        if False:
            return 10
        '\n\n        @param notifier: callback to remove from list of disconnect callbacks.\n        '
        try:
            self.disconnects.remove(notifier)
        except ValueError:
            pass

    def localObjectForID(self, luid):
        if False:
            print('Hello World!')
        '\n        Get a local object for a locally unique ID.\n\n        @return: An object previously stored with L{registerReference} or\n            L{None} if there is no object which corresponds to the given\n            identifier.\n        '
        if isinstance(luid, str):
            luid = luid.encode('utf8')
        lob = self.localObjects.get(luid)
        if lob is None:
            return
        return lob.object
    maxBrokerRefsViolations = 0

    def registerReference(self, object):
        if False:
            return 10
        '\n        Store a persistent reference to a local object and map its\n        id() to a generated, session-unique ID.\n\n        @param object: a local object\n        @return: the generated ID\n        '
        assert object is not None
        puid = object.processUniqueID()
        luid = self.luids.get(puid)
        if luid is None:
            if len(self.localObjects) > MAX_BROKER_REFS:
                self.maxBrokerRefsViolations = self.maxBrokerRefsViolations + 1
                if self.maxBrokerRefsViolations > 3:
                    self.transport.loseConnection()
                    raise Error('Maximum PB reference count exceeded.  Goodbye.')
                raise Error('Maximum PB reference count exceeded.')
            luid = self.newLocalID()
            self.localObjects[luid] = Local(object)
            self.luids[puid] = luid
        else:
            self.localObjects[luid].incref()
        return luid

    def setNameForLocal(self, name, object):
        if False:
            print('Hello World!')
        "\n        Store a special (string) ID for this object.\n\n        This is how you specify a 'base' set of objects that the remote\n        protocol can connect to.\n\n        @param name: An ID.\n        @param object: The object.\n        "
        if isinstance(name, str):
            name = name.encode('utf8')
        assert object is not None
        self.localObjects[name] = Local(object)

    def remoteForName(self, name):
        if False:
            while True:
                i = 10
        '\n        Returns an object from the remote name mapping.\n\n        Note that this does not check the validity of the name, only\n        creates a translucent reference for it.\n\n        @param name: The name to look up.\n        @return: An object which maps to the name.\n        '
        if isinstance(name, str):
            name = name.encode('utf8')
        return RemoteReference(None, self, name, 0)

    def cachedRemotelyAs(self, instance, incref=0):
        if False:
            print('Hello World!')
        "\n\n        @param instance: The instance to look up.\n        @param incref: Flag to specify whether to increment the\n                       reference.\n        @return: An ID that says what this instance is cached as\n                 remotely, or L{None} if it's not.\n        "
        puid = instance.processUniqueID()
        luid = self.remotelyCachedLUIDs.get(puid)
        if luid is not None and incref:
            self.remotelyCachedObjects[luid].incref()
        return luid

    def remotelyCachedForLUID(self, luid):
        if False:
            while True:
                i = 10
        '\n\n        @param luid: The LUID to look up.\n        @return: An instance which is cached remotely.\n        '
        return self.remotelyCachedObjects[luid].object

    def cacheRemotely(self, instance):
        if False:
            while True:
                i = 10
        '\n        XXX\n\n        @return: A new LUID.\n        '
        puid = instance.processUniqueID()
        luid = self.newLocalID()
        if len(self.remotelyCachedObjects) > MAX_BROKER_REFS:
            self.maxBrokerRefsViolations = self.maxBrokerRefsViolations + 1
            if self.maxBrokerRefsViolations > 3:
                self.transport.loseConnection()
                raise Error('Maximum PB cache count exceeded.  Goodbye.')
            raise Error('Maximum PB cache count exceeded.')
        self.remotelyCachedLUIDs[puid] = luid
        self.remotelyCachedObjects[luid] = Local(instance, self.serializingPerspective)
        return luid

    def cacheLocally(self, cid, instance):
        if False:
            return 10
        '(internal)\n\n        Store a non-filled-out cached instance locally.\n        '
        self.locallyCachedObjects[cid] = instance

    def cachedLocallyAs(self, cid):
        if False:
            while True:
                i = 10
        instance = self.locallyCachedObjects[cid]
        return instance

    def serialize(self, object, perspective=None, method=None, args=None, kw=None):
        if False:
            while True:
                i = 10
        '\n        Jelly an object according to the remote security rules for this broker.\n\n        @param object: The object to jelly.\n        @param perspective: The perspective.\n        @param method: The method.\n        @param args: Arguments.\n        @param kw: Keyword arguments.\n        '
        if isinstance(object, defer.Deferred):
            object.addCallbacks(self.serialize, lambda x: x, callbackKeywords={'perspective': perspective, 'method': method, 'args': args, 'kw': kw})
            return object
        self.serializingPerspective = perspective
        self.jellyMethod = method
        self.jellyArgs = args
        self.jellyKw = kw
        try:
            return jelly(object, self.security, None, self)
        finally:
            self.serializingPerspective = None
            self.jellyMethod = None
            self.jellyArgs = None
            self.jellyKw = None

    def unserialize(self, sexp, perspective=None):
        if False:
            while True:
                i = 10
        '\n        Unjelly an sexp according to the local security rules for this broker.\n\n        @param sexp: The object to unjelly.\n        @param perspective: The perspective.\n        '
        self.unserializingPerspective = perspective
        try:
            return unjelly(sexp, self.security, None, self)
        finally:
            self.unserializingPerspective = None

    def newLocalID(self):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        @return: A newly generated LUID.\n        '
        self.currentLocalID = self.currentLocalID + 1
        return self.currentLocalID

    def newRequestID(self):
        if False:
            return 10
        '\n\n        @return: A newly generated request ID.\n        '
        self.currentRequestID = self.currentRequestID + 1
        return self.currentRequestID

    def _sendMessage(self, prefix, perspective, objectID, message, args, kw):
        if False:
            i = 10
            return i + 15
        pbc = None
        pbe = None
        answerRequired = 1
        if 'pbcallback' in kw:
            pbc = kw['pbcallback']
            del kw['pbcallback']
        if 'pberrback' in kw:
            pbe = kw['pberrback']
            del kw['pberrback']
        if 'pbanswer' in kw:
            assert not pbe and (not pbc), "You can't specify a no-answer requirement."
            answerRequired = kw['pbanswer']
            del kw['pbanswer']
        if self.disconnected:
            raise DeadReferenceError('Calling Stale Broker')
        try:
            netArgs = self.serialize(args, perspective=perspective, method=message)
            netKw = self.serialize(kw, perspective=perspective, method=message)
        except BaseException:
            return defer.fail(failure.Failure())
        requestID = self.newRequestID()
        if answerRequired:
            rval = defer.Deferred()
            self.waitingForAnswers[requestID] = rval
            if pbc or pbe:
                log.msg('warning! using deprecated "pbcallback"')
                rval.addCallbacks(pbc, pbe)
        else:
            rval = None
        self.sendCall(prefix + b'message', requestID, objectID, message, answerRequired, netArgs, netKw)
        return rval

    def proto_message(self, requestID, objectID, message, answerRequired, netArgs, netKw):
        if False:
            for i in range(10):
                print('nop')
        self._recvMessage(self.localObjectForID, requestID, objectID, message, answerRequired, netArgs, netKw)

    def proto_cachemessage(self, requestID, objectID, message, answerRequired, netArgs, netKw):
        if False:
            print('Hello World!')
        self._recvMessage(self.cachedLocallyAs, requestID, objectID, message, answerRequired, netArgs, netKw)

    def _recvMessage(self, findObjMethod, requestID, objectID, message, answerRequired, netArgs, netKw):
        if False:
            return 10
        "\n        Received a message-send.\n\n        Look up message based on object, unserialize the arguments, and\n        invoke it with args, and send an 'answer' or 'error' response.\n\n        @param findObjMethod: A callable which takes C{objectID} as argument.\n        @param requestID: The requiest ID.\n        @param objectID: The object ID.\n        @param message: The message.\n        @param answerRequired:\n        @param netArgs: Arguments.\n        @param netKw: Keyword arguments.\n        "
        if not isinstance(message, str):
            message = message.decode('utf8')
        try:
            object = findObjMethod(objectID)
            if object is None:
                raise Error('Invalid Object ID')
            netResult = object.remoteMessageReceived(self, message, netArgs, netKw)
        except Error as e:
            if answerRequired:
                if isinstance(e, Jellyable) or self.security.isClassAllowed(e.__class__):
                    self._sendError(e, requestID)
                else:
                    self._sendError(CopyableFailure(e), requestID)
        except BaseException:
            if answerRequired:
                log.msg('Peer will receive following PB traceback:', isError=True)
                f = CopyableFailure()
                self._sendError(f, requestID)
            log.err()
        else:
            if answerRequired:
                if isinstance(netResult, defer.Deferred):
                    args = (requestID,)
                    netResult.addCallbacks(self._sendAnswer, self._sendFailureOrError, callbackArgs=args, errbackArgs=args)
                else:
                    self._sendAnswer(netResult, requestID)

    def _sendAnswer(self, netResult, requestID):
        if False:
            for i in range(10):
                print('nop')
        '\n        (internal) Send an answer to a previously sent message.\n\n        @param netResult: The answer.\n        @param requestID: The request ID.\n        '
        self.sendCall(b'answer', requestID, netResult)

    def proto_answer(self, requestID, netResult):
        if False:
            return 10
        '\n        (internal) Got an answer to a previously sent message.\n\n        Look up the appropriate callback and call it.\n\n        @param requestID: The request ID.\n        @param netResult: The answer.\n        '
        d = self.waitingForAnswers[requestID]
        del self.waitingForAnswers[requestID]
        d.callback(self.unserialize(netResult))

    def _sendFailureOrError(self, fail, requestID):
        if False:
            for i in range(10):
                print('nop')
        '\n        Call L{_sendError} or L{_sendFailure}, depending on whether C{fail}\n        represents an L{Error} subclass or not.\n\n        @param fail: The failure.\n        @param requestID: The request ID.\n        '
        if fail.check(Error) is None:
            self._sendFailure(fail, requestID)
        else:
            self._sendError(fail, requestID)

    def _sendFailure(self, fail, requestID):
        if False:
            return 10
        '\n        Log error and then send it.\n\n        @param fail: The failure.\n        @param requestID: The request ID.\n        '
        log.msg('Peer will receive following PB traceback:')
        log.err(fail)
        self._sendError(fail, requestID)

    def _sendError(self, fail, requestID):
        if False:
            while True:
                i = 10
        '\n        (internal) Send an error for a previously sent message.\n\n        @param fail: The failure.\n        @param requestID: The request ID.\n        '
        if isinstance(fail, failure.Failure):
            if isinstance(fail.value, Jellyable) or self.security.isClassAllowed(fail.value.__class__):
                fail = fail.value
            elif not isinstance(fail, CopyableFailure):
                fail = failure2Copyable(fail, self.factory.unsafeTracebacks)
        if isinstance(fail, CopyableFailure):
            fail.unsafeTracebacks = self.factory.unsafeTracebacks
        self.sendCall(b'error', requestID, self.serialize(fail))

    def proto_error(self, requestID, fail):
        if False:
            while True:
                i = 10
        '\n        (internal) Deal with an error.\n\n        @param requestID: The request ID.\n        @param fail: The failure.\n        '
        d = self.waitingForAnswers[requestID]
        del self.waitingForAnswers[requestID]
        d.errback(self.unserialize(fail))

    def sendDecRef(self, objectID):
        if False:
            return 10
        '\n        (internal) Send a DECREF directive.\n\n        @param objectID: The object ID.\n        '
        self.sendCall(b'decref', objectID)

    def proto_decref(self, objectID):
        if False:
            while True:
                i = 10
        '\n        (internal) Decrement the reference count of an object.\n\n        If the reference count is zero, it will free the reference to this\n        object.\n\n        @param objectID: The object ID.\n        '
        if isinstance(objectID, str):
            objectID = objectID.encode('utf8')
        refs = self.localObjects[objectID].decref()
        if refs == 0:
            puid = self.localObjects[objectID].object.processUniqueID()
            del self.luids[puid]
            del self.localObjects[objectID]
            self._localCleanup.pop(puid, lambda : None)()

    def decCacheRef(self, objectID):
        if False:
            while True:
                i = 10
        '\n        (internal) Send a DECACHE directive.\n\n        @param objectID: The object ID.\n        '
        self.sendCall(b'decache', objectID)

    def proto_decache(self, objectID):
        if False:
            while True:
                i = 10
        "\n        (internal) Decrement the reference count of a cached object.\n\n        If the reference count is zero, free the reference, then send an\n        'uncached' directive.\n\n        @param objectID: The object ID.\n        "
        refs = self.remotelyCachedObjects[objectID].decref()
        if refs == 0:
            lobj = self.remotelyCachedObjects[objectID]
            cacheable = lobj.object
            perspective = lobj.perspective
            try:
                cacheable.stoppedObserving(perspective, RemoteCacheObserver(self, cacheable, perspective))
            except BaseException:
                log.deferr()
            puid = cacheable.processUniqueID()
            del self.remotelyCachedLUIDs[puid]
            del self.remotelyCachedObjects[objectID]
            self.sendCall(b'uncache', objectID)

    def proto_uncache(self, objectID):
        if False:
            for i in range(10):
                print('nop')
        '\n        (internal) Tell the client it is now OK to uncache an object.\n\n        @param objectID: The object ID.\n        '
        obj = self.locallyCachedObjects[objectID]
        obj.broker = None
        del self.locallyCachedObjects[objectID]

def respond(challenge, password):
    if False:
        return 10
    '\n    Respond to a challenge.\n\n    This is useful for challenge/response authentication.\n\n    @param challenge: A challenge.\n    @param password: A password.\n    @return: The password hashed twice.\n    '
    m = md5()
    m.update(password)
    hashedPassword = m.digest()
    m = md5()
    m.update(hashedPassword)
    m.update(challenge)
    doubleHashedPassword = m.digest()
    return doubleHashedPassword

def challenge():
    if False:
        return 10
    '\n\n    @return: Some random data.\n    '
    crap = bytes((random.randint(65, 90) for x in range(random.randrange(15, 25))))
    crap = md5(crap).digest()
    return crap

class PBClientFactory(protocol.ClientFactory):
    """
    Client factory for PB brokers.

    As with all client factories, use with reactor.connectTCP/SSL/etc..
    getPerspective and getRootObject can be called either before or
    after the connect.
    """
    protocol = Broker
    unsafeTracebacks = False

    def __init__(self, unsafeTracebacks=False, security=globalSecurity):
        if False:
            i = 10
            return i + 15
        '\n        @param unsafeTracebacks: if set, tracebacks for exceptions will be sent\n            over the wire.\n        @type unsafeTracebacks: C{bool}\n\n        @param security: security options used by the broker, default to\n            C{globalSecurity}.\n        @type security: L{twisted.spread.jelly.SecurityOptions}\n        '
        self.unsafeTracebacks = unsafeTracebacks
        self.security = security
        self._reset()

    def buildProtocol(self, addr):
        if False:
            for i in range(10):
                print('nop')
        '\n        Build the broker instance, passing the security options to it.\n        '
        p = self.protocol(isClient=True, security=self.security)
        p.factory = self
        return p

    def _reset(self):
        if False:
            print('Hello World!')
        self.rootObjectRequests = []
        self._broker = None
        self._root = None

    def _failAll(self, reason):
        if False:
            i = 10
            return i + 15
        deferreds = self.rootObjectRequests
        self._reset()
        for d in deferreds:
            d.errback(reason)

    def clientConnectionFailed(self, connector, reason):
        if False:
            for i in range(10):
                print('nop')
        self._failAll(reason)

    def clientConnectionLost(self, connector, reason, reconnecting=0):
        if False:
            i = 10
            return i + 15
        '\n        Reconnecting subclasses should call with reconnecting=1.\n        '
        if reconnecting:
            self._broker = None
            self._root = None
        else:
            self._failAll(reason)

    def clientConnectionMade(self, broker):
        if False:
            return 10
        self._broker = broker
        self._root = broker.remoteForName('root')
        ds = self.rootObjectRequests
        self.rootObjectRequests = []
        for d in ds:
            d.callback(self._root)

    def getRootObject(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get root object of remote PB server.\n\n        @return: Deferred of the root object.\n        '
        if self._broker and (not self._broker.disconnected):
            return defer.succeed(self._root)
        d = defer.Deferred()
        self.rootObjectRequests.append(d)
        return d

    def disconnect(self):
        if False:
            print('Hello World!')
        '\n        If the factory is connected, close the connection.\n\n        Note that if you set up the factory to reconnect, you will need to\n        implement extra logic to prevent automatic reconnection after this\n        is called.\n        '
        if self._broker:
            self._broker.transport.loseConnection()

    def _cbSendUsername(self, root, username, password, client):
        if False:
            print('Hello World!')
        return root.callRemote('login', username).addCallback(self._cbResponse, password, client)

    def _cbResponse(self, challenges, password, client):
        if False:
            return 10
        (challenge, challenger) = challenges
        return challenger.callRemote('respond', respond(challenge, password), client)

    def _cbLoginAnonymous(self, root, client):
        if False:
            return 10
        '\n        Attempt an anonymous login on the given remote root object.\n\n        @type root: L{RemoteReference}\n        @param root: The object on which to attempt the login, most likely\n            returned by a call to L{PBClientFactory.getRootObject}.\n\n        @param client: A jellyable object which will be used as the I{mind}\n            parameter for the login attempt.\n\n        @rtype: L{Deferred}\n        @return: A L{Deferred} which will be called back with a\n            L{RemoteReference} to an avatar when anonymous login succeeds, or\n            which will errback if anonymous login fails.\n        '
        return root.callRemote('loginAnonymous', client)

    def login(self, credentials, client=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Login and get perspective from remote PB server.\n\n        Currently the following credentials are supported::\n\n            L{twisted.cred.credentials.IUsernamePassword}\n            L{twisted.cred.credentials.IAnonymous}\n\n        @rtype: L{Deferred}\n        @return: A L{Deferred} which will be called back with a\n            L{RemoteReference} for the avatar logged in to, or which will\n            errback if login fails.\n        '
        d = self.getRootObject()
        if IAnonymous.providedBy(credentials):
            d.addCallback(self._cbLoginAnonymous, client)
        else:
            d.addCallback(self._cbSendUsername, credentials.username, credentials.password, client)
        return d

class PBServerFactory(protocol.ServerFactory):
    """
    Server factory for perspective broker.

    Login is done using a Portal object, whose realm is expected to return
    avatars implementing IPerspective. The credential checkers in the portal
    should accept IUsernameHashedPassword or IUsernameMD5Password.

    Alternatively, any object providing or adaptable to L{IPBRoot} can be
    used instead of a portal to provide the root object of the PB server.
    """
    unsafeTracebacks = False
    protocol = Broker

    def __init__(self, root, unsafeTracebacks=False, security=globalSecurity):
        if False:
            for i in range(10):
                print('nop')
        '\n        @param root: factory providing the root Referenceable used by the broker.\n        @type root: object providing or adaptable to L{IPBRoot}.\n\n        @param unsafeTracebacks: if set, tracebacks for exceptions will be sent\n            over the wire.\n        @type unsafeTracebacks: C{bool}\n\n        @param security: security options used by the broker, default to\n            C{globalSecurity}.\n        @type security: L{twisted.spread.jelly.SecurityOptions}\n        '
        self.root = IPBRoot(root)
        self.unsafeTracebacks = unsafeTracebacks
        self.security = security

    def buildProtocol(self, addr):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a Broker attached to the factory (as the service provider).\n        '
        proto = self.protocol(isClient=False, security=self.security)
        proto.factory = self
        proto.setNameForLocal('root', self.root.rootObject(proto))
        return proto

    def clientConnectionMade(self, protocol):
        if False:
            for i in range(10):
                print('nop')
        pass

class IUsernameMD5Password(ICredentials):
    """
    I encapsulate a username and a hashed password.

    This credential is used for username/password over PB. CredentialCheckers
    which check this kind of credential must store the passwords in plaintext
    form or as a MD5 digest.

    @type username: C{str} or C{Deferred}
    @ivar username: The username associated with these credentials.
    """

    def checkPassword(password):
        if False:
            while True:
                i = 10
        '\n        Validate these credentials against the correct password.\n\n        @type password: C{str}\n        @param password: The correct, plaintext password against which to\n            check.\n\n        @rtype: C{bool} or L{Deferred}\n        @return: C{True} if the credentials represented by this object match the\n            given password, C{False} if they do not, or a L{Deferred} which will\n            be called back with one of these values.\n        '

    def checkMD5Password(password):
        if False:
            i = 10
            return i + 15
        '\n        Validate these credentials against the correct MD5 digest of the\n        password.\n\n        @type password: C{str}\n        @param password: The correct MD5 digest of a password against which to\n            check.\n\n        @rtype: C{bool} or L{Deferred}\n        @return: C{True} if the credentials represented by this object match the\n            given digest, C{False} if they do not, or a L{Deferred} which will\n            be called back with one of these values.\n        '

@implementer(IPBRoot)
class _PortalRoot:
    """
    Root object, used to login to portal.
    """

    def __init__(self, portal):
        if False:
            print('Hello World!')
        self.portal = portal

    def rootObject(self, broker):
        if False:
            print('Hello World!')
        return _PortalWrapper(self.portal, broker)
registerAdapter(_PortalRoot, Portal, IPBRoot)

class _JellyableAvatarMixin:
    """
    Helper class for code which deals with avatars which PB must be capable of
    sending to a peer.
    """

    def _cbLogin(self, result):
        if False:
            print('Hello World!')
        "\n        Ensure that the avatar to be returned to the client is jellyable and\n        set up disconnection notification to call the realm's logout object.\n        "
        (interface, avatar, logout) = result
        if not IJellyable.providedBy(avatar):
            avatar = AsReferenceable(avatar, 'perspective')
        puid = avatar.processUniqueID()
        logout = [logout]

        def maybeLogout():
            if False:
                print('Hello World!')
            if not logout:
                return
            fn = logout[0]
            del logout[0]
            fn()
        self.broker._localCleanup[puid] = maybeLogout
        self.broker.notifyOnDisconnect(maybeLogout)
        return avatar

class _PortalWrapper(Referenceable, _JellyableAvatarMixin):
    """
    Root Referenceable object, used to login to portal.
    """

    def __init__(self, portal, broker):
        if False:
            i = 10
            return i + 15
        self.portal = portal
        self.broker = broker

    def remote_login(self, username):
        if False:
            print('Hello World!')
        '\n        Start of username/password login.\n\n        @param username: The username.\n        '
        c = challenge()
        return (c, _PortalAuthChallenger(self.portal, self.broker, username, c))

    def remote_loginAnonymous(self, mind):
        if False:
            while True:
                i = 10
        '\n        Attempt an anonymous login.\n\n        @param mind: An object to use as the mind parameter to the portal login\n            call (possibly None).\n\n        @rtype: L{Deferred}\n        @return: A Deferred which will be called back with an avatar when login\n            succeeds or which will be errbacked if login fails somehow.\n        '
        d = self.portal.login(Anonymous(), mind, IPerspective)
        d.addCallback(self._cbLogin)
        return d

@implementer(IUsernameHashedPassword, IUsernameMD5Password)
class _PortalAuthChallenger(Referenceable, _JellyableAvatarMixin):
    """
    Called with response to password challenge.
    """

    def __init__(self, portal, broker, username, challenge):
        if False:
            for i in range(10):
                print('nop')
        self.portal = portal
        self.broker = broker
        self.username = username
        self.challenge = challenge

    def remote_respond(self, response, mind):
        if False:
            for i in range(10):
                print('nop')
        self.response = response
        d = self.portal.login(self, mind, IPerspective)
        d.addCallback(self._cbLogin)
        return d

    def checkPassword(self, password):
        if False:
            return 10
        '\n        L{IUsernameHashedPassword}\n\n        @param password: The password.\n        @return: L{_PortalAuthChallenger.checkMD5Password}\n        '
        return self.checkMD5Password(md5(password).digest())

    def checkMD5Password(self, md5Password):
        if False:
            print('Hello World!')
        '\n        L{IUsernameMD5Password}\n\n        @param md5Password:\n        @rtype: L{bool}\n        @return: L{True} if password matches.\n        '
        md = md5()
        md.update(md5Password)
        md.update(self.challenge)
        correct = md.digest()
        return self.response == correct
__all__ = ['IPBRoot', 'Serializable', 'Referenceable', 'NoSuchMethod', 'Root', 'ViewPoint', 'Viewable', 'Copyable', 'Jellyable', 'Cacheable', 'RemoteCopy', 'RemoteCache', 'RemoteCacheObserver', 'copyTags', 'setUnjellyableForClass', 'setUnjellyableFactoryForClass', 'setUnjellyableForClassTree', 'setCopierForClass', 'setFactoryForClass', 'setCopierForClassTree', 'MAX_BROKER_REFS', 'portno', 'ProtocolError', 'DeadReferenceError', 'Error', 'PBConnectionLost', 'RemoteMethod', 'IPerspective', 'Avatar', 'AsReferenceable', 'RemoteReference', 'CopyableFailure', 'CopiedFailure', 'failure2Copyable', 'Broker', 'respond', 'challenge', 'PBClientFactory', 'PBServerFactory', 'IUsernameMD5Password']