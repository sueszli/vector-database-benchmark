from __future__ import annotations
__all__ = ['StateVar', 'FunctionCall', 'EnterExit', 'Pulse', 'EventPulse', 'EventArgument']
from direct.showbase.DirectObject import DirectObject

class PushesStateChanges:

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self._value = value
        self._subscribers = set()

    def destroy(self):
        if False:
            return 10
        if len(self._subscribers) != 0:
            raise Exception('%s object still has subscribers in destroy(): %s' % (self.__class__.__name__, self._subscribers))
        del self._subscribers
        del self._value

    def getState(self):
        if False:
            for i in range(10):
                print('nop')
        return self._value

    def pushCurrentState(self):
        if False:
            for i in range(10):
                print('nop')
        self._handleStateChange()
        return self

    def _addSubscription(self, subscriber):
        if False:
            return 10
        self._subscribers.add(subscriber)
        subscriber._recvStatePush(self)

    def _removeSubscription(self, subscriber):
        if False:
            i = 10
            return i + 15
        self._subscribers.remove(subscriber)

    def _handlePotentialStateChange(self, value):
        if False:
            while True:
                i = 10
        oldValue = self._value
        self._value = value
        if oldValue != value:
            self._handleStateChange()

    def _handleStateChange(self):
        if False:
            while True:
                i = 10
        for subscriber in self._subscribers:
            subscriber._recvStatePush(self)

class ReceivesStateChanges:

    def __init__(self, source):
        if False:
            i = 10
            return i + 15
        self._source = None
        self._initSource = source

    def _finishInit(self):
        if False:
            while True:
                i = 10
        self._subscribeTo(self._initSource)
        del self._initSource

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        self._unsubscribe()
        del self._source

    def _subscribeTo(self, source):
        if False:
            print('Hello World!')
        self._unsubscribe()
        self._source = source
        if self._source:
            self._source._addSubscription(self)

    def _unsubscribe(self):
        if False:
            return 10
        if self._source:
            self._source._removeSubscription(self)
            self._source = None

    def _recvStatePush(self, source):
        if False:
            while True:
                i = 10
        pass

class StateVar(PushesStateChanges):

    def set(self, value):
        if False:
            i = 10
            return i + 15
        PushesStateChanges._handlePotentialStateChange(self, value)

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        return PushesStateChanges.getState(self)

class StateChangeNode(PushesStateChanges, ReceivesStateChanges):

    def __init__(self, source):
        if False:
            return 10
        ReceivesStateChanges.__init__(self, source)
        PushesStateChanges.__init__(self, source.getState())
        ReceivesStateChanges._finishInit(self)

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        PushesStateChanges.destroy(self)
        ReceivesStateChanges.destroy(self)

    def _recvStatePush(self, source):
        if False:
            while True:
                i = 10
        self._handlePotentialStateChange(source._value)

class ReceivesMultipleStateChanges:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._key2source = {}
        self._source2key = {}

    def destroy(self):
        if False:
            return 10
        keys = list(self._key2source.keys())
        for key in keys:
            self._unsubscribe(key)
        del self._key2source
        del self._source2key

    def _subscribeTo(self, source, key):
        if False:
            i = 10
            return i + 15
        self._unsubscribe(key)
        self._key2source[key] = source
        self._source2key[source] = key
        source._addSubscription(self)

    def _unsubscribe(self, key):
        if False:
            for i in range(10):
                print('nop')
        if key in self._key2source:
            source = self._key2source[key]
            source._removeSubscription(self)
            del self._key2source[key]
            del self._source2key[source]

    def _recvStatePush(self, source):
        if False:
            print('Hello World!')
        self._recvMultiStatePush(self._source2key[source], source)

    def _recvMultiStatePush(self, key, source):
        if False:
            while True:
                i = 10
        pass

class FunctionCall(ReceivesMultipleStateChanges, PushesStateChanges):

    def __init__(self, func, *args, **kArgs):
        if False:
            i = 10
            return i + 15
        self._initialized = False
        ReceivesMultipleStateChanges.__init__(self)
        PushesStateChanges.__init__(self, None)
        self._func = func
        self._args = args
        self._kArgs = kArgs
        self._bakedArgs = []
        self._bakedKargs = {}
        for (i, arg) in enumerate(self._args):
            key = i
            if isinstance(arg, PushesStateChanges):
                self._bakedArgs.append(arg.getState())
                self._subscribeTo(arg, key)
            else:
                self._bakedArgs.append(self._args[i])
        for (key, arg) in self._kArgs.items():
            if isinstance(arg, PushesStateChanges):
                self._bakedKargs[key] = arg.getState()
                self._subscribeTo(arg, key)
            else:
                self._bakedKargs[key] = arg
        self._initialized = True

    def destroy(self):
        if False:
            while True:
                i = 10
        ReceivesMultipleStateChanges.destroy(self)
        PushesStateChanges.destroy(self)
        del self._func
        del self._args
        del self._kArgs
        del self._bakedArgs
        del self._bakedKargs

    def getState(self):
        if False:
            while True:
                i = 10
        return (tuple(self._bakedArgs), dict(self._bakedKargs))

    def _recvMultiStatePush(self, key, source):
        if False:
            while True:
                i = 10
        if isinstance(key, str):
            self._bakedKargs[key] = source.getState()
        else:
            self._bakedArgs[key] = source.getState()
        self._handlePotentialStateChange(self.getState())

    def _handleStateChange(self):
        if False:
            i = 10
            return i + 15
        if self._initialized:
            self._func(*self._bakedArgs, **self._bakedKargs)
            PushesStateChanges._handleStateChange(self)

class EnterExit(StateChangeNode):

    def __init__(self, source, enterFunc, exitFunc):
        if False:
            i = 10
            return i + 15
        self._enterFunc = enterFunc
        self._exitFunc = exitFunc
        StateChangeNode.__init__(self, source)

    def destroy(self):
        if False:
            i = 10
            return i + 15
        StateChangeNode.destroy(self)
        del self._exitFunc
        del self._enterFunc

    def _handlePotentialStateChange(self, value):
        if False:
            return 10
        StateChangeNode._handlePotentialStateChange(self, bool(value))

    def _handleStateChange(self):
        if False:
            for i in range(10):
                print('nop')
        if self._value:
            self._enterFunc()
        else:
            self._exitFunc()
        StateChangeNode._handleStateChange(self)

class Pulse(PushesStateChanges):

    def __init__(self):
        if False:
            print('Hello World!')
        PushesStateChanges.__init__(self, False)

    def sendPulse(self):
        if False:
            return 10
        self._handlePotentialStateChange(True)
        self._handlePotentialStateChange(False)

class EventPulse(Pulse, DirectObject):

    def __init__(self, event):
        if False:
            return 10
        Pulse.__init__(self)
        self.accept(event, self.sendPulse)

    def destroy(self):
        if False:
            return 10
        self.ignoreAll()
        Pulse.destroy(self)

class EventArgument(PushesStateChanges, DirectObject):

    def __init__(self, event, index=0):
        if False:
            for i in range(10):
                print('nop')
        PushesStateChanges.__init__(self, None)
        self._index = index
        self.accept(event, self._handleEvent)

    def destroy(self):
        if False:
            while True:
                i = 10
        self.ignoreAll()
        del self._index
        PushesStateChanges.destroy(self)

    def _handleEvent(self, *args):
        if False:
            i = 10
            return i + 15
        self._handlePotentialStateChange(args[self._index])

class AttrSetter(StateChangeNode):

    def __init__(self, source, object, attrName):
        if False:
            while True:
                i = 10
        self._object = object
        self._attrName = attrName
        StateChangeNode.__init__(self, source)
        self._handleStateChange()

    def _handleStateChange(self):
        if False:
            return 10
        setattr(self._object, self._attrName, self._value)
        StateChangeNode._handleStateChange(self)