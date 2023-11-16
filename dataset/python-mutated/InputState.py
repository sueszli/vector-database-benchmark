from direct.directnotify import DirectNotifyGlobal
from direct.showbase import DirectObject
from direct.showbase.PythonUtil import SerialNumGen
from direct.showbase.MessengerGlobal import messenger

class InputStateToken:
    _SerialGen = SerialNumGen()
    Inval = 'invalidatedToken'

    def __init__(self, inputState):
        if False:
            for i in range(10):
                print('nop')
        self._id = InputStateToken._SerialGen.next()
        self._hash = self._id
        self._inputState = inputState

    def release(self):
        if False:
            while True:
                i = 10
        assert False

    def isValid(self):
        if False:
            for i in range(10):
                print('nop')
        return self._id != InputStateToken.Inval

    def invalidate(self):
        if False:
            print('Hello World!')
        self._id = InputStateToken.Inval

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._hash
    is_valid = isValid

class InputStateWatchToken(InputStateToken, DirectObject.DirectObject):

    def release(self):
        if False:
            return 10
        self._inputState._ignore(self)
        self.ignoreAll()

class InputStateForceToken(InputStateToken):

    def release(self):
        if False:
            i = 10
            return i + 15
        self._inputState._unforce(self)

class InputStateTokenGroup:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._tokens = []

    def addToken(self, token):
        if False:
            i = 10
            return i + 15
        self._tokens.append(token)

    def release(self):
        if False:
            return 10
        for token in self._tokens:
            token.release()
        self._tokens = []
    add_token = addToken

class InputState(DirectObject.DirectObject):
    """
    InputState is for tracking the on/off state of some events.
    The initial usage is to watch some keyboard keys so that another
    task can poll the key states.  By the way, in general polling is
    not a good idea, but it is useful in some situations.  Know when
    to use it:)  If in doubt, don't use this class and listen for
    events instead.
    """
    notify = DirectNotifyGlobal.directNotify.newCategory('InputState')
    WASD = 'WASD'
    QE = 'QE'
    ArrowKeys = 'ArrowKeys'
    Keyboard = 'Keyboard'
    Mouse = 'Mouse'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._state = {}
        self._forcingOn = {}
        self._forcingOff = {}
        self._token2inputSource = {}
        self._token2forceInfo = {}
        self._watching = {}
        assert self.debugPrint('InputState()')

    def delete(self):
        if False:
            while True:
                i = 10
        del self._watching
        del self._token2forceInfo
        del self._token2inputSource
        del self._forcingOff
        del self._forcingOn
        del self._state
        self.ignoreAll()

    def isSet(self, name, inputSource=None):
        if False:
            i = 10
            return i + 15
        '\n        returns True/False\n        '
        if name in self._forcingOn:
            return True
        elif name in self._forcingOff:
            return False
        if inputSource:
            s = self._state.get(name)
            if s:
                return inputSource in s
            else:
                return False
        else:
            return name in self._state

    def getEventName(self, name):
        if False:
            for i in range(10):
                print('nop')
        return 'InputState-%s' % (name,)

    def set(self, name, isActive, inputSource=None):
        if False:
            while True:
                i = 10
        assert self.debugPrint('set(name=%s, isActive=%s, inputSource=%s)' % (name, isActive, inputSource))
        if inputSource is None:
            inputSource = 'anonymous'
        if isActive:
            self._state.setdefault(name, set())
            self._state[name].add(inputSource)
        elif name in self._state:
            self._state[name].discard(inputSource)
            if len(self._state[name]) == 0:
                del self._state[name]
        messenger.send(self.getEventName(name), [self.isSet(name)])

    def releaseInputs(self, name):
        if False:
            i = 10
            return i + 15
        del self._state[name]

    def watch(self, name, eventOn, eventOff, startState=False, inputSource=None):
        if False:
            i = 10
            return i + 15
        "\n        This returns a token; hold onto the token and call token.release() when\n        you no longer want to watch for these events.\n\n        Example::\n\n            # set up\n            token = inputState.watch('forward', 'w', 'w-up', inputSource=inputState.WASD)\n            ...\n            # tear down\n            token.release()\n        "
        assert self.debugPrint('watch(name=%s, eventOn=%s, eventOff=%s, startState=%s)' % (name, eventOn, eventOff, startState))
        if inputSource is None:
            inputSource = "eventPair('%s','%s')" % (eventOn, eventOff)
        self.set(name, startState, inputSource)
        token = InputStateWatchToken(self)
        token.accept(eventOn, self.set, [name, True, inputSource])
        token.accept(eventOff, self.set, [name, False, inputSource])
        self._token2inputSource[token] = inputSource
        self._watching.setdefault(inputSource, {})
        self._watching[inputSource][token] = (name, eventOn, eventOff)
        return token

    def watchWithModifiers(self, name, event, startState=False, inputSource=None):
        if False:
            for i in range(10):
                print('nop')
        patterns = ('%s', 'control-%s', 'shift-control-%s', 'alt-%s', 'control-alt-%s', 'shift-%s', 'shift-alt-%s')
        tGroup = InputStateTokenGroup()
        for pattern in patterns:
            tGroup.addToken(self.watch(name, pattern % event, '%s-up' % event, startState=startState, inputSource=inputSource))
        return tGroup

    def _ignore(self, token):
        if False:
            return 10
        "\n        Undo a watch(). Don't call this directly, call release() on the token that watch() returned.\n        "
        inputSource = self._token2inputSource.pop(token)
        (name, eventOn, eventOff) = self._watching[inputSource].pop(token)
        token.invalidate()
        DirectObject.DirectObject.ignore(self, eventOn)
        DirectObject.DirectObject.ignore(self, eventOff)
        if len(self._watching[inputSource]) == 0:
            del self._watching[inputSource]

    def force(self, name, value, inputSource):
        if False:
            print('Hello World!')
        "\n        Force isSet(name) to return 'value'.\n\n        This returns a token; hold onto the token and call token.release() when\n        you no longer want to force the state.\n\n        Example::\n\n            # set up\n            token = inputState.force('forward', True, inputSource='myForwardForcer')\n            ...\n            # tear down\n            token.release()\n        "
        token = InputStateForceToken(self)
        self._token2forceInfo[token] = (name, inputSource)
        if value:
            if name in self._forcingOff:
                self.notify.error("%s is trying to force '%s' to ON, but '%s' is already being forced OFF by %s" % (inputSource, name, name, self._forcingOff[name]))
            self._forcingOn.setdefault(name, set())
            self._forcingOn[name].add(inputSource)
        else:
            if name in self._forcingOn:
                self.notify.error("%s is trying to force '%s' to OFF, but '%s' is already being forced ON by %s" % (inputSource, name, name, self._forcingOn[name]))
            self._forcingOff.setdefault(name, set())
            self._forcingOff[name].add(inputSource)
        return token

    def _unforce(self, token):
        if False:
            while True:
                i = 10
        "\n        Stop forcing a value. Don't call this directly, call release() on your token.\n        "
        (name, inputSource) = self._token2forceInfo[token]
        token.invalidate()
        if name in self._forcingOn:
            self._forcingOn[name].discard(inputSource)
            if len(self._forcingOn[name]) == 0:
                del self._forcingOn[name]
        if name in self._forcingOff:
            self._forcingOff[name].discard(inputSource)
            if len(self._forcingOff[name]) == 0:
                del self._forcingOff[name]

    def debugPrint(self, message):
        if False:
            while True:
                i = 10
        'for debugging'
        return self.notify.debug('%s (%s) %s' % (id(self), len(self._state), message))
    watch_with_modifiers = watchWithModifiers
    is_set = isSet
    get_event_name = getEventName
    debug_print = debugPrint
    release_inputs = releaseInputs