"""This defines the Messenger class, which is responsible for most of the
:ref:`event handling <event-handlers>` that happens on the Python side.
"""
__all__ = ['Messenger']
from direct.stdpy.threading import Lock
from direct.directnotify import DirectNotifyGlobal
from .PythonUtil import safeRepr
import types

class Messenger:
    notify = DirectNotifyGlobal.directNotify.newCategory('Messenger')

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        One is keyed off the event name. It has the following structure::\n\n            {event1: {object1: [method, extraArgs, persistent],\n                       object2: [method, extraArgs, persistent]},\n             event2: {object1: [method, extraArgs, persistent],\n                       object2: [method, extraArgs, persistent]}}\n\n        This dictionary allows for efficient callbacks when the\n        messenger hears an event.\n\n        A second dictionary remembers which objects are accepting which\n        events. This allows for efficient ignoreAll commands.\n\n        Or, for an example with more real data::\n\n            {'mouseDown': {avatar: [avatar.jump, [2.0], 1]}}\n        "
        self.__callbacks = {}
        self.__objectEvents = {}
        self._messengerIdGen = 0
        self._id2object = {}
        self._eventQueuesByTaskChain = {}
        self.lock = Lock()
        if __debug__:
            self.__isWatching = 0
            self.__watching = {}
        self.quieting = {'NewFrame': 1, 'avatarMoving': 1, 'event-loop-done': 1, 'collisionLoopFinished': 1}

    def _getMessengerId(self, object):
        if False:
            print('Hello World!')
        if not hasattr(object, '_MSGRmessengerId'):
            object._MSGRmessengerId = (object.__class__.__name__, self._messengerIdGen)
            self._messengerIdGen += 1
        return object._MSGRmessengerId

    def _storeObject(self, object):
        if False:
            for i in range(10):
                print('nop')
        id = self._getMessengerId(object)
        if id not in self._id2object:
            self._id2object[id] = [1, object]
        else:
            self._id2object[id][0] += 1

    def _getObject(self, id):
        if False:
            return 10
        return self._id2object[id][1]

    def _getObjects(self):
        if False:
            return 10
        self.lock.acquire()
        try:
            objs = []
            for (refCount, obj) in self._id2object.values():
                objs.append(obj)
            return objs
        finally:
            self.lock.release()

    def _getNumListeners(self, event):
        if False:
            i = 10
            return i + 15
        return len(self.__callbacks.get(event, {}))

    def _getEvents(self):
        if False:
            return 10
        return list(self.__callbacks.keys())

    def _releaseObject(self, object):
        if False:
            i = 10
            return i + 15
        id = self._getMessengerId(object)
        if id in self._id2object:
            record = self._id2object[id]
            record[0] -= 1
            if record[0] <= 0:
                del self._id2object[id]

    def future(self, event):
        if False:
            while True:
                i = 10
        ' Returns a future that is triggered by the given event name.  This\n        will function only once. '
        from .EventManagerGlobal import eventMgr
        return eventMgr.eventHandler.get_future(event)

    def accept(self, event, object, method, extraArgs=[], persistent=1):
        if False:
            for i in range(10):
                print('nop')
        ' accept(self, string, DirectObject, Function, List, Boolean)\n\n        Make this object accept this event. When the event is\n        sent (using Messenger.send or from C++), method will be executed,\n        optionally passing in extraArgs.\n\n        If the persistent flag is set, it will continue to respond\n        to this event, otherwise it will respond only once.\n        '
        notifyDebug = Messenger.notify.getDebug()
        if notifyDebug:
            Messenger.notify.debug('object: %s (%s)\n accepting: %s\n method: %s\n extraArgs: %s\n persistent: %s' % (safeRepr(object), self._getMessengerId(object), event, safeRepr(method), safeRepr(extraArgs), persistent))
        assert hasattr(method, '__call__'), 'method not callable in accept (ignoring): %s %s' % (safeRepr(method), safeRepr(extraArgs))
        if not (isinstance(extraArgs, list) or isinstance(extraArgs, tuple) or isinstance(extraArgs, set)):
            raise TypeError('A list is required as extraArgs argument')
        self.lock.acquire()
        try:
            acceptorDict = self.__callbacks.setdefault(event, {})
            id = self._getMessengerId(object)
            if id in acceptorDict:
                if notifyDebug:
                    oldMethod = acceptorDict[id][0]
                    if oldMethod == method:
                        self.notify.warning('object: %s was already accepting: "%s" with same callback: %s()' % (object.__class__.__name__, safeRepr(event), method.__name__))
                    else:
                        self.notify.warning('object: %s accept: "%s" new callback: %s() supplanting old callback: %s()' % (object.__class__.__name__, safeRepr(event), method.__name__, oldMethod.__name__))
            acceptorDict[id] = [method, extraArgs, persistent]
            eventDict = self.__objectEvents.setdefault(id, {})
            if event not in eventDict:
                self._storeObject(object)
                eventDict[event] = None
        finally:
            self.lock.release()

    def ignore(self, event, object):
        if False:
            for i in range(10):
                print('nop')
        ' ignore(self, string, DirectObject)\n        Make this object no longer respond to this event.\n        It is safe to call even if it was not already accepting\n        '
        if Messenger.notify.getDebug():
            Messenger.notify.debug(safeRepr(object) + ' (%s)\n now ignoring: ' % (self._getMessengerId(object),) + safeRepr(event))
        self.lock.acquire()
        try:
            id = self._getMessengerId(object)
            acceptorDict = self.__callbacks.get(event)
            if acceptorDict and id in acceptorDict:
                del acceptorDict[id]
                if len(acceptorDict) == 0:
                    del self.__callbacks[event]
            eventDict = self.__objectEvents.get(id)
            if eventDict and event in eventDict:
                del eventDict[event]
                if len(eventDict) == 0:
                    del self.__objectEvents[id]
                self._releaseObject(object)
        finally:
            self.lock.release()

    def ignoreAll(self, object):
        if False:
            while True:
                i = 10
        '\n        Make this object no longer respond to any events it was accepting\n        Useful for cleanup\n        '
        if Messenger.notify.getDebug():
            Messenger.notify.debug(safeRepr(object) + ' (%s)\n now ignoring all events' % (self._getMessengerId(object),))
        self.lock.acquire()
        try:
            id = self._getMessengerId(object)
            eventDict = self.__objectEvents.get(id)
            if eventDict:
                for event in list(eventDict.keys()):
                    acceptorDict = self.__callbacks.get(event)
                    if acceptorDict and id in acceptorDict:
                        del acceptorDict[id]
                        if len(acceptorDict) == 0:
                            del self.__callbacks[event]
                    self._releaseObject(object)
                del self.__objectEvents[id]
        finally:
            self.lock.release()

    def getAllAccepting(self, object):
        if False:
            return 10
        '\n        Returns the list of all events accepted by the indicated object.\n        '
        self.lock.acquire()
        try:
            id = self._getMessengerId(object)
            eventDict = self.__objectEvents.get(id)
            if eventDict:
                return list(eventDict.keys())
            return []
        finally:
            self.lock.release()

    def isAccepting(self, event, object):
        if False:
            for i in range(10):
                print('nop')
        ' isAccepting(self, string, DirectOject)\n        Is this object accepting this event?\n        '
        self.lock.acquire()
        try:
            acceptorDict = self.__callbacks.get(event)
            id = self._getMessengerId(object)
            if acceptorDict and id in acceptorDict:
                return 1
            return 0
        finally:
            self.lock.release()

    def whoAccepts(self, event):
        if False:
            return 10
        '\n        Return objects accepting the given event\n        '
        return self.__callbacks.get(event)

    def isIgnoring(self, event, object):
        if False:
            while True:
                i = 10
        ' isIgnorning(self, string, DirectObject)\n        Is this object ignoring this event?\n        '
        return not self.isAccepting(event, object)

    def send(self, event, sentArgs=[], taskChain=None):
        if False:
            return 10
        '\n        Send this event, optionally passing in arguments.\n\n        Args:\n            event (str): The name of the event.\n            sentArgs (list): A list of arguments to be passed along to the\n                handlers listening to this event.\n            taskChain (str, optional): If not None, the name of the task chain\n                which should receive the event.  If None, then the event is\n                handled immediately. Setting a non-None taskChain will defer\n                the event (possibly till next frame or even later) and create a\n                new, temporary task within the named taskChain, but this is the\n                only way to send an event across threads.\n        '
        if Messenger.notify.getDebug() and (not self.quieting.get(event)):
            assert Messenger.notify.debug('sent event: %s sentArgs = %s, taskChain = %s' % (event, sentArgs, taskChain))
        self.lock.acquire()
        try:
            foundWatch = 0
            if __debug__:
                if self.__isWatching:
                    for i in self.__watching:
                        if str(event).find(i) >= 0:
                            foundWatch = 1
                            break
            acceptorDict = self.__callbacks.get(event)
            if not acceptorDict:
                if __debug__:
                    if foundWatch:
                        print('Messenger: "%s" was sent, but no function in Python listened.' % (event,))
                return
            if taskChain:
                from direct.task.TaskManagerGlobal import taskMgr
                queue = self._eventQueuesByTaskChain.setdefault(taskChain, [])
                queue.append((acceptorDict, event, sentArgs, foundWatch))
                if len(queue) == 1:
                    taskMgr.add(self.__taskChainDispatch, name='Messenger-%s' % taskChain, extraArgs=[taskChain], taskChain=taskChain, appendTask=True)
            else:
                self.__dispatch(acceptorDict, event, sentArgs, foundWatch)
        finally:
            self.lock.release()

    def __taskChainDispatch(self, taskChain, task):
        if False:
            for i in range(10):
                print('nop')
        ' This task is spawned each time an event is sent across\n        task chains.  Its job is to empty the task events on the queue\n        for this particular task chain.  This guarantees that events\n        are still delivered in the same order they were sent. '
        while True:
            eventTuple = None
            self.lock.acquire()
            try:
                queue = self._eventQueuesByTaskChain.get(taskChain, None)
                if queue:
                    eventTuple = queue[0]
                    del queue[0]
                if not queue:
                    if queue is not None:
                        del self._eventQueuesByTaskChain[taskChain]
                if not eventTuple:
                    return task.done
                self.__dispatch(*eventTuple)
            finally:
                self.lock.release()
        return task.done

    def __dispatch(self, acceptorDict, event, sentArgs, foundWatch):
        if False:
            for i in range(10):
                print('nop')
        for id in list(acceptorDict.keys()):
            callInfo = acceptorDict.get(id)
            if callInfo:
                (method, extraArgs, persistent) = callInfo
                if not persistent:
                    eventDict = self.__objectEvents.get(id)
                    if eventDict and event in eventDict:
                        del eventDict[event]
                        if len(eventDict) == 0:
                            del self.__objectEvents[id]
                        self._releaseObject(self._getObject(id))
                    del acceptorDict[id]
                    if event in self.__callbacks and len(self.__callbacks[event]) == 0:
                        del self.__callbacks[event]
                if __debug__:
                    if foundWatch:
                        print('Messenger: "%s" --> %s%s' % (event, self.__methodRepr(method), tuple(extraArgs + sentArgs)))
                assert hasattr(method, '__call__')
                self.lock.release()
                try:
                    result = method(*extraArgs + sentArgs)
                finally:
                    self.lock.acquire()
                if hasattr(result, 'cr_await'):
                    from direct.task.TaskManagerGlobal import taskMgr
                    taskMgr.add(result)

    def clear(self):
        if False:
            return 10
        '\n        Start fresh with a clear dict\n        '
        self.lock.acquire()
        try:
            self.__callbacks.clear()
            self.__objectEvents.clear()
            self._id2object.clear()
        finally:
            self.lock.release()

    def isEmpty(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.__callbacks) == 0

    def getEvents(self):
        if False:
            return 10
        return list(self.__callbacks.keys())

    def replaceMethod(self, oldMethod, newFunction):
        if False:
            while True:
                i = 10
        '\n        This is only used by Finder.py - the module that lets\n        you redefine functions with Control-c-Control-v\n        '
        retFlag = 0
        for entry in list(self.__callbacks.items()):
            (event, objectDict) = entry
            for objectEntry in list(objectDict.items()):
                (object, params) = objectEntry
                method = params[0]
                if isinstance(method, types.MethodType):
                    function = method.__func__
                else:
                    function = method
                if function == oldMethod:
                    newMethod = types.MethodType(newFunction, method.__self__)
                    params[0] = newMethod
                    retFlag += 1
        return retFlag

    def toggleVerbose(self):
        if False:
            for i in range(10):
                print('nop')
        isVerbose = 1 - Messenger.notify.getDebug()
        Messenger.notify.setDebug(isVerbose)
        if isVerbose:
            print('Verbose mode true.  quiet list = %s' % (list(self.quieting.keys()),))
    if __debug__:

        def watch(self, needle):
            if False:
                i = 10
                return i + 15
            '\n            return a matching event (needle) if found (in haystack).\n            This is primarily a debugging tool.\n\n            This is intended for debugging use only.\n            This function is not defined if python is ran with -O (optimize).\n\n            See Also: `unwatch`\n            '
            if not self.__watching.get(needle):
                self.__isWatching += 1
                self.__watching[needle] = 1

        def unwatch(self, needle):
            if False:
                return 10
            '\n            return a matching event (needle) if found (in haystack).\n            This is primarily a debugging tool.\n\n            This is intended for debugging use only.\n            This function is not defined if python is ran with -O (optimize).\n\n            See Also: `watch`\n            '
            if self.__watching.get(needle):
                self.__isWatching -= 1
                del self.__watching[needle]

        def quiet(self, message):
            if False:
                i = 10
                return i + 15
            "\n            When verbose mode is on, don't spam the output with messages\n            marked as quiet.\n            This is primarily a debugging tool.\n\n            This is intended for debugging use only.\n            This function is not defined if python is ran with -O (optimize).\n\n            See Also: `unquiet`\n            "
            if not self.quieting.get(message):
                self.quieting[message] = 1

        def unquiet(self, message):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Remove a message from the list of messages that are not reported\n            in verbose mode.\n            This is primarily a debugging tool.\n\n            This is intended for debugging use only.\n            This function is not defined if python is ran with -O (optimize).\n\n            See Also: `quiet`\n            '
            if self.quieting.get(message):
                del self.quieting[message]

    def find(self, needle):
        if False:
            print('Hello World!')
        '\n        return a matching event (needle) if found (in haystack).\n        This is primarily a debugging tool.\n        '
        for event in sorted(self.__callbacks):
            if repr(event).find(needle) >= 0:
                return {event: self.__callbacks[event]}

    def findAll(self, needle, limit=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        return a dict of events (needle) if found (in haystack).\n        limit may be None or an integer (e.g. 1).\n        This is primarily a debugging tool.\n        '
        matches = {}
        for event in sorted(self.__callbacks):
            if repr(event).find(needle) >= 0:
                matches[event] = self.__callbacks[event]
                if limit > 0:
                    limit -= 1
                    if limit == 0:
                        break
        return matches

    def __methodRepr(self, method):
        if False:
            return 10
        '\n        return string version of class.method or method.\n        '
        if isinstance(method, types.MethodType):
            functionName = method.__self__.__class__.__name__ + '.' + method.__func__.__name__
        elif hasattr(method, '__name__'):
            functionName = method.__name__
        else:
            return ''
        return functionName

    def __eventRepr(self, event):
        if False:
            print('Hello World!')
        '\n        Compact version of event, acceptor pairs\n        '
        str = event.ljust(32) + '\t'
        acceptorDict = self.__callbacks[event]
        for (key, (method, extraArgs, persistent)) in list(acceptorDict.items()):
            str = str + self.__methodRepr(method) + ' '
        str = str + '\n'
        return str

    def __repr__(self):
        if False:
            return 10
        '\n        Compact version of event, acceptor pairs\n        '
        str = 'The messenger is currently handling:\n' + '=' * 64 + '\n'
        for event in sorted(self.__callbacks):
            str += self.__eventRepr(event)
        str += '=' * 64 + '\n'
        for (key, eventDict) in list(self.__objectEvents.items()):
            object = self._getObject(key)
            str += '%s:\n' % repr(object)
            for event in list(eventDict.keys()):
                str += '     %s\n' % repr(event)
        str += '=' * 64 + '\n' + 'End of messenger info.\n'
        return str

    def detailedRepr(self):
        if False:
            print('Hello World!')
        '\n        Print out the table in a detailed readable format\n        '
        str = 'Messenger\n'
        str = str + '=' * 50 + '\n'
        for event in sorted(self.__callbacks):
            acceptorDict = self.__callbacks[event]
            str = str + 'Event: ' + event + '\n'
            for key in list(acceptorDict.keys()):
                (function, extraArgs, persistent) = acceptorDict[key]
                object = self._getObject(key)
                objectClass = getattr(object, '__class__', None)
                if objectClass:
                    className = objectClass.__name__
                else:
                    className = 'Not a class'
                functionName = function.__name__
                str = str + '\t' + 'Acceptor:     ' + className + ' instance' + '\n\t' + 'Function name:' + functionName + '\n\t' + 'Extra Args:   ' + repr(extraArgs) + '\n\t' + 'Persistent:   ' + repr(persistent) + '\n'
                if isinstance(function, types.MethodType):
                    str = str + '\t' + 'Method:       ' + repr(function) + '\n\t' + 'Function:     ' + repr(function.__func__) + '\n'
                else:
                    str = str + '\t' + 'Function:     ' + repr(function) + '\n'
        str = str + '=' * 50 + '\n'
        return str
    get_events = getEvents
    is_ignoring = isIgnoring
    who_accepts = whoAccepts
    find_all = findAll
    replace_method = replaceMethod
    ignore_all = ignoreAll
    is_accepting = isAccepting
    is_empty = isEmpty
    detailed_repr = detailedRepr
    get_all_accepting = getAllAccepting
    toggle_verbose = toggleVerbose