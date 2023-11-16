from __future__ import annotations
from direct.directnotify import DirectNotifyGlobal
from direct.showbase.DirectObject import DirectObject
from direct.showbase.MessengerGlobal import messenger
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import ConfigVariableDouble, ConfigVariableInt, ConfigVariableBool
ASYNC_REQUEST_DEFAULT_TIMEOUT_IN_SECONDS = 8.0
ASYNC_REQUEST_INFINITE_RETRIES = -1
ASYNC_REQUEST_DEFAULT_NUM_RETRIES = 0
if __debug__:
    _overrideTimeoutTimeForAllAsyncRequests = ConfigVariableDouble('async-request-timeout', -1.0)
    _overrideNumRetriesForAllAsyncRequests = ConfigVariableInt('async-request-num-retries', -1)
    _breakOnTimeout = ConfigVariableBool('async-request-break-on-timeout', False)

class AsyncRequest(DirectObject):
    """
    This class is used to make asynchronous reads and creates to a database.

    You can create a list of self.neededObjects and then ask for each to be
    read or created, or if you only have one object that you need you can
    skip the self.neededObjects because calling askForObject or createObject
    will set the self.neededObjects value for you.

    Once all the objects have been read or created, the self.finish() method
    will be called.  You may override this function to run your code in a
    derived class.

    If you wish to queue up several items that you all need before the finish
    method is called, you can put items in self.neededObjects and then call
    askForObject or createObject afterwards.  That way the _checkCompletion
    will not call finish until after all the requests have been done.

    If you need to chain serveral object reads or creates, just add more
    entries to the self.neededObjects dictionary in the self.finish function
    and return without calling AsyncRequest.finish().  Your finish method
    will be called again when the new self.neededObjects is complete.  You
    may repeat this as necessary.
    """
    _asyncRequests: dict[int, AsyncRequest] = {}
    notify = DirectNotifyGlobal.directNotify.newCategory('AsyncRequest')

    def __init__(self, air, replyToChannelId=None, timeoutTime=ASYNC_REQUEST_DEFAULT_TIMEOUT_IN_SECONDS, numRetries=ASYNC_REQUEST_DEFAULT_NUM_RETRIES):
        if False:
            print('Hello World!')
        '\n        air is the AI Respository.\n        replyToChannelId may be an avatarId, an accountId, or a channelId.\n        timeoutTime is how many seconds to wait before aborting the request.\n        numRetries is the number of times to retry the request before giving up.\n        '
        assert AsyncRequest.notify.debugCall()
        if __debug__:
            if _overrideTimeoutTimeForAllAsyncRequests.getValue() >= 0.0:
                timeoutTime = _overrideTimeoutTimeForAllAsyncRequests.getValue()
            if _overrideNumRetriesForAllAsyncRequests.getValue() >= 0:
                numRetries = _overrideNumRetriesForAllAsyncRequests.getValue()
        AsyncRequest._asyncRequests[id(self)] = self
        self.deletingMessage = 'AsyncRequest-deleting-%s' % id(self)
        self.air = air
        self.replyToChannelId = replyToChannelId
        self.timeoutTask = None
        self.neededObjects = {}
        self._timeoutTime = timeoutTime
        self._initialNumRetries = numRetries

    def delete(self):
        if False:
            print('Hello World!')
        assert AsyncRequest.notify.debugCall()
        del AsyncRequest._asyncRequests[id(self)]
        self.ignoreAll()
        self._resetTimeoutTask(False)
        messenger.send(self.deletingMessage, [])
        del self.neededObjects
        del self.air
        del self.replyToChannelId

    def askForObjectField(self, dclassName, fieldName, doId, key=None, context=None):
        if False:
            while True:
                i = 10
        '\n        Request an already created object, i.e. read from database.\n        '
        assert AsyncRequest.notify.debugCall()
        if key is None:
            key = fieldName
        assert doId
        if context is None:
            context = self.air.allocateContext()
        self.air.contextToClassName[context] = dclassName
        self.acceptOnce('doFieldResponse-%s' % (context,), self._checkCompletion, [key])
        self.neededObjects[key] = None
        self.air.queryObjectField(dclassName, fieldName, doId, context)
        self._resetTimeoutTask()

    def askForObjectFields(self, dclassName, fieldNames, doId, key=None, context=None):
        if False:
            return 10
        '\n        Request an already created object, i.e. read from database.\n        '
        assert AsyncRequest.notify.debugCall()
        if key is None:
            key = fieldNames[0]
        assert doId
        if context is None:
            context = self.air.allocateContext()
        self.air.contextToClassName[context] = dclassName
        self.acceptOnce('doFieldResponse-%s' % (context,), self._checkCompletion, [key])
        self.air.queryObjectFields(dclassName, fieldNames, doId, context)
        self._resetTimeoutTask()

    def askForObjectFieldsByString(self, dbId, dclassName, objString, fieldNames, key=None, context=None):
        if False:
            for i in range(10):
                print('nop')
        assert AsyncRequest.notify.debugCall()
        assert dbId
        if key is None:
            key = fieldNames
        if context is None:
            context = self.air.allocateContext()
        self.air.contextToClassName[context] = dclassName
        self.acceptOnce('doFieldResponse-%s' % (context,), self._checkCompletion, [key])
        self.air.queryObjectStringFields(dbId, dclassName, objString, fieldNames, context)
        self._resetTimeoutTask()

    def askForObject(self, doId, context=None):
        if False:
            return 10
        '\n        Request an already created object, i.e. read from database.\n        '
        assert AsyncRequest.notify.debugCall()
        assert doId
        if context is None:
            context = self.air.allocateContext()
        self.acceptOnce('doRequestResponse-%s' % (context,), self._checkCompletion, [None])
        self.air.queryObjectAll(doId, context)
        self._resetTimeoutTask()

    def createObject(self, name, className, databaseId=None, values=None, context=None):
        if False:
            print('Hello World!')
        '\n        Create a new database object.  You can get the doId from within\n        your self.finish() function.\n\n        This functions is different from createObjectId in that it does\n        generate the object when the response comes back.  The object is\n        added to the doId2do and so forth and treated as a full regular\n        object (which it is).  This is useful on the AI where we really\n        do want the object on the AI.\n        '
        assert AsyncRequest.notify.debugCall()
        assert name
        assert className
        self.neededObjects[name] = None
        if context is None:
            context = self.air.allocateContext()
        self.accept(self.air.getDatabaseGenerateResponseEvent(context), self._doCreateObject, [name, className, values])
        self.air.requestDatabaseGenerate(className, context, databaseId=databaseId, values=values)
        self._resetTimeoutTask()

    def createObjectId(self, name, className, values=None, context=None):
        if False:
            return 10
        "\n        Create a new database object.  You can get the doId from within\n        your self.finish() function.\n\n        This functions is different from createObject in that it does not\n        generate the object when the response comes back.  It only tells you\n        the doId.  This is useful on the UD where we don't really want the\n        object on the UD, we just want the object created and the UD wants\n        to send messages to it using the ID.\n        "
        assert AsyncRequest.notify.debugCall()
        assert name
        assert className
        self.neededObjects[name] = None
        if context is None:
            context = self.air.allocateContext()
        self.accept(self.air.getDatabaseGenerateResponseEvent(context), self._checkCompletion, [name, None])
        self.air.requestDatabaseGenerate(className, context, values=values)
        self._resetTimeoutTask()

    def finish(self):
        if False:
            print('Hello World!')
        '\n        This is the function that gets called when all of the needed objects\n        are in (i.e. all the askForObject and createObject requests have\n        been satisfied).\n        If the other requests timeout, finish will not be called.\n        '
        assert self.notify.debugCall()
        self.delete()

    def _doCreateObject(self, name, className, values, doId):
        if False:
            print('Hello World!')
        isInDoId2do = doId in self.air.doId2do
        distObj = self.air.generateGlobalObject(doId, className, values)
        if not isInDoId2do and game.name == 'uberDog':
            self.air.doId2do.pop(doId, None)
        self._checkCompletion(name, None, distObj)

    def _checkCompletion(self, name, context, distObj):
        if False:
            print('Hello World!')
        '\n        This checks whether we have all the needed objects and calls\n        finish() if we do.\n        '
        if name is not None:
            self.neededObjects[name] = distObj
        else:
            self.neededObjects[distObj.doId] = distObj
        for i in self.neededObjects.values():
            if i is None:
                return
        self.finish()

    def _resetTimeoutTask(self, createAnew=True):
        if False:
            i = 10
            return i + 15
        if self.timeoutTask:
            taskMgr.remove(self.timeoutTask)
            self.timeoutTask = None
        if createAnew:
            self.numRetries = self._initialNumRetries
            self.timeoutTask = taskMgr.doMethodLater(self._timeoutTime, self.timeout, 'AsyncRequestTimer-%s' % id(self))

    def timeout(self, task):
        if False:
            for i in range(10):
                print('nop')
        assert AsyncRequest.notify.debugCall('neededObjects: %s' % (self.neededObjects,))
        if self.numRetries > 0:
            assert AsyncRequest.notify.debug('Timed out. Trying %d more time(s) : %s' % (self.numRetries + 1, repr(self.neededObjects)))
            self.numRetries -= 1
            return Task.again
        else:
            if __debug__:
                if _breakOnTimeout:
                    if hasattr(self, 'avatarId'):
                        print('\n\nself.avatarId =', self.avatarId)
                    print('\nself.neededObjects =', self.neededObjects)
                    print('\ntimed out after %s seconds.\n\n' % (task.delayTime,))
                    import pdb
                    pdb.set_trace()
            self.delete()
            return Task.done

def cleanupAsyncRequests():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only call this when the application is shuting down.\n    '
    for asyncRequest in AsyncRequest._asyncRequests:
        asyncRequest.delete()
    assert not AsyncRequest._asyncRequests