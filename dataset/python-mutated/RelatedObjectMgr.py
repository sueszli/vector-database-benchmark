"""RelatedObjectMgr module: contains the RelatedObjectMgr class"""
from direct.showbase import DirectObject
from direct.task.TaskManagerGlobal import taskMgr
from direct.directnotify import DirectNotifyGlobal

class RelatedObjectMgr(DirectObject.DirectObject):
    """
    This class manages a relationship between DistributedObjects that
    know about each other, and are expected to be generated together.
    Ideally, we should be able to guarantee the ordering of the
    generate calls, but there are certain cases in which the objects
    may not be generated in the correct order as defined by the
    toon.dc file.

    To handle cases like these robustly, it is necessary for each
    object to deal with the possibility that its companion object has
    not yet been generated.  This may mean deferring some operations
    until the expected companion object has been generated.

    This class helps manage that process.  To use it, an object should
    register its desire to be associated with the other object's doId.
    When the other object is generated (or immediately, if the object
    already exists), the associated callback will be called.  There is
    also a timeout callback in case the object never appears.
    """
    notify = DirectNotifyGlobal.directNotify.newCategory('RelatedObjectMgr')
    doLaterSequence = 1

    def __init__(self, cr):
        if False:
            for i in range(10):
                print('nop')
        self.cr = cr
        self.pendingObjects = {}

    def destroy(self):
        if False:
            return 10
        self.abortAllRequests()
        del self.cr
        del self.pendingObjects

    def requestObjects(self, doIdList, allCallback=None, eachCallback=None, timeout=None, timeoutCallback=None):
        if False:
            while True:
                i = 10
        '\n        Requests a callback to be called when the objects in the\n        doIdList are generated.  The allCallback will be called only\n        when all the objects have been generated (and it receives a\n        list of objects, in the order given in doIdList).  The\n        eachCallback is called as each object is generated, and\n        receives only the object itself.\n\n        If the objects already exist, the appropriate callback is\n        called immediately.\n\n        If all of the objects are not generated within the indicated\n        timeout time, the timeoutCallback is called instead, with the\n        original doIdList as the parameter.  If the timeoutCallback is\n        None, then allCallback is called on timeout, with the list of\n        objects that have been generated so far, and None for objects\n        that have not been generated.\n\n        If any element of doIdList is None or 0, it is ignored, and\n        None is passed in its place in the object list passed to the\n        callback.\n\n        The return value may be saved and passed to a future call to\n        abortRequest(), in order to abort a pending request before the\n        timeout expires.\n\n        Actually, you should be careful to call abortRequest() if you\n        have made a call to requestObjects() that has not been resolved.\n        To find examples, do a search for abortRequest() to find out\n        how other code is using it.  A common idiom is to store the\n        result from requestObjects() and call abortRequest() if delete()\n        or destroy() is called on the requesting object.\n\n        See Also: abortRequest()\n        '
        assert self.notify.debug('requestObjects(%s, timeout=%s)' % (doIdList, timeout))
        (objects, doIdsPending) = self.__generateObjectList(doIdList)
        if eachCallback:
            for object in objects:
                if object:
                    eachCallback(object)
        if len(doIdsPending) == 0:
            assert self.notify.debug('All objects already exist.')
            if allCallback:
                allCallback(objects)
            return
        assert self.notify.debug('Some objects pending: %s' % doIdsPending)
        doIdList = doIdList[:]
        doLaterName = None
        if timeout is not None:
            doLaterName = 'RelatedObject-%s' % RelatedObjectMgr.doLaterSequence
            assert self.notify.debug('doLaterName = %s' % doLaterName)
            RelatedObjectMgr.doLaterSequence += 1
        tuple = (allCallback, eachCallback, timeoutCallback, doIdsPending, doIdList, doLaterName)
        for doId in doIdsPending:
            pendingList = self.pendingObjects.get(doId)
            if pendingList is None:
                pendingList = []
                self.pendingObjects[doId] = pendingList
                self.__listenFor(doId)
            pendingList.append(tuple)
        if doLaterName:
            taskMgr.doMethodLater(timeout, self.__timeoutExpired, doLaterName, extraArgs=[tuple])
        return tuple

    def abortRequest(self, tuple):
        if False:
            return 10
        '\n        Aborts a previous request.  The parameter is the return value\n        from a previous call to requestObjects().  The pending request\n        is removed from the queue and no further callbacks will be called.\n\n        See Also: requestObjects()\n        '
        if tuple:
            (allCallback, eachCallback, timeoutCallback, doIdsPending, doIdList, doLaterName) = tuple
            assert self.notify.debug('aborting request for %s (remaining: %s)' % (doIdList, doIdsPending))
            if doLaterName:
                taskMgr.remove(doLaterName)
            self.__removePending(tuple, doIdsPending)

    def abortAllRequests(self):
        if False:
            while True:
                i = 10
        '\n        Call this method to abruptly abort all pending requests, but\n        leave the RelatedObjectMgr in a state for accepting more\n        requests.\n        '
        self.ignoreAll()
        for pendingList in self.pendingObjects.values():
            for tuple in pendingList:
                (allCallback, eachCallback, timeoutCallback, doIdsPending, doIdList, doLaterName) = tuple
                if doLaterName:
                    taskMgr.remove(doLaterName)
        self.pendingObjects = {}

    def __timeoutExpired(self, tuple):
        if False:
            print('Hello World!')
        (allCallback, eachCallback, timeoutCallback, doIdsPending, doIdList, doLaterName) = tuple
        assert self.notify.debug('timeout expired for %s (remaining: %s)' % (doIdList, doIdsPending))
        self.__removePending(tuple, doIdsPending)
        if timeoutCallback:
            timeoutCallback(doIdList)
        else:
            (objects, doIdsPending) = self.__generateObjectList(doIdList)
            if allCallback:
                allCallback(objects)

    def __removePending(self, tuple, doIdsPending):
        if False:
            print('Hello World!')
        while len(doIdsPending) > 0:
            doId = doIdsPending.pop()
            pendingList = self.pendingObjects[doId]
            pendingList.remove(tuple)
            if len(pendingList) == 0:
                del self.pendingObjects[doId]
                self.__noListenFor(doId)

    def __listenFor(self, doId):
        if False:
            while True:
                i = 10
        assert self.notify.debug('Now listening for generate from %s' % doId)
        announceGenerateName = 'generate-%s' % doId
        self.acceptOnce(announceGenerateName, self.__generated)

    def __noListenFor(self, doId):
        if False:
            for i in range(10):
                print('nop')
        assert self.notify.debug('No longer listening for generate from %s' % doId)
        announceGenerateName = 'generate-%s' % doId
        self.ignore(announceGenerateName)

    def __generated(self, object):
        if False:
            for i in range(10):
                print('nop')
        doId = object.doId
        assert self.notify.debug('Got generate from %s' % doId)
        pendingList = self.pendingObjects[doId]
        del self.pendingObjects[doId]
        for tuple in pendingList:
            (allCallback, eachCallback, timeoutCallback, doIdsPending, doIdList, doLaterName) = tuple
            doIdsPending.remove(doId)
            if eachCallback:
                eachCallback(object)
            if len(doIdsPending) == 0:
                assert self.notify.debug('All objects generated on list: %s' % (doIdList,))
                if doLaterName:
                    taskMgr.remove(doLaterName)
                (objects, doIdsPending) = self.__generateObjectList(doIdList)
                if None in objects:
                    assert self.notify.warning('calling %s with None.\n objects=%s\n doIdsPending=%s\n doIdList=%s\n' % (allCallback, objects, doIdsPending, doIdList))
                if allCallback:
                    allCallback(objects)
            else:
                assert self.notify.debug('Objects still pending: %s' % doIdsPending)

    def __generateObjectList(self, doIdList):
        if False:
            for i in range(10):
                print('nop')
        objects = []
        doIdsPending = []
        for doId in doIdList:
            if doId:
                object = self.cr.doId2do.get(doId)
                objects.append(object)
                if object is None:
                    doIdsPending.append(doId)
            else:
                objects.append(None)
        return (objects, doIdsPending)