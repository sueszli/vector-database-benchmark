"""CRCache module: contains the CRCache class"""
from direct.directnotify import DirectNotifyGlobal
from direct.showbase.MessengerGlobal import messenger
from direct.showbase.PythonUtil import safeRepr, itype
from . import DistributedObject

class CRCache:
    notify = DirectNotifyGlobal.directNotify.newCategory('CRCache')

    def __init__(self, maxCacheItems=10):
        if False:
            i = 10
            return i + 15
        self.maxCacheItems = maxCacheItems
        self.storedCacheItems = maxCacheItems
        self.dict = {}
        self.fifo = []

    def isEmpty(self):
        if False:
            return 10
        return len(self.fifo) == 0

    def flush(self):
        if False:
            return 10
        '\n        Delete each item in the cache then clear all references to them\n        '
        assert self.checkCache()
        CRCache.notify.debug('Flushing the cache')
        messenger.send('clientCleanup')
        delayDeleted = []
        for distObj in self.dict.values():
            distObj.deleteOrDelay()
            if distObj.getDelayDeleteCount() != 0:
                delayDeleted.append(distObj)
            if distObj.getDelayDeleteCount() <= 0:
                distObj.detectLeaks()
        delayDeleteLeaks = []
        for distObj in delayDeleted:
            if distObj.getDelayDeleteCount() != 0:
                delayDeleteLeaks.append(distObj)
        if len(delayDeleteLeaks) > 0:
            s = 'CRCache.flush:'
            for obj in delayDeleteLeaks:
                s += '\n  could not delete %s (%s), delayDeletes=%s' % (safeRepr(obj), itype(obj), obj.getDelayDeleteNames())
            self.notify.error(s)
        self.dict = {}
        self.fifo = []

    def cache(self, distObj):
        if False:
            while True:
                i = 10
        assert isinstance(distObj, DistributedObject.DistributedObject)
        assert self.checkCache()
        doId = distObj.getDoId()
        success = False
        if doId in self.dict:
            CRCache.notify.warning('Double cache attempted for distObj ' + str(doId))
        else:
            distObj.disableAndAnnounce()
            self.fifo.append(distObj)
            self.dict[doId] = distObj
            success = True
            if len(self.fifo) > self.maxCacheItems:
                oldestDistObj = self.fifo.pop(0)
                del self.dict[oldestDistObj.getDoId()]
                oldestDistObj.deleteOrDelay()
                if oldestDistObj.getDelayDeleteCount() <= 0:
                    oldestDistObj.detectLeaks()
        assert len(self.dict) == len(self.fifo)
        return success

    def retrieve(self, doId):
        if False:
            i = 10
            return i + 15
        assert self.checkCache()
        if doId in self.dict:
            distObj = self.dict[doId]
            del self.dict[doId]
            self.fifo.remove(distObj)
            return distObj
        else:
            return None

    def contains(self, doId):
        if False:
            return 10
        return doId in self.dict

    def delete(self, doId):
        if False:
            print('Hello World!')
        assert self.checkCache()
        assert doId in self.dict
        distObj = self.dict[doId]
        del self.dict[doId]
        self.fifo.remove(distObj)
        distObj.deleteOrDelay()
        if distObj.getDelayDeleteCount() <= 0:
            distObj.detectLeaks()

    def checkCache(self):
        if False:
            print('Hello World!')
        from panda3d.core import NodePath
        for obj in self.dict.values():
            if isinstance(obj, NodePath):
                assert not obj.isEmpty() and obj.getTopNode() != render.node()
        return 1

    def turnOff(self):
        if False:
            for i in range(10):
                print('nop')
        self.flush()
        self.storedMaxCache = self.maxCacheItems
        self.maxCacheItems = 0

    def turnOn(self):
        if False:
            for i in range(10):
                print('nop')
        self.maxCacheItems = self.storedMaxCache