"""
>>> from direct.showbase import ObjectReport

>>> o=ObjectReport.ObjectReport('baseline')
>>> run()
...

>>> o2=ObjectReport.ObjectReport('')
>>> o.diff(o2)
"""
from __future__ import annotations
__all__ = ['ExclusiveObjectPool', 'ObjectReport']
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.showbase.DirectObject import DirectObject
from direct.showbase.ObjectPool import ObjectPool
from direct.showbase.GarbageReport import GarbageReport
from direct.showbase.PythonUtil import makeList, Sync, SerialNumGen
import gc
import sys
import builtins

class ExclusiveObjectPool(DirectObject):
    _ExclObjs: list[object] = []
    _ExclObjIds: dict[int, int] = {}
    _SyncMaster = Sync('ExclusiveObjectPool.ExcludedObjectList')
    _SerialNumGen = SerialNumGen()

    @classmethod
    def addExclObjs(cls, *objs):
        if False:
            return 10
        for obj in makeList(objs):
            if id(obj) not in cls._ExclObjIds:
                cls._ExclObjs.append(obj)
            cls._ExclObjIds.setdefault(id(obj), 0)
            cls._ExclObjIds[id(obj)] += 1
        cls._SyncMaster.change()

    @classmethod
    def removeExclObjs(cls, *objs):
        if False:
            for i in range(10):
                print('nop')
        for obj in makeList(objs):
            assert id(obj) in cls._ExclObjIds
            cls._ExclObjIds[id(obj)] -= 1
            if cls._ExclObjIds[id(obj)] == 0:
                del cls._ExclObjIds[id(obj)]
                cls._ExclObjs.remove(obj)
        cls._SyncMaster.change()

    def __init__(self, objects):
        if False:
            print('Hello World!')
        self._objects = list(objects)
        self._postFilterObjs = []
        self._sync = Sync('%s-%s' % (self.__class__.__name__, self._SerialNumGen.next()), self._SyncMaster)
        self._sync.invalidate()
        ExclusiveObjectPool.addExclObjs(self._objects, self._postFilterObjs, self._sync)

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        self.ignoreAll()
        ExclusiveObjectPool.removeExclObjs(self._objects, self._postFilterObjs, self._sync)
        del self._objects
        del self._postFilterObjs
        del self._sync

    def _resync(self):
        if False:
            print('Hello World!')
        if self._sync.isSynced(self._SyncMaster):
            return
        if hasattr(self, '_filteredPool'):
            ExclusiveObjectPool.removeExclObjs(*self._filteredPool._getInternalObjs())
            ExclusiveObjectPool.removeExclObjs(self._filteredPool)
            del self._filteredPool
        del self._postFilterObjs[:]
        for obj in self._objects:
            if id(obj) not in ExclusiveObjectPool._ExclObjIds:
                self._postFilterObjs.append(obj)
        self._filteredPool = ExclusiveObjectPool(self._postFilterObjs)
        ExclusiveObjectPool.addExclObjs(self._filteredPool)
        ExclusiveObjectPool.addExclObjs(*self._filteredPool._getInternalObjs())
        self._sync.sync(self._SyncMaster)

    def getObjsOfType(self, type):
        if False:
            print('Hello World!')
        self._resync()
        return self._filteredPool.getObjsOfType(type)

    def printObjsOfType(self, type):
        if False:
            print('Hello World!')
        self._resync()
        return self._filteredPool.printObjsOfType(type)

    def diff(self, other):
        if False:
            i = 10
            return i + 15
        self._resync()
        return self._filteredPool.diff(other._filteredPool)

    def typeFreqStr(self):
        if False:
            print('Hello World!')
        self._resync()
        return self._filteredPool.typeFreqStr()

    def __len__(self):
        if False:
            return 10
        self._resync()
        return len(self._filteredPool)

class ObjectReport:
    """report on every Python object in the current process"""
    notify = directNotify.newCategory('ObjectReport')

    def __init__(self, name, log=True):
        if False:
            for i in range(10):
                print('nop')
        gr = GarbageReport("ObjectReport's GarbageReport: %s" % name, log=log)
        gr.destroy()
        del gr
        self._name = name
        self._pool = ObjectPool(self._getObjectList())
        if log:
            self.notify.info("===== ObjectReport: '%s' =====\n%s" % (self._name, self.typeFreqStr()))

    def destroy(self):
        if False:
            i = 10
            return i + 15
        self._pool.destroy()
        del self._pool
        del self._name

    def typeFreqStr(self):
        if False:
            return 10
        return self._pool.typeFreqStr()

    def diff(self, other):
        if False:
            return 10
        return self._pool.diff(other._pool)

    def getObjectPool(self):
        if False:
            print('Hello World!')
        return self._pool

    def _getObjectList(self):
        if False:
            return 10
        if hasattr(sys, 'getobjects'):
            return sys.getobjects(0)
        else:
            gc.collect()
            gc_objects = gc.get_objects()
            objects = gc_objects
            objects.append(builtins.__dict__)
            nextObjList = gc_objects
            found = set()
            found.add(id(objects))
            found.add(id(found))
            found.add(id(gc_objects))
            for obj in objects:
                found.add(id(obj))
            while len(nextObjList) > 0:
                curObjList = nextObjList
                nextObjList = []
                for obj in curObjList:
                    refs = gc.get_referents(obj)
                    for ref in refs:
                        if id(ref) not in found:
                            found.add(id(ref))
                            objects.append(ref)
                            nextObjList.append(ref)
            return objects