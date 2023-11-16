from direct.distributed import DoHierarchy
import re
BAD_DO_ID = BAD_ZONE_ID = 0
BAD_CHANNEL_ID = 0

class DoCollectionManager:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.doId2do = {}
        if self.hasOwnerView():
            self.doId2ownerView = {}
        self._doHierarchy = DoHierarchy.DoHierarchy()

    def getDo(self, doId):
        if False:
            i = 10
            return i + 15
        return self.doId2do.get(doId)

    def getGameDoId(self):
        if False:
            i = 10
            return i + 15
        return self.GameGlobalsId

    def callbackWithDo(self, doId, callback):
        if False:
            print('Hello World!')
        do = self.doId2do.get(doId)
        if do is not None:
            callback(do)
        else:
            relatedObjectMgr(doId, allCallback=callback)

    def getOwnerView(self, doId):
        if False:
            return 10
        assert self.hasOwnerView()
        return self.doId2ownerView.get(doId)

    def callbackWithOwnerView(self, doId, callback):
        if False:
            for i in range(10):
                print('nop')
        assert self.hasOwnerView()
        do = self.doId2ownerView.get(doId)
        if do is not None:
            callback(do)
        else:
            pass

    def getDoTable(self, ownerView):
        if False:
            for i in range(10):
                print('nop')
        if ownerView:
            assert self.hasOwnerView()
            return self.doId2ownerView
        else:
            return self.doId2do

    def doFind(self, str):
        if False:
            return 10
        '\n        Returns list of distributed objects with matching str in value.\n        '
        for value in self.doId2do.values():
            if repr(value).find(str) >= 0:
                return value

    def doFindAll(self, str):
        if False:
            print('Hello World!')
        '\n        Returns list of distributed objects with matching str in value.\n        '
        matches = []
        for value in self.doId2do.values():
            if repr(value).find(str) >= 0:
                matches.append(value)
        return matches

    def doFindAllMatching(self, str):
        if False:
            i = 10
            return i + 15
        '\n        Returns list of distributed objects with matching str in value.\n        '
        matches = []
        for value in self.doId2do.values():
            if re.search(str, repr(value)):
                matches.append(value)
        return matches

    def doFindAllOfType(self, query):
        if False:
            return 10
        '\n        Useful method for searching through the Distributed Object collection\n        for objects of a particular type\n        '
        matches = []
        for value in self.doId2do.values():
            if query in str(value.__class__):
                matches.append(value)
        return (matches, len(matches))

    def doFindAllInstances(self, cls):
        if False:
            while True:
                i = 10
        matches = []
        for value in self.doId2do.values():
            if isinstance(value, cls):
                matches.append(value)
        return matches

    def _getDistanceFromLA(self, do):
        if False:
            print('Hello World!')
        if hasattr(do, 'getPos'):
            return do.getPos(localAvatar).length()
        return None

    def _compareDistance(self, do1, do2):
        if False:
            return 10
        dist1 = self._getDistanceFromLA(do1)
        dist2 = self._getDistanceFromLA(do2)
        if dist1 is None and dist2 is None:
            return 0
        if dist1 is None:
            return 1
        if dist2 is None:
            return -1
        if dist1 < dist2:
            return -1
        return 1

    def dosByDistance(self):
        if False:
            while True:
                i = 10
        objs = list(self.doId2do.values())
        objs.sort(cmp=self._compareDistance)
        return objs

    def doByDistance(self):
        if False:
            return 10
        objs = self.dosByDistance()
        for obj in objs:
            print('%s\t%s\t%s' % (obj.doId, self._getDistanceFromLA(obj), obj.dclass.getName()))
    if __debug__:

        def printObjects(self):
            if False:
                print('Hello World!')
            format = '%10s %10s %10s %30s %20s'
            title = format % ('parentId', 'zoneId', 'doId', 'dclass', 'name')
            print(title)
            print('-' * len(title))
            for distObj in self.doId2do.values():
                print(format % (distObj.__dict__.get('parentId'), distObj.__dict__.get('zoneId'), distObj.__dict__.get('doId'), distObj.dclass.getName(), distObj.__dict__.get('name')))

    def _printObjects(self, table):
        if False:
            return 10
        class2count = {}
        for obj in self.getDoTable(ownerView=False).values():
            className = obj.__class__.__name__
            class2count.setdefault(className, 0)
            class2count[className] += 1
        count2classes = invertDictLossless(class2count)
        for count in sorted(count2classes, reverse=True):
            count2classes[count].sort()
            for name in count2classes[count]:
                print('%s %s' % (count, name))
        print('')

    def _returnObjects(self, table):
        if False:
            for i in range(10):
                print('nop')
        class2count = {}
        stringToReturn = ''
        for obj in self.getDoTable(ownerView=False).values():
            className = obj.__class__.__name__
            class2count.setdefault(className, 0)
            class2count[className] += 1
        count2classes = invertDictLossless(class2count)
        for count in sorted(count2classes, reverse=True):
            count2classes[count].sort()
            for name in count2classes[count]:
                stringToReturn = '%s%s %s\n' % (stringToReturn, count, name)
        return stringToReturn

    def webPrintObjectCount(self):
        if False:
            while True:
                i = 10
        strToReturn = '==== OBJECT COUNT ====\n'
        if self.hasOwnerView():
            strToReturn = '%s == doId2do\n' % strToReturn
        strToReturn = '%s%s' % (strToReturn, self._returnObjects(self.getDoTable(ownerView=False)))
        if self.hasOwnerView():
            strToReturn = '%s\n== doId2ownerView\n' % strToReturn
            strToReturn = '%s%s' % (strToReturn, self._returnObjects(self.getDoTable(ownerView=False)))
        return strToReturn

    def printObjectCount(self):
        if False:
            while True:
                i = 10
        print('==== OBJECT COUNT ====')
        if self.hasOwnerView():
            print('== doId2do')
        self._printObjects(self.getDoTable(ownerView=False))
        if self.hasOwnerView():
            print('== doId2ownerView')
            self._printObjects(self.getDoTable(ownerView=True))

    def getDoList(self, parentId, zoneId=None, classType=None):
        if False:
            print('Hello World!')
        "\n        Args:\n            parentId: any distributed object id.\n            zoneId: a uint32, defaults to None (all zones).  Try zone 2 if\n                you're not sure which zone to use (0 is a bad/null zone and\n                1 has had reserved use in the past as a no messages zone, while\n                2 has traditionally been a global, uber, misc stuff zone).\n            dclassType: a distributed class type filter, defaults to None\n                (no filter).\n\n        If dclassName is None then all objects in the zone are returned;\n        otherwise the list is filtered to only include objects of that type.\n        "
        return [self.doId2do.get(i) for i in self.getDoIdList(parentId, zoneId, classType)]

    def getDoIdList(self, parentId, zoneId=None, classType=None):
        if False:
            print('Hello World!')
        return self._doHierarchy.getDoIds(self.getDo, parentId, zoneId, classType)

    def hasOwnerViewDoId(self, doId):
        if False:
            print('Hello World!')
        assert self.hasOwnerView()
        return doId in self.doId2ownerView

    def getOwnerViewDoList(self, classType):
        if False:
            return 10
        assert self.hasOwnerView()
        l = []
        for obj in self.doId2ownerView.values():
            if isinstance(obj, classType):
                l.append(obj)
        return l

    def getOwnerViewDoIdList(self, classType):
        if False:
            i = 10
            return i + 15
        assert self.hasOwnerView()
        l = []
        for (doId, obj) in self.doId2ownerView.items():
            if isinstance(obj, classType):
                l.append(doId)
        return l

    def countObjects(self, classType):
        if False:
            while True:
                i = 10
        '\n        Counts the number of objects of the given type in the\n        repository (for testing purposes)\n        '
        count = 0
        for dobj in self.doId2do.values():
            if isinstance(dobj, classType):
                count += 1
        return count

    def getAllOfType(self, type):
        if False:
            return 10
        result = []
        for obj in self.doId2do.values():
            if isinstance(obj, type):
                result.append(obj)
        return result

    def findAnyOfType(self, type):
        if False:
            i = 10
            return i + 15
        for obj in self.doId2do.values():
            if isinstance(obj, type):
                return obj
        return None

    def deleteDistributedObjects(self):
        if False:
            for i in range(10):
                print('nop')
        for do in list(self.doId2do.values()):
            self.deleteDistObject(do)
        self.deleteObjects()
        if not self._doHierarchy.isEmpty():
            self.notify.warning('_doHierarchy table not empty: %s' % self._doHierarchy)
            self._doHierarchy.clear()

    def handleObjectLocation(self, di):
        if False:
            i = 10
            return i + 15
        doId = di.getUint32()
        parentId = di.getUint32()
        zoneId = di.getUint32()
        obj = self.doId2do.get(doId)
        if obj is not None:
            self.notify.debug('handleObjectLocation: doId: %s parentId: %s zoneId: %s' % (doId, parentId, zoneId))
            obj.setLocation(parentId, zoneId)
        else:
            self.notify.warning('handleObjectLocation: Asked to update non-existent obj: %s' % doId)

    def handleSetLocation(self, di):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugStateCall(self)
        parentId = di.getUint32()
        zoneId = di.getUint32()
        distObj = self.doId2do.get(self.getMsgChannel())
        if distObj is not None:
            distObj.setLocation(parentId, zoneId)
        else:
            self.notify.warning('handleSetLocation: object %s not present' % self.getMsgChannel())

    def storeObjectLocation(self, object, parentId, zoneId):
        if False:
            for i in range(10):
                print('nop')
        oldParentId = object.parentId
        oldZoneId = object.zoneId
        if oldParentId != parentId:
            oldParentObj = self.doId2do.get(oldParentId)
            if oldParentObj is not None:
                oldParentObj.handleChildLeave(object, oldZoneId)
            self.deleteObjectLocation(object, oldParentId, oldZoneId)
        elif oldZoneId != zoneId:
            oldParentObj = self.doId2do.get(oldParentId)
            if oldParentObj is not None:
                oldParentObj.handleChildLeaveZone(object, oldZoneId)
            self.deleteObjectLocation(object, oldParentId, oldZoneId)
        else:
            return
        if parentId is None or zoneId is None or parentId == zoneId == 0:
            object.parentId = None
            object.zoneId = None
        else:
            self._doHierarchy.storeObjectLocation(object, parentId, zoneId)
            object.parentId = parentId
            object.zoneId = zoneId
        if oldParentId != parentId:
            parentObj = self.doId2do.get(parentId)
            if parentObj is not None:
                parentObj.handleChildArrive(object, zoneId)
            elif parentId not in (None, 0, self.getGameDoId()):
                self.notify.warning('storeObjectLocation(%s): parent %s not present' % (object.doId, parentId))
        elif oldZoneId != zoneId:
            parentObj = self.doId2do.get(parentId)
            if parentObj is not None:
                parentObj.handleChildArriveZone(object, zoneId)
            elif parentId not in (None, 0, self.getGameDoId()):
                self.notify.warning('storeObjectLocation(%s): parent %s not present' % (object.doId, parentId))

    def deleteObjectLocation(self, object, parentId, zoneId):
        if False:
            for i in range(10):
                print('nop')
        if parentId is None or zoneId is None or parentId == zoneId == 0:
            return
        self._doHierarchy.deleteObjectLocation(object, parentId, zoneId)

    def addDOToTables(self, do, location=None, ownerView=False):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugStateCall(self)
        if not ownerView:
            if location is None:
                location = (do.parentId, do.zoneId)
        doTable = self.getDoTable(ownerView)
        if do.doId in doTable:
            if ownerView:
                tableName = 'doId2ownerView'
            else:
                tableName = 'doId2do'
            self.notify.error('doId %s already in %s [%s stomping %s]' % (do.doId, tableName, do.__class__.__name__, doTable[do.doId].__class__.__name__))
        doTable[do.doId] = do
        if not ownerView:
            if self.isValidLocationTuple(location):
                self.storeObjectLocation(do, location[0], location[1])

    def isValidLocationTuple(self, location):
        if False:
            for i in range(10):
                print('nop')
        return location is not None and location != (4294967295, 4294967295) and (location != (0, 0))
    if __debug__:

        def isInDoTables(self, doId):
            if False:
                i = 10
                return i + 15
            assert self.notify.debugStateCall(self)
            return doId in self.doId2do

    def removeDOFromTables(self, do):
        if False:
            i = 10
            return i + 15
        assert self.notify.debugStateCall(self)
        location = do.getLocation()
        if location:
            (oldParentId, oldZoneId) = location
            oldParentObj = self.doId2do.get(oldParentId)
            if oldParentObj:
                oldParentObj.handleChildLeave(do, oldZoneId)
        self.deleteObjectLocation(do, do.parentId, do.zoneId)
        if do.doId in self.doId2do:
            del self.doId2do[do.doId]

    def getObjectsInZone(self, parentId, zoneId):
        if False:
            return 10
        '\n        returns dict of doId:distObj for a zone.\n        returned dict is safely mutable.\n        '
        assert self.notify.debugStateCall(self)
        doDict = {}
        for doId in self.getDoIdList(parentId, zoneId):
            doDict[doId] = self.getDo(doId)
        return doDict

    def getObjectsOfClassInZone(self, parentId, zoneId, objClass):
        if False:
            i = 10
            return i + 15
        "\n        returns dict of doId:object for a zone, containing all objects\n        that inherit from 'class'. returned dict is safely mutable.\n        "
        assert self.notify.debugStateCall(self)
        doDict = {}
        for doId in self.getDoIdList(parentId, zoneId, objClass):
            doDict[doId] = self.getDo(doId)
        return doDict