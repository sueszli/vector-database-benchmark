from direct.directnotify.DirectNotifyGlobal import directNotify

class DoHierarchy:
    """
    This table has been a source of memory leaks, with DoIds getting left in the table indefinitely.
    DoHierarchy guards access to the table and ensures correctness.
    """
    notify = directNotify.newCategory('DoHierarchy')

    def __init__(self):
        if False:
            while True:
                i = 10
        self._table = {}
        self._allDoIds = set()

    def isEmpty(self):
        if False:
            while True:
                i = 10
        assert (len(self._table) == 0) == (len(self._allDoIds) == 0)
        return len(self._table) == 0 and len(self._allDoIds) == 0

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._allDoIds)

    def clear(self):
        if False:
            return 10
        assert self.notify.debugCall()
        self._table = {}
        self._allDoIds = set()

    def getDoIds(self, getDo, parentId, zoneId=None, classType=None):
        if False:
            i = 10
            return i + 15
        "\n        Args:\n            parentId: any distributed object id.\n            zoneId: a uint32, defaults to None (all zones).  Try zone 2 if\n                you're not sure which zone to use (0 is a bad/null zone and\n                1 has had reserved use in the past as a no messages zone, while\n                2 has traditionally been a global, uber, misc stuff zone).\n            dclassType: a distributed class type filter, defaults to None\n                (no filter).\n\n        If dclassName is None then all objects in the zone are returned;\n        otherwise the list is filtered to only include objects of that type.\n        "
        parent = self._table.get(parentId)
        if parent is None:
            return []
        if zoneId is None:
            r = []
            for zone in parent.values():
                for obj in zone:
                    r.append(obj)
        else:
            r = parent.get(zoneId, [])
        if classType is not None:
            a = []
            for doId in r:
                obj = getDo(doId)
                if isinstance(obj, classType):
                    a.append(doId)
            r = a
        return r

    def storeObjectLocation(self, do, parentId, zoneId):
        if False:
            i = 10
            return i + 15
        doId = do.doId
        if doId in self._allDoIds:
            self.notify.error("storeObjectLocation(%s %s) already in _allDoIds; duplicate generate()? or didn't clean up previous instance of DO?" % (do.__class__.__name__, do.doId))
        parentZoneDict = self._table.setdefault(parentId, {})
        zoneDoSet = parentZoneDict.setdefault(zoneId, set())
        zoneDoSet.add(doId)
        self._allDoIds.add(doId)
        self.notify.debug('storeObjectLocation: %s(%s) @ (%s, %s)' % (do.__class__.__name__, doId, parentId, zoneId))

    def deleteObjectLocation(self, do, parentId, zoneId):
        if False:
            i = 10
            return i + 15
        doId = do.doId
        if doId not in self._allDoIds:
            self.notify.error('deleteObjectLocation(%s %s) not in _allDoIds; duplicate delete()? or invalid previous location on a new object?' % (do.__class__.__name__, do.doId))
        if doId not in self._allDoIds:
            return
        parentZoneDict = self._table.get(parentId)
        if parentZoneDict is not None:
            zoneDoSet = parentZoneDict.get(zoneId)
            if zoneDoSet is not None:
                if doId in zoneDoSet:
                    zoneDoSet.remove(doId)
                    self._allDoIds.remove(doId)
                    self.notify.debug('deleteObjectLocation: %s(%s) @ (%s, %s)' % (do.__class__.__name__, doId, parentId, zoneId))
                    if len(zoneDoSet) == 0:
                        del parentZoneDict[zoneId]
                        if len(parentZoneDict) == 0:
                            del self._table[parentId]
                else:
                    self.notify.error('deleteObjectLocation: objId: %s not found' % doId)
            else:
                self.notify.error('deleteObjectLocation: zoneId: %s not found' % zoneId)
        else:
            self.notify.error('deleteObjectLocation: parentId: %s not found' % parentId)