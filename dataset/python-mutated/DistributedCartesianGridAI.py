from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from .DistributedNodeAI import DistributedNodeAI
from .CartesianGridBase import CartesianGridBase

class DistributedCartesianGridAI(DistributedNodeAI, CartesianGridBase):
    notify = directNotify.newCategory('DistributedCartesianGridAI')
    RuleSeparator = ':'

    def __init__(self, air, startingZone, gridSize, gridRadius, cellWidth, style='Cartesian'):
        if False:
            return 10
        DistributedNodeAI.__init__(self, air)
        self.style = style
        self.startingZone = startingZone
        self.gridSize = gridSize
        self.gridRadius = gridRadius
        self.cellWidth = cellWidth
        self.gridObjects = {}
        self.updateTaskStarted = 0

    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        DistributedNodeAI.delete(self)
        self.stopUpdateGridTask()

    def isGridParent(self):
        if False:
            print('Hello World!')
        return 1

    def getCellWidth(self):
        if False:
            for i in range(10):
                print('nop')
        return self.cellWidth

    def getParentingRules(self):
        if False:
            print('Hello World!')
        self.notify.debug('calling getter')
        rule = '%i%s%i%s%i' % (self.startingZone, self.RuleSeparator, self.gridSize, self.RuleSeparator, self.gridRadius)
        return [self.style, rule]

    def addObjectToGrid(self, av, useZoneId=-1, startAutoUpdate=True):
        if False:
            return 10
        self.notify.debug('setting parent to grid %s' % self)
        avId = av.doId
        self.gridObjects[avId] = av
        self.handleAvatarZoneChange(av, useZoneId)
        if startAutoUpdate and (not self.updateTaskStarted):
            self.startUpdateGridTask()

    def removeObjectFromGrid(self, av):
        if False:
            while True:
                i = 10
        avId = av.doId
        if avId in self.gridObjects:
            del self.gridObjects[avId]
        if len(self.gridObjects) == 0:
            self.stopUpdateGridTask()

    def startUpdateGridTask(self):
        if False:
            print('Hello World!')
        self.stopUpdateGridTask()
        self.updateTaskStarted = 1
        taskMgr.add(self.updateGridTask, self.taskName('updateGridTask'))

    def stopUpdateGridTask(self):
        if False:
            print('Hello World!')
        taskMgr.remove(self.taskName('updateGridTask'))
        self.updateTaskStarted = 0

    def updateGridTask(self, task=None):
        if False:
            for i in range(10):
                print('nop')
        missingObjs = []
        for avId in list(self.gridObjects.keys()):
            av = self.gridObjects[avId]
            if av.isEmpty():
                task.setDelay(1.0)
                del self.gridObjects[avId]
                continue
            pos = av.getPos()
            if (pos[0] < 0 or pos[1] < 0) or (pos[0] > self.cellWidth or pos[1] > self.cellWidth):
                self.handleAvatarZoneChange(av)
        if task:
            task.setDelay(1.0)
        return Task.again

    def handleAvatarZoneChange(self, av, useZoneId=-1):
        if False:
            return 10
        if useZoneId == -1:
            pos = av.getPos(self)
            zoneId = self.getZoneFromXYZ(pos)
        else:
            pos = None
            zoneId = useZoneId
        if not self.isValidZone(zoneId):
            self.notify.warning('%s handleAvatarZoneChange %s: not a valid zone (%s) for pos %s' % (self.doId, av.doId, zoneId, pos))
            return
        av.b_setLocation(self.doId, zoneId)

    def handleSetLocation(self, av, parentId, zoneId):
        if False:
            while True:
                i = 10
        pass