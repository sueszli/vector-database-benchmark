"""DistributedSmoothNodeBase module: contains the DistributedSmoothNodeBase class"""
from .ClockDelta import globalClockDelta
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from direct.showbase.PythonUtil import randFloat
from panda3d.direct import CDistributedSmoothNodeBase
from enum import IntEnum

class DummyTaskClass:

    def setDelay(self, blah):
        if False:
            print('Hello World!')
        pass
DummyTask = DummyTaskClass()

class DistributedSmoothNodeBase:
    """common base class for DistributedSmoothNode and DistributedSmoothNodeAI
    """

    class BroadcastTypes(IntEnum):
        FULL = 0
        XYH = 1
        XY = 2

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.__broadcastPeriod = None

    def generate(self):
        if False:
            print('Hello World!')
        self.cnode = CDistributedSmoothNodeBase()
        self.cnode.setClockDelta(globalClockDelta)
        self.d_broadcastPosHpr = None

    def disable(self):
        if False:
            while True:
                i = 10
        del self.cnode
        self.stopPosHprBroadcast()

    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def b_clearSmoothing(self):
        if False:
            return 10
        self.d_clearSmoothing()
        self.clearSmoothing()

    def d_clearSmoothing(self):
        if False:
            return 10
        self.sendUpdate('clearSmoothing', [0])

    def getPosHprBroadcastTaskName(self):
        if False:
            print('Hello World!')
        return 'sendPosHpr-%s' % self.doId

    def setPosHprBroadcastPeriod(self, period):
        if False:
            print('Hello World!')
        self.__broadcastPeriod = period

    def getPosHprBroadcastPeriod(self):
        if False:
            print('Hello World!')
        return self.__broadcastPeriod

    def stopPosHprBroadcast(self):
        if False:
            while True:
                i = 10
        taskMgr.remove(self.getPosHprBroadcastTaskName())
        self.d_broadcastPosHpr = None

    def posHprBroadcastStarted(self):
        if False:
            for i in range(10):
                print('nop')
        return self.d_broadcastPosHpr is not None

    def wantSmoothPosBroadcastTask(self):
        if False:
            return 10
        return True

    def startPosHprBroadcast(self, period=0.2, stagger=0, type=None):
        if False:
            while True:
                i = 10
        if self.cnode is None:
            self.initializeCnode()
        BT = DistributedSmoothNodeBase.BroadcastTypes
        if type is None:
            type = BT.FULL
        self.broadcastType = type
        broadcastFuncs = {BT.FULL: self.cnode.broadcastPosHprFull, BT.XYH: self.cnode.broadcastPosHprXyh, BT.XY: self.cnode.broadcastPosHprXy}
        self.d_broadcastPosHpr = broadcastFuncs[self.broadcastType]
        taskName = self.getPosHprBroadcastTaskName()
        self.cnode.initialize(self, self.dclass, self.doId)
        self.setPosHprBroadcastPeriod(period)
        self.b_clearSmoothing()
        self.cnode.sendEverything()
        taskMgr.remove(taskName)
        delay = 0.0
        if stagger:
            delay = randFloat(period)
        if self.wantSmoothPosBroadcastTask():
            taskMgr.doMethodLater(self.__broadcastPeriod + delay, self._posHprBroadcast, taskName)

    def _posHprBroadcast(self, task=DummyTask):
        if False:
            print('Hello World!')
        self.d_broadcastPosHpr()
        task.setDelay(self.__broadcastPeriod)
        return Task.again

    def sendCurrentPosition(self):
        if False:
            while True:
                i = 10
        if self.d_broadcastPosHpr is None:
            self.cnode.initialize(self, self.dclass, self.doId)
        self.cnode.sendEverything()