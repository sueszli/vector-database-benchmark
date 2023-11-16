""" Class used to create and control radamec device """
from panda3d.core import Vec3
from direct.showbase.DirectObject import DirectObject
from direct.task.Task import Task
from direct.task.TaskManagerGlobal import taskMgr
from .DirectDeviceManager import DirectDeviceManager
from direct.directnotify import DirectNotifyGlobal
NULL_AXIS = -1
FAST_X = 0
FAST_Y = 1
FAST_Z = 2

class DirectFastrak(DirectObject):
    fastrakCount = 0
    notify = DirectNotifyGlobal.directNotify.newCategory('DirectFastrak')

    def __init__(self, device='Tracker0', nodePath=None):
        if False:
            print('Hello World!')
        if base.direct.deviceManager is None:
            base.direct.deviceManager = DirectDeviceManager()
        self.name = 'Fastrak-' + repr(DirectFastrak.fastrakCount)
        self.deviceNo = DirectFastrak.fastrakCount
        DirectFastrak.fastrakCount += 1
        self.device = device
        self.tracker = None
        self.trackerPos = None
        self.updateFunc = self.fastrakUpdate
        self.enable()

    def enable(self):
        if False:
            print('Hello World!')
        self.disable()
        self.tracker = base.direct.deviceManager.createTracker(self.device)
        taskMgr.add(self.updateTask, self.name + '-updateTask')

    def disable(self):
        if False:
            while True:
                i = 10
        taskMgr.remove(self.name + '-updateTask')

    def destroy(self):
        if False:
            while True:
                i = 10
        self.disable()
        self.tempCS.removeNode()

    def updateTask(self, state):
        if False:
            print('Hello World!')
        self.updateFunc()
        return Task.cont

    def fastrakUpdate(self):
        if False:
            print('Hello World!')
        pos = base.direct.fastrak[self.deviceNo].tracker.getPos()
        self.trackerPos = Vec3(3.280839895013123 * pos[2], 3.280839895013123 * pos[1], 3.280839895013123 * pos[0])
        self.notify.debug('Tracker(%d) Pos = %s' % (self.deviceNo, repr(self.trackerPos)))