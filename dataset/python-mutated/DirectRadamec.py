""" Class used to create and control radamec device """
from direct.showbase.DirectObject import DirectObject
from direct.task.Task import Task
from direct.task.TaskManagerGlobal import taskMgr
from .DirectDeviceManager import DirectDeviceManager
from direct.directnotify import DirectNotifyGlobal
RAD_PAN = 0
RAD_TILT = 1
RAD_ZOOM = 2
RAD_FOCUS = 3

class DirectRadamec(DirectObject):
    radamecCount = 0
    notify = DirectNotifyGlobal.directNotify.newCategory('DirectRadamec')

    def __init__(self, device='Analog0', nodePath=None):
        if False:
            return 10
        if base.direct.deviceManager is None:
            base.direct.deviceManager = DirectDeviceManager()
        self.name = 'Radamec-' + repr(DirectRadamec.radamecCount)
        DirectRadamec.radamecCount += 1
        self.device = device
        self.analogs = base.direct.deviceManager.createAnalogs(self.device)
        self.numAnalogs = len(self.analogs)
        self.aList = [0, 0, 0, 0, 0, 0, 0, 0]
        self.minRange = [-180.0, -90, 522517.0, 494762.0]
        self.maxRange = [180.0, 90, 547074.0, 533984.0]
        self.enable()

    def enable(self):
        if False:
            while True:
                i = 10
        self.disable()
        taskMgr.add(self.updateTask, self.name + '-updateTask')

    def disable(self):
        if False:
            i = 10
            return i + 15
        taskMgr.remove(self.name + '-updateTask')

    def destroy(self):
        if False:
            print('Hello World!')
        self.disable()

    def updateTask(self, state):
        if False:
            return 10
        for i in range(len(self.analogs)):
            self.aList[i] = self.analogs.getControlState(i)
        return Task.cont

    def radamecDebug(self):
        if False:
            while True:
                i = 10
        panVal = self.normalizeChannel(RAD_PAN, -180, 180)
        tiltVal = self.normalizeChannel(RAD_TILT, -90, 90)
        self.notify.debug('PAN = %s' % self.aList[RAD_PAN])
        self.notify.debug('TILT = %s' % self.aList[RAD_TILT])
        self.notify.debug('ZOOM = %s' % self.aList[RAD_ZOOM])
        self.notify.debug('FOCUS = %s' % self.aList[RAD_FOCUS])
        self.notify.debug('Normalized: panVal: %s  tiltVal: %s' % (panVal, tiltVal))

    def normalizeChannel(self, chan, minVal=-1, maxVal=1):
        if False:
            while True:
                i = 10
        if chan < 0 or chan >= min(len(self.maxRange), len(self.minRange)):
            raise RuntimeError("can't normalize this channel (channel %d)" % chan)
        maxRange = self.maxRange[chan]
        minRange = self.minRange[chan]
        diff = maxRange - minRange
        clampedVal = max(min(self.aList[chan], maxRange), maxRange)
        return (maxVal - minVal) * (clampedVal - minRange) / diff + minVal