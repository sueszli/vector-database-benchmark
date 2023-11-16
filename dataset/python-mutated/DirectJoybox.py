""" Class used to create and control joybox device """
from direct.showbase.DirectObject import DirectObject
from .DirectDeviceManager import ANALOG_DEADBAND, ANALOG_MAX, ANALOG_MIN, DirectDeviceManager
from direct.directtools.DirectUtil import CLAMP
from direct.gui import OnscreenText
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import ButtonRegistry, ButtonThrower, ClockObject, NodePath, VBase3, Vec3
import math
L_STICK = 0
L_UPPER = 1
L_LOWER = 2
R_STICK = 3
R_UPPER = 4
R_LOWER = 5
NULL_AXIS = -1
L_LEFT_RIGHT = 0
L_FWD_BACK = 1
L_TWIST = 2
L_SLIDE = 3
R_LEFT_RIGHT = 4
R_FWD_BACK = 5
R_TWIST = 6
R_SLIDE = 7
JOYBOX_MIN = ANALOG_MIN + ANALOG_DEADBAND
JOYBOX_MAX = ANALOG_MAX - ANALOG_DEADBAND
JOYBOX_RANGE = JOYBOX_MAX - JOYBOX_MIN
JOYBOX_TREAD_SEPERATION = 1.0

class DirectJoybox(DirectObject):
    joyboxCount = 0
    xyzMultiplier = 1.0
    hprMultiplier = 1.0

    def __init__(self, device='CerealBox', nodePath=None, headingNP=None):
        if False:
            for i in range(10):
                print('nop')
        from direct.showbase.ShowBaseGlobal import base
        if nodePath is None:
            nodePath = base.direct.camera
        if headingNP is None:
            headingNP = base.direct.camera
        if base.direct.deviceManager is None:
            base.direct.deviceManager = DirectDeviceManager()
        DirectJoybox.joyboxCount += 1
        self.name = 'Joybox-' + repr(DirectJoybox.joyboxCount)
        self.device = device
        self.analogs = base.direct.deviceManager.createAnalogs(self.device)
        self.buttons = base.direct.deviceManager.createButtons(self.device)
        self.aList = [0, 0, 0, 0, 0, 0, 0, 0]
        self.bList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.mapping = [R_LEFT_RIGHT, R_FWD_BACK, L_FWD_BACK, R_TWIST, L_TWIST, NULL_AXIS]
        self.modifier = [1, 1, 1, -1, -1, 0]
        self.lastTime = ClockObject.getGlobalClock().getFrameTime()
        self.nodePath = nodePath
        self.headingNP = headingNP
        self.useHeadingNP = False
        self.rotateInPlace = False
        self.floatingNP = NodePath('floating')
        self.refCS = base.direct.cameraControl.coaMarker
        self.tempCS = base.direct.group.attachNewNode('JoyboxTempCS')
        self.readout = OnscreenText.OnscreenText(pos=(-0.9, 0.95), font=base.direct.font, mayChange=1)
        self.modeList = [self.joeMode, self.driveMode, self.orbitMode]
        self.updateFunc = self.joyboxFly
        self.modeName = 'Joe Mode'
        self.auxData = []
        self.addButtonEvents()
        self.enable()

    def setHeadingNodePath(self, np):
        if False:
            i = 10
            return i + 15
        self.headingNP = np

    def enable(self):
        if False:
            while True:
                i = 10
        self.disable()
        self.acceptSwitchModeEvent()
        self.acceptUprightCameraEvent()
        taskMgr.add(self.updateTask, self.name + '-updateTask')

    def disable(self):
        if False:
            i = 10
            return i + 15
        taskMgr.remove(self.name + '-updateTask')
        self.ignoreSwitchModeEvent()
        self.ignoreUprightCameraEvent()

    def destroy(self):
        if False:
            while True:
                i = 10
        self.disable()
        self.tempCS.removeNode()

    def addButtonEvents(self):
        if False:
            while True:
                i = 10
        self.breg = ButtonRegistry.ptr()
        for i in range(8):
            self.buttons.setButtonMap(i, self.breg.getButton(self.getEventName(i)))
        self.eventThrower = self.buttons.getNodePath().attachNewNode(ButtonThrower('JB Button Thrower'))

    def setNodePath(self, nodePath):
        if False:
            i = 10
            return i + 15
        self.nodePath = nodePath

    def getNodePath(self):
        if False:
            for i in range(10):
                print('nop')
        return self.nodePath

    def setRefCS(self, refCS):
        if False:
            for i in range(10):
                print('nop')
        self.refCS = refCS

    def getRefCS(self):
        if False:
            for i in range(10):
                print('nop')
        return self.refCS

    def getEventName(self, index):
        if False:
            for i in range(10):
                print('nop')
        return self.name + '-button-' + repr(index)

    def setXyzMultiplier(self, multiplier):
        if False:
            while True:
                i = 10
        DirectJoybox.xyzMultiplier = multiplier

    def getXyzMultiplier(self):
        if False:
            print('Hello World!')
        return DirectJoybox.xyzMultiplier

    def setHprMultiplier(self, multiplier):
        if False:
            while True:
                i = 10
        DirectJoybox.hprMultiplier = multiplier

    def getHprMultiplier(self):
        if False:
            while True:
                i = 10
        return DirectJoybox.hprMultiplier

    def updateTask(self, state):
        if False:
            print('Hello World!')
        self.updateVals()
        self.updateFunc()
        return Task.cont

    def updateVals(self):
        if False:
            return 10
        cTime = ClockObject.getGlobalClock().getFrameTime()
        self.deltaTime = cTime - self.lastTime
        self.lastTime = cTime
        for i in range(len(self.analogs)):
            self.aList[i] = self.normalizeChannel(i)
        for i in range(len(self.buttons)):
            try:
                self.bList[i] = self.buttons[i]
            except IndexError:
                self.bList[i] = 0

    def updateValsUnrolled(self):
        if False:
            i = 10
            return i + 15
        cTime = ClockObject.getGlobalClock().getFrameTime()
        self.deltaTime = cTime - self.lastTime
        self.lastTime = cTime
        for chan in range(len(self.analogs)):
            val = self.analogs.getControlState(chan)
            if val < 0:
                val = min(val + ANALOG_DEADBAND, 0.0)
            else:
                val = max(val - ANALOG_DEADBAND, 0.0)
            if chan == L_TWIST or chan == R_TWIST:
                val *= 3.0
            val = CLAMP(val, JOYBOX_MIN, JOYBOX_MAX)
            self.aList[chan] = 2.0 * ((val - JOYBOX_MIN) / JOYBOX_RANGE) - 1
        for i in range(len(self.buttons)):
            try:
                self.bList[i] = self.buttons.getButtonState(i)
            except IndexError:
                self.bList[i] = 0

    def acceptSwitchModeEvent(self, button=R_UPPER):
        if False:
            return 10
        self.accept(self.getEventName(button), self.switchMode)

    def ignoreSwitchModeEvent(self, button=R_UPPER):
        if False:
            for i in range(10):
                print('nop')
        self.ignore(self.getEventName(button))

    def switchMode(self):
        if False:
            i = 10
            return i + 15
        try:
            self.modeFunc = self.modeList[0]
            self.modeList = self.modeList[1:] + self.modeList[:1]
            self.modeFunc()
        except IndexError:
            pass

    def showMode(self, modeText):
        if False:
            return 10

        def hideText(state, s=self):
            if False:
                print('Hello World!')
            s.readout.setText('')
            return Task.done
        taskMgr.remove(self.name + '-showMode')
        self.readout.setText(modeText)
        t = taskMgr.doMethodLater(3.0, hideText, self.name + '-showMode')
        t.setUponDeath(hideText)

    def acceptUprightCameraEvent(self, button=L_UPPER):
        if False:
            for i in range(10):
                print('nop')
        self.accept(self.getEventName(button), base.direct.cameraControl.orbitUprightCam)

    def ignoreUprightCameraEvent(self, button=L_UPPER):
        if False:
            while True:
                i = 10
        self.ignore(self.getEventName(button))

    def setMode(self, func, name):
        if False:
            i = 10
            return i + 15
        self.disable()
        self.updateFunc = func
        self.modeName = name
        self.showMode(self.modeName)
        self.enable()

    def setUseHeadingNP(self, enabled):
        if False:
            return 10
        self.useHeadingNP = enabled

    def setRotateInPlace(self, enabled):
        if False:
            i = 10
            return i + 15
        self.rotateInPlace = enabled

    def joyboxFly(self):
        if False:
            print('Hello World!')
        if self.nodePath is None:
            return
        hprScale = (self.aList[L_SLIDE] + 1.0) * 50.0 * DirectJoybox.hprMultiplier
        posScale = (self.aList[R_SLIDE] + 1.0) * 50.0 * DirectJoybox.xyzMultiplier

        def getAxisVal(index, s=self):
            if False:
                print('Hello World!')
            try:
                return s.aList[s.mapping[index]]
            except IndexError:
                return 0.0
        x = getAxisVal(0) * self.modifier[0]
        y = getAxisVal(1) * self.modifier[1]
        z = getAxisVal(2) * self.modifier[2]
        pos = Vec3(x, y, z) * (posScale * self.deltaTime)
        h = getAxisVal(3) * self.modifier[3]
        p = getAxisVal(4) * self.modifier[4]
        r = getAxisVal(5) * self.modifier[5]
        hpr = Vec3(h, p, r) * (hprScale * self.deltaTime)
        if self.useHeadingNP and self.headingNP is not None:
            oldZ = pos.getZ()
            pos = self.nodePath.getRelativeVector(self.headingNP, pos)
            pos.setZ(oldZ)
            if self.rotateInPlace:
                parent = self.nodePath.getParent()
                self.floatingNP.reparentTo(parent)
                self.floatingNP.setPos(self.headingNP, 0, 0, 0)
                self.floatingNP.setHpr(0, 0, 0)
                self.nodePath.wrtReparentTo(self.floatingNP)
                self.floatingNP.setHpr(hpr)
                self.nodePath.wrtReparentTo(parent)
                hpr = Vec3(0, 0, 0)
        self.nodePath.setPosHpr(self.nodePath, pos, hpr)

    def joeMode(self):
        if False:
            return 10
        self.mapping = [R_LEFT_RIGHT, R_FWD_BACK, L_FWD_BACK, R_TWIST, L_TWIST, NULL_AXIS]
        self.modifier = [1, 1, 1, -1, -1, 0]
        self.setMode(self.joyboxFly, 'Joe Mode')

    def basicMode(self):
        if False:
            while True:
                i = 10
        self.mapping = [NULL_AXIS, R_FWD_BACK, NULL_AXIS, R_LEFT_RIGHT, NULL_AXIS, NULL_AXIS]
        self.modifier = [0, 1, 0, -1, 0, 0]
        self.setMode(self.joyboxFly, 'Basic Mode')

    def fpsMode(self):
        if False:
            return 10
        self.mapping = [L_LEFT_RIGHT, R_FWD_BACK, L_FWD_BACK, R_LEFT_RIGHT, NULL_AXIS, NULL_AXIS]
        self.modifier = [1, 1, 1, -1, 0, 0]
        self.setMode(self.joyboxFly, 'FPS Mode')

    def tankMode(self):
        if False:
            for i in range(10):
                print('nop')
        self.setMode(self.tankFly, 'Tank Mode')

    def nullMode(self):
        if False:
            print('Hello World!')
        self.setMode(self.nullFly, 'Null Mode')

    def lucMode(self):
        if False:
            print('Hello World!')
        self.mapping = [R_LEFT_RIGHT, R_FWD_BACK, L_FWD_BACK, R_TWIST, L_TWIST, L_LEFT_RIGHT]
        self.modifier = [1, 1, 1, -1, -1, 0]
        self.setMode(self.joyboxFly, 'Luc Mode')

    def driveMode(self):
        if False:
            print('Hello World!')
        self.mapping = [L_LEFT_RIGHT, R_FWD_BACK, R_TWIST, R_LEFT_RIGHT, L_FWD_BACK, NULL_AXIS]
        self.modifier = [1, 1, -1, -1, -1, 0]
        self.setMode(self.joyboxFly, 'Drive Mode')

    def lookAtMode(self):
        if False:
            return 10
        self.mapping = [R_LEFT_RIGHT, R_TWIST, R_FWD_BACK, L_LEFT_RIGHT, L_FWD_BACK, NULL_AXIS]
        self.modifier = [1, 1, 1, -1, 1, 0]
        self.setMode(self.joyboxFly, 'Look At Mode')

    def lookAroundMode(self):
        if False:
            return 10
        self.mapping = [NULL_AXIS, NULL_AXIS, NULL_AXIS, R_LEFT_RIGHT, R_FWD_BACK, NULL_AXIS]
        self.modifier = [0, 0, 0, -1, -1, 0]
        self.setMode(self.joyboxFly, 'Lookaround Mode')

    def demoMode(self):
        if False:
            return 10
        self.mapping = [R_LEFT_RIGHT, R_FWD_BACK, L_FWD_BACK, R_TWIST, NULL_AXIS, NULL_AXIS]
        self.modifier = [1, 1, 1, -1, 0, 0]
        self.setMode(self.joyboxFly, 'Demo Mode')

    def hprXyzMode(self):
        if False:
            return 10
        self.mapping = [R_LEFT_RIGHT, R_FWD_BACK, R_TWIST, L_TWIST, L_FWD_BACK, L_LEFT_RIGHT]
        self.modifier = [1, 1, -1, -1, -1, 1]
        self.setMode(self.joyboxFly, 'HprXyz Mode')

    def mopathMode(self):
        if False:
            for i in range(10):
                print('nop')
        self.mapping = [R_LEFT_RIGHT, R_FWD_BACK, R_TWIST, L_LEFT_RIGHT, L_FWD_BACK, L_LEFT_RIGHT]
        self.modifier = [1, 1, -1, -1, 1, 0]
        self.setMode(self.joyboxFly, 'Mopath Mode')

    def walkthruMode(self):
        if False:
            for i in range(10):
                print('nop')
        self.mapping = [R_LEFT_RIGHT, R_FWD_BACK, L_TWIST, R_TWIST, L_FWD_BACK, L_LEFT_RIGHT]
        self.modifier = [1, 1, -1, -1, -1, 1]
        self.setMode(self.joyboxFly, 'Walkthru Mode')

    def spaceMode(self):
        if False:
            for i in range(10):
                print('nop')
        self.setMode(self.spaceFly, 'Space Mode')

    def nullFly(self):
        if False:
            print('Hello World!')
        return

    def tankFly(self):
        if False:
            return 10
        leftTreadSpeed = self.normalizeChannel(L_SLIDE, 0.1, 100) * DirectJoybox.xyzMultiplier * self.aList[L_FWD_BACK]
        rightTreadSpeed = self.normalizeChannel(R_SLIDE, 0.1, 100) * DirectJoybox.xyzMultiplier * self.aList[R_FWD_BACK]
        forwardSpeed = (leftTreadSpeed + rightTreadSpeed) * 0.5
        headingSpeed = math.atan2(leftTreadSpeed - rightTreadSpeed, JOYBOX_TREAD_SEPERATION)
        headingSpeed = 180 / 3.14159 * headingSpeed
        dh = -1.0 * headingSpeed * self.deltaTime * 0.3
        dy = forwardSpeed * self.deltaTime
        self.nodePath.setH(self.nodePath, dh)
        self.nodePath.setY(self.nodePath, dy)

    def spaceFly(self):
        if False:
            i = 10
            return i + 15
        if self.nodePath is None:
            return
        hprScale = self.normalizeChannel(L_SLIDE, 0.1, 100) * DirectJoybox.hprMultiplier
        posScale = self.normalizeChannel(R_SLIDE, 0.1, 100) * DirectJoybox.xyzMultiplier
        dr = -1 * hprScale * self.aList[R_TWIST] * self.deltaTime
        dp = -1 * hprScale * self.aList[R_FWD_BACK] * self.deltaTime
        dh = -1 * hprScale * self.aList[R_LEFT_RIGHT] * self.deltaTime
        self.nodePath.setHpr(self.nodePath, dh, dp, dr)
        dy = posScale * self.aList[L_FWD_BACK] * self.deltaTime
        self.nodePath.setY(self.nodePath, dy)

    def planetMode(self, auxData=[]):
        if False:
            print('Hello World!')
        self.auxData = auxData
        self.setMode(self.planetFly, 'Space Mode')

    def planetFly(self):
        if False:
            print('Hello World!')
        if self.nodePath is None:
            return
        hprScale = self.normalizeChannel(L_SLIDE, 0.1, 100) * DirectJoybox.hprMultiplier
        posScale = self.normalizeChannel(R_SLIDE, 0.1, 100) * DirectJoybox.xyzMultiplier
        dr = -1 * hprScale * self.aList[R_TWIST] * self.deltaTime
        dp = -1 * hprScale * self.aList[R_FWD_BACK] * self.deltaTime
        dh = -1 * hprScale * self.aList[R_LEFT_RIGHT] * self.deltaTime
        self.nodePath.setHpr(self.nodePath, dh, dp, dr)
        dy = posScale * self.aList[L_FWD_BACK] * self.deltaTime
        dPos = VBase3(0, dy, 0)
        for (planet, radius) in self.auxData:
            np2planet = Vec3(self.nodePath.getPos(planet))
            offsetDist = np2planet.length()
            if offsetDist > 1.2 * radius:
                pass
            else:
                oNorm = Vec3()
                oNorm.assign(np2planet)
                oNorm.normalize()
                dPlanet = self.nodePath.getMat(planet).xformVec(Vec3(0, dy, 0))
                dotProd = oNorm.dot(dPlanet)
                if dotProd < 0:
                    radialComponent = oNorm * dotProd
                    above = offsetDist - radius
                    sf = max(1.0 - max(above, 0.0) / (0.2 * radius), 0.0)
                    dPlanet -= radialComponent * (sf * sf)
                    dPos.assign(planet.getMat(self.nodePath).xformVec(dPlanet))
        self.nodePath.setPos(self.nodePath, dPos)

    def orbitMode(self):
        if False:
            return 10
        self.setMode(self.orbitFly, 'Orbit Mode')

    def orbitFly(self):
        if False:
            for i in range(10):
                print('nop')
        if self.nodePath is None:
            return
        hprScale = self.normalizeChannel(L_SLIDE, 0.1, 100) * DirectJoybox.hprMultiplier
        posScale = self.normalizeChannel(R_SLIDE, 0.1, 100) * DirectJoybox.xyzMultiplier
        r = -0.01 * posScale * self.aList[R_TWIST] * self.deltaTime
        rx = hprScale * self.aList[R_LEFT_RIGHT] * self.deltaTime
        ry = -hprScale * self.aList[R_FWD_BACK] * self.deltaTime
        x = posScale * self.aList[L_LEFT_RIGHT] * self.deltaTime
        z = posScale * self.aList[L_FWD_BACK] * self.deltaTime
        h = -1 * hprScale * self.aList[L_TWIST] * self.deltaTime
        self.nodePath.setX(self.nodePath, x)
        self.nodePath.setZ(self.nodePath, z)
        self.nodePath.setH(self.nodePath, h)
        self.orbitNode(rx, ry, 0)
        pos = self.nodePath.getPos(self.refCS)
        if Vec3(pos).length() < 0.005:
            pos.set(0, -0.01, 0)
        pos.assign(pos * (1 + r))
        self.nodePath.setPos(self.refCS, pos)

    def orbitNode(self, h, p, r):
        if False:
            return 10
        self.tempCS.setPos(self.refCS, 0, 0, 0)
        self.tempCS.setHpr(self.nodePath, 0, 0, 0)
        pos = self.nodePath.getPos(self.tempCS)
        self.tempCS.setHpr(self.tempCS, h, p, r)
        self.nodePath.setPos(self.tempCS, pos)
        self.nodePath.setHpr(self.tempCS, 0, 0, 0)

    def normalizeChannel(self, chan, minVal=-1, maxVal=1):
        if False:
            print('Hello World!')
        try:
            if chan == L_TWIST or chan == R_TWIST:
                return self.analogs.normalize(self.analogs.getControlState(chan), minVal, maxVal, 3.0)
            else:
                return self.analogs.normalize(self.analogs.getControlState(chan), minVal, maxVal)
        except IndexError:
            return 0.0