from direct.showbase.DirectObject import DirectObject
from direct.directtools.DirectUtil import *
from seGeometry import *
from direct.directtools.DirectGlobals import *
from direct.task import Task
import math
CAM_MOVE_DURATION = 1.2
COA_MARKER_SF = 0.0075
Y_AXIS = Vec3(0, 1, 0)

class DirectCameraControl(DirectObject):

    def __init__(self):
        if False:
            return 10
        self.startT = 0.0
        self.startF = 0
        self.orthoViewRoll = 0.0
        self.lastView = 0
        self.coa = Point3(0, 100, 0)
        self.coaMarker = loader.loadModel('models/misc/sphere')
        self.coaMarker.setName('DirectCameraCOAMarker')
        self.coaMarker.setTransparency(1)
        self.coaMarker.setColor(1, 0, 0, 0)
        self.coaMarker.setPos(0, 100, 0)
        useDirectRenderStyle(self.coaMarker)
        self.coaMarkerPos = Point3(0)
        self.fLockCOA = 0
        self.nullHitPointCount = 0
        self.cqEntries = []
        self.coaMarkerRef = SEditor.group.attachNewNode('coaMarkerRef')
        self.camManipRef = SEditor.group.attachNewNode('camManipRef')
        t = CAM_MOVE_DURATION
        self.actionEvents = [['DIRECT-mouse2', self.mouseFlyStart], ['DIRECT-mouse2Up', self.mouseFlyStop]]
        self.keyEvents = [['c', self.centerCamIn, 0.5], ['f', self.fitOnWidget], ['h', self.homeCam], ['shift-v', self.toggleMarkerVis], ['m', self.moveToFit], ['n', self.pickNextCOA], ['u', self.orbitUprightCam], ['shift-u', self.uprightCam], ['1', self.spawnMoveToView, 1], ['2', self.spawnMoveToView, 2], ['3', self.spawnMoveToView, 3], ['4', self.spawnMoveToView, 4], ['5', self.spawnMoveToView, 5], ['6', self.spawnMoveToView, 6], ['7', self.spawnMoveToView, 7], ['8', self.spawnMoveToView, 8], ['9', self.swingCamAboutWidget, -90.0, t], ['0', self.swingCamAboutWidget, 90.0, t], ['`', self.removeManipulateCameraTask], ['=', self.zoomCam, 0.5, t], ['+', self.zoomCam, 0.5, t], ['-', self.zoomCam, -2.0, t], ['_', self.zoomCam, -2.0, t]]

    def toggleMarkerVis(self):
        if False:
            while True:
                i = 10
        if SEditor.cameraControl.coaMarker.isHidden():
            SEditor.cameraControl.coaMarker.show()
        else:
            SEditor.cameraControl.coaMarker.hide()

    def mouseFlyStart(self, modifiers):
        if False:
            return 10
        SEditor.pushUndo([SEditor.camera])
        if abs(SEditor.dr.mouseX) < 0.9 and abs(SEditor.dr.mouseY) < 0.9:
            self.coaMarker.hide()
            self.startT = globalClock.getFrameTime()
            self.startF = globalClock.getFrameCount()
            self.spawnXZTranslateOrHPanYZoom()
        elif abs(SEditor.dr.mouseX) > 0.9 and abs(SEditor.dr.mouseY) > 0.9:
            self.spawnMouseRollTask()
        else:
            self.spawnMouseRotateTask()

    def mouseFlyStop(self):
        if False:
            i = 10
            return i + 15
        taskMgr.remove('manipulateCamera')
        stopT = globalClock.getFrameTime()
        deltaT = stopT - self.startT
        stopF = globalClock.getFrameCount()
        deltaF = stopF - self.startF
        if deltaT <= 0.25 or deltaF <= 1:
            skipFlags = SKIP_HIDDEN | SKIP_BACKFACE
            skipFlags |= SKIP_CAMERA * (1 - base.getControl())
            self.computeCOA(SEditor.iRay.pickGeom(skipFlags=skipFlags))
            self.coaMarkerRef.iPosHprScale(base.cam)
            self.cqEntries = []
            for i in range(SEditor.iRay.getNumEntries()):
                self.cqEntries.append(SEditor.iRay.getEntry(i))
        self.coaMarker.show()
        self.updateCoaMarkerSize()

    def spawnXZTranslateOrHPanYZoom(self):
        if False:
            return 10
        taskMgr.remove('manipulateCamera')
        t = Task.Task(self.XZTranslateOrHPanYZoomTask)
        t.zoomSF = Vec3(self.coaMarker.getPos(SEditor.camera)).length()
        taskMgr.add(t, 'manipulateCamera')

    def spawnXZTranslateOrHPPan(self):
        if False:
            while True:
                i = 10
        taskMgr.remove('manipulateCamera')
        taskMgr.add(self.XZTranslateOrHPPanTask, 'manipulateCamera')

    def spawnXZTranslate(self):
        if False:
            while True:
                i = 10
        taskMgr.remove('manipulateCamera')
        taskMgr.add(self.XZTranslateTask, 'manipulateCamera')

    def spawnHPanYZoom(self):
        if False:
            print('Hello World!')
        taskMgr.remove('manipulateCamera')
        t = Task.Task(self.HPanYZoomTask)
        t.zoomSF = Vec3(self.coaMarker.getPos(SEditor.camera)).length()
        taskMgr.add(t, 'manipulateCamera')

    def spawnHPPan(self):
        if False:
            print('Hello World!')
        taskMgr.remove('manipulateCamera')
        taskMgr.add(self.HPPanTask, 'manipulateCamera')

    def XZTranslateOrHPanYZoomTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        if SEditor.fShift:
            return self.XZTranslateTask(state)
        else:
            return self.HPanYZoomTask(state)

    def XZTranslateOrHPPanTask(self, state):
        if False:
            while True:
                i = 10
        if SEditor.fShift:
            return self.HPPanTask(state)
        else:
            return self.XZTranslateTask(state)

    def XZTranslateTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        coaDist = Vec3(self.coaMarker.getPos(SEditor.camera)).length()
        xlateSF = coaDist / SEditor.dr.near
        SEditor.camera.setPos(SEditor.camera, -0.5 * SEditor.dr.mouseDeltaX * SEditor.dr.nearWidth * xlateSF, 0.0, -0.5 * SEditor.dr.mouseDeltaY * SEditor.dr.nearHeight * xlateSF)
        return Task.cont

    def HPanYZoomTask(self, state):
        if False:
            return 10
        if SEditor.fControl:
            moveDir = Vec3(self.coaMarker.getPos(SEditor.camera))
            if moveDir[1] < 0.0:
                moveDir.assign(moveDir * -1)
            moveDir.normalize()
        else:
            moveDir = Vec3(Y_AXIS)
        moveDir.assign(moveDir * (-1.0 * SEditor.dr.mouseDeltaY * state.zoomSF))
        if SEditor.dr.mouseDeltaY > 0.0:
            moveDir.setY(moveDir[1] * 1.0)
        SEditor.camera.setPosHpr(SEditor.camera, moveDir[0], moveDir[1], moveDir[2], 0.5 * SEditor.dr.mouseDeltaX * SEditor.dr.fovH, 0.0, 0.0)
        return Task.cont

    def HPPanTask(self, state):
        if False:
            return 10
        SEditor.camera.setHpr(SEditor.camera, 0.5 * SEditor.dr.mouseDeltaX * SEditor.dr.fovH, -0.5 * SEditor.dr.mouseDeltaY * SEditor.dr.fovV, 0.0)
        return Task.cont

    def spawnMouseRotateTask(self):
        if False:
            print('Hello World!')
        taskMgr.remove('manipulateCamera')
        self.camManipRef.setPos(self.coaMarkerPos)
        self.camManipRef.setHpr(SEditor.camera, ZERO_POINT)
        t = Task.Task(self.mouseRotateTask)
        if abs(SEditor.dr.mouseX) > 0.9:
            t.constrainedDir = 'y'
        else:
            t.constrainedDir = 'x'
        taskMgr.add(t, 'manipulateCamera')

    def mouseRotateTask(self, state):
        if False:
            return 10
        if state.constrainedDir == 'y' and abs(SEditor.dr.mouseX) > 0.9:
            deltaX = 0
            deltaY = SEditor.dr.mouseDeltaY
        elif state.constrainedDir == 'x' and abs(SEditor.dr.mouseY) > 0.9:
            deltaX = SEditor.dr.mouseDeltaX
            deltaY = 0
        else:
            deltaX = SEditor.dr.mouseDeltaX
            deltaY = SEditor.dr.mouseDeltaY
        if SEditor.fShift:
            SEditor.camera.setHpr(SEditor.camera, deltaX * SEditor.dr.fovH, -deltaY * SEditor.dr.fovV, 0.0)
            self.camManipRef.setPos(self.coaMarkerPos)
            self.camManipRef.setHpr(SEditor.camera, ZERO_POINT)
        else:
            wrt = SEditor.camera.getTransform(self.camManipRef)
            self.camManipRef.setHpr(self.camManipRef, -1 * deltaX * 180.0, deltaY * 180.0, 0.0)
            SEditor.camera.setTransform(self.camManipRef, wrt)
        return Task.cont

    def spawnMouseRollTask(self):
        if False:
            return 10
        taskMgr.remove('manipulateCamera')
        self.camManipRef.setPos(self.coaMarkerPos)
        self.camManipRef.setHpr(SEditor.camera, ZERO_POINT)
        t = Task.Task(self.mouseRollTask)
        t.coaCenter = getScreenXY(self.coaMarker)
        t.lastAngle = getCrankAngle(t.coaCenter)
        t.wrt = SEditor.camera.getTransform(self.camManipRef)
        taskMgr.add(t, 'manipulateCamera')

    def mouseRollTask(self, state):
        if False:
            print('Hello World!')
        wrt = state.wrt
        angle = getCrankAngle(state.coaCenter)
        deltaAngle = angle - state.lastAngle
        state.lastAngle = angle
        self.camManipRef.setHpr(self.camManipRef, 0, 0, deltaAngle)
        SEditor.camera.setTransform(self.camManipRef, wrt)
        return Task.cont

    def lockCOA(self):
        if False:
            print('Hello World!')
        self.fLockCOA = 1
        SEditor.message('COA Lock On')

    def unlockCOA(self):
        if False:
            print('Hello World!')
        self.fLockCOA = 0
        SEditor.message('COA Lock Off')

    def toggleCOALock(self):
        if False:
            print('Hello World!')
        self.fLockCOA = 1 - self.fLockCOA
        if self.fLockCOA:
            SEditor.message('COA Lock On')
        else:
            SEditor.message('COA Lock Off')

    def pickNextCOA(self):
        if False:
            i = 10
            return i + 15
        ' Cycle through collision handler entries '
        if self.cqEntries:
            entry = self.cqEntries[0]
            self.cqEntries = self.cqEntries[1:] + self.cqEntries[:1]
            nodePath = entry.getIntoNodePath()
            if SEditor.camera not in nodePath.getAncestors():
                hitPt = entry.getSurfacePoint(entry.getFromNodePath())
                self.updateCoa(hitPt, ref=self.coaMarkerRef)
            else:
                self.cqEntries = self.cqEntries[:-1]
                self.pickNextCOA()

    def computeCOA(self, entry):
        if False:
            print('Hello World!')
        coa = Point3(0)
        dr = SEditor.drList.getCurrentDr()
        if self.fLockCOA:
            coa.assign(self.coaMarker.getPos(SEditor.camera))
            self.nullHitPointCount = 0
        elif entry:
            hitPt = entry.getSurfacePoint(entry.getFromNodePath())
            hitPtDist = Vec3(hitPt).length()
            coa.assign(hitPt)
            if hitPtDist < 1.1 * dr.near or hitPtDist > dr.far:
                coa.assign(self.coaMarker.getPos(SEditor.camera))
            self.nullHitPointCount = 0
        else:
            self.nullHitPointCount = (self.nullHitPointCount + 1) % 7
            dist = pow(10.0, self.nullHitPointCount)
            SEditor.message('COA Distance: ' + repr(dist))
            coa.set(0, dist, 0)
        coaDist = Vec3(coa - ZERO_POINT).length()
        if coaDist < 1.1 * dr.near:
            coa.set(0, 100, 0)
            coaDist = 100
        self.updateCoa(coa, coaDist=coaDist)

    def updateCoa(self, ref2point, coaDist=None, ref=None):
        if False:
            return 10
        self.coa.set(ref2point[0], ref2point[1], ref2point[2])
        if not coaDist:
            coaDist = Vec3(self.coa - ZERO_POINT).length()
        if ref == None:
            ref = SEditor.drList.getCurrentDr().cam
        self.coaMarker.setPos(ref, self.coa)
        pos = self.coaMarker.getPos()
        self.coaMarker.setPosHprScale(pos, Vec3(0), Vec3(1))
        self.updateCoaMarkerSize(coaDist)
        self.coaMarkerPos.assign(self.coaMarker.getPos())

    def updateCoaMarkerSizeOnDeath(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.updateCoaMarkerSize()

    def updateCoaMarkerSize(self, coaDist=None):
        if False:
            return 10
        if not coaDist:
            coaDist = Vec3(self.coaMarker.getPos(SEditor.camera)).length()
        sf = COA_MARKER_SF * coaDist * math.tan(deg2Rad(SEditor.drList.getCurrentDr().fovV))
        if sf == 0.0:
            sf = 0.1
        self.coaMarker.setScale(sf)
        self.coaMarker.colorInterval(3.0, VBase4(1, 0, 0, 0), name='fadeAway').start()

    def homeCam(self):
        if False:
            for i in range(10):
                print('nop')
        SEditor.pushUndo([SEditor.camera])
        SEditor.camera.reparentTo(render)
        SEditor.camera.clearMat()
        self.updateCoaMarkerSize()

    def uprightCam(self):
        if False:
            for i in range(10):
                print('nop')
        taskMgr.remove('manipulateCamera')
        SEditor.pushUndo([SEditor.camera])
        currH = SEditor.camera.getH()
        SEditor.camera.lerpHpr(currH, 0, 0, CAM_MOVE_DURATION, other=render, blendType='easeInOut', task='manipulateCamera')

    def orbitUprightCam(self):
        if False:
            print('Hello World!')
        taskMgr.remove('manipulateCamera')
        SEditor.pushUndo([SEditor.camera])
        mCam2Render = Mat4()
        mCam2Render.assign(SEditor.camera.getMat(render))
        zAxis = Vec3(mCam2Render.xformVec(Z_AXIS))
        zAxis.normalize()
        orbitAngle = rad2Deg(math.acos(CLAMP(zAxis.dot(Z_AXIS), -1, 1)))
        if orbitAngle < 0.1:
            return
        rotAxis = Vec3(zAxis.cross(Z_AXIS))
        rotAxis.normalize()
        rotAngle = rad2Deg(math.acos(CLAMP(rotAxis.dot(X_AXIS), -1, 1)))
        if rotAxis[1] < 0:
            rotAngle *= -1
        self.camManipRef.setPos(self.coaMarker, Vec3(0))
        self.camManipRef.setHpr(render, rotAngle, 0, 0)
        parent = SEditor.camera.getParent()
        SEditor.camera.wrtReparentTo(self.camManipRef)
        t = self.camManipRef.lerpHpr(rotAngle, orbitAngle, 0, CAM_MOVE_DURATION, other=render, blendType='easeInOut', task='manipulateCamera')
        t.parent = parent
        t.uponDeath = self.reparentCam

    def centerCam(self):
        if False:
            i = 10
            return i + 15
        self.centerCamIn(1.0)

    def centerCamNow(self):
        if False:
            return 10
        self.centerCamIn(0.0)

    def centerCamIn(self, t):
        if False:
            return 10
        taskMgr.remove('manipulateCamera')
        SEditor.pushUndo([SEditor.camera])
        markerToCam = self.coaMarker.getPos(SEditor.camera)
        dist = Vec3(markerToCam - ZERO_POINT).length()
        scaledCenterVec = Y_AXIS * dist
        delta = markerToCam - scaledCenterVec
        self.camManipRef.setPosHpr(SEditor.camera, Point3(0), Point3(0))
        t = SEditor.camera.lerpPos(Point3(delta), CAM_MOVE_DURATION, other=self.camManipRef, blendType='easeInOut', task='manipulateCamera')
        t.uponDeath = self.updateCoaMarkerSizeOnDeath

    def zoomCam(self, zoomFactor, t):
        if False:
            return 10
        taskMgr.remove('manipulateCamera')
        SEditor.pushUndo([SEditor.camera])
        zoomPtToCam = self.coaMarker.getPos(SEditor.camera) * zoomFactor
        self.camManipRef.setPos(SEditor.camera, zoomPtToCam)
        t = SEditor.camera.lerpPos(ZERO_POINT, CAM_MOVE_DURATION, other=self.camManipRef, blendType='easeInOut', task='manipulateCamera')
        t.uponDeath = self.updateCoaMarkerSizeOnDeath

    def spawnMoveToView(self, view):
        if False:
            i = 10
            return i + 15
        taskMgr.remove('manipulateCamera')
        SEditor.pushUndo([SEditor.camera])
        hprOffset = VBase3()
        if view == 8:
            self.orthoViewRoll = (self.orthoViewRoll + 90.0) % 360.0
            view = self.lastView
        else:
            self.orthoViewRoll = 0.0
        if view == 1:
            hprOffset.set(180.0, 0.0, 0.0)
        elif view == 2:
            hprOffset.set(0.0, 0.0, 0.0)
        elif view == 3:
            hprOffset.set(90.0, 0.0, 0.0)
        elif view == 4:
            hprOffset.set(-90.0, 0.0, 0.0)
        elif view == 5:
            hprOffset.set(0.0, -90.0, 0.0)
        elif view == 6:
            hprOffset.set(0.0, 90.0, 0.0)
        elif view == 7:
            hprOffset.set(135.0, -35.264, 0.0)
        self.camManipRef.setPosHpr(self.coaMarker, ZERO_VEC, hprOffset)
        offsetDistance = Vec3(SEditor.camera.getPos(self.camManipRef) - ZERO_POINT).length()
        scaledCenterVec = Y_AXIS * (-1.0 * offsetDistance)
        self.camManipRef.setPosHpr(self.camManipRef, scaledCenterVec, ZERO_VEC)
        self.lastView = view
        t = SEditor.camera.lerpPosHpr(ZERO_POINT, VBase3(0, 0, self.orthoViewRoll), CAM_MOVE_DURATION, other=self.camManipRef, blendType='easeInOut', task='manipulateCamera')
        t.uponDeath = self.updateCoaMarkerSizeOnDeath

    def swingCamAboutWidget(self, degrees, t):
        if False:
            i = 10
            return i + 15
        taskMgr.remove('manipulateCamera')
        SEditor.pushUndo([SEditor.camera])
        self.camManipRef.setPos(self.coaMarker, ZERO_POINT)
        self.camManipRef.setHpr(ZERO_POINT)
        parent = SEditor.camera.getParent()
        SEditor.camera.wrtReparentTo(self.camManipRef)
        manipTask = self.camManipRef.lerpHpr(VBase3(degrees, 0, 0), CAM_MOVE_DURATION, blendType='easeInOut', task='manipulateCamera')
        manipTask.parent = parent
        manipTask.uponDeath = self.reparentCam

    def reparentCam(self, state):
        if False:
            print('Hello World!')
        SEditor.camera.wrtReparentTo(state.parent)
        self.updateCoaMarkerSize()

    def fitOnWidget(self, nodePath='None Given'):
        if False:
            return 10
        taskMgr.remove('manipulateCamera')
        nodeScale = SEditor.widget.scalingNode.getScale(render)
        maxScale = max(nodeScale[0], nodeScale[1], nodeScale[2])
        maxDim = min(SEditor.dr.nearWidth, SEditor.dr.nearHeight)
        camY = SEditor.dr.near * (2.0 * maxScale) / (0.3 * maxDim)
        centerVec = Y_AXIS * camY
        vWidget2Camera = SEditor.widget.getPos(SEditor.camera)
        deltaMove = vWidget2Camera - centerVec
        self.camManipRef.setPos(SEditor.camera, deltaMove)
        parent = SEditor.camera.getParent()
        SEditor.camera.wrtReparentTo(self.camManipRef)
        fitTask = SEditor.camera.lerpPos(Point3(0, 0, 0), CAM_MOVE_DURATION, blendType='easeInOut', task='manipulateCamera')
        fitTask.parent = parent
        fitTask.uponDeath = self.reparentCam

    def moveToFit(self):
        if False:
            return 10
        widgetScale = SEditor.widget.scalingNode.getScale(render)
        maxScale = max(widgetScale[0], widgetScale[1], widgetScale[2])
        camY = 2 * SEditor.dr.near * (1.5 * maxScale) / min(SEditor.dr.nearWidth, SEditor.dr.nearHeight)
        centerVec = Y_AXIS * camY
        SEditor.selected.getWrtAll()
        SEditor.pushUndo(SEditor.selected)
        taskMgr.remove('followSelectedNodePath')
        taskMgr.add(self.stickToWidgetTask, 'stickToWidget')
        t = SEditor.widget.lerpPos(Point3(centerVec), CAM_MOVE_DURATION, other=SEditor.camera, blendType='easeInOut', task='moveToFitTask')
        t.uponDeath = lambda state: taskMgr.remove('stickToWidget')

    def stickToWidgetTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        SEditor.selected.moveWrtWidgetAll()
        return Task.cont

    def enableMouseFly(self, fKeyEvents=1):
        if False:
            while True:
                i = 10
        base.disableMouse()
        for event in self.actionEvents:
            self.accept(event[0], event[1], extraArgs=event[2:])
        if fKeyEvents:
            for event in self.keyEvents:
                self.accept(event[0], event[1], extraArgs=event[2:])
        self.coaMarker.reparentTo(SEditor.group)

    def disableMouseFly(self):
        if False:
            print('Hello World!')
        self.coaMarker.reparentTo(hidden)
        for event in self.actionEvents:
            self.ignore(event[0])
        for event in self.keyEvents:
            self.ignore(event[0])
        self.removeManipulateCameraTask()
        taskMgr.remove('stickToWidget')
        base.enableMouse()

    def removeManipulateCameraTask(self):
        if False:
            print('Hello World!')
        taskMgr.remove('manipulateCamera')