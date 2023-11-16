import math
from panda3d.core import BitMask32, Mat4, NodePath, Point3, VBase3, Vec3, Vec4, rad2Deg
from direct.showbase.DirectObject import DirectObject
from direct.showbase import ShowBaseGlobal
from .DirectUtil import CLAMP, useDirectRenderStyle
from .DirectGeometry import getCrankAngle, getScreenXY
from . import DirectGlobals as DG
from .DirectSelection import SelectionRay
from direct.interval.IntervalGlobal import Sequence, Func
from direct.directnotify import DirectNotifyGlobal
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
CAM_MOVE_DURATION = 1.2
COA_MARKER_SF = 0.0075
Y_AXIS = Vec3(0, 1, 0)

class DirectCameraControl(DirectObject):
    notify = DirectNotifyGlobal.directNotify.newCategory('DirectCameraControl')

    def __init__(self):
        if False:
            while True:
                i = 10
        self.startT = 0.0
        self.startF = 0
        self.orthoViewRoll = 0.0
        self.lastView = 0
        self.coa = Point3(0, 100, 0)
        self.coaMarker = ShowBaseGlobal.loader.loadModel('models/misc/sphere')
        self.coaMarker.setName('DirectCameraCOAMarker')
        self.coaMarker.setTransparency(1)
        self.coaMarker.setColor(1, 0, 0, 0)
        self.coaMarker.setPos(0, 100, 0)
        useDirectRenderStyle(self.coaMarker)
        self.coaMarkerPos = Point3(0)
        self.coaMarkerColorIval = None
        self.fLockCOA = 0
        self.nullHitPointCount = 0
        self.cqEntries = []
        self.coaMarkerRef = ShowBaseGlobal.direct.group.attachNewNode('coaMarkerRef')
        self.camManipRef = ShowBaseGlobal.direct.group.attachNewNode('camManipRef')
        self.switchDirBelowZero = True
        self.manipulateCameraTask = None
        self.manipulateCameraInterval = None
        t = CAM_MOVE_DURATION
        self.actionEvents = [['DIRECT-mouse1', self.mouseRotateStart], ['DIRECT-mouse1Up', self.mouseDollyStop], ['DIRECT-mouse2', self.mouseFlyStart], ['DIRECT-mouse2Up', self.mouseFlyStop], ['DIRECT-mouse3', self.mouseDollyStart], ['DIRECT-mouse3Up', self.mouseDollyStop]]
        self.keyEvents = [['DIRECT-centerCamIn', self.centerCamIn, 0.5], ['DIRECT-fitOnWidget', self.fitOnWidget], ['DIRECT-homeCam', self.homeCam], ['DIRECT-toggleMarkerVis', self.toggleMarkerVis], ['DIRECT-moveToFit', self.moveToFit], ['DIRECT-pickNextCOA', self.pickNextCOA], ['DIRECT-orbitUprightCam', self.orbitUprightCam], ['DIRECT-uprightCam', self.uprightCam], ['DIRECT-spwanMoveToView-1', self.spawnMoveToView, 1], ['DIRECT-spwanMoveToView-2', self.spawnMoveToView, 2], ['DIRECT-spwanMoveToView-3', self.spawnMoveToView, 3], ['DIRECT-spwanMoveToView-4', self.spawnMoveToView, 4], ['DIRECT-spwanMoveToView-5', self.spawnMoveToView, 5], ['DIRECT-spwanMoveToView-6', self.spawnMoveToView, 6], ['DIRECT-spwanMoveToView-7', self.spawnMoveToView, 7], ['DIRECT-spwanMoveToView-8', self.spawnMoveToView, 8], ['DIRECT-swingCamAboutWidget-0', self.swingCamAboutWidget, -90.0, t], ['DIRECT-swingCamAboutWidget-1', self.swingCamAboutWidget, 90.0, t], ['DIRECT-removeManipulateCameraTask', self.removeManipulateCameraTask], ['DIRECT-zoomInCam', self.zoomCam, 0.5, t], ['DIRECT-zoomOutCam', self.zoomCam, -2.0, t]]
        self.lockRoll = False
        self.useMayaCamControls = 0
        self.altDown = 0
        self.perspCollPlane = None
        self.perspCollPlane2 = None

    def toggleMarkerVis(self):
        if False:
            while True:
                i = 10
        if self.coaMarker.isHidden():
            self.coaMarker.show()
        else:
            self.coaMarker.hide()

    def mouseRotateStart(self, modifiers):
        if False:
            return 10
        if self.useMayaCamControls and modifiers == 4:
            self.spawnMouseRotateTask()

    def mouseDollyStart(self, modifiers):
        if False:
            for i in range(10):
                print('nop')
        if self.useMayaCamControls and modifiers == 4:
            self.coaMarker.hide()
            base = ShowBaseGlobal.base
            self.startT = base.clock.getFrameTime()
            self.startF = base.clock.getFrameCount()
            direct = ShowBaseGlobal.direct
            if hasattr(direct, 'manipulationControl') and direct.manipulationControl.fMultiView and (direct.camera.getName() != 'persp'):
                self.spawnOrthoZoom()
            else:
                self.spawnHPanYZoom()

    def __stopManipulateCamera(self):
        if False:
            print('Hello World!')
        if self.manipulateCameraTask:
            taskMgr.remove(self.manipulateCameraTask)
            self.manipulateCameraTask = None
        if self.manipulateCameraInterval:
            self.manipulateCameraInterval.finish()
            self.manipulateCameraInterval = None

    def __startManipulateCamera(self, func=None, task=None, ival=None):
        if False:
            for i in range(10):
                print('nop')
        self.__stopManipulateCamera()
        if func:
            assert task is None
            task = Task.Task(func)
        if task:
            self.manipulateCameraTask = taskMgr.add(task, 'manipulateCamera')
        if ival:
            ival.start()
            self.manipulateCameraInterval = ival

    def mouseDollyStop(self):
        if False:
            return 10
        self.__stopManipulateCamera()

    def mouseFlyStart(self, modifiers):
        if False:
            for i in range(10):
                print('nop')
        base = ShowBaseGlobal.base
        direct = ShowBaseGlobal.direct
        if self.useMayaCamControls and modifiers == 4:
            self.coaMarker.hide()
            self.startT = base.clock.getFrameTime()
            self.startF = base.clock.getFrameCount()
            if hasattr(direct, 'manipulationControl') and direct.manipulationControl.fMultiView and (direct.camera.getName() != 'persp'):
                self.spawnOrthoTranslate()
            else:
                self.spawnXZTranslate()
            self.altDown = 1
        elif not self.useMayaCamControls:
            if abs(direct.dr.mouseX) < 0.9 and abs(direct.dr.mouseY) < 0.9:
                self.coaMarker.hide()
                self.startT = base.clock.getFrameTime()
                self.startF = base.clock.getFrameCount()
                self.spawnXZTranslateOrHPanYZoom()
            elif abs(direct.dr.mouseX) > 0.9 and abs(direct.dr.mouseY) > 0.9:
                self.spawnMouseRollTask()
            else:
                self.spawnMouseRotateTask()
        if not modifiers == 4:
            self.altDown = 0

    def mouseFlyStop(self):
        if False:
            print('Hello World!')
        self.__stopManipulateCamera()
        base = ShowBaseGlobal.base
        stopT = base.clock.getFrameTime()
        deltaT = stopT - self.startT
        stopF = base.clock.getFrameCount()
        deltaF = stopF - self.startF
        direct = ShowBaseGlobal.direct
        if not self.altDown and len(direct.selected.getSelectedAsList()) == 0:
            skipFlags = DG.SKIP_HIDDEN | DG.SKIP_BACKFACE
            skipFlags |= DG.SKIP_CAMERA * (1 - base.getControl())
            self.computeCOA(direct.iRay.pickGeom(skipFlags=skipFlags))
            self.coaMarkerRef.setPosHprScale(base.cam, 0, 0, 0, 0, 0, 0, 1, 1, 1)
            self.cqEntries = []
            for i in range(direct.iRay.getNumEntries()):
                self.cqEntries.append(direct.iRay.getEntry(i))
        self.coaMarker.show()
        self.updateCoaMarkerSize()

    def mouseFlyStartTopWin(self):
        if False:
            i = 10
            return i + 15
        print('Moving mouse 2 in new window')

    def mouseFlyStopTopWin(self):
        if False:
            while True:
                i = 10
        print('Stopping mouse 2 in new window')

    def spawnXZTranslateOrHPanYZoom(self):
        if False:
            return 10
        self.__stopManipulateCamera()
        t = Task.Task(self.XZTranslateOrHPanYZoomTask)
        t.zoomSF = Vec3(self.coaMarker.getPos(ShowBaseGlobal.direct.camera)).length()
        self.__startManipulateCamera(task=t)

    def spawnXZTranslateOrHPPan(self):
        if False:
            i = 10
            return i + 15
        self.__stopManipulateCamera()
        self.__startManipulateCamera(func=self.XZTranslateOrHPPanTask)

    def spawnXZTranslate(self):
        if False:
            return 10
        self.__stopManipulateCamera()
        self.__startManipulateCamera(func=self.XZTranslateTask)

    def spawnOrthoTranslate(self):
        if False:
            i = 10
            return i + 15
        self.__stopManipulateCamera()
        self.__startManipulateCamera(func=self.OrthoTranslateTask)

    def spawnHPanYZoom(self):
        if False:
            return 10
        self.__stopManipulateCamera()
        t = Task.Task(self.HPanYZoomTask)
        t.zoomSF = Vec3(self.coaMarker.getPos(ShowBaseGlobal.direct.camera)).length()
        self.__startManipulateCamera(task=t)

    def spawnOrthoZoom(self):
        if False:
            for i in range(10):
                print('nop')
        self.__stopManipulateCamera()
        t = Task.Task(self.OrthoZoomTask)
        self.__startManipulateCamera(task=t)

    def spawnHPPan(self):
        if False:
            return 10
        self.__stopManipulateCamera()
        self.__startManipulateCamera(func=self.HPPanTask)

    def XZTranslateOrHPanYZoomTask(self, state):
        if False:
            while True:
                i = 10
        if ShowBaseGlobal.direct.fShift:
            return self.XZTranslateTask(state)
        else:
            return self.HPanYZoomTask(state)

    def XZTranslateOrHPPanTask(self, state):
        if False:
            while True:
                i = 10
        if ShowBaseGlobal.direct.fShift:
            return self.HPPanTask(state)
        else:
            return self.XZTranslateTask(state)

    def XZTranslateTask(self, state):
        if False:
            i = 10
            return i + 15
        direct = ShowBaseGlobal.direct
        coaDist = Vec3(self.coaMarker.getPos(direct.camera)).length()
        xlateSF = coaDist / direct.dr.near
        direct.camera.setPos(direct.camera, -0.5 * direct.dr.mouseDeltaX * direct.dr.nearWidth * xlateSF, 0.0, -0.5 * direct.dr.mouseDeltaY * direct.dr.nearHeight * xlateSF)
        return Task.cont

    def OrthoTranslateTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        direct = ShowBaseGlobal.direct
        iRay = SelectionRay(direct.camera)
        iRay.collider.setFromLens(direct.camNode, direct.dr.mouseX, direct.dr.mouseY)
        iRay.collideWithBitMask(BitMask32.bit(21))
        iRay.ct.traverse(direct.grid)
        entry = iRay.getEntry(0)
        hitPt = entry.getSurfacePoint(entry.getFromNodePath())
        iRay.collisionNodePath.removeNode()
        del iRay
        if hasattr(state, 'prevPt'):
            direct.camera.setPos(direct.camera, state.prevPt - hitPt)
        state.prevPt = hitPt
        return Task.cont

    def HPanYZoomTask(self, state):
        if False:
            while True:
                i = 10
        direct = ShowBaseGlobal.direct
        if hasattr(direct.cam.node(), 'getLens') and direct.cam.node().getLens().__class__.__name__ == 'OrthographicLens':
            return
        if direct.fControl:
            moveDir = Vec3(self.coaMarker.getPos(direct.camera))
            if moveDir[1] < 0.0:
                moveDir.assign(moveDir * -1)
            moveDir.normalize()
        else:
            moveDir = Vec3(Y_AXIS)
        if self.useMayaCamControls:
            moveDir.assign(moveDir * ((direct.dr.mouseDeltaX - 1.0 * direct.dr.mouseDeltaY) * state.zoomSF))
            hVal = 0.0
        else:
            moveDir.assign(moveDir * (-1.0 * direct.dr.mouseDeltaY * state.zoomSF))
            if direct.dr.mouseDeltaY > 0.0:
                moveDir.setY(moveDir[1] * 1.0)
            hVal = 0.5 * direct.dr.mouseDeltaX * direct.dr.fovH
        direct.camera.setPosHpr(direct.camera, moveDir[0], moveDir[1], moveDir[2], hVal, 0.0, 0.0)
        if self.lockRoll:
            direct.camera.setR(0)
        return Task.cont

    def OrthoZoomTask(self, state):
        if False:
            while True:
                i = 10
        direct = ShowBaseGlobal.direct
        filmSize = direct.camNode.getLens().getFilmSize()
        factor = (direct.dr.mouseDeltaX - 1.0 * direct.dr.mouseDeltaY) * 0.1
        x = direct.dr.getWidth()
        y = direct.dr.getHeight()
        direct.dr.orthoFactor -= factor
        if direct.dr.orthoFactor < 0:
            direct.dr.orthoFactor = 0.0001
        direct.dr.updateFilmSize(x, y)
        return Task.cont

    def HPPanTask(self, state):
        if False:
            while True:
                i = 10
        direct = ShowBaseGlobal.direct
        direct.camera.setHpr(direct.camera, 0.5 * direct.dr.mouseDeltaX * direct.dr.fovH, -0.5 * direct.dr.mouseDeltaY * direct.dr.fovV, 0.0)
        return Task.cont

    def spawnMouseRotateTask(self):
        if False:
            i = 10
            return i + 15
        self.__stopManipulateCamera()
        direct = ShowBaseGlobal.direct
        if self.perspCollPlane:
            iRay = SelectionRay(direct.camera)
            iRay.collider.setFromLens(direct.camNode, 0.0, 0.0)
            iRay.collideWithBitMask(1)
            if direct.camera.getPos().getZ() >= 0:
                iRay.ct.traverse(self.perspCollPlane)
            else:
                iRay.ct.traverse(self.perspCollPlane2)
            if iRay.getNumEntries() > 0:
                entry = iRay.getEntry(0)
                hitPt = entry.getSurfacePoint(entry.getFromNodePath())
                np = NodePath('temp')
                np.setPos(direct.camera, hitPt)
                self.coaMarkerPos = np.getPos()
                np.removeNode()
                self.coaMarker.setPos(self.coaMarkerPos)
            iRay.collisionNodePath.removeNode()
            del iRay
        self.camManipRef.setPos(self.coaMarkerPos)
        self.camManipRef.setHpr(direct.camera, DG.ZERO_POINT)
        t = Task.Task(self.mouseRotateTask)
        if abs(direct.dr.mouseX) > 0.9:
            t.constrainedDir = 'y'
        else:
            t.constrainedDir = 'x'
        self.__startManipulateCamera(task=t)

    def mouseRotateTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        direct = ShowBaseGlobal.direct
        if hasattr(direct.cam.node(), 'getLens') and direct.cam.node().getLens().__class__.__name__ == 'OrthographicLens':
            return
        if state.constrainedDir == 'y' and abs(direct.dr.mouseX) > 0.9:
            deltaX = 0
            deltaY = direct.dr.mouseDeltaY
        elif state.constrainedDir == 'x' and abs(direct.dr.mouseY) > 0.9:
            deltaX = direct.dr.mouseDeltaX
            deltaY = 0
        else:
            deltaX = direct.dr.mouseDeltaX
            deltaY = direct.dr.mouseDeltaY
        if direct.fShift:
            direct.camera.setHpr(direct.camera, deltaX * direct.dr.fovH, -deltaY * direct.dr.fovV, 0.0)
            if self.lockRoll:
                direct.camera.setR(0)
            self.camManipRef.setPos(self.coaMarkerPos)
            self.camManipRef.setHpr(direct.camera, DG.ZERO_POINT)
        else:
            if direct.camera.getPos().getZ() >= 0 or not self.switchDirBelowZero:
                dirX = -1
            else:
                dirX = 1
            wrt = direct.camera.getTransform(self.camManipRef)
            self.camManipRef.setHpr(self.camManipRef, dirX * deltaX * 180.0, deltaY * 180.0, 0.0)
            if self.lockRoll:
                self.camManipRef.setR(0)
            direct.camera.setTransform(self.camManipRef, wrt)
        return Task.cont

    def spawnMouseRollTask(self):
        if False:
            for i in range(10):
                print('nop')
        self.__stopManipulateCamera()
        direct = ShowBaseGlobal.direct
        self.camManipRef.setPos(self.coaMarkerPos)
        self.camManipRef.setHpr(direct.camera, DG.ZERO_POINT)
        t = Task.Task(self.mouseRollTask)
        t.coaCenter = getScreenXY(self.coaMarker)
        t.lastAngle = getCrankAngle(t.coaCenter)
        t.wrt = direct.camera.getTransform(self.camManipRef)
        self.__startManipulateCamera(task=t)

    def mouseRollTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        wrt = state.wrt
        angle = getCrankAngle(state.coaCenter)
        deltaAngle = angle - state.lastAngle
        state.lastAngle = angle
        self.camManipRef.setHpr(self.camManipRef, 0, 0, deltaAngle)
        if self.lockRoll:
            self.camManipRef.setR(0)
        ShowBaseGlobal.direct.camera.setTransform(self.camManipRef, wrt)
        return Task.cont

    def lockCOA(self):
        if False:
            return 10
        self.fLockCOA = 1
        ShowBaseGlobal.direct.message('COA Lock On')

    def unlockCOA(self):
        if False:
            while True:
                i = 10
        self.fLockCOA = 0
        ShowBaseGlobal.direct.message('COA Lock Off')

    def toggleCOALock(self):
        if False:
            for i in range(10):
                print('nop')
        self.fLockCOA = 1 - self.fLockCOA
        if self.fLockCOA:
            ShowBaseGlobal.direct.message('COA Lock On')
        else:
            ShowBaseGlobal.direct.message('COA Lock Off')

    def pickNextCOA(self):
        if False:
            print('Hello World!')
        ' Cycle through collision handler entries '
        if self.cqEntries:
            entry = self.cqEntries[0]
            self.cqEntries = self.cqEntries[1:] + self.cqEntries[:1]
            nodePath = entry.getIntoNodePath()
            if ShowBaseGlobal.direct.camera not in nodePath.getAncestors():
                hitPt = entry.getSurfacePoint(entry.getFromNodePath())
                self.updateCoa(hitPt, ref=self.coaMarkerRef)
            else:
                self.cqEntries = self.cqEntries[:-1]
                self.pickNextCOA()

    def computeCOA(self, entry):
        if False:
            i = 10
            return i + 15
        coa = Point3(0)
        dr = ShowBaseGlobal.direct.drList.getCurrentDr()
        if self.fLockCOA:
            coa.assign(self.coaMarker.getPos(ShowBaseGlobal.direct.camera))
            self.nullHitPointCount = 0
        elif entry:
            hitPt = entry.getSurfacePoint(entry.getFromNodePath())
            hitPtDist = Vec3(hitPt).length()
            coa.assign(hitPt)
            if hitPtDist < 1.1 * dr.near or hitPtDist > dr.far:
                coa.assign(self.coaMarker.getPos(ShowBaseGlobal.direct.camera))
            self.nullHitPointCount = 0
        else:
            self.nullHitPointCount = (self.nullHitPointCount + 1) % 7
            dist = pow(10.0, self.nullHitPointCount)
            ShowBaseGlobal.direct.message('COA Distance: ' + repr(dist))
            coa.set(0, dist, 0)
        coaDist = Vec3(coa - DG.ZERO_POINT).length()
        if coaDist < 1.1 * dr.near:
            coa.set(0, 100, 0)
            coaDist = 100
        self.updateCoa(coa, coaDist=coaDist)

    def updateCoa(self, ref2point, coaDist=None, ref=None):
        if False:
            for i in range(10):
                print('nop')
        self.coa.set(ref2point[0], ref2point[1], ref2point[2])
        if not coaDist:
            coaDist = Vec3(self.coa - DG.ZERO_POINT).length()
        if ref is None:
            ref = ShowBaseGlobal.direct.drList.getCurrentDr().cam
        self.coaMarker.setPos(ref, self.coa)
        pos = self.coaMarker.getPos()
        self.coaMarker.setPosHprScale(pos, Vec3(0), Vec3(1))
        self.updateCoaMarkerSize(coaDist)
        self.coaMarkerPos.assign(self.coaMarker.getPos())

    def updateCoaMarkerSizeOnDeath(self):
        if False:
            print('Hello World!')
        self.updateCoaMarkerSize()

    def updateCoaMarkerSize(self, coaDist=None):
        if False:
            return 10
        if not coaDist:
            coaDist = Vec3(self.coaMarker.getPos(ShowBaseGlobal.direct.camera)).length()
        sf = COA_MARKER_SF * coaDist * (ShowBaseGlobal.direct.drList.getCurrentDr().fovV / 30.0)
        if sf == 0.0:
            sf = 0.1
        self.coaMarker.setScale(sf)
        if self.coaMarkerColorIval:
            self.coaMarkerColorIval.finish()
        self.coaMarkerColorIval = Sequence(Func(self.coaMarker.unstash), self.coaMarker.colorInterval(1.5, Vec4(1, 0, 0, 0), startColor=Vec4(1, 0, 0, 1), blendType='easeInOut'), Func(self.coaMarker.stash))
        self.coaMarkerColorIval.start()

    def homeCam(self):
        if False:
            i = 10
            return i + 15
        direct = ShowBaseGlobal.direct
        direct.pushUndo([direct.camera])
        direct.camera.reparentTo(ShowBaseGlobal.base.render)
        direct.camera.clearMat()
        self.updateCoaMarkerSize()

    def uprightCam(self):
        if False:
            i = 10
            return i + 15
        self.__stopManipulateCamera()
        direct = ShowBaseGlobal.direct
        direct.pushUndo([direct.camera])
        currH = direct.camera.getH()
        ival = direct.camera.hprInterval(CAM_MOVE_DURATION, (currH, 0, 0), other=ShowBaseGlobal.base.render, blendType='easeInOut', name='manipulateCamera')
        self.__startManipulateCamera(ival=ival)

    def orbitUprightCam(self):
        if False:
            for i in range(10):
                print('nop')
        self.__stopManipulateCamera()
        direct = ShowBaseGlobal.direct
        direct.pushUndo([direct.camera])
        render = ShowBaseGlobal.base.render
        mCam2Render = Mat4(Mat4.identMat())
        mCam2Render.assign(direct.camera.getMat(render))
        zAxis = Vec3(mCam2Render.xformVec(DG.Z_AXIS))
        zAxis.normalize()
        orbitAngle = rad2Deg(math.acos(CLAMP(zAxis.dot(DG.Z_AXIS), -1, 1)))
        if orbitAngle < 0.1:
            return
        rotAxis = Vec3(zAxis.cross(DG.Z_AXIS))
        rotAxis.normalize()
        rotAngle = rad2Deg(math.acos(CLAMP(rotAxis.dot(DG.X_AXIS), -1, 1)))
        if rotAxis[1] < 0:
            rotAngle *= -1
        self.camManipRef.setPos(self.coaMarker, Vec3(0))
        self.camManipRef.setHpr(render, rotAngle, 0, 0)
        parent = direct.camera.getParent()
        direct.camera.wrtReparentTo(self.camManipRef)
        ival = self.camManipRef.hprInterval(CAM_MOVE_DURATION, (rotAngle, orbitAngle, 0), other=render, blendType='easeInOut')
        ival = Sequence(ival, Func(self.reparentCam, parent), name='manipulateCamera')
        self.__startManipulateCamera(ival=ival)

    def centerCam(self):
        if False:
            return 10
        self.centerCamIn(1.0)

    def centerCamNow(self):
        if False:
            while True:
                i = 10
        self.centerCamIn(0.0)

    def centerCamIn(self, t):
        if False:
            return 10
        self.__stopManipulateCamera()
        direct = ShowBaseGlobal.direct
        direct.pushUndo([direct.camera])
        markerToCam = self.coaMarker.getPos(direct.camera)
        dist = Vec3(markerToCam - DG.ZERO_POINT).length()
        scaledCenterVec = Y_AXIS * dist
        delta = markerToCam - scaledCenterVec
        self.camManipRef.setPosHpr(direct.camera, Point3(0), Point3(0))
        ival = direct.camera.posInterval(CAM_MOVE_DURATION, Point3(delta), other=self.camManipRef, blendType='easeInOut')
        ival = Sequence(ival, Func(self.updateCoaMarkerSizeOnDeath), name='manipulateCamera')
        self.__startManipulateCamera(ival=ival)

    def zoomCam(self, zoomFactor, t):
        if False:
            i = 10
            return i + 15
        self.__stopManipulateCamera()
        direct = ShowBaseGlobal.direct
        direct.pushUndo([direct.camera])
        zoomPtToCam = self.coaMarker.getPos(direct.camera) * zoomFactor
        self.camManipRef.setPos(direct.camera, zoomPtToCam)
        ival = direct.camera.posInterval(CAM_MOVE_DURATION, DG.ZERO_POINT, other=self.camManipRef, blendType='easeInOut')
        ival = Sequence(ival, Func(self.updateCoaMarkerSizeOnDeath), name='manipulateCamera')
        self.__startManipulateCamera(ival=ival)

    def spawnMoveToView(self, view):
        if False:
            while True:
                i = 10
        self.__stopManipulateCamera()
        direct = ShowBaseGlobal.direct
        direct.pushUndo([direct.camera])
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
        self.camManipRef.setPosHpr(self.coaMarker, DG.ZERO_VEC, hprOffset)
        offsetDistance = Vec3(direct.camera.getPos(self.camManipRef) - DG.ZERO_POINT).length()
        scaledCenterVec = Y_AXIS * (-1.0 * offsetDistance)
        self.camManipRef.setPosHpr(self.camManipRef, scaledCenterVec, DG.ZERO_VEC)
        self.lastView = view
        ival = direct.camera.posHprInterval(CAM_MOVE_DURATION, pos=DG.ZERO_POINT, hpr=VBase3(0, 0, self.orthoViewRoll), other=self.camManipRef, blendType='easeInOut')
        ival = Sequence(ival, Func(self.updateCoaMarkerSizeOnDeath), name='manipulateCamera')
        self.__startManipulateCamera(ival=ival)

    def swingCamAboutWidget(self, degrees, t):
        if False:
            for i in range(10):
                print('nop')
        self.__stopManipulateCamera()
        direct = ShowBaseGlobal.direct
        direct.pushUndo([direct.camera])
        self.camManipRef.setPos(self.coaMarker, DG.ZERO_POINT)
        self.camManipRef.setHpr(DG.ZERO_POINT)
        parent = direct.camera.getParent()
        direct.camera.wrtReparentTo(self.camManipRef)
        ival = self.camManipRef.hprInterval(CAM_MOVE_DURATION, VBase3(degrees, 0, 0), blendType='easeInOut')
        ival = Sequence(ival, Func(self.reparentCam, parent), name='manipulateCamera')
        self.__startManipulateCamera(ival=ival)

    def reparentCam(self, parent):
        if False:
            for i in range(10):
                print('nop')
        ShowBaseGlobal.direct.camera.wrtReparentTo(parent)
        self.updateCoaMarkerSize()

    def fitOnWidget(self, nodePath='None Given'):
        if False:
            while True:
                i = 10
        self.__stopManipulateCamera()
        direct = ShowBaseGlobal.direct
        nodeScale = direct.widget.scalingNode.getScale(ShowBaseGlobal.base.render)
        maxScale = max(nodeScale[0], nodeScale[1], nodeScale[2])
        maxDim = min(direct.dr.nearWidth, direct.dr.nearHeight)
        camY = direct.dr.near * (2.0 * maxScale) / (0.3 * maxDim)
        centerVec = Y_AXIS * camY
        vWidget2Camera = direct.widget.getPos(direct.camera)
        deltaMove = vWidget2Camera - centerVec
        try:
            self.camManipRef.setPos(direct.camera, deltaMove)
        except Exception:
            pass
        parent = direct.camera.getParent()
        direct.camera.wrtReparentTo(self.camManipRef)
        ival = direct.camera.posInterval(CAM_MOVE_DURATION, Point3(0, 0, 0), blendType='easeInOut')
        ival = Sequence(ival, Func(self.reparentCam, parent), name='manipulateCamera')
        self.__startManipulateCamera(ival=ival)

    def moveToFit(self):
        if False:
            i = 10
            return i + 15
        direct = ShowBaseGlobal.direct
        widgetScale = direct.widget.scalingNode.getScale(ShowBaseGlobal.base.render)
        maxScale = max(widgetScale[0], widgetScale[1], widgetScale[2])
        camY = 2 * direct.dr.near * (1.5 * maxScale) / min(direct.dr.nearWidth, direct.dr.nearHeight)
        centerVec = Y_AXIS * camY
        direct.selected.getWrtAll()
        direct.pushUndo(direct.selected)
        taskMgr.remove('followSelectedNodePath')
        taskMgr.add(self.stickToWidgetTask, 'stickToWidget')
        ival = direct.widget.posInterval(CAM_MOVE_DURATION, Point3(centerVec), other=direct.camera, blendType='easeInOut')
        ival = Sequence(ival, Func(lambda : taskMgr.remove('stickToWidget')), name='moveToFit')
        ival.start()

    def stickToWidgetTask(self, state):
        if False:
            i = 10
            return i + 15
        ShowBaseGlobal.direct.selected.moveWrtWidgetAll()
        return Task.cont

    def enableMouseFly(self, fKeyEvents=1):
        if False:
            print('Hello World!')
        base = ShowBaseGlobal.base
        base.disableMouse()
        for event in self.actionEvents:
            self.accept(event[0], event[1], extraArgs=event[2:])
        if fKeyEvents:
            for event in self.keyEvents:
                self.accept(event[0], event[1], extraArgs=event[2:])
        self.coaMarker.reparentTo(ShowBaseGlobal.direct.group)

    def disableMouseFly(self):
        if False:
            print('Hello World!')
        self.coaMarker.reparentTo(ShowBaseGlobal.hidden)
        for event in self.actionEvents:
            self.ignore(event[0])
        for event in self.keyEvents:
            self.ignore(event[0])
        self.removeManipulateCameraTask()
        taskMgr.remove('stickToWidget')
        ShowBaseGlobal.base.enableMouse()

    def removeManipulateCameraTask(self):
        if False:
            print('Hello World!')
        self.__stopManipulateCamera()