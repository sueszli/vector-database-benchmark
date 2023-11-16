from direct.showbase.DirectObject import *
from direct.directtools.DirectGlobals import *
from direct.directtools.DirectUtil import *
from seGeometry import *
from direct.task import Task
import math

class DirectManipulationControl(DirectObject):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.objectHandles = ObjectHandles()
        self.hitPt = Point3(0)
        self.prevHit = Vec3(0)
        self.rotationCenter = Point3(0)
        self.initScaleMag = 1
        self.manipRef = SEditor.group.attachNewNode('manipRef')
        self.hitPtDist = 0
        self.constraint = None
        self.rotateAxis = 'x'
        self.lastCrankAngle = 0
        self.fSetCoa = 0
        self.fHitInit = 1
        self.fScaleInit = 1
        self.fWidgetTop = 0
        self.fFreeManip = 1
        self.fScaling = 0
        self.mode = None
        self.actionEvents = [['DIRECT-mouse1', self.manipulationStart], ['DIRECT-mouse1Up', self.manipulationStop], ['tab', self.toggleObjectHandlesMode], ['.', self.objectHandles.multiplyScalingFactorBy, 2.0], ['>', self.objectHandles.multiplyScalingFactorBy, 2.0], [',', self.objectHandles.multiplyScalingFactorBy, 0.5], ['<', self.objectHandles.multiplyScalingFactorBy, 0.5], ['shift-f', self.objectHandles.growToFit], ['i', self.plantSelectedNodePath]]

    def manipulationStart(self, modifiers):
        if False:
            i = 10
            return i + 15
        self.mode = 'select'
        entry = SEditor.iRay.pickWidget()
        if entry:
            self.hitPt.assign(entry.getSurfacePoint(entry.getFromNodePath()))
            self.hitPtDist = Vec3(self.hitPt).length()
            self.constraint = entry.getIntoNodePath().getName()
        else:
            self.constraint = None
        taskMgr.doMethodLater(MANIPULATION_MOVE_DELAY, self.switchToMoveMode, 'manip-move-wait')
        self.moveDir = None
        watchMouseTask = Task.Task(self.watchMouseTask)
        watchMouseTask.initX = SEditor.dr.mouseX
        watchMouseTask.initY = SEditor.dr.mouseY
        taskMgr.add(watchMouseTask, 'manip-watch-mouse')

    def switchToMoveMode(self, state):
        if False:
            i = 10
            return i + 15
        taskMgr.remove('manip-watch-mouse')
        self.mode = 'move'
        self.manipulateObject()
        return Task.done

    def watchMouseTask(self, state):
        if False:
            while True:
                i = 10
        if abs(state.initX - SEditor.dr.mouseX) > 0.01 or abs(state.initY - SEditor.dr.mouseY) > 0.01:
            taskMgr.remove('manip-move-wait')
            self.mode = 'move'
            self.manipulateObject()
            return Task.done
        else:
            return Task.cont

    def manipulationStop(self, xy=[]):
        if False:
            for i in range(10):
                print('nop')
        taskMgr.remove('manipulateObject')
        taskMgr.remove('manip-move-wait')
        taskMgr.remove('manip-watch-mouse')
        if self.mode == 'select':
            skipFlags = SKIP_HIDDEN | SKIP_BACKFACE
            skipFlags |= SKIP_CAMERA * (1 - base.getControl())
            entry = SEditor.iRay.pickGeom(skipFlags=skipFlags)
            if entry:
                self.hitPt.assign(entry.getSurfacePoint(entry.getFromNodePath()))
                self.hitPtDist = Vec3(self.hitPt).length()
                SEditor.select(entry.getIntoNodePath(), SEditor.fShift)
            else:
                SEditor.deselectAll()
        else:
            self.manipulateObjectCleanup()

    def manipulateObjectCleanup(self):
        if False:
            return 10
        if self.fScaling:
            self.objectHandles.transferObjectHandlesScale()
            self.fScaling = 0
        SEditor.selected.highlightAll()
        self.objectHandles.showAllHandles()
        self.objectHandles.hideGuides()
        self.spawnFollowSelectedNodePathTask()
        messenger.send('DIRECT_manipulateObjectCleanup')

    def spawnFollowSelectedNodePathTask(self):
        if False:
            return 10
        if not SEditor.selected.last:
            return
        taskMgr.remove('followSelectedNodePath')
        pos = VBase3(0)
        hpr = VBase3(0)
        decomposeMatrix(SEditor.selected.last.mCoa2Dnp, VBase3(0), hpr, pos, CSDefault)
        t = Task.Task(self.followSelectedNodePathTask)
        t.pos = pos
        t.hpr = hpr
        t.base = SEditor.selected.last
        taskMgr.add(t, 'followSelectedNodePath')

    def followSelectedNodePathTask(self, state):
        if False:
            return 10
        SEditor.widget.setPosHpr(state.base, state.pos, state.hpr)
        return Task.cont

    def enableManipulation(self):
        if False:
            print('Hello World!')
        for event in self.actionEvents:
            self.accept(event[0], event[1], extraArgs=event[2:])

    def disableManipulation(self):
        if False:
            while True:
                i = 10
        for event in self.actionEvents:
            self.ignore(event[0])
        self.removeManipulateObjectTask()
        taskMgr.remove('manipulateObject')
        taskMgr.remove('manip-move-wait')
        taskMgr.remove('manip-watch-mouse')
        taskMgr.remove('highlightWidgetTask')
        taskMgr.remove('resizeObjectHandles')

    def toggleObjectHandlesMode(self):
        if False:
            print('Hello World!')
        self.fSetCoa = 1 - self.fSetCoa
        if self.fSetCoa:
            self.objectHandles.coaModeColor()
        else:
            self.objectHandles.manipModeColor()

    def removeManipulateObjectTask(self):
        if False:
            while True:
                i = 10
        taskMgr.remove('manipulateObject')

    def manipulateObject(self):
        if False:
            for i in range(10):
                print('nop')
        if SEditor.selected:
            taskMgr.remove('followSelectedNodePath')
            taskMgr.remove('highlightWidgetTask')
            self.fManip = 1
            SEditor.pushUndo(SEditor.selected)
            self.objectHandles.showGuides()
            self.objectHandles.hideAllHandles()
            self.objectHandles.showHandle(self.constraint)
            SEditor.selected.getWrtAll()
            SEditor.selected.dehighlightAll()
            messenger.send('DIRECT_manipulateObjectStart')
            self.spawnManipulateObjectTask()

    def spawnManipulateObjectTask(self):
        if False:
            for i in range(10):
                print('nop')
        self.fHitInit = 1
        self.fScaleInit = 1
        t = Task.Task(self.manipulateObjectTask)
        t.fMouseX = abs(SEditor.dr.mouseX) > 0.9
        t.fMouseY = abs(SEditor.dr.mouseY) > 0.9
        if t.fMouseX:
            t.constrainedDir = 'y'
        else:
            t.constrainedDir = 'x'
        t.coaCenter = getScreenXY(SEditor.widget)
        if t.fMouseX and t.fMouseY:
            t.lastAngle = getCrankAngle(t.coaCenter)
        taskMgr.add(t, 'manipulateObject')

    def manipulateObjectTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        if self.constraint:
            type = self.constraint[2:]
            if type == 'post':
                self.xlate1D(state)
            elif type == 'disc':
                self.xlate2D(state)
            elif type == 'ring':
                self.rotate1D(state)
        elif self.fFreeManip:
            if 0 and self.fScaling and (not SEditor.fAlt):
                self.objectHandles.transferObjectHandlesScale()
                self.fScaling = 0
            if SEditor.fControl:
                self.fScaling = 1
                self.scale3D(state)
            elif state.fMouseX and state.fMouseY:
                self.rotateAboutViewVector(state)
            elif state.fMouseX or state.fMouseY:
                self.rotate2D(state)
            elif SEditor.fShift or SEditor.fControl:
                self.xlateCamXY(state)
            else:
                self.xlateCamXZ(state)
        if self.fSetCoa:
            SEditor.selected.last.mCoa2Dnp.assign(SEditor.widget.getMat(SEditor.selected.last))
        else:
            SEditor.selected.moveWrtWidgetAll()
        return Task.cont

    def xlate1D(self, state):
        if False:
            return 10
        self.hitPt.assign(self.objectHandles.getAxisIntersectPt(self.constraint[:1]))
        if self.fHitInit:
            self.fHitInit = 0
            self.prevHit.assign(self.hitPt)
        else:
            offset = self.hitPt - self.prevHit
            SEditor.widget.setPos(SEditor.widget, offset)

    def xlate2D(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.hitPt.assign(self.objectHandles.getWidgetIntersectPt(SEditor.widget, self.constraint[:1]))
        if self.fHitInit:
            self.fHitInit = 0
            self.prevHit.assign(self.hitPt)
        else:
            offset = self.hitPt - self.prevHit
            SEditor.widget.setPos(SEditor.widget, offset)

    def rotate1D(self, state):
        if False:
            for i in range(10):
                print('nop')
        if self.fHitInit:
            self.fHitInit = 0
            self.rotateAxis = self.constraint[:1]
            self.fWidgetTop = self.widgetCheck('top?')
            self.rotationCenter = getScreenXY(SEditor.widget)
            self.lastCrankAngle = getCrankAngle(self.rotationCenter)
        newAngle = getCrankAngle(self.rotationCenter)
        deltaAngle = self.lastCrankAngle - newAngle
        if self.fWidgetTop:
            deltaAngle = -1 * deltaAngle
        if self.rotateAxis == 'x':
            SEditor.widget.setP(SEditor.widget, deltaAngle)
        elif self.rotateAxis == 'y':
            SEditor.widget.setR(SEditor.widget, deltaAngle)
        elif self.rotateAxis == 'z':
            SEditor.widget.setH(SEditor.widget, deltaAngle)
        self.lastCrankAngle = newAngle

    def widgetCheck(self, type):
        if False:
            for i in range(10):
                print('nop')
        axis = self.constraint[:1]
        mWidget2Cam = SEditor.widget.getMat(SEditor.camera)
        pos = VBase3(0)
        decomposeMatrix(mWidget2Cam, VBase3(0), VBase3(0), pos, CSDefault)
        widgetDir = Vec3(pos)
        widgetDir.normalize()
        if axis == 'x':
            widgetAxis = Vec3(mWidget2Cam.xformVec(X_AXIS))
        elif axis == 'y':
            widgetAxis = Vec3(mWidget2Cam.xformVec(Y_AXIS))
        elif axis == 'z':
            widgetAxis = Vec3(mWidget2Cam.xformVec(Z_AXIS))
        widgetAxis.normalize()
        if type == 'top?':
            return widgetDir.dot(widgetAxis) < 0.0
        elif type == 'edge?':
            return abs(widgetDir.dot(widgetAxis)) < 0.2

    def xlateCamXZ(self, state):
        if False:
            i = 10
            return i + 15
        "Constrained 2D motion parallel to the camera's image plane\n        This moves the object in the camera's XZ plane"
        self.fHitInit = 1
        self.fScaleInit = 1
        vWidget2Camera = SEditor.widget.getPos(SEditor.camera)
        x = vWidget2Camera[0]
        y = vWidget2Camera[1]
        z = vWidget2Camera[2]
        dr = SEditor.dr
        SEditor.widget.setX(SEditor.camera, x + 0.5 * dr.mouseDeltaX * dr.nearWidth * (y / dr.near))
        SEditor.widget.setZ(SEditor.camera, z + 0.5 * dr.mouseDeltaY * dr.nearHeight * (y / dr.near))

    def xlateCamXY(self, state):
        if False:
            return 10
        "Constrained 2D motion perpendicular to camera's image plane\n        This moves the object in the camera's XY plane if shift is held\n        Moves object toward camera if control is held\n        "
        self.fScaleInit = 1
        vWidget2Camera = SEditor.widget.getPos(SEditor.camera)
        if self.fHitInit:
            self.fHitInit = 0
            self.xlateSF = Vec3(vWidget2Camera).length()
            coaCenter = getNearProjectionPoint(SEditor.widget)
            self.deltaNearX = coaCenter[0] - SEditor.dr.nearVec[0]
        if SEditor.fControl:
            moveDir = Vec3(vWidget2Camera)
            if moveDir[1] < 0.0:
                moveDir.assign(moveDir * -1)
            moveDir.normalize()
        else:
            moveDir = Vec3(Y_AXIS)
        dr = SEditor.dr
        moveDir.assign(moveDir * (2.0 * dr.mouseDeltaY * self.xlateSF))
        vWidget2Camera += moveDir
        vWidget2Camera.setX((dr.nearVec[0] + self.deltaNearX) * (vWidget2Camera[1] / dr.near))
        SEditor.widget.setPos(SEditor.camera, vWidget2Camera)

    def rotate2D(self, state):
        if False:
            i = 10
            return i + 15
        ' Virtual trackball rotation of widget '
        self.fHitInit = 1
        self.fScaleInit = 1
        tumbleRate = 360
        if state.constrainedDir == 'y' and abs(SEditor.dr.mouseX) > 0.9:
            deltaX = 0
            deltaY = SEditor.dr.mouseDeltaY
        elif state.constrainedDir == 'x' and abs(SEditor.dr.mouseY) > 0.9:
            deltaX = SEditor.dr.mouseDeltaX
            deltaY = 0
        else:
            deltaX = SEditor.dr.mouseDeltaX
            deltaY = SEditor.dr.mouseDeltaY
        relHpr(SEditor.widget, SEditor.camera, deltaX * tumbleRate, -deltaY * tumbleRate, 0)

    def rotateAboutViewVector(self, state):
        if False:
            return 10
        self.fHitInit = 1
        self.fScaleInit = 1
        angle = getCrankAngle(state.coaCenter)
        deltaAngle = angle - state.lastAngle
        state.lastAngle = angle
        relHpr(SEditor.widget, SEditor.camera, 0, 0, -deltaAngle)

    def scale3D(self, state):
        if False:
            for i in range(10):
                print('nop')
        if self.fScaleInit:
            self.fScaleInit = 0
            self.manipRef.setPos(SEditor.widget, 0, 0, 0)
            self.manipRef.setHpr(SEditor.camera, 0, 0, 0)
            self.initScaleMag = Vec3(self.objectHandles.getWidgetIntersectPt(self.manipRef, 'y')).length()
            self.initScale = SEditor.widget.getScale()
        self.fHitInit = 1
        currScale = self.initScale * (self.objectHandles.getWidgetIntersectPt(self.manipRef, 'y').length() / self.initScaleMag)
        SEditor.widget.setScale(currScale)

    def plantSelectedNodePath(self):
        if False:
            print('Hello World!')
        ' Move selected object to intersection point of cursor on scene '
        entry = SEditor.iRay.pickGeom(skipFlags=SKIP_HIDDEN | SKIP_BACKFACE | SKIP_CAMERA)
        if entry != None and SEditor.selected.last != None:
            SEditor.pushUndo(SEditor.selected)
            SEditor.selected.getWrtAll()
            SEditor.widget.setPos(SEditor.camera, entry.getSurfacePoint(entry.getFromNodePath()))
            SEditor.selected.moveWrtWidgetAll()
            messenger.send('DIRECT_manipulateObjectCleanup')

class ObjectHandles(NodePath, DirectObject):

    def __init__(self):
        if False:
            print('Hello World!')
        NodePath.__init__(self)
        self.assign(loader.loadModel('models/misc/objectHandles'))
        self.setName('objectHandles')
        self.scalingNode = self.getChild(0)
        self.scalingNode.setName('ohScalingNode')
        self.ohScalingFactor = 1.0
        self.hitPt = Vec3(0)
        self.xHandles = self.find('**/ohScalingNode')
        self.xPostGroup = self.xHandles.find('**/x-post-group')
        self.xPostCollision = self.xHandles.find('**/x-post')
        self.xRingGroup = self.xHandles.find('**/x-ring-group')
        self.xRingCollision = self.xHandles.find('**/x-ring')
        self.xDiscGroup = self.xHandles.find('**/x-disc-group')
        self.xDisc = self.xHandles.find('**/x-disc-visible')
        self.xDiscCollision = self.xHandles.find('**/x-disc')
        self.yHandles = self.find('**/Y')
        self.yPostGroup = self.yHandles.find('**/y-post-group')
        self.yPostCollision = self.yHandles.find('**/y-post')
        self.yRingGroup = self.yHandles.find('**/y-ring-group')
        self.yRingCollision = self.yHandles.find('**/y-ring')
        self.yDiscGroup = self.yHandles.find('**/y-disc-group')
        self.yDisc = self.yHandles.find('**/y-disc-visible')
        self.yDiscCollision = self.yHandles.find('**/y-disc')
        self.zHandles = self.find('**/Z')
        self.zPostGroup = self.zHandles.find('**/z-post-group')
        self.zPostCollision = self.zHandles.find('**/z-post')
        self.zRingGroup = self.zHandles.find('**/z-ring-group')
        self.zRingCollision = self.zHandles.find('**/z-ring')
        self.zDiscGroup = self.zHandles.find('**/z-disc-group')
        self.zDisc = self.zHandles.find('**/z-disc-visible')
        self.zDiscCollision = self.zHandles.find('**/z-disc')
        self.xPostCollision.hide()
        self.xRingCollision.hide()
        self.xDisc.setColor(1, 0, 0, 0.2)
        self.yPostCollision.hide()
        self.yRingCollision.hide()
        self.yDisc.setColor(0, 1, 0, 0.2)
        self.zPostCollision.hide()
        self.zRingCollision.hide()
        self.zDisc.setColor(0, 0, 1, 0.2)
        self.createObjectHandleLines()
        self.createGuideLines()
        self.hideGuides()
        self.fActive = 1
        self.toggleWidget()
        useDirectRenderStyle(self)

    def coaModeColor(self):
        if False:
            for i in range(10):
                print('nop')
        self.setColor(0.5, 0.5, 0.5, 1)

    def manipModeColor(self):
        if False:
            i = 10
            return i + 15
        self.clearColor()

    def toggleWidget(self):
        if False:
            print('Hello World!')
        if self.fActive:
            self.deactivate()
        else:
            self.activate()

    def activate(self):
        if False:
            return 10
        self.scalingNode.reparentTo(self)
        self.fActive = 1

    def deactivate(self):
        if False:
            while True:
                i = 10
        self.scalingNode.reparentTo(hidden)
        self.fActive = 0

    def showWidgetIfActive(self):
        if False:
            return 10
        if self.fActive:
            self.reparentTo(SEditor.group)

    def showWidget(self):
        if False:
            while True:
                i = 10
        self.reparentTo(SEditor.group)

    def hideWidget(self):
        if False:
            i = 10
            return i + 15
        self.reparentTo(hidden)

    def enableHandles(self, handles):
        if False:
            i = 10
            return i + 15
        if type(handles) is list:
            for handle in handles:
                self.enableHandle(handle)
        elif handles == 'x':
            self.enableHandles(['x-post', 'x-ring', 'x-disc'])
        elif handles == 'y':
            self.enableHandles(['y-post', 'y-ring', 'y-disc'])
        elif handles == 'z':
            self.enableHandles(['z-post', 'z-ring', 'z-disc'])
        elif handles == 'post':
            self.enableHandles(['x-post', 'y-post', 'z-post'])
        elif handles == 'ring':
            self.enableHandles(['x-ring', 'y-ring', 'z-ring'])
        elif handles == 'disc':
            self.enableHandles(['x-disc', 'y-disc', 'z-disc'])
        elif handles == 'all':
            self.enableHandles(['x-post', 'x-ring', 'x-disc', 'y-post', 'y-ring', 'y-disc', 'z-post', 'z-ring', 'z-disc'])

    def enableHandle(self, handle):
        if False:
            print('Hello World!')
        if handle == 'x-post':
            self.xPostGroup.reparentTo(self.xHandles)
        elif handle == 'x-ring':
            self.xRingGroup.reparentTo(self.xHandles)
        elif handle == 'x-disc':
            self.xDiscGroup.reparentTo(self.xHandles)
        if handle == 'y-post':
            self.yPostGroup.reparentTo(self.yHandles)
        elif handle == 'y-ring':
            self.yRingGroup.reparentTo(self.yHandles)
        elif handle == 'y-disc':
            self.yDiscGroup.reparentTo(self.yHandles)
        if handle == 'z-post':
            self.zPostGroup.reparentTo(self.zHandles)
        elif handle == 'z-ring':
            self.zRingGroup.reparentTo(self.zHandles)
        elif handle == 'z-disc':
            self.zDiscGroup.reparentTo(self.zHandles)

    def disableHandles(self, handles):
        if False:
            return 10
        if type(handles) is list:
            for handle in handles:
                self.disableHandle(handle)
        elif handles == 'x':
            self.disableHandles(['x-post', 'x-ring', 'x-disc'])
        elif handles == 'y':
            self.disableHandles(['y-post', 'y-ring', 'y-disc'])
        elif handles == 'z':
            self.disableHandles(['z-post', 'z-ring', 'z-disc'])
        elif handles == 'post':
            self.disableHandles(['x-post', 'y-post', 'z-post'])
        elif handles == 'ring':
            self.disableHandles(['x-ring', 'y-ring', 'z-ring'])
        elif handles == 'disc':
            self.disableHandles(['x-disc', 'y-disc', 'z-disc'])
        elif handles == 'all':
            self.disableHandles(['x-post', 'x-ring', 'x-disc', 'y-post', 'y-ring', 'y-disc', 'z-post', 'z-ring', 'z-disc'])

    def disableHandle(self, handle):
        if False:
            print('Hello World!')
        if handle == 'x-post':
            self.xPostGroup.reparentTo(hidden)
        elif handle == 'x-ring':
            self.xRingGroup.reparentTo(hidden)
        elif handle == 'x-disc':
            self.xDiscGroup.reparentTo(hidden)
        if handle == 'y-post':
            self.yPostGroup.reparentTo(hidden)
        elif handle == 'y-ring':
            self.yRingGroup.reparentTo(hidden)
        elif handle == 'y-disc':
            self.yDiscGroup.reparentTo(hidden)
        if handle == 'z-post':
            self.zPostGroup.reparentTo(hidden)
        elif handle == 'z-ring':
            self.zRingGroup.reparentTo(hidden)
        elif handle == 'z-disc':
            self.zDiscGroup.reparentTo(hidden)

    def showAllHandles(self):
        if False:
            for i in range(10):
                print('nop')
        self.xPost.show()
        self.xRing.show()
        self.xDisc.show()
        self.yPost.show()
        self.yRing.show()
        self.yDisc.show()
        self.zPost.show()
        self.zRing.show()
        self.zDisc.show()

    def hideAllHandles(self):
        if False:
            return 10
        self.xPost.hide()
        self.xRing.hide()
        self.xDisc.hide()
        self.yPost.hide()
        self.yRing.hide()
        self.yDisc.hide()
        self.zPost.hide()
        self.zRing.hide()
        self.zDisc.hide()

    def showHandle(self, handle):
        if False:
            return 10
        if handle == 'x-post':
            self.xPost.show()
        elif handle == 'x-ring':
            self.xRing.show()
        elif handle == 'x-disc':
            self.xDisc.show()
        elif handle == 'y-post':
            self.yPost.show()
        elif handle == 'y-ring':
            self.yRing.show()
        elif handle == 'y-disc':
            self.yDisc.show()
        elif handle == 'z-post':
            self.zPost.show()
        elif handle == 'z-ring':
            self.zRing.show()
        elif handle == 'z-disc':
            self.zDisc.show()

    def showGuides(self):
        if False:
            for i in range(10):
                print('nop')
        self.guideLines.show()

    def hideGuides(self):
        if False:
            return 10
        self.guideLines.hide()

    def setScalingFactor(self, scaleFactor):
        if False:
            print('Hello World!')
        self.ohScalingFactor = scaleFactor
        self.scalingNode.setScale(self.ohScalingFactor)

    def getScalingFactor(self):
        if False:
            for i in range(10):
                print('nop')
        return self.scalingNode.getScale()

    def transferObjectHandlesScale(self):
        if False:
            print('Hello World!')
        ohs = self.getScale()
        sns = self.scalingNode.getScale()
        self.scalingNode.setScale(ohs[0] * sns[0], ohs[1] * sns[1], ohs[2] * sns[2])
        self.setScale(1)

    def multiplyScalingFactorBy(self, factor):
        if False:
            while True:
                i = 10
        taskMgr.remove('resizeObjectHandles')
        sf = self.ohScalingFactor = self.ohScalingFactor * factor
        self.scalingNode.lerpScale(sf, sf, sf, 0.5, blendType='easeInOut', task='resizeObjectHandles')

    def growToFit(self):
        if False:
            while True:
                i = 10
        taskMgr.remove('resizeObjectHandles')
        pos = SEditor.widget.getPos(SEditor.camera)
        minDim = min(SEditor.dr.nearWidth, SEditor.dr.nearHeight)
        sf = 0.4 * minDim * (pos[1] / SEditor.dr.near)
        self.ohScalingFactor = sf
        self.scalingNode.lerpScale(sf, sf, sf, 0.5, blendType='easeInOut', task='resizeObjectHandles')

    def createObjectHandleLines(self):
        if False:
            print('Hello World!')
        self.xPost = self.xPostGroup.attachNewNode('x-post-visible')
        lines = LineNodePath(self.xPost)
        lines.setColor(VBase4(1, 0, 0, 1))
        lines.setThickness(5)
        lines.moveTo(0, 0, 0)
        lines.drawTo(1.5, 0, 0)
        lines.create()
        lines = LineNodePath(self.xPost)
        lines.setColor(VBase4(1, 0, 0, 1))
        lines.setThickness(1.5)
        lines.moveTo(0, 0, 0)
        lines.drawTo(-1.5, 0, 0)
        lines.create()
        self.xRing = self.xRingGroup.attachNewNode('x-ring-visible')
        lines = LineNodePath(self.xRing)
        lines.setColor(VBase4(1, 0, 0, 1))
        lines.setThickness(3)
        lines.moveTo(0, 1, 0)
        for ang in range(15, 370, 15):
            lines.drawTo(0, math.cos(deg2Rad(ang)), math.sin(deg2Rad(ang)))
        lines.create()
        self.yPost = self.yPostGroup.attachNewNode('y-post-visible')
        lines = LineNodePath(self.yPost)
        lines.setColor(VBase4(0, 1, 0, 1))
        lines.setThickness(5)
        lines.moveTo(0, 0, 0)
        lines.drawTo(0, 1.5, 0)
        lines.create()
        lines = LineNodePath(self.yPost)
        lines.setColor(VBase4(0, 1, 0, 1))
        lines.setThickness(1.5)
        lines.moveTo(0, 0, 0)
        lines.drawTo(0, -1.5, 0)
        lines.create()
        self.yRing = self.yRingGroup.attachNewNode('y-ring-visible')
        lines = LineNodePath(self.yRing)
        lines.setColor(VBase4(0, 1, 0, 1))
        lines.setThickness(3)
        lines.moveTo(1, 0, 0)
        for ang in range(15, 370, 15):
            lines.drawTo(math.cos(deg2Rad(ang)), 0, math.sin(deg2Rad(ang)))
        lines.create()
        self.zPost = self.zPostGroup.attachNewNode('z-post-visible')
        lines = LineNodePath(self.zPost)
        lines.setColor(VBase4(0, 0, 1, 1))
        lines.setThickness(5)
        lines.moveTo(0, 0, 0)
        lines.drawTo(0, 0, 1.5)
        lines.create()
        lines = LineNodePath(self.zPost)
        lines.setColor(VBase4(0, 0, 1, 1))
        lines.setThickness(1.5)
        lines.moveTo(0, 0, 0)
        lines.drawTo(0, 0, -1.5)
        lines.create()
        self.zRing = self.zRingGroup.attachNewNode('z-ring-visible')
        lines = LineNodePath(self.zRing)
        lines.setColor(VBase4(0, 0, 1, 1))
        lines.setThickness(3)
        lines.moveTo(1, 0, 0)
        for ang in range(15, 370, 15):
            lines.drawTo(math.cos(deg2Rad(ang)), math.sin(deg2Rad(ang)), 0)
        lines.create()

    def createGuideLines(self):
        if False:
            return 10
        self.guideLines = self.attachNewNode('guideLines')
        lines = LineNodePath(self.guideLines)
        lines.setColor(VBase4(1, 0, 0, 1))
        lines.setThickness(0.5)
        lines.moveTo(-500, 0, 0)
        lines.drawTo(500, 0, 0)
        lines.create()
        lines.setName('x-guide')
        lines = LineNodePath(self.guideLines)
        lines.setColor(VBase4(0, 1, 0, 1))
        lines.setThickness(0.5)
        lines.moveTo(0, -500, 0)
        lines.drawTo(0, 500, 0)
        lines.create()
        lines.setName('y-guide')
        lines = LineNodePath(self.guideLines)
        lines.setColor(VBase4(0, 0, 1, 1))
        lines.setThickness(0.5)
        lines.moveTo(0, 0, -500)
        lines.drawTo(0, 0, 500)
        lines.create()
        lines.setName('z-guide')

    def getAxisIntersectPt(self, axis):
        if False:
            i = 10
            return i + 15
        mCam2Widget = SEditor.camera.getMat(SEditor.widget)
        lineDir = Vec3(mCam2Widget.xformVec(SEditor.dr.nearVec))
        lineDir.normalize()
        lineOrigin = VBase3(0)
        decomposeMatrix(mCam2Widget, VBase3(0), VBase3(0), lineOrigin, CSDefault)
        if axis == 'x':
            if abs(lineDir.dot(Y_AXIS)) > abs(lineDir.dot(Z_AXIS)):
                self.hitPt.assign(planeIntersect(lineOrigin, lineDir, ORIGIN, Y_AXIS))
            else:
                self.hitPt.assign(planeIntersect(lineOrigin, lineDir, ORIGIN, Z_AXIS))
            self.hitPt.setY(0)
            self.hitPt.setZ(0)
        elif axis == 'y':
            if abs(lineDir.dot(X_AXIS)) > abs(lineDir.dot(Z_AXIS)):
                self.hitPt.assign(planeIntersect(lineOrigin, lineDir, ORIGIN, X_AXIS))
            else:
                self.hitPt.assign(planeIntersect(lineOrigin, lineDir, ORIGIN, Z_AXIS))
            self.hitPt.setX(0)
            self.hitPt.setZ(0)
        elif axis == 'z':
            if abs(lineDir.dot(X_AXIS)) > abs(lineDir.dot(Y_AXIS)):
                self.hitPt.assign(planeIntersect(lineOrigin, lineDir, ORIGIN, X_AXIS))
            else:
                self.hitPt.assign(planeIntersect(lineOrigin, lineDir, ORIGIN, Y_AXIS))
            self.hitPt.setX(0)
            self.hitPt.setY(0)
        return self.hitPt

    def getWidgetIntersectPt(self, nodePath, plane):
        if False:
            return 10
        mCam2NodePath = SEditor.camera.getMat(nodePath)
        lineOrigin = VBase3(0)
        decomposeMatrix(mCam2NodePath, VBase3(0), VBase3(0), lineOrigin, CSDefault)
        lineDir = Vec3(mCam2NodePath.xformVec(SEditor.dr.nearVec))
        lineDir.normalize()
        if plane == 'x':
            self.hitPt.assign(planeIntersect(lineOrigin, lineDir, ORIGIN, X_AXIS))
        elif plane == 'y':
            self.hitPt.assign(planeIntersect(lineOrigin, lineDir, ORIGIN, Y_AXIS))
        elif plane == 'z':
            self.hitPt.assign(planeIntersect(lineOrigin, lineDir, ORIGIN, Z_AXIS))
        return self.hitPt