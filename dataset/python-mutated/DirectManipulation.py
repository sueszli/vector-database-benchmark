import math
from panda3d.core import BitMask32, BoundingHexahedron, CSDefault, Mat4, NodePath, Point3, VBase3, VBase4, Vec3, decomposeMatrix, deg2Rad
from direct.showbase.DirectObject import DirectObject
from direct.showbase.MessengerGlobal import messenger
from direct.showbase.ShowBaseGlobal import hidden
from direct.showbase import ShowBaseGlobal
from . import DirectGlobals as DG
from .DirectUtil import useDirectRenderStyle
from .DirectGeometry import LineNodePath, getCrankAngle, getNearProjectionPoint, getScreenXY, planeIntersect, relHpr
from .DirectSelection import SelectionRay
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from copy import deepcopy
from typing import Optional

class DirectManipulationControl(DirectObject):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.objectHandles = ObjectHandles()
        self.hitPt = Point3(0)
        self.prevHit = Vec3(0)
        self.widgetList: list[ObjectHandles] = []
        self.hitPtScale = Point3(0)
        self.prevHitScale = Vec3(0)
        self.rotationCenter = Point3(0)
        self.initScaleMag = 1
        self.manipRef = ShowBaseGlobal.direct.group.attachNewNode('manipRef')
        self.hitPtDist = 0
        self.constraint: Optional[str] = None
        self.rotateAxis = 'x'
        self.lastCrankAngle = 0
        self.fSetCoa = 0
        self.fHitInit = 1
        self.fScaleInit = 1
        self.fScaleInit1 = 1
        self.fWidgetTop = 0
        self.fFreeManip = 1
        self.fScaling3D = 0
        self.fScaling1D = 0
        self.fMovable = 1
        self.mode = None
        self.worldSpaceManip = False
        self.useSeparateScaleHandles = False
        self.actionEvents = [['DIRECT-mouse1', self.manipulationStart], ['DIRECT-mouse1Up', self.manipulationStop], ['tab', self.toggleObjectHandlesMode], ['DIRECT-widgetScaleUp', self.scaleWidget, 2.0], ['DIRECT-widgetScaleDown', self.scaleWidget, 0.5], ['shift-f', self.objectHandles.growToFit], ['i', self.plantSelectedNodePath]]
        self.defaultSkipFlags = DG.SKIP_HIDDEN | DG.SKIP_BACKFACE
        self.optionalSkipFlags = 0
        self.unmovableTagList = []
        self.fAllowSelectionOnly = 0
        self.fAllowMarquee = 0
        self.marquee = None
        self.fMultiView = 0
        self.fGridSnap = 0

    def scaleWidget(self, factor):
        if False:
            return 10
        if hasattr(ShowBaseGlobal.direct, 'widget'):
            ShowBaseGlobal.direct.widget.multiplyScalingFactorBy(factor)
        else:
            self.objectHandles.multiplyScalingFactorBy(factor)

    def supportMultiView(self):
        if False:
            i = 10
            return i + 15
        if self.fMultiView:
            return
        self.objectHandles.hide(BitMask32.bit(0))
        self.objectHandles.hide(BitMask32.bit(1))
        self.objectHandles.hide(BitMask32.bit(2))
        self.topViewWidget = ObjectHandles('topViewWidget')
        self.frontViewWidget = ObjectHandles('frontViewWidget')
        self.leftViewWidget = ObjectHandles('leftViewWidget')
        self.widgetList = [self.topViewWidget, self.frontViewWidget, self.leftViewWidget, self.objectHandles]
        self.topViewWidget.hide(BitMask32.bit(1))
        self.topViewWidget.hide(BitMask32.bit(2))
        self.topViewWidget.hide(BitMask32.bit(3))
        self.frontViewWidget.hide(BitMask32.bit(0))
        self.frontViewWidget.hide(BitMask32.bit(2))
        self.frontViewWidget.hide(BitMask32.bit(3))
        self.leftViewWidget.hide(BitMask32.bit(0))
        self.leftViewWidget.hide(BitMask32.bit(1))
        self.leftViewWidget.hide(BitMask32.bit(3))
        self.fMultiView = 1

    def manipulationStart(self, modifiers):
        if False:
            while True:
                i = 10
        self.mode = 'select'
        if ShowBaseGlobal.direct.cameraControl.useMayaCamControls and modifiers == 4:
            self.mode = 'camera'
        if self.fAllowSelectionOnly:
            return
        if self.fScaling1D == 0 and self.fScaling3D == 0:
            entry = ShowBaseGlobal.direct.iRay.pickWidget(skipFlags=DG.SKIP_WIDGET)
            if entry:
                self.hitPt.assign(entry.getSurfacePoint(entry.getFromNodePath()))
                self.hitPtDist = Vec3(self.hitPt).length()
                self.constraint = entry.getIntoNodePath().getName()
            else:
                self.constraint = None
                if ShowBaseGlobal.direct.cameraControl.useMayaCamControls and (not ShowBaseGlobal.direct.gotControl(modifiers)) and (not self.fAllowMarquee):
                    return
        else:
            entry = None
        if not ShowBaseGlobal.direct.gotAlt(modifiers):
            if entry:
                taskMgr.doMethodLater(DG.MANIPULATION_MOVE_DELAY, self.switchToMoveMode, 'manip-move-wait')
                self.moveDir = None
                watchMouseTask = Task.Task(self.watchMouseTask)
                watchMouseTask.initX = ShowBaseGlobal.direct.dr.mouseX
                watchMouseTask.initY = ShowBaseGlobal.direct.dr.mouseY
                taskMgr.add(watchMouseTask, 'manip-watch-mouse')
            elif ShowBaseGlobal.direct.fControl:
                self.mode = 'move'
                self.manipulateObject()
            elif not ShowBaseGlobal.direct.fAlt and self.fAllowMarquee:
                self.moveDir = None
                watchMarqueeTask = Task.Task(self.watchMarqueeTask)
                watchMarqueeTask.initX = ShowBaseGlobal.direct.dr.mouseX
                watchMarqueeTask.initY = ShowBaseGlobal.direct.dr.mouseY
                taskMgr.add(watchMarqueeTask, 'manip-marquee-mouse')

    def switchToWorldSpaceMode(self):
        if False:
            print('Hello World!')
        self.worldSpaceManip = True

    def switchToLocalSpaceMode(self):
        if False:
            for i in range(10):
                print('nop')
        self.worldSpaceManip = False

    def switchToMoveMode(self, state):
        if False:
            return 10
        taskMgr.remove('manip-watch-mouse')
        self.mode = 'move'
        self.manipulateObject()
        return Task.done

    def watchMouseTask(self, state):
        if False:
            return 10
        if abs(state.initX - ShowBaseGlobal.direct.dr.mouseX) > 0.01 or abs(state.initY - ShowBaseGlobal.direct.dr.mouseY) > 0.01:
            taskMgr.remove('manip-move-wait')
            self.mode = 'move'
            self.manipulateObject()
            return Task.done
        else:
            return Task.cont

    def watchMarqueeTask(self, state):
        if False:
            while True:
                i = 10
        taskMgr.remove('manip-watch-mouse')
        taskMgr.remove('manip-move-wait')
        self.mode = 'select'
        self.drawMarquee(state.initX, state.initY)
        return Task.cont

    def drawMarquee(self, startX, startY):
        if False:
            while True:
                i = 10
        if self.marquee:
            self.marquee.removeNode()
            self.marquee = None
        if ShowBaseGlobal.direct.cameraControl.useMayaCamControls and ShowBaseGlobal.direct.fAlt:
            return
        if ShowBaseGlobal.direct.fControl:
            return
        endX = ShowBaseGlobal.direct.dr.mouseX
        endY = ShowBaseGlobal.direct.dr.mouseY
        if abs(endX - startX) < 0.01 and abs(endY - startY) < 0.01:
            return
        self.marquee = LineNodePath(ShowBaseGlobal.base.render2d, 'marquee', 0.5, VBase4(0.8, 0.6, 0.6, 1))
        self.marqueeInfo = (startX, startY, endX, endY)
        self.marquee.drawLines([[(startX, 0, startY), (startX, 0, endY)], [(startX, 0, endY), (endX, 0, endY)], [(endX, 0, endY), (endX, 0, startY)], [(endX, 0, startY), (startX, 0, startY)]])
        self.marquee.create()
        if self.fMultiView:
            DG.LE_showInOneCam(self.marquee, ShowBaseGlobal.direct.camera.getName())

    def manipulationStop(self):
        if False:
            return 10
        taskMgr.remove('manipulateObject')
        taskMgr.remove('manip-move-wait')
        taskMgr.remove('manip-watch-mouse')
        taskMgr.remove('manip-marquee-mouse')
        direct = ShowBaseGlobal.direct
        if self.mode == 'select':
            base = ShowBaseGlobal.base
            skipFlags = self.defaultSkipFlags | self.optionalSkipFlags
            skipFlags |= DG.SKIP_CAMERA * (1 - base.getControl())
            if self.marquee:
                self.marquee.removeNode()
                self.marquee = None
                direct.deselectAll()
                startX = self.marqueeInfo[0]
                startY = self.marqueeInfo[1]
                endX = self.marqueeInfo[2]
                endY = self.marqueeInfo[3]
                fll = Point3(0, 0, 0)
                flr = Point3(0, 0, 0)
                fur = Point3(0, 0, 0)
                ful = Point3(0, 0, 0)
                nll = Point3(0, 0, 0)
                nlr = Point3(0, 0, 0)
                nur = Point3(0, 0, 0)
                nul = Point3(0, 0, 0)
                lens = direct.cam.node().getLens()
                lens.extrude((startX, startY), nul, ful)
                lens.extrude((endX, startY), nur, fur)
                lens.extrude((endX, endY), nlr, flr)
                lens.extrude((startX, endY), nll, fll)
                marqueeFrustum = BoundingHexahedron(fll, flr, fur, ful, nll, nlr, nur, nul)
                marqueeFrustum.xform(direct.cam.getNetTransform().getMat())
                base.marqueeFrustum = marqueeFrustum

                def findTaggedNodePath(nodePath):
                    if False:
                        return 10
                    for tag in direct.selected.tagList:
                        if nodePath.hasNetTag(tag):
                            nodePath = nodePath.findNetTag(tag)
                            return nodePath
                    return None
                selectionList = []
                for geom in base.render.findAllMatches('**/+GeomNode'):
                    if skipFlags & DG.SKIP_HIDDEN and geom.isHidden():
                        continue
                    elif skipFlags & DG.SKIP_CAMERA and base.camera in geom.getAncestors():
                        continue
                    elif skipFlags & DG.SKIP_UNPICKABLE and geom.getName() in direct.iRay.unpickable:
                        continue
                    nodePath = findTaggedNodePath(geom)
                    if nodePath in selectionList:
                        continue
                    bb = geom.getBounds()
                    bbc = bb.makeCopy()
                    bbc.xform(geom.getParent().getNetTransform().getMat())
                    boundingSphereTest = marqueeFrustum.contains(bbc)
                    if boundingSphereTest > 1:
                        if boundingSphereTest == 7:
                            if nodePath not in selectionList:
                                selectionList.append(nodePath)
                        else:
                            tMat = Mat4(geom.getMat())
                            geom.clearMat()
                            min = Point3(0)
                            max = Point3(0)
                            geom.calcTightBounds(min, max)
                            geom.setMat(tMat)
                            fll = Point3(min[0], max[1], min[2])
                            flr = Point3(max[0], max[1], min[2])
                            fur = max
                            ful = Point3(min[0], max[1], max[2])
                            nll = min
                            nlr = Point3(max[0], min[1], min[2])
                            nur = Point3(max[0], min[1], max[2])
                            nul = Point3(min[0], min[1], max[2])
                            tbb = BoundingHexahedron(fll, flr, fur, ful, nll, nlr, nur, nul)
                            tbb.xform(geom.getNetTransform().getMat())
                            tightBoundTest = marqueeFrustum.contains(tbb)
                            if tightBoundTest > 1:
                                if nodePath not in selectionList:
                                    selectionList.append(nodePath)
                for nodePath in selectionList:
                    direct.select(nodePath, 1)
            else:
                entry = direct.iRay.pickGeom(skipFlags=skipFlags)
                if entry:
                    self.hitPt.assign(entry.getSurfacePoint(entry.getFromNodePath()))
                    self.hitPtDist = Vec3(self.hitPt).length()
                    direct.select(entry.getIntoNodePath(), direct.fShift)
                else:
                    direct.deselectAll()
        self.manipulateObjectCleanup()
        self.mode = None

    def manipulateObjectCleanup(self):
        if False:
            print('Hello World!')
        direct = ShowBaseGlobal.direct
        if self.fScaling3D or self.fScaling1D:
            if hasattr(direct, 'widget'):
                direct.widget.transferObjectHandlesScale()
            else:
                self.objectHandles.transferObjectHandlesScale()
            self.fScaling3D = 0
            self.fScaling1D = 0
        direct.selected.highlightAll()
        if hasattr(direct, 'widget'):
            direct.widget.showAllHandles()
        else:
            self.objectHandles.showAllHandles()
        if direct.clusterMode == 'client':
            direct.cluster('direct.manipulationControl.objectHandles.showAllHandles()')
        if hasattr(direct, 'widget'):
            direct.widget.hideGuides()
        else:
            self.objectHandles.hideGuides()
        self.spawnFollowSelectedNodePathTask()
        messenger.send('DIRECT_manipulateObjectCleanup', [direct.selected.getSelectedAsList()])

    def spawnFollowSelectedNodePathTask(self):
        if False:
            while True:
                i = 10
        if not ShowBaseGlobal.direct.selected.last:
            return
        taskMgr.remove('followSelectedNodePath')
        pos = VBase3(0)
        hpr = VBase3(0)
        decomposeMatrix(ShowBaseGlobal.direct.selected.last.mCoa2Dnp, VBase3(0), hpr, pos, CSDefault)
        t = Task.Task(self.followSelectedNodePathTask)
        t.pos = pos
        t.hpr = hpr
        t.base = ShowBaseGlobal.direct.selected.last
        taskMgr.add(t, 'followSelectedNodePath')

    def followSelectedNodePathTask(self, state):
        if False:
            return 10
        if hasattr(ShowBaseGlobal.direct, 'manipulationControl') and ShowBaseGlobal.direct.manipulationControl.fMultiView:
            for widget in ShowBaseGlobal.direct.manipulationControl.widgetList:
                if self.worldSpaceManip:
                    widget.setPos(state.base, state.pos)
                    widget.setHpr(ShowBaseGlobal.base.render, VBase3(0))
                else:
                    widget.setPosHpr(state.base, state.pos, state.hpr)
        else:
            widget = ShowBaseGlobal.direct.widget
            if self.worldSpaceManip:
                widget.setPos(state.base, state.pos)
                widget.setHpr(ShowBaseGlobal.base.render, VBase3(0))
            else:
                widget.setPosHpr(state.base, state.pos, state.hpr)
        return Task.cont

    def enableManipulation(self):
        if False:
            i = 10
            return i + 15
        for event in self.actionEvents:
            self.accept(event[0], event[1], extraArgs=event[2:])
        self.fAllowSelectionOnly = 0

    def disableManipulation(self, allowSelectionOnly=False):
        if False:
            return 10
        for event in self.actionEvents:
            self.ignore(event[0])
        if allowSelectionOnly:
            self.fAllowSelectionOnly = allowSelectionOnly
            self.accept('DIRECT-mouse1', self.manipulationStart)
            self.accept('DIRECT-mouse1Up', self.manipulationStop)
        self.removeManipulateObjectTask()
        taskMgr.remove('manipulateObject')
        taskMgr.remove('manip-move-wait')
        taskMgr.remove('manip-watch-mouse')
        taskMgr.remove('highlightWidgetTask')

    def toggleObjectHandlesMode(self):
        if False:
            for i in range(10):
                print('nop')
        if self.fMovable:
            self.fSetCoa = 1 - self.fSetCoa
            if self.fSetCoa:
                if hasattr(ShowBaseGlobal.direct, 'widget'):
                    ShowBaseGlobal.direct.widget.coaModeColor()
                else:
                    self.objectHandles.coaModeColor()
            elif hasattr(ShowBaseGlobal.direct, 'widget'):
                ShowBaseGlobal.direct.widget.manipModeColor()
            else:
                self.objectHandles.manipModeColor()
        elif hasattr(ShowBaseGlobal.direct, 'widget'):
            ShowBaseGlobal.direct.widget.disabledModeColor()
        else:
            self.objectHandles.disabledModeColor()

    def removeManipulateObjectTask(self):
        if False:
            i = 10
            return i + 15
        taskMgr.remove('manipulateObject')

    def enableWidgetMove(self):
        if False:
            i = 10
            return i + 15
        self.fMovable = 1
        if self.fSetCoa:
            if hasattr(ShowBaseGlobal.direct, 'widget'):
                ShowBaseGlobal.direct.widget.coaModeColor()
            else:
                self.objectHandles.coaModeColor()
        elif hasattr(ShowBaseGlobal.direct, 'widget'):
            ShowBaseGlobal.direct.widget.manipModeColor()
        else:
            self.objectHandles.manipModeColor()

    def disableWidgetMove(self):
        if False:
            return 10
        self.fMovable = 0
        if hasattr(ShowBaseGlobal.direct, 'widget'):
            ShowBaseGlobal.direct.widget.disabledModeColor()
        else:
            self.objectHandles.disabledModeColor()

    def getEditTypes(self, objects):
        if False:
            while True:
                i = 10
        editTypes = 0
        for tag in self.unmovableTagList:
            for selected in objects:
                unmovableTag = selected.getTag(tag)
                if unmovableTag:
                    editTypes |= int(unmovableTag)
        return editTypes

    def manipulateObject(self):
        if False:
            while True:
                i = 10
        direct = ShowBaseGlobal.direct
        selectedList = direct.selected.getSelectedAsList()
        editTypes = self.getEditTypes(selectedList)
        if editTypes & DG.EDIT_TYPE_UNEDITABLE == DG.EDIT_TYPE_UNEDITABLE:
            return
        self.currEditTypes = editTypes
        if selectedList:
            taskMgr.remove('followSelectedNodePath')
            taskMgr.remove('highlightWidgetTask')
            self.fManip = 1
            direct.pushUndo(direct.selected)
            if hasattr(direct, 'widget'):
                direct.widget.showGuides()
                direct.widget.hideAllHandles()
                direct.widget.showHandle(self.constraint)
            else:
                self.objectHandles.showGuides()
                self.objectHandles.hideAllHandles()
                self.objectHandles.showHandle(self.constraint)
            if direct.clusterMode == 'client':
                oh = 'direct.manipulationControl.objectHandles'
                cluster = direct.cluster
                cluster(oh + '.showGuides()', 0)
                cluster(oh + '.hideAllHandles()', 0)
                cluster(oh + '.showHandle("%s")' % self.constraint, 0)
            direct.selected.getWrtAll()
            direct.selected.dehighlightAll()
            messenger.send('DIRECT_manipulateObjectStart')
            self.spawnManipulateObjectTask()

    def spawnManipulateObjectTask(self):
        if False:
            return 10
        self.fHitInit = 1
        self.fScaleInit = 1
        if not self.fScaling1D and (not self.fScaling3D):
            self.fScaleInit1 = 1
        t = Task.Task(self.manipulateObjectTask)
        t.fMouseX = abs(ShowBaseGlobal.direct.dr.mouseX) > 0.9
        t.fMouseY = abs(ShowBaseGlobal.direct.dr.mouseY) > 0.9
        if t.fMouseX:
            t.constrainedDir = 'y'
        else:
            t.constrainedDir = 'x'
        t.coaCenter = getScreenXY(ShowBaseGlobal.direct.widget)
        if t.fMouseX and t.fMouseY:
            t.lastAngle = getCrankAngle(t.coaCenter)
        taskMgr.add(t, 'manipulateObject')

    def manipulateObjectTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        if self.fScaling1D:
            self.scale1D(state)
        elif self.fScaling3D:
            self.scale3D(state)
        elif self.constraint:
            type = self.constraint[2:]
            if self.useSeparateScaleHandles:
                if type == 'post' and (not self.currEditTypes & DG.EDIT_TYPE_UNMOVABLE):
                    self.xlate1D(state)
                elif type == 'disc' and (not self.currEditTypes & DG.EDIT_TYPE_UNMOVABLE):
                    self.xlate2D(state)
                elif type == 'ring' and (not self.currEditTypes & DG.EDIT_TYPE_UNROTATABLE):
                    self.rotate1D(state)
                elif type == 'scale' and (not self.currEditTypes & DG.EDIT_TYPE_UNSCALABLE):
                    if ShowBaseGlobal.direct.fShift:
                        self.fScaling3D = 1
                        self.scale3D(state)
                    else:
                        self.fScaling1D = 1
                        self.scale1D(state)
            elif ShowBaseGlobal.direct.fControl and (not self.currEditTypes & DG.EDIT_TYPE_UNSCALABLE):
                if type == 'post':
                    self.fScaling1D = 1
                    self.scale1D(state)
                else:
                    self.fScaling3D = 1
                    self.scale3D(state)
            elif type == 'post' and (not self.currEditTypes & DG.EDIT_TYPE_UNMOVABLE):
                self.xlate1D(state)
            elif type == 'disc' and (not self.currEditTypes & DG.EDIT_TYPE_UNMOVABLE):
                self.xlate2D(state)
            elif type == 'ring' and (not self.currEditTypes & DG.EDIT_TYPE_UNROTATABLE):
                self.rotate1D(state)
        elif self.fFreeManip and (not self.useSeparateScaleHandles):
            if 0 and (self.fScaling1D or self.fScaling3D) and (not ShowBaseGlobal.direct.fAlt):
                if hasattr(ShowBaseGlobal.direct, 'widget'):
                    ShowBaseGlobal.direct.widget.transferObjectHandleScale()
                else:
                    self.objectHandles.transferObjectHandlesScale()
                self.fScaling1D = 0
                self.fScaling3D = 0
            if ShowBaseGlobal.direct.fControl and (not self.currEditTypes & DG.EDIT_TYPE_UNSCALABLE):
                self.fScaling3D = 1
                self.scale3D(state)
            elif state.fMouseX and state.fMouseY and (not self.currEditTypes & DG.EDIT_TYPE_UNROTATABLE):
                self.rotateAboutViewVector(state)
            elif state.fMouseX or (state.fMouseY and (not self.currEditTypes & DG.EDIT_TYPE_UNMOVABLE)):
                self.rotate2D(state)
            elif not self.currEditTypes & DG.EDIT_TYPE_UNMOVABLE:
                if ShowBaseGlobal.direct.fShift or ShowBaseGlobal.direct.fControl:
                    self.xlateCamXY(state)
                else:
                    self.xlateCamXZ(state)
        else:
            return Task.done
        if self.fSetCoa:
            ShowBaseGlobal.direct.selected.last.mCoa2Dnp.assign(ShowBaseGlobal.direct.widget.getMat(ShowBaseGlobal.direct.selected.last))
        else:
            ShowBaseGlobal.direct.selected.moveWrtWidgetAll()
        return Task.cont

    def addTag(self, tag):
        if False:
            i = 10
            return i + 15
        if tag not in self.unmovableTagList:
            self.unmovableTagList.append(tag)

    def removeTag(self, tag):
        if False:
            return 10
        self.unmovableTagList.remove(tag)

    def gridSnapping(self, nodePath, offset):
        if False:
            i = 10
            return i + 15
        offsetX = nodePath.getX() + offset.getX()
        offsetY = nodePath.getY() + offset.getY()
        offsetZ = nodePath.getZ() + offset.getZ()
        if offsetX < 0.0:
            signX = -1.0
        else:
            signX = 1.0
        modX = math.fabs(offsetX) % ShowBaseGlobal.direct.grid.gridSpacing
        floorX = math.floor(math.fabs(offsetX) / ShowBaseGlobal.direct.grid.gridSpacing)
        if modX < ShowBaseGlobal.direct.grid.gridSpacing / 2.0:
            offsetX = signX * floorX * ShowBaseGlobal.direct.grid.gridSpacing
        else:
            offsetX = signX * (floorX + 1) * ShowBaseGlobal.direct.grid.gridSpacing
        if offsetY < 0.0:
            signY = -1.0
        else:
            signY = 1.0
        modY = math.fabs(offsetY) % ShowBaseGlobal.direct.grid.gridSpacing
        floorY = math.floor(math.fabs(offsetY) / ShowBaseGlobal.direct.grid.gridSpacing)
        if modY < ShowBaseGlobal.direct.grid.gridSpacing / 2.0:
            offsetY = signY * floorY * ShowBaseGlobal.direct.grid.gridSpacing
        else:
            offsetY = signY * (floorY + 1) * ShowBaseGlobal.direct.grid.gridSpacing
        if offsetZ < 0.0:
            signZ = -1.0
        else:
            signZ = 1.0
        modZ = math.fabs(offsetZ) % ShowBaseGlobal.direct.grid.gridSpacing
        floorZ = math.floor(math.fabs(offsetZ) / ShowBaseGlobal.direct.grid.gridSpacing)
        if modZ < ShowBaseGlobal.direct.grid.gridSpacing / 2.0:
            offsetZ = signZ * floorZ * ShowBaseGlobal.direct.grid.gridSpacing
        else:
            offsetZ = signZ * (floorZ + 1) * ShowBaseGlobal.direct.grid.gridSpacing
        return Point3(offsetX, offsetY, offsetZ)

    def xlate1D(self, state):
        if False:
            for i in range(10):
                print('nop')
        assert self.constraint is not None
        self.hitPt.assign(self.objectHandles.getAxisIntersectPt(self.constraint[:1]))
        if self.fHitInit:
            self.fHitInit = 0
            self.prevHit.assign(self.hitPt)
        else:
            offset = self.hitPt - self.prevHit
            if hasattr(ShowBaseGlobal.direct, 'manipulationControl') and ShowBaseGlobal.direct.manipulationControl.fMultiView:
                for widget in ShowBaseGlobal.direct.manipulationControl.widgetList:
                    if self.fGridSnap:
                        widget.setPos(self.gridSnapping(widget, offset))
                    else:
                        widget.setPos(widget, offset)
            elif self.fGridSnap:
                ShowBaseGlobal.direct.widget.setPos(self.gridSnapping(ShowBaseGlobal.direct.widget, offset))
            else:
                ShowBaseGlobal.direct.widget.setPos(ShowBaseGlobal.direct.widget, offset)

    def xlate2D(self, state):
        if False:
            while True:
                i = 10
        assert self.constraint is not None
        self.hitPt.assign(self.objectHandles.getWidgetIntersectPt(ShowBaseGlobal.direct.widget, self.constraint[:1]))
        if self.fHitInit:
            self.fHitInit = 0
            self.prevHit.assign(self.hitPt)
        else:
            offset = self.hitPt - self.prevHit
            if hasattr(ShowBaseGlobal.direct, 'manipulationControl') and ShowBaseGlobal.direct.manipulationControl.fMultiView:
                for widget in ShowBaseGlobal.direct.manipulationControl.widgetList:
                    if self.fGridSnap:
                        widget.setPos(self.gridSnapping(widget, offset))
                    else:
                        widget.setPos(widget, offset)
                if ShowBaseGlobal.direct.camera.getName() != 'persp':
                    self.prevHit.assign(self.hitPt)
            elif self.fGridSnap:
                ShowBaseGlobal.direct.widget.setPos(self.gridSnapping(ShowBaseGlobal.direct.widget, offset))
            else:
                ShowBaseGlobal.direct.widget.setPos(ShowBaseGlobal.direct.widget, offset)

    def rotate1D(self, state):
        if False:
            print('Hello World!')
        assert self.constraint is not None
        if self.fHitInit:
            self.fHitInit = 0
            self.rotateAxis = self.constraint[:1]
            self.fWidgetTop = self.widgetCheck('top?')
            self.rotationCenter = getScreenXY(ShowBaseGlobal.direct.widget)
            self.lastCrankAngle = getCrankAngle(self.rotationCenter)
        newAngle = getCrankAngle(self.rotationCenter)
        deltaAngle = self.lastCrankAngle - newAngle
        if self.fWidgetTop:
            deltaAngle = -1 * deltaAngle
        if self.rotateAxis == 'x':
            if hasattr(ShowBaseGlobal.direct, 'manipulationControl') and ShowBaseGlobal.direct.manipulationControl.fMultiView:
                for widget in ShowBaseGlobal.direct.manipulationControl.widgetList:
                    widget.setP(widget, deltaAngle)
            else:
                ShowBaseGlobal.direct.widget.setP(ShowBaseGlobal.direct.widget, deltaAngle)
        elif self.rotateAxis == 'y':
            if hasattr(ShowBaseGlobal.direct, 'manipulationControl') and ShowBaseGlobal.direct.manipulationControl.fMultiView:
                for widget in ShowBaseGlobal.direct.manipulationControl.widgetList:
                    widget.setR(widget, deltaAngle)
            else:
                ShowBaseGlobal.direct.widget.setR(ShowBaseGlobal.direct.widget, deltaAngle)
        elif self.rotateAxis == 'z':
            if hasattr(ShowBaseGlobal.direct, 'manipulationControl') and ShowBaseGlobal.direct.manipulationControl.fMultiView:
                for widget in ShowBaseGlobal.direct.manipulationControl.widgetList:
                    widget.setH(widget, deltaAngle)
            else:
                ShowBaseGlobal.direct.widget.setH(ShowBaseGlobal.direct.widget, deltaAngle)
        self.lastCrankAngle = newAngle

    def widgetCheck(self, type):
        if False:
            return 10
        assert self.constraint is not None
        axis = self.constraint[:1]
        mWidget2Cam = ShowBaseGlobal.direct.widget.getMat(ShowBaseGlobal.direct.camera)
        pos = VBase3(0)
        decomposeMatrix(mWidget2Cam, VBase3(0), VBase3(0), pos, CSDefault)
        widgetDir = Vec3(pos)
        widgetDir.normalize()
        if axis == 'x':
            widgetAxis = Vec3(mWidget2Cam.xformVec(DG.X_AXIS))
        elif axis == 'y':
            widgetAxis = Vec3(mWidget2Cam.xformVec(DG.Y_AXIS))
        elif axis == 'z':
            widgetAxis = Vec3(mWidget2Cam.xformVec(DG.Z_AXIS))
        widgetAxis.normalize()
        if type == 'top?':
            return widgetDir.dot(widgetAxis) < 0.0
        elif type == 'edge?':
            return abs(widgetDir.dot(widgetAxis)) < 0.2

    def xlateCamXZ(self, state):
        if False:
            while True:
                i = 10
        "Constrained 2D motion parallel to the camera's image plane\n        This moves the object in the camera's XZ plane"
        self.fHitInit = 1
        self.fScaleInit = 1
        vWidget2Camera = ShowBaseGlobal.direct.widget.getPos(ShowBaseGlobal.direct.camera)
        x = vWidget2Camera[0]
        y = vWidget2Camera[1]
        z = vWidget2Camera[2]
        dr = ShowBaseGlobal.direct.dr
        ShowBaseGlobal.direct.widget.setX(ShowBaseGlobal.direct.camera, x + 0.5 * dr.mouseDeltaX * dr.nearWidth * (y / dr.near))
        ShowBaseGlobal.direct.widget.setZ(ShowBaseGlobal.direct.camera, z + 0.5 * dr.mouseDeltaY * dr.nearHeight * (y / dr.near))

    def xlateCamXY(self, state):
        if False:
            i = 10
            return i + 15
        "Constrained 2D motion perpendicular to camera's image plane\n        This moves the object in the camera's XY plane if shift is held\n        Moves object toward camera if control is held\n        "
        self.fScaleInit = 1
        vWidget2Camera = ShowBaseGlobal.direct.widget.getPos(ShowBaseGlobal.direct.camera)
        if self.fHitInit:
            self.fHitInit = 0
            self.xlateSF = Vec3(vWidget2Camera).length()
            coaCenter = getNearProjectionPoint(ShowBaseGlobal.direct.widget)
            self.deltaNearX = coaCenter[0] - ShowBaseGlobal.direct.dr.nearVec[0]
        if ShowBaseGlobal.direct.fControl:
            moveDir = Vec3(vWidget2Camera)
            if moveDir[1] < 0.0:
                moveDir.assign(moveDir * -1)
            moveDir.normalize()
        else:
            moveDir = Vec3(DG.Y_AXIS)
        dr = ShowBaseGlobal.direct.dr
        moveDir.assign(moveDir * (2.0 * dr.mouseDeltaY * self.xlateSF))
        vWidget2Camera += moveDir
        vWidget2Camera.setX((dr.nearVec[0] + self.deltaNearX) * (vWidget2Camera[1] / dr.near))
        ShowBaseGlobal.direct.widget.setPos(ShowBaseGlobal.direct.camera, vWidget2Camera)

    def rotate2D(self, state):
        if False:
            print('Hello World!')
        ' Virtual trackball rotation of widget '
        self.fHitInit = 1
        self.fScaleInit = 1
        tumbleRate = 360
        if state.constrainedDir == 'y' and abs(ShowBaseGlobal.direct.dr.mouseX) > 0.9:
            deltaX = 0
            deltaY = ShowBaseGlobal.direct.dr.mouseDeltaY
        elif state.constrainedDir == 'x' and abs(ShowBaseGlobal.direct.dr.mouseY) > 0.9:
            deltaX = ShowBaseGlobal.direct.dr.mouseDeltaX
            deltaY = 0
        else:
            deltaX = ShowBaseGlobal.direct.dr.mouseDeltaX
            deltaY = ShowBaseGlobal.direct.dr.mouseDeltaY
        relHpr(ShowBaseGlobal.direct.widget, ShowBaseGlobal.direct.camera, deltaX * tumbleRate, -deltaY * tumbleRate, 0)

    def rotateAboutViewVector(self, state):
        if False:
            i = 10
            return i + 15
        self.fHitInit = 1
        self.fScaleInit = 1
        angle = getCrankAngle(state.coaCenter)
        deltaAngle = angle - state.lastAngle
        state.lastAngle = angle
        relHpr(ShowBaseGlobal.direct.widget, ShowBaseGlobal.direct.camera, 0, 0, -deltaAngle)

    def scale1D(self, state):
        if False:
            for i in range(10):
                print('nop')
        assert self.constraint is not None
        direct = ShowBaseGlobal.direct
        if hasattr(direct, 'manipulationControl') and direct.manipulationControl.fMultiView:
            self.hitPtScale.assign(self.objectHandles.getAxisIntersectPt(self.constraint[:1]))
            self.hitPtScale = self.objectHandles.getMat().xformVec(self.hitPtScale)
            if self.fScaleInit1:
                self.fScaleInit1 = 0
                self.prevHitScale.assign(self.hitPtScale)
                self.origScale = direct.widget.getScale()
            else:
                widgetPos = direct.widget.getPos()
                d0 = self.prevHitScale.length()
                if d0 == 0:
                    d0 = 0.001
                d1 = self.hitPtScale.length()
                if d1 == 0:
                    d1 = 0.001
                currScale = self.origScale
                if self.constraint[:1] == 'x':
                    currScale = Vec3(currScale.getX() * d1 / d0, currScale.getY(), currScale.getZ())
                elif self.constraint[:1] == 'y':
                    currScale = Vec3(currScale.getX(), currScale.getY() * d1 / d0, currScale.getZ())
                elif self.constraint[:1] == 'z':
                    currScale = Vec3(currScale.getX(), currScale.getY(), currScale.getZ() * d1 / d0)
                direct.widget.setScale(currScale)
            return
        if self.fScaleInit:
            self.fScaleInit = 0
            self.initScaleMag = Vec3(self.objectHandles.getAxisIntersectPt(self.constraint[:1])).length()
            self.initScale = direct.widget.getScale()
        self.fHitInit = 1
        direct.widget.setScale(1, 1, 1)
        if self.constraint[:1] == 'x':
            currScale = Vec3(self.initScale.getX() * self.objectHandles.getAxisIntersectPt('x').length() / self.initScaleMag, self.initScale.getY(), self.initScale.getZ())
        elif self.constraint[:1] == 'y':
            currScale = Vec3(self.initScale.getX(), self.initScale.getY() * self.objectHandles.getAxisIntersectPt('y').length() / self.initScaleMag, self.initScale.getZ())
        elif self.constraint[:1] == 'z':
            currScale = Vec3(self.initScale.getX(), self.initScale.getY(), self.initScale.getZ() * self.objectHandles.getAxisIntersectPt('z').length() / self.initScaleMag)
        direct.widget.setScale(currScale)

    def scale3D(self, state):
        if False:
            print('Hello World!')
        direct = ShowBaseGlobal.direct
        if hasattr(direct, 'manipulationControl') and direct.manipulationControl.fMultiView:
            if self.useSeparateScaleHandles:
                assert self.constraint is not None
                self.hitPtScale.assign(self.objectHandles.getAxisIntersectPt(self.constraint[:1]))
                self.hitPtScale = self.objectHandles.getMat().xformVec(self.hitPtScale)
                if self.fScaleInit1:
                    self.fScaleInit1 = 0
                    self.prevHitScale.assign(self.hitPtScale)
                    self.origScale = direct.widget.getScale()
                else:
                    widgetPos = direct.widget.getPos()
                    d0 = self.prevHitScale.length()
                    if d0 == 0:
                        d0 = 0.001
                    d1 = self.hitPtScale.length()
                    if d1 == 0:
                        d1 = 0.001
                    currScale = self.origScale
                    currScale = Vec3(currScale.getX() * d1 / d0, currScale.getY() * d1 / d0, currScale.getZ() * d1 / d0)
                    direct.widget.setScale(currScale)
                return
            else:
                self.hitPtScale.assign(self.objectHandles.getMouseIntersectPt())
                if self.fScaleInit1:
                    self.fScaleInit1 = 0
                    self.prevHitScale.assign(self.hitPtScale)
                    self.origScale = direct.widget.getScale()
                else:
                    widgetPos = direct.widget.getPos()
                    d0 = (self.prevHitScale - widgetPos).length()
                    if d0 == 0:
                        d0 = 0.001
                    d1 = (self.hitPtScale - widgetPos).length()
                    if d1 == 0:
                        d1 = 0.001
                    currScale = self.origScale
                    currScale = currScale * d1 / d0
                    direct.widget.setScale(currScale)
                return
        if self.fScaleInit:
            self.fScaleInit = 0
            self.manipRef.setPos(direct.widget, 0, 0, 0)
            self.manipRef.setHpr(direct.camera, 0, 0, 0)
            self.initScaleMag = Vec3(self.objectHandles.getWidgetIntersectPt(self.manipRef, 'y')).length()
            self.initScale = direct.widget.getScale()
        self.fHitInit = 1
        currScale = self.initScale * (self.objectHandles.getWidgetIntersectPt(self.manipRef, 'y').length() / self.initScaleMag)
        direct.widget.setScale(currScale)

    def plantSelectedNodePath(self):
        if False:
            print('Hello World!')
        ' Move selected object to intersection point of cursor on scene '
        entry = ShowBaseGlobal.direct.iRay.pickGeom(skipFlags=DG.SKIP_HIDDEN | DG.SKIP_BACKFACE | DG.SKIP_CAMERA)
        if entry is not None and ShowBaseGlobal.direct.selected.last is not None:
            ShowBaseGlobal.direct.pushUndo(ShowBaseGlobal.direct.selected)
            ShowBaseGlobal.direct.selected.getWrtAll()
            ShowBaseGlobal.direct.widget.setPos(ShowBaseGlobal.direct.camera, entry.getSurfacePoint(entry.getFromNodePath()))
            ShowBaseGlobal.direct.selected.moveWrtWidgetAll()
            messenger.send('DIRECT_manipulateObjectCleanup', [ShowBaseGlobal.direct.selected.getSelectedAsList()])

class ObjectHandles(NodePath, DirectObject):

    def __init__(self, name='objectHandles'):
        if False:
            i = 10
            return i + 15
        NodePath.__init__(self)
        self.assign(ShowBaseGlobal.loader.loadModel('models/misc/objectHandles'))
        self.setName(name)
        self.scalingNode = NodePath(self)
        self.scalingNode.setName('ohScalingNode')
        self.ohScalingFactor = 1.0
        self.directScalingFactor = 1.0
        self.hitPt = Vec3(0)
        self.xHandles = self.find('**/X')
        self.xPostGroup = self.xHandles.find('**/x-post-group')
        self.xPostCollision = self.xHandles.find('**/x-post')
        self.xRingGroup = self.xHandles.find('**/x-ring-group')
        self.xRingCollision = self.xHandles.find('**/x-ring')
        self.xDiscGroup = self.xHandles.find('**/x-disc-group')
        self.xDisc = self.xHandles.find('**/x-disc-visible')
        self.xDiscCollision = self.xHandles.find('**/x-disc')
        self.xScaleGroup = deepcopy(self.xPostGroup)
        self.xScaleGroup.setName('x-scale-group')
        self.xScaleCollision = self.xScaleGroup.find('**/x-post')
        self.xScaleCollision.setName('x-scale')
        self.yHandles = self.find('**/Y')
        self.yPostGroup = self.yHandles.find('**/y-post-group')
        self.yPostCollision = self.yHandles.find('**/y-post')
        self.yRingGroup = self.yHandles.find('**/y-ring-group')
        self.yRingCollision = self.yHandles.find('**/y-ring')
        self.yDiscGroup = self.yHandles.find('**/y-disc-group')
        self.yDisc = self.yHandles.find('**/y-disc-visible')
        self.yDiscCollision = self.yHandles.find('**/y-disc')
        self.yScaleGroup = deepcopy(self.yPostGroup)
        self.yScaleGroup.setName('y-scale-group')
        self.yScaleCollision = self.yScaleGroup.find('**/y-post')
        self.yScaleCollision.setName('y-scale')
        self.zHandles = self.find('**/Z')
        self.zPostGroup = self.zHandles.find('**/z-post-group')
        self.zPostCollision = self.zHandles.find('**/z-post')
        self.zRingGroup = self.zHandles.find('**/z-ring-group')
        self.zRingCollision = self.zHandles.find('**/z-ring')
        self.zDiscGroup = self.zHandles.find('**/z-disc-group')
        self.zDisc = self.zHandles.find('**/z-disc-visible')
        self.zDiscCollision = self.zHandles.find('**/z-disc')
        self.zScaleGroup = deepcopy(self.zPostGroup)
        self.zScaleGroup.setName('z-scale-group')
        self.zScaleCollision = self.zScaleGroup.find('**/z-post')
        self.zScaleCollision.setName('z-scale')
        self.xPostCollision.hide()
        self.xRingCollision.hide()
        self.xScaleCollision.hide()
        self.xDisc.setColor(1, 0, 0, 0.2)
        self.yPostCollision.hide()
        self.yRingCollision.hide()
        self.yScaleCollision.hide()
        self.yDisc.setColor(0, 1, 0, 0.2)
        self.zPostCollision.hide()
        self.zRingCollision.hide()
        self.zScaleCollision.hide()
        self.zDisc.setColor(0, 0, 1, 0.2)
        self.createObjectHandleLines()
        self.createGuideLines()
        self.hideGuides()
        self.xPostCollision.setTag('WidgetName', name)
        self.yPostCollision.setTag('WidgetName', name)
        self.zPostCollision.setTag('WidgetName', name)
        self.xRingCollision.setTag('WidgetName', name)
        self.yRingCollision.setTag('WidgetName', name)
        self.zRingCollision.setTag('WidgetName', name)
        self.xDiscCollision.setTag('WidgetName', name)
        self.yDiscCollision.setTag('WidgetName', name)
        self.zDiscCollision.setTag('WidgetName', name)
        self.xScaleCollision.setTag('WidgetName', name)
        self.yScaleCollision.setTag('WidgetName', name)
        self.zScaleCollision.setTag('WidgetName', name)
        self.xDisc.find('**/+GeomNode').setName('x-disc-geom')
        self.yDisc.find('**/+GeomNode').setName('y-disc-geom')
        self.zDisc.find('**/+GeomNode').setName('z-disc-geom')
        self.disableHandles('scale')
        self.fActive = 1
        self.toggleWidget()
        useDirectRenderStyle(self)

    def coaModeColor(self):
        if False:
            i = 10
            return i + 15
        self.setColor(0.5, 0.5, 0.5, 0.5, 1)

    def disabledModeColor(self):
        if False:
            for i in range(10):
                print('nop')
        self.setColor(0.1, 0.1, 0.1, 0.1, 1)

    def manipModeColor(self):
        if False:
            return 10
        self.clearColor()

    def toggleWidget(self):
        if False:
            return 10
        if self.fActive:
            if hasattr(ShowBaseGlobal.direct, 'manipulationControl') and ShowBaseGlobal.direct.manipulationControl.fMultiView:
                for widget in ShowBaseGlobal.direct.manipulationControl.widgetList:
                    widget.deactivate()
            else:
                self.deactivate()
        elif hasattr(ShowBaseGlobal.direct, 'manipulationControl') and ShowBaseGlobal.direct.manipulationControl.fMultiView:
            for widget in ShowBaseGlobal.direct.manipulationControl.widgetList:
                widget.activate()
                widget.showWidgetIfActive()
        else:
            self.activate()

    def activate(self):
        if False:
            print('Hello World!')
        self.scalingNode.reparentTo(self)
        self.fActive = 1

    def deactivate(self):
        if False:
            for i in range(10):
                print('nop')
        self.scalingNode.reparentTo(hidden)
        self.fActive = 0

    def showWidgetIfActive(self):
        if False:
            return 10
        if self.fActive:
            self.reparentTo(ShowBaseGlobal.direct.group)

    def showWidget(self):
        if False:
            print('Hello World!')
        self.reparentTo(ShowBaseGlobal.direct.group)

    def hideWidget(self):
        if False:
            for i in range(10):
                print('nop')
        self.reparentTo(hidden)

    def enableHandles(self, handles):
        if False:
            return 10
        if isinstance(handles, list):
            for handle in handles:
                self.enableHandle(handle)
        elif handles == 'x':
            self.enableHandles(['x-post', 'x-ring', 'x-disc', 'x-scale'])
        elif handles == 'y':
            self.enableHandles(['y-post', 'y-ring', 'y-disc', 'y-scale'])
        elif handles == 'z':
            self.enableHandles(['z-post', 'z-ring', 'z-disc', 'z-scale'])
        elif handles == 'post':
            self.enableHandles(['x-post', 'y-post', 'z-post'])
        elif handles == 'ring':
            self.enableHandles(['x-ring', 'y-ring', 'z-ring'])
        elif handles == 'disc':
            self.enableHandles(['x-disc', 'y-disc', 'z-disc'])
        elif handles == 'scale':
            self.enableHandles(['x-scale', 'y-scale', 'z-scale'])
        elif handles == 'all':
            self.enableHandles(['x-post', 'x-ring', 'x-disc', 'x-scale', 'y-post', 'y-ring', 'y-disc', 'y-scale', 'z-post', 'z-ring', 'z-disc', 'z-scale'])

    def enableHandle(self, handle):
        if False:
            return 10
        if handle == 'x-post':
            self.xPostGroup.reparentTo(self.xHandles)
        elif handle == 'x-ring':
            self.xRingGroup.reparentTo(self.xHandles)
        elif handle == 'x-disc':
            self.xDiscGroup.reparentTo(self.xHandles)
        elif handle == 'x-scale' and ShowBaseGlobal.direct.manipulationControl.useSeparateScaleHandles:
            self.xScaleGroup.reparentTo(self.xHandles)
        elif handle == 'y-post':
            self.yPostGroup.reparentTo(self.yHandles)
        elif handle == 'y-ring':
            self.yRingGroup.reparentTo(self.yHandles)
        elif handle == 'y-disc':
            self.yDiscGroup.reparentTo(self.yHandles)
        elif handle == 'y-scale' and ShowBaseGlobal.direct.manipulationControl.useSeparateScaleHandles:
            self.yScaleGroup.reparentTo(self.yHandles)
        elif handle == 'z-post':
            self.zPostGroup.reparentTo(self.zHandles)
        elif handle == 'z-ring':
            self.zRingGroup.reparentTo(self.zHandles)
        elif handle == 'z-disc':
            self.zDiscGroup.reparentTo(self.zHandles)
        elif handle == 'z-scale' and ShowBaseGlobal.direct.manipulationControl.useSeparateScaleHandles:
            self.zScaleGroup.reparentTo(self.zHandles)

    def disableHandles(self, handles):
        if False:
            while True:
                i = 10
        if isinstance(handles, list):
            for handle in handles:
                self.disableHandle(handle)
        elif handles == 'x':
            self.disableHandles(['x-post', 'x-ring', 'x-disc', 'x-scale'])
        elif handles == 'y':
            self.disableHandles(['y-post', 'y-ring', 'y-disc', 'y-scale'])
        elif handles == 'z':
            self.disableHandles(['z-post', 'z-ring', 'z-disc', 'z-scale'])
        elif handles == 'post':
            self.disableHandles(['x-post', 'y-post', 'z-post'])
        elif handles == 'ring':
            self.disableHandles(['x-ring', 'y-ring', 'z-ring'])
        elif handles == 'disc':
            self.disableHandles(['x-disc', 'y-disc', 'z-disc'])
        elif handles == 'scale':
            self.disableHandles(['x-scale', 'y-scale', 'z-scale'])
        elif handles == 'all':
            self.disableHandles(['x-post', 'x-ring', 'x-disc', 'x-scale', 'y-post', 'y-ring', 'y-disc', 'y-scale', 'z-post', 'z-ring', 'z-disc', 'z-scale'])

    def disableHandle(self, handle):
        if False:
            for i in range(10):
                print('nop')
        if handle == 'x-post':
            self.xPostGroup.reparentTo(hidden)
        elif handle == 'x-ring':
            self.xRingGroup.reparentTo(hidden)
        elif handle == 'x-disc':
            self.xDiscGroup.reparentTo(hidden)
        elif handle == 'x-scale':
            self.xScaleGroup.reparentTo(hidden)
        if handle == 'y-post':
            self.yPostGroup.reparentTo(hidden)
        elif handle == 'y-ring':
            self.yRingGroup.reparentTo(hidden)
        elif handle == 'y-disc':
            self.yDiscGroup.reparentTo(hidden)
        elif handle == 'y-scale':
            self.yScaleGroup.reparentTo(hidden)
        if handle == 'z-post':
            self.zPostGroup.reparentTo(hidden)
        elif handle == 'z-ring':
            self.zRingGroup.reparentTo(hidden)
        elif handle == 'z-disc':
            self.zDiscGroup.reparentTo(hidden)
        elif handle == 'z-scale':
            self.zScaleGroup.reparentTo(hidden)

    def showAllHandles(self):
        if False:
            i = 10
            return i + 15
        self.xPost.show()
        self.xRing.show()
        self.xDisc.show()
        self.xScale.show()
        self.yPost.show()
        self.yRing.show()
        self.yDisc.show()
        self.yScale.show()
        self.zPost.show()
        self.zRing.show()
        self.zDisc.show()
        self.zScale.show()

    def hideAllHandles(self):
        if False:
            return 10
        self.xPost.hide()
        self.xRing.hide()
        self.xDisc.hide()
        self.xScale.hide()
        self.yPost.hide()
        self.yRing.hide()
        self.yDisc.hide()
        self.yScale.hide()
        self.zPost.hide()
        self.zRing.hide()
        self.zDisc.hide()
        self.zScale.hide()

    def showHandle(self, handle):
        if False:
            print('Hello World!')
        if handle == 'x-post':
            self.xPost.show()
        elif handle == 'x-ring':
            self.xRing.show()
        elif handle == 'x-disc':
            self.xDisc.show()
        elif handle == 'x-scale':
            self.xScale.show()
        elif handle == 'y-post':
            self.yPost.show()
        elif handle == 'y-ring':
            self.yRing.show()
        elif handle == 'y-disc':
            self.yDisc.show()
        elif handle == 'y-scale':
            self.yScale.show()
        elif handle == 'z-post':
            self.zPost.show()
        elif handle == 'z-ring':
            self.zRing.show()
        elif handle == 'z-disc':
            self.zDisc.show()
        elif handle == 'z-scale':
            self.zScale.show()

    def showGuides(self):
        if False:
            for i in range(10):
                print('nop')
        self.guideLines.show()

    def hideGuides(self):
        if False:
            i = 10
            return i + 15
        self.guideLines.hide()

    def setDirectScalingFactor(self, factor):
        if False:
            while True:
                i = 10
        self.directScalingFactor = factor
        self.setScalingFactor(1)

    def setScalingFactor(self, scaleFactor):
        if False:
            i = 10
            return i + 15
        self.ohScalingFactor = scaleFactor
        self.scalingNode.setScale(self.ohScalingFactor * self.directScalingFactor)

    def getScalingFactor(self):
        if False:
            print('Hello World!')
        return self.scalingNode.getScale()

    def transferObjectHandlesScale(self):
        if False:
            while True:
                i = 10
        ohs = self.getScale()
        sns = self.scalingNode.getScale()
        self.scalingNode.setScale(ohs[0] * sns[0], ohs[1] * sns[1], ohs[2] * sns[2])
        self.setScale(1)

    def multiplyScalingFactorBy(self, factor):
        if False:
            while True:
                i = 10
        self.ohScalingFactor = self.ohScalingFactor * factor
        sf = self.ohScalingFactor * self.directScalingFactor
        ival = self.scalingNode.scaleInterval(0.5, (sf, sf, sf), blendType='easeInOut', name='resizeObjectHandles')
        ival.start()

    def growToFit(self):
        if False:
            return 10
        pos = ShowBaseGlobal.direct.widget.getPos(ShowBaseGlobal.direct.camera)
        minDim = min(ShowBaseGlobal.direct.dr.nearWidth, ShowBaseGlobal.direct.dr.nearHeight)
        sf = 0.15 * minDim * (pos[1] / ShowBaseGlobal.direct.dr.near)
        self.ohScalingFactor = sf
        sf = sf * self.directScalingFactor
        ival = self.scalingNode.scaleInterval(0.5, (sf, sf, sf), blendType='easeInOut', name='resizeObjectHandles')
        ival.start()

    def createObjectHandleLines(self):
        if False:
            while True:
                i = 10
        self.xPost = self.xPostGroup.attachNewNode('x-post-visible')
        lines = LineNodePath(self.xPost)
        lines.setColor(VBase4(1, 0, 0, 1))
        lines.setThickness(5)
        lines.moveTo(1.5, 0, 0)
        lines.drawTo(-1.5, 0, 0)
        arrowInfo0 = 1.3
        arrowInfo1 = 0.1
        lines.moveTo(1.5, 0, 0)
        lines.drawTo(arrowInfo0, arrowInfo1, arrowInfo1)
        lines.moveTo(1.5, 0, 0)
        lines.drawTo(arrowInfo0, arrowInfo1, -1 * arrowInfo1)
        lines.moveTo(1.5, 0, 0)
        lines.drawTo(arrowInfo0, -1 * arrowInfo1, arrowInfo1)
        lines.moveTo(1.5, 0, 0)
        lines.drawTo(arrowInfo0, -1 * arrowInfo1, -1 * arrowInfo1)
        lines.create()
        lines.setName('x-post-line')
        self.xScale = self.xScaleGroup.attachNewNode('x-scale-visible')
        lines = LineNodePath(self.xScale)
        lines.setColor(VBase4(1, 0, 0, 1))
        lines.setThickness(5)
        lines.moveTo(1.3, 0, 0)
        lines.drawTo(-1.5, 0, 0)
        drawBox(lines, (1.3, 0, 0), 0.2)
        lines.create()
        lines.setName('x-scale-line')
        self.xRing = self.xRingGroup.attachNewNode('x-ring-visible')
        lines = LineNodePath(self.xRing)
        lines.setColor(VBase4(1, 0, 0, 1))
        lines.setThickness(3)
        lines.moveTo(0, 1, 0)
        for ang in range(15, 370, 15):
            lines.drawTo(0, math.cos(deg2Rad(ang)), math.sin(deg2Rad(ang)))
        lines.create()
        lines.setName('x-ring-line')
        self.yPost = self.yPostGroup.attachNewNode('y-post-visible')
        lines = LineNodePath(self.yPost)
        lines.setColor(VBase4(0, 1, 0, 1))
        lines.setThickness(5)
        lines.moveTo(0, 1.5, 0)
        lines.drawTo(0, -1.5, 0)
        lines.moveTo(0, 1.5, 0)
        lines.drawTo(arrowInfo1, arrowInfo0, arrowInfo1)
        lines.moveTo(0, 1.5, 0)
        lines.drawTo(arrowInfo1, arrowInfo0, -1 * arrowInfo1)
        lines.moveTo(0, 1.5, 0)
        lines.drawTo(-1 * arrowInfo1, arrowInfo0, arrowInfo1)
        lines.moveTo(0, 1.5, 0)
        lines.drawTo(-1 * arrowInfo1, arrowInfo0, -1 * arrowInfo1)
        lines.create()
        lines.setName('y-post-line')
        self.yScale = self.yScaleGroup.attachNewNode('y-scale-visible')
        lines = LineNodePath(self.yScale)
        lines.setColor(VBase4(0, 1, 0, 1))
        lines.setThickness(5)
        lines.moveTo(0, 1.3, 0)
        lines.drawTo(0, -1.5, 0)
        drawBox(lines, (0, 1.4, 0), 0.2)
        lines.create()
        lines.setName('y-scale-line')
        self.yRing = self.yRingGroup.attachNewNode('y-ring-visible')
        lines = LineNodePath(self.yRing)
        lines.setColor(VBase4(0, 1, 0, 1))
        lines.setThickness(3)
        lines.moveTo(1, 0, 0)
        for ang in range(15, 370, 15):
            lines.drawTo(math.cos(deg2Rad(ang)), 0, math.sin(deg2Rad(ang)))
        lines.create()
        lines.setName('y-ring-line')
        self.zPost = self.zPostGroup.attachNewNode('z-post-visible')
        lines = LineNodePath(self.zPost)
        lines.setColor(VBase4(0, 0, 1, 1))
        lines.setThickness(5)
        lines.moveTo(0, 0, 1.5)
        lines.drawTo(0, 0, -1.5)
        lines.moveTo(0, 0, 1.5)
        lines.drawTo(arrowInfo1, arrowInfo1, arrowInfo0)
        lines.moveTo(0, 0, 1.5)
        lines.drawTo(arrowInfo1, -1 * arrowInfo1, arrowInfo0)
        lines.moveTo(0, 0, 1.5)
        lines.drawTo(-1 * arrowInfo1, arrowInfo1, arrowInfo0)
        lines.moveTo(0, 0, 1.5)
        lines.drawTo(-1 * arrowInfo1, -1 * arrowInfo1, arrowInfo0)
        lines.create()
        lines.setName('z-post-line')
        self.zScale = self.zScaleGroup.attachNewNode('z-scale-visible')
        lines = LineNodePath(self.zScale)
        lines.setColor(VBase4(0, 0, 1, 1))
        lines.setThickness(5)
        lines.moveTo(0, 0, 1.3)
        lines.drawTo(0, 0, -1.5)
        drawBox(lines, (0, 0, 1.4), 0.2)
        lines.create()
        lines.setName('y-scale-line')
        self.zRing = self.zRingGroup.attachNewNode('z-ring-visible')
        lines = LineNodePath(self.zRing)
        lines.setColor(VBase4(0, 0, 1, 1))
        lines.setThickness(3)
        lines.moveTo(1, 0, 0)
        for ang in range(15, 370, 15):
            lines.drawTo(math.cos(deg2Rad(ang)), math.sin(deg2Rad(ang)), 0)
        lines.create()
        lines.setName('z-ring-line')

    def createGuideLines(self):
        if False:
            print('Hello World!')
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
            while True:
                i = 10
        if hasattr(ShowBaseGlobal.direct, 'manipulationControl') and ShowBaseGlobal.direct.manipulationControl.fMultiView and (ShowBaseGlobal.direct.camera.getName() != 'persp'):
            iRay = SelectionRay(ShowBaseGlobal.direct.camera)
            iRay.collider.setFromLens(ShowBaseGlobal.direct.camNode, ShowBaseGlobal.direct.dr.mouseX, ShowBaseGlobal.direct.dr.mouseY)
            iRay.collideWithBitMask(BitMask32.bit(21))
            iRay.ct.traverse(ShowBaseGlobal.direct.grid)
            if iRay.getNumEntries() == 0:
                del iRay
                return self.hitPt
            entry = iRay.getEntry(0)
            self.hitPt = entry.getSurfacePoint(self)
            del iRay
            if axis == 'x':
                self.hitPt.setY(0)
                self.hitPt.setZ(0)
            elif axis == 'y':
                self.hitPt.setX(0)
                self.hitPt.setZ(0)
            elif axis == 'z':
                self.hitPt.setX(0)
                self.hitPt.setY(0)
            return self.hitPt
        mCam2Widget = ShowBaseGlobal.direct.camera.getMat(ShowBaseGlobal.direct.widget)
        lineDir = Vec3(mCam2Widget.xformVec(ShowBaseGlobal.direct.dr.nearVec))
        lineDir.normalize()
        lineOrigin = VBase3(0)
        decomposeMatrix(mCam2Widget, VBase3(0), VBase3(0), lineOrigin, CSDefault)
        if axis == 'x':
            if abs(lineDir.dot(DG.Y_AXIS)) > abs(lineDir.dot(DG.Z_AXIS)):
                self.hitPt.assign(planeIntersect(lineOrigin, lineDir, DG.ORIGIN, DG.Y_AXIS))
            else:
                self.hitPt.assign(planeIntersect(lineOrigin, lineDir, DG.ORIGIN, DG.Z_AXIS))
            self.hitPt.setY(0)
            self.hitPt.setZ(0)
        elif axis == 'y':
            if abs(lineDir.dot(DG.X_AXIS)) > abs(lineDir.dot(DG.Z_AXIS)):
                self.hitPt.assign(planeIntersect(lineOrigin, lineDir, DG.ORIGIN, DG.X_AXIS))
            else:
                self.hitPt.assign(planeIntersect(lineOrigin, lineDir, DG.ORIGIN, DG.Z_AXIS))
            self.hitPt.setX(0)
            self.hitPt.setZ(0)
        elif axis == 'z':
            if abs(lineDir.dot(DG.X_AXIS)) > abs(lineDir.dot(DG.Y_AXIS)):
                self.hitPt.assign(planeIntersect(lineOrigin, lineDir, DG.ORIGIN, DG.X_AXIS))
            else:
                self.hitPt.assign(planeIntersect(lineOrigin, lineDir, DG.ORIGIN, DG.Y_AXIS))
            self.hitPt.setX(0)
            self.hitPt.setY(0)
        return self.hitPt

    def getMouseIntersectPt(self):
        if False:
            return 10
        iRay = SelectionRay(ShowBaseGlobal.direct.camera)
        iRay.collider.setFromLens(ShowBaseGlobal.direct.camNode, ShowBaseGlobal.direct.dr.mouseX, ShowBaseGlobal.direct.dr.mouseY)
        iRay.collideWithBitMask(BitMask32.bit(21))
        iRay.ct.traverse(ShowBaseGlobal.direct.grid)
        if iRay.getNumEntries() == 0:
            del iRay
            return Point3(0)
        entry = iRay.getEntry(0)
        hitPt = entry.getSurfacePoint(entry.getFromNodePath())
        np = NodePath('temp')
        np.setPos(ShowBaseGlobal.direct.camera, hitPt)
        resultPt = Point3(0)
        resultPt.assign(np.getPos())
        np.removeNode()
        del iRay
        return resultPt

    def getWidgetIntersectPt(self, nodePath, plane):
        if False:
            print('Hello World!')
        if hasattr(ShowBaseGlobal.direct, 'manipulationControl') and ShowBaseGlobal.direct.manipulationControl.fMultiView and (ShowBaseGlobal.direct.camera.getName() != 'persp'):
            self.hitPt.assign(self.getMouseIntersectPt())
            return self.hitPt
        mCam2NodePath = ShowBaseGlobal.direct.camera.getMat(nodePath)
        lineOrigin = VBase3(0)
        decomposeMatrix(mCam2NodePath, VBase3(0), VBase3(0), lineOrigin, CSDefault)
        lineDir = Vec3(mCam2NodePath.xformVec(ShowBaseGlobal.direct.dr.nearVec))
        lineDir.normalize()
        if plane == 'x':
            self.hitPt.assign(planeIntersect(lineOrigin, lineDir, DG.ORIGIN, DG.X_AXIS))
        elif plane == 'y':
            self.hitPt.assign(planeIntersect(lineOrigin, lineDir, DG.ORIGIN, DG.Y_AXIS))
        elif plane == 'z':
            self.hitPt.assign(planeIntersect(lineOrigin, lineDir, DG.ORIGIN, DG.Z_AXIS))
        return self.hitPt

def drawBox(lines, center, sideLength):
    if False:
        i = 10
        return i + 15
    l = sideLength * 0.5
    lines.moveTo(center[0] + l, center[1] + l, center[2] + l)
    lines.drawTo(center[0] + l, center[1] + l, center[2] - l)
    lines.drawTo(center[0] + l, center[1] - l, center[2] - l)
    lines.drawTo(center[0] + l, center[1] - l, center[2] + l)
    lines.drawTo(center[0] + l, center[1] + l, center[2] + l)
    lines.moveTo(center[0] - l, center[1] + l, center[2] + l)
    lines.drawTo(center[0] - l, center[1] + l, center[2] - l)
    lines.drawTo(center[0] - l, center[1] - l, center[2] - l)
    lines.drawTo(center[0] - l, center[1] - l, center[2] + l)
    lines.drawTo(center[0] - l, center[1] + l, center[2] + l)
    lines.moveTo(center[0] + l, center[1] + l, center[2] + l)
    lines.drawTo(center[0] + l, center[1] + l, center[2] - l)
    lines.drawTo(center[0] - l, center[1] + l, center[2] - l)
    lines.drawTo(center[0] - l, center[1] + l, center[2] + l)
    lines.drawTo(center[0] + l, center[1] + l, center[2] + l)
    lines.moveTo(center[0] + l, center[1] - l, center[2] + l)
    lines.drawTo(center[0] + l, center[1] - l, center[2] - l)
    lines.drawTo(center[0] - l, center[1] - l, center[2] - l)
    lines.drawTo(center[0] - l, center[1] - l, center[2] + l)
    lines.drawTo(center[0] + l, center[1] - l, center[2] + l)
    lines.moveTo(center[0] + l, center[1] + l, center[2] + l)
    lines.drawTo(center[0] - l, center[1] + l, center[2] + l)
    lines.drawTo(center[0] - l, center[1] - l, center[2] + l)
    lines.drawTo(center[0] + l, center[1] - l, center[2] + l)
    lines.drawTo(center[0] + l, center[1] + l, center[2] + l)
    lines.moveTo(center[0] + l, center[1] + l, center[2] - l)
    lines.drawTo(center[0] - l, center[1] + l, center[2] - l)
    lines.drawTo(center[0] - l, center[1] - l, center[2] - l)
    lines.drawTo(center[0] + l, center[1] - l, center[2] - l)
    lines.drawTo(center[0] + l, center[1] + l, center[2] - l)