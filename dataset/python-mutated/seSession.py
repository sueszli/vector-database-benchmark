from direct.showbase.DirectObject import *
from direct.directtools.DirectGlobals import *
from direct.directtools.DirectUtil import *
from direct.interval.IntervalGlobal import *
from seCameraControl import *
from seManipulation import *
from seSelection import *
from seGrid import *
from seGeometry import *
from direct.tkpanels import Placer
from direct.tkwidgets import Slider
from direct.gui import OnscreenText
from direct.task import Task
from direct.showbase import Loader
import math

class SeSession(DirectObject):

    def __init__(self):
        if False:
            print('Hello World!')
        __builtins__['SEditor'] = self
        self.group = render.attachNewNode('SEditor')
        self.font = TextNode.getDefaultFont()
        self.fEnabled = 0
        self.drList = DisplayRegionList()
        self.iRayList = map(lambda x: x.iRay, self.drList)
        self.dr = self.drList[0]
        self.camera = base.camera
        self.trueCamera = self.camera
        self.iRay = self.dr.iRay
        self.coaMode = COA_ORIGIN
        self.enableAutoCamera = True
        self.cameraControl = DirectCameraControl()
        self.manipulationControl = DirectManipulationControl()
        self.useObjectHandles()
        self.grid = DirectGrid()
        self.grid.disable()
        self.selected = SelectedNodePaths()
        self.ancestry = []
        self.ancestryIndex = 0
        self.activeParent = None
        self.selectedNPReadout = OnscreenText.OnscreenText(pos=(-1.0, -0.9), bg=Vec4(1, 1, 1, 1), scale=0.05, align=TextNode.ALeft, mayChange=1, font=self.font)
        useDirectRenderStyle(self.selectedNPReadout)
        self.selectedNPReadout.reparentTo(hidden)
        self.activeParentReadout = OnscreenText.OnscreenText(pos=(-1.0, -0.975), bg=Vec4(1, 1, 1, 1), scale=0.05, align=TextNode.ALeft, mayChange=1, font=self.font)
        useDirectRenderStyle(self.activeParentReadout)
        self.activeParentReadout.reparentTo(hidden)
        self.directMessageReadout = OnscreenText.OnscreenText(pos=(-1.0, 0.9), bg=Vec4(1, 1, 1, 1), scale=0.05, align=TextNode.ALeft, mayChange=1, font=self.font)
        useDirectRenderStyle(self.directMessageReadout)
        self.directMessageReadout.reparentTo(hidden)
        self.fControl = 0
        self.fAlt = 0
        self.fShift = 0
        self.pos = VBase3()
        self.hpr = VBase3()
        self.scale = VBase3()
        self.hitPt = Point3(0.0)
        self.undoList = []
        self.redoList = []
        self.drList.updateContext()
        for dr in self.drList:
            dr.camUpdate()
        self.modifierEvents = ['control', 'control-up', 'shift', 'shift-up', 'alt', 'alt-up']
        self.keyEvents = ['escape', 'delete', 'page_up', 'page_down', '[', '{', ']', '}', 'shift-a', 'b', 'control-f', 'l', 'shift-l', 'o', 'p', 'r', 'shift-r', 's', 't', 'v', 'w']
        self.mouseEvents = ['mouse1', 'mouse1-up', 'shift-mouse1', 'shift-mouse1-up', 'control-mouse1', 'control-mouse1-up', 'alt-mouse1', 'alt-mouse1-up', 'mouse2', 'mouse2-up', 'shift-mouse2', 'shift-mouse2-up', 'control-mouse2', 'control-mouse2-up', 'alt-mouse2', 'alt-mouse2-up', 'mouse3', 'mouse3-up', 'shift-mouse3', 'shift-mouse3-up', 'control-mouse3', 'control-mouse3-up', 'alt-mouse3', 'alt-mouse3-up']

    def enable(self):
        if False:
            print('Hello World!')
        if self.fEnabled:
            return
        self.disable()
        self.drList.spawnContextTask()
        self.cameraControl.enableMouseFly()
        self.manipulationControl.enableManipulation()
        self.selected.reset()
        self.enableKeyEvents()
        self.enableModifierEvents()
        self.enableMouseEvents()
        self.fEnabled = 1
        if self.enableAutoCamera:
            self.accept('DH_LoadingComplete', self.autoCameraMove)

    def disable(self):
        if False:
            return 10
        self.drList.removeContextTask()
        self.cameraControl.disableMouseFly()
        self.manipulationControl.disableManipulation()
        self.disableKeyEvents()
        self.disableModifierEvents()
        self.disableMouseEvents()
        self.ignore('DH_LoadingComplete')
        taskMgr.remove('flashNodePath')
        taskMgr.remove('hideDirectMessage')
        taskMgr.remove('hideDirectMessageLater')
        self.fEnabled = 0

    def minimumConfiguration(self):
        if False:
            while True:
                i = 10
        self.drList.removeContextTask()
        self.cameraControl.disableMouseFly()
        self.disableKeyEvents()
        self.disableActionEvents()
        self.enableMouseEvents()
        self.enableModifierEvents()

    def oobe(self):
        if False:
            i = 10
            return i + 15
        try:
            self.oobeMode
        except:
            self.oobeMode = 0
            self.oobeCamera = hidden.attachNewNode('oobeCamera')
            self.oobeVis = loader.loadModelOnce('models/misc/camera')
            if self.oobeVis:
                self.oobeVis.node().setFinal(1)
        if self.oobeMode:
            self.cameraControl.camManipRef.iPosHpr(self.trueCamera)
            t = self.oobeCamera.lerpPosHpr(Point3(0), Vec3(0), 2.0, other=self.cameraControl.camManipRef, task='manipulateCamera', blendType='easeInOut')
            t.uponDeath = self.endOOBE
        else:
            self.oobeVis.reparentTo(self.trueCamera)
            self.oobeVis.clearMat()
            cameraParent = self.camera.getParent()
            self.oobeCamera.reparentTo(cameraParent)
            self.oobeCamera.iPosHpr(self.trueCamera)
            base.cam.reparentTo(self.oobeCamera)
            self.cameraControl.camManipRef.setPos(self.trueCamera, Vec3(-2, -20, 5))
            self.cameraControl.camManipRef.lookAt(self.trueCamera)
            t = self.oobeCamera.lerpPosHpr(Point3(0), Vec3(0), 2.0, other=self.cameraControl.camManipRef, task='manipulateCamera', blendType='easeInOut')
            t.uponDeath = self.beginOOBE

    def beginOOBE(self, state):
        if False:
            return 10
        self.oobeCamera.iPosHpr(self.cameraControl.camManipRef)
        self.camera = self.oobeCamera
        self.oobeMode = 1

    def endOOBE(self, state):
        if False:
            i = 10
            return i + 15
        self.oobeCamera.iPosHpr(self.trueCamera)
        base.cam.reparentTo(self.trueCamera)
        self.camera = self.trueCamera
        self.oobeVis.reparentTo(hidden)
        self.oobeCamera.reparentTo(hidden)
        self.oobeMode = 0

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        self.disable()

    def reset(self):
        if False:
            print('Hello World!')
        self.enable()

    def enableModifierEvents(self):
        if False:
            i = 10
            return i + 15
        for event in self.modifierEvents:
            self.accept(event, self.inputHandler, [event])

    def enableKeyEvents(self):
        if False:
            while True:
                i = 10
        for event in self.keyEvents:
            self.accept(event, self.inputHandler, [event])

    def enableMouseEvents(self):
        if False:
            while True:
                i = 10
        for event in self.mouseEvents:
            self.accept(event, self.inputHandler, [event])

    def disableModifierEvents(self):
        if False:
            return 10
        for event in self.modifierEvents:
            self.ignore(event)

    def disableKeyEvents(self):
        if False:
            print('Hello World!')
        for event in self.keyEvents:
            self.ignore(event)

    def disableMouseEvents(self):
        if False:
            print('Hello World!')
        for event in self.mouseEvents:
            self.ignore(event)

    def inputHandler(self, input):
        if False:
            print('Hello World!')
        if input == 'mouse1-up':
            messenger.send('DIRECT-mouse1Up')
            if SEditor.widget.fActive:
                messenger.send('shift-f')
        elif input.find('mouse1') != -1:
            modifiers = self.getModifiers(input, 'mouse1')
            messenger.send('DIRECT-mouse1', sentArgs=[modifiers])
        elif input == 'mouse2-up':
            messenger.send('DIRECT-mouse2Up')
            if SEditor.widget.fActive:
                messenger.send('shift-f')
        elif input.find('mouse2') != -1:
            modifiers = self.getModifiers(input, 'mouse2')
            messenger.send('DIRECT-mouse2', sentArgs=[modifiers])
        elif input == 'mouse3-up':
            messenger.send('DIRECT-mouse3Up')
            if SEditor.widget.fActive:
                messenger.send('shift-f')
        elif input.find('mouse3') != -1:
            modifiers = self.getModifiers(input, 'mouse3')
            messenger.send('DIRECT-mouse3', sentArgs=[modifiers])
        elif input == 'shift':
            self.fShift = 1
        elif input == 'shift-up':
            self.fShift = 0
        elif input == 'control':
            self.fControl = 1
        elif input == 'control-up':
            self.fControl = 0
        elif input == 'alt':
            self.fAlt = 1
        elif input == 'alt-up':
            self.fAlt = 0
        elif input == 'page_up':
            self.upAncestry()
        elif input == 'page_down':
            self.downAncestry()
        elif input == 'escape':
            self.deselectAll()
        elif input == 'delete':
            taskMgr.remove('followSelectedNodePath')
            messenger.send('SGE_Remove', [None])
            self.deselectAll()
        elif input == 'v':
            messenger.send('SEditor-ToggleWidgetVis')
            self.toggleWidgetVis()
            if SEditor.widget.fActive:
                messenger.send('shift-f')
        elif input == 'b':
            messenger.send('SEditor-ToggleBackface')
            base.toggleBackface()
        elif input == 'shift-l':
            self.cameraControl.toggleCOALock()
        elif input == 'o':
            self.oobe()
        elif input == 'p':
            if self.selected.last:
                self.setActiveParent(self.selected.last)
        elif input == 'r':
            if self.selected.last:
                self.reparent(self.selected.last, fWrt=1)
        elif input == 'shift-r':
            if self.selected.last:
                self.reparent(self.selected.last)
        elif input == 's':
            if self.selected.last:
                self.select(self.selected.last)
        elif input == 't':
            messenger.send('SEditor-ToggleTexture')
            base.toggleTexture()
        elif input == 'shift-a':
            self.selected.toggleVisAll()
        elif input == 'w':
            messenger.send('SEditor-ToggleWireframe')
            base.toggleWireframe()
        elif input == '[' or input == '{':
            self.undo()
        elif input == ']' or input == '}':
            self.redo()

    def getModifiers(self, input, base):
        if False:
            for i in range(10):
                print('nop')
        modifiers = DIRECT_NO_MOD
        modifierString = input[:input.find(base)]
        if modifierString.find('shift') != -1:
            modifiers |= DIRECT_SHIFT_MOD
        if modifierString.find('control') != -1:
            modifiers |= DIRECT_CONTROL_MOD
        if modifierString.find('alt') != -1:
            modifiers |= DIRECT_ALT_MOD
        return modifiers

    def gotShift(self, modifiers):
        if False:
            print('Hello World!')
        return modifiers & DIRECT_SHIFT_MOD

    def gotControl(self, modifiers):
        if False:
            i = 10
            return i + 15
        return modifiers & DIRECT_CONTROL_MOD

    def gotAlt(self, modifiers):
        if False:
            while True:
                i = 10
        return modifiers & DIRECT_ALT_MOD

    def select(self, nodePath, fMultiSelect=0, fResetAncestry=1, callback=False):
        if False:
            i = 10
            return i + 15
        dnp = self.selected.select(nodePath, fMultiSelect)
        if dnp:
            messenger.send('DIRECT_preSelectNodePath', [dnp])
            if fResetAncestry:
                self.ancestry = list(dnp.getAncestors())
                self.ancestry.reverse()
                self.ancestryIndex = 0
            self.selectedNPReadout.reparentTo(aspect2d)
            self.selectedNPReadout.setText('Selected:' + dnp.getName())
            self.widget.showWidget()
            mCoa2Camera = dnp.mCoa2Dnp * dnp.getMat(self.camera)
            row = mCoa2Camera.getRow(3)
            coa = Vec3(row[0], row[1], row[2])
            self.cameraControl.updateCoa(coa)
            self.widget.setScalingFactor(dnp.getRadius())
            taskMgr.remove('followSelectedNodePath')
            t = Task.Task(self.followSelectedNodePathTask)
            t.dnp = dnp
            taskMgr.add(t, 'followSelectedNodePath')
            messenger.send('DIRECT_selectedNodePath', [dnp])
            if callback:
                messenger.send('se_selectedNodePath', [dnp, False])
            else:
                messenger.send('se_selectedNodePath', [dnp])
            self.upAncestry()
            if SEditor.widget.fActive:
                messenger.send('shift-f')

    def followSelectedNodePathTask(self, state):
        if False:
            i = 10
            return i + 15
        mCoa2Render = state.dnp.mCoa2Dnp * state.dnp.getMat(render)
        decomposeMatrix(mCoa2Render, self.scale, self.hpr, self.pos, CSDefault)
        self.widget.setPosHpr(self.pos, self.hpr)
        return Task.cont

    def deselect(self, nodePath):
        if False:
            i = 10
            return i + 15
        dnp = self.selected.deselect(nodePath)
        if dnp:
            self.widget.hideWidget()
            self.selectedNPReadout.reparentTo(hidden)
            self.selectedNPReadout.setText(' ')
            taskMgr.remove('followSelectedNodePath')
            self.ancestry = []
            messenger.send('DIRECT_deselectedNodePath', [dnp])

    def deselectAll(self):
        if False:
            for i in range(10):
                print('nop')
        self.selected.deselectAll()
        self.widget.hideWidget()
        self.selectedNPReadout.reparentTo(hidden)
        self.selectedNPReadout.setText(' ')
        taskMgr.remove('followSelectedNodePath')
        messenger.send('se_deselectedAll')

    def setActiveParent(self, nodePath=None):
        if False:
            print('Hello World!')
        self.activeParent = nodePath
        self.activeParentReadout.reparentTo(aspect2d)
        self.activeParentReadout.setText('Active Reparent Target:' + nodePath.getName())
        self.activeParentReadout.show()

    def reparent(self, nodePath=None, fWrt=0):
        if False:
            while True:
                i = 10
        if nodePath and self.activeParent and self.isNotCycle(nodePath, self.activeParent):
            oldParent = nodePath.getParent()
            if fWrt:
                nodePath.wrtReparentTo(self.activeParent)
            else:
                nodePath.reparentTo(self.activeParent)
            messenger.send('DIRECT_reparent', [nodePath, oldParent, self.activeParent])
            messenger.send('SGE_Update Explorer', [render])
            self.activeParentReadout.hide()

    def isNotCycle(self, nodePath, parent):
        if False:
            for i in range(10):
                print('nop')
        if nodePath.get_key() == parent.get_key():
            print('DIRECT.reparent: Invalid parent')
            return 0
        elif parent.hasParent():
            return self.isNotCycle(nodePath, parent.getParent())
        else:
            return 1

    def fitOnNodePath(self, nodePath='None Given'):
        if False:
            for i in range(10):
                print('nop')
        if nodePath == 'None Given':
            nodePath = self.selected.last
        SEditor.select(nodePath)

        def fitTask(state, self=self):
            if False:
                for i in range(10):
                    print('nop')
            self.cameraControl.fitOnWidget()
            return Task.done
        taskMgr.doMethodLater(0.1, fitTask, 'manipulateCamera')

    def isolate(self, nodePath='None Given'):
        if False:
            for i in range(10):
                print('nop')
        ' Show a node path and hide its siblings '
        taskMgr.remove('flashNodePath')
        if nodePath == 'None Given':
            nodePath = self.selected.last
        if nodePath:
            self.showAllDescendants(nodePath.getParent())
            nodePath.hideSiblings()

    def toggleVis(self, nodePath='None Given'):
        if False:
            return 10
        ' Toggle visibility of node path '
        taskMgr.remove('flashNodePath')
        if nodePath == 'None Given':
            nodePath = self.selected.last
        if nodePath:
            if nodePath.is_hidden():
                nodePath.show()
            else:
                nodePath.hide()

    def removeNodePath(self, nodePath='None Given'):
        if False:
            i = 10
            return i + 15
        if nodePath == 'None Given':
            nodePath = self.selected.last
        if nodePath:
            nodePath.remove()

    def removeAllSelected(self):
        if False:
            i = 10
            return i + 15
        self.selected.removeAll()

    def showAllDescendants(self, nodePath=render):
        if False:
            i = 10
            return i + 15
        ' Show the level and its descendants '
        nodePath.showAllDescendants()
        nodePath.hideCS()

    def upAncestry(self):
        if False:
            while True:
                i = 10
        if self.ancestry:
            l = len(self.ancestry)
            i = self.ancestryIndex + 1
            if i < l:
                np = self.ancestry[i]
                name = np.getName()
                if i > 0:
                    type = self.ancestry[i - 1].node().getType().getName()
                else:
                    type = self.ancestry[0].node().getType().getName()
                ntype = np.node().getType().getName()
                if name != 'render' and name != 'renderTop' and self.checkTypeNameForAncestry(type, ntype):
                    self.ancestryIndex = i
                    self.select(np, 0, 0, True)

    def checkTypeNameForAncestry(self, type, nextType):
        if False:
            for i in range(10):
                print('nop')
        if type == 'ModelRoot':
            if nextType == 'AmbientLight' or nextType == 'PointLight' or nextType == 'DirectionalLight' or (nextType == 'Spotlight'):
                return True
            return False
        elif type == 'ModelNode':
            if nextType == 'ModelNode':
                return True
            return False
        elif type == 'CollisionNode':
            return False
        elif type == 'ActorNode':
            return False
        elif type == 'AmbientLight' or type == 'PointLight' or type == 'DirectionalLight' or (type == 'Spotlight'):
            return False
        else:
            return True

    def downAncestry(self):
        if False:
            i = 10
            return i + 15
        if self.ancestry:
            l = len(self.ancestry)
            i = self.ancestryIndex - 1
            if i >= 0:
                np = self.ancestry[i]
                name = np.getName()
                if name != 'render' and name != 'renderTop':
                    self.ancestryIndex = i
                    self.select(np, 0, 0, True)

    def getAndSetName(self, nodePath):
        if False:
            while True:
                i = 10
        ' Prompt user for new node path name '
        from tkSimpleDialog import askstring
        newName = askstring('Node Path: ' + nodePath.getName(), 'Enter new name:')
        if newName:
            nodePath.setName(newName)
            messenger.send('DIRECT_nodePathSetName', [nodePath, newName])

    def pushUndo(self, nodePathList, fResetRedo=1):
        if False:
            for i in range(10):
                print('nop')
        undoGroup = []
        for nodePath in nodePathList:
            t = nodePath.getTransform()
            undoGroup.append([nodePath, t])
        self.undoList.append(undoGroup)
        self.undoList = self.undoList[-25:]
        messenger.send('DIRECT_pushUndo')
        if fResetRedo and nodePathList != []:
            self.redoList = []
            messenger.send('DIRECT_redoListEmpty')

    def popUndoGroup(self):
        if False:
            i = 10
            return i + 15
        undoGroup = self.undoList[-1]
        self.undoList = self.undoList[:-1]
        if not self.undoList:
            messenger.send('DIRECT_undoListEmpty')
        return undoGroup

    def pushRedo(self, nodePathList):
        if False:
            print('Hello World!')
        redoGroup = []
        for nodePath in nodePathList:
            t = nodePath.getTransform()
            redoGroup.append([nodePath, t])
        self.redoList.append(redoGroup)
        self.redoList = self.redoList[-25:]
        messenger.send('DIRECT_pushRedo')

    def popRedoGroup(self):
        if False:
            for i in range(10):
                print('nop')
        redoGroup = self.redoList[-1]
        self.redoList = self.redoList[:-1]
        if not self.redoList:
            messenger.send('DIRECT_redoListEmpty')
        return redoGroup

    def undo(self):
        if False:
            return 10
        if self.undoList:
            undoGroup = self.popUndoGroup()
            nodePathList = map(lambda x: x[0], undoGroup)
            self.pushRedo(nodePathList)
            for pose in undoGroup:
                pose[0].setTransform(pose[1])
            messenger.send('DIRECT_undo')

    def redo(self):
        if False:
            print('Hello World!')
        if self.redoList:
            redoGroup = self.popRedoGroup()
            nodePathList = map(lambda x: x[0], redoGroup)
            self.pushUndo(nodePathList, fResetRedo=0)
            for pose in redoGroup:
                pose[0].setTransform(pose[1])
            messenger.send('DIRECT_redo')

    def message(self, text):
        if False:
            return 10
        taskMgr.remove('hideDirectMessage')
        taskMgr.remove('hideDirectMessageLater')
        self.directMessageReadout.reparentTo(aspect2d)
        self.directMessageReadout.setText(text)
        self.hideDirectMessageLater()

    def hideDirectMessageLater(self):
        if False:
            for i in range(10):
                print('nop')
        taskMgr.doMethodLater(3.0, self.hideDirectMessage, 'hideDirectMessage')

    def hideDirectMessage(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.directMessageReadout.reparentTo(hidden)
        return Task.done

    def useObjectHandles(self):
        if False:
            i = 10
            return i + 15
        self.widget = self.manipulationControl.objectHandles
        self.widget.reparentTo(SEditor.group)

    def hideSelectedNPReadout(self):
        if False:
            return 10
        self.selectedNPReadout.reparentTo(hidden)

    def hideActiveParentReadout(self):
        if False:
            for i in range(10):
                print('nop')
        self.activeParentReadout.reparentTo(hidden)

    def toggleWidgetVis(self):
        if False:
            while True:
                i = 10
        self.widget.toggleWidget()

    def setCOAMode(self, mode):
        if False:
            for i in range(10):
                print('nop')
        self.coaMode = mode

    def isEnabled(self):
        if False:
            print('Hello World!')
        return self.fEnabled

    def addUnpickable(self, item):
        if False:
            while True:
                i = 10
        for iRay in self.iRayList:
            iRay.addUnpickable(item)

    def removeUnpickable(self, item):
        if False:
            for i in range(10):
                print('nop')
        for iRay in self.iRayList:
            iRay.removeUnpickable(item)

    def toggleAutoCamera(self):
        if False:
            i = 10
            return i + 15
        self.enableAutoCamera = (self.enableAutoCamera + 1) % 2
        if self.enableAutoCamera == 1:
            self.accept('DH_LoadingComplete', self.autoCameraMove)
        else:
            self.ignore('DH_LoadingComplete')
        return

    def autoCameraMove(self, nodePath):
        if False:
            return 10
        time = 1
        node = DirectNodePath(nodePath)
        radius = node.getRadius()
        center = node.getCenter()
        node.dehighlight()
        posB = base.camera.getPos()
        hprB = base.camera.getHpr()
        posE = Point3(radius * -1.41 + center.getX(), radius * -1.41 + center.getY(), radius * 1.41 + center.getZ())
        hprE = Point3(-45, -38, 0)
        print(posB, hprB)
        print(posE, hprE)
        posInterval1 = base.camera.posInterval(time, posE, bakeInStart=1)
        posInterval2 = base.camera.posInterval(time, posB, bakeInStart=1)
        hprInterval1 = base.camera.hprInterval(time, hprE, bakeInStart=1)
        hprInterval2 = base.camera.hprInterval(time, hprB, bakeInStart=1)
        parallel1 = Parallel(posInterval1, hprInterval1)
        parallel2 = Parallel(posInterval2, hprInterval2)
        Sequence(Wait(7), parallel1, Wait(1), parallel2).start()
        return

class DisplayRegionContext(DirectObject):
    regionCount = 0

    def __init__(self, cam):
        if False:
            print('Hello World!')
        self.cam = cam
        self.camNode = self.cam.node()
        self.camLens = self.camNode.getLens()
        changeEvent = 'dr%d-change-event' % DisplayRegionContext.regionCount
        DisplayRegionContext.regionCount += 1
        self.camLens.setChangeEvent(changeEvent)
        self.accept(changeEvent, self.camUpdate)
        self.iRay = SelectionRay(self.cam)
        self.nearVec = Vec3(0)
        self.mouseX = 0.0
        self.mouseY = 0.0
        try:
            self.dr = self.camNode.getDr(0)
        except:
            self.dr = self.camNode.getDisplayRegion(0)
        left = self.dr.getLeft()
        right = self.dr.getRight()
        bottom = self.dr.getBottom()
        top = self.dr.getTop()
        self.originX = left + right - 1
        self.originY = top + bottom - 1
        self.scaleX = 1.0 / (right - left)
        self.scaleY = 1.0 / (top - bottom)
        self.setOrientation()
        self.camUpdate()

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self.__dict__[key]

    def setOrientation(self):
        if False:
            print('Hello World!')
        hpr = self.cam.getHpr()
        if hpr[2] < 135 and hpr[2] > 45 or (hpr[2] > 225 and hpr[2] < 315):
            self.isSideways = 1
        elif hpr[2] > -135 and hpr[2] < -45 or (hpr[2] < -225 and hpr[2] > -315):
            self.isSideways = 1
        else:
            self.isSideways = 0

    def getHfov(self):
        if False:
            for i in range(10):
                print('nop')
        if self.isSideways:
            return self.camLens.getVfov()
        else:
            return self.camLens.getHfov()

    def getVfov(self):
        if False:
            for i in range(10):
                print('nop')
        if self.isSideways:
            return self.camLens.getHfov()
        else:
            return self.camLens.getVfov()

    def setHfov(self, hfov):
        if False:
            return 10
        if self.isSideways:
            self.camLens.setFov(self.camLens.getHfov(), hfov)
        else:
            self.camLens.setFov(hfov, self.camLens.getVfov())

    def setVfov(self, vfov):
        if False:
            i = 10
            return i + 15
        if self.isSideways:
            self.camLens.setFov(vfov, self.camLens.getVfov())
        else:
            self.camLens.setFov(self.camLens.getHfov(), vfov)

    def setFov(self, hfov, vfov):
        if False:
            for i in range(10):
                print('nop')
        if self.isSideways:
            self.camLens.setFov(vfov, hfov)
        else:
            self.camLens.setFov(hfov, vfov)

    def getWidth(self):
        if False:
            print('Hello World!')
        prop = base.win.getProperties()
        if prop.hasSize():
            return prop.getXSize()
        else:
            return 640

    def getHeight(self):
        if False:
            while True:
                i = 10
        prop = base.win.getProperties()
        if prop.hasSize():
            return prop.getYSize()
        else:
            return 480

    def camUpdate(self, lens=None):
        if False:
            for i in range(10):
                print('nop')
        self.near = self.camLens.getNear()
        self.far = self.camLens.getFar()
        self.fovH = self.camLens.getHfov()
        self.fovV = self.camLens.getVfov()
        self.nearWidth = math.tan(deg2Rad(self.fovH * 0.5)) * self.near * 2.0
        self.nearHeight = math.tan(deg2Rad(self.fovV * 0.5)) * self.near * 2.0
        self.left = -self.nearWidth * 0.5
        self.right = self.nearWidth * 0.5
        self.top = self.nearHeight * 0.5
        self.bottom = -self.nearHeight * 0.5

    def mouseUpdate(self):
        if False:
            i = 10
            return i + 15
        self.mouseLastX = self.mouseX
        self.mouseLastY = self.mouseY
        if base.mouseWatcherNode.hasMouse():
            self.mouseX = base.mouseWatcherNode.getMouseX()
            self.mouseY = base.mouseWatcherNode.getMouseY()
            self.mouseX = (self.mouseX - self.originX) * self.scaleX
            self.mouseY = (self.mouseY - self.originY) * self.scaleY
        self.mouseDeltaX = self.mouseX - self.mouseLastX
        self.mouseDeltaY = self.mouseY - self.mouseLastY
        self.nearVec.set(self.nearWidth * 0.5 * self.mouseX, self.near, self.nearHeight * 0.5 * self.mouseY)

class DisplayRegionList(DirectObject):

    def __init__(self):
        if False:
            print('Hello World!')
        self.displayRegionList = []
        i = 0
        if hasattr(base, 'oobeMode') and base.oobeMode:
            drc = DisplayRegionContext(base.cam)
            self.displayRegionList.append(drc)
        else:
            for camIndex in range(len(base.camList)):
                cam = base.camList[camIndex]
                if cam.getName() == '<noname>':
                    cam.setName('Camera%d' % camIndex)
                drc = DisplayRegionContext(cam)
                self.displayRegionList.append(drc)
        self.accept('DIRECT-mouse1', self.mouseUpdate)
        self.accept('DIRECT-mouse2', self.mouseUpdate)
        self.accept('DIRECT-mouse3', self.mouseUpdate)
        self.accept('DIRECT-mouse1Up', self.mouseUpdate)
        self.accept('DIRECT-mouse2Up', self.mouseUpdate)
        self.accept('DIRECT-mouse3Up', self.mouseUpdate)

    def __getitem__(self, index):
        if False:
            return 10
        return self.displayRegionList[index]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.displayRegionList)

    def updateContext(self):
        if False:
            i = 10
            return i + 15
        self.contextTask(None)

    def setNearFar(self, near, far):
        if False:
            print('Hello World!')
        for dr in self.displayRegionList:
            dr.camLens.setNearFar(near, far)

    def setNear(self, near):
        if False:
            return 10
        for dr in self.displayRegionList:
            dr.camLens.setNear(near)

    def setFar(self, far):
        if False:
            for i in range(10):
                print('nop')
        for dr in self.displayRegionList:
            dr.camLens.setFar(far)

    def setFov(self, hfov, vfov):
        if False:
            print('Hello World!')
        for dr in self.displayRegionList:
            dr.setFov(hfov, vfov)

    def setHfov(self, fov):
        if False:
            print('Hello World!')
        for dr in self.displayRegionList:
            dr.setHfov(fov)

    def setVfov(self, fov):
        if False:
            for i in range(10):
                print('nop')
        for dr in self.displayRegionList:
            dr.setVfov(fov)

    def mouseUpdate(self, modifiers=DIRECT_NO_MOD):
        if False:
            while True:
                i = 10
        for dr in self.displayRegionList:
            dr.mouseUpdate()
        SEditor.dr = self.getCurrentDr()

    def getCurrentDr(self):
        if False:
            return 10
        for dr in self.displayRegionList:
            if dr.mouseX >= -1.0 and dr.mouseX <= 1.0 and (dr.mouseY >= -1.0) and (dr.mouseY <= 1.0):
                return dr
        return self.displayRegionList[0]

    def start(self):
        if False:
            return 10
        self.stop()
        self.spawnContextTask()

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        taskMgr.remove('DIRECTContextTask')

    def spawnContextTask(self):
        if False:
            while True:
                i = 10
        taskMgr.add(self.contextTask, 'DIRECTContextTask')

    def removeContextTask(self):
        if False:
            print('Hello World!')
        taskMgr.remove('DIRECTContextTask')

    def contextTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.mouseUpdate()
        return Task.cont