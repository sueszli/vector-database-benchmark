import math
from panda3d.core import CollisionNode, ConfigVariableBool, ConfigVariableString, CSDefault, GraphicsWindow, NodePath, Point3, TextNode, VBase3, VBase4, Vec3, Vec4, decomposeMatrix, deg2Rad
from .DirectUtil import useDirectRenderStyle
from direct.showbase.DirectObject import DirectObject
from direct.showbase.BulletinBoardGlobal import bulletinBoard as bboard
from direct.task import Task
from . import DirectGlobals as DG
from .DirectCameraControl import DirectCameraControl
from .DirectManipulation import DirectManipulationControl
from .DirectSelection import SelectionRay, COA_ORIGIN, SelectedNodePaths
from .DirectGrid import DirectGrid
from .DirectLights import DirectLights
from direct.gui import OnscreenText
from direct.interval.IntervalGlobal import Func, Sequence
from direct.task.TaskManagerGlobal import taskMgr
from direct.showbase.MessengerGlobal import messenger
from direct.showbase import ShowBaseGlobal
from direct.showbase.ShowBaseGlobal import ShowBase, hidden
import builtins
base: ShowBase

class DirectSession(DirectObject):
    DIRECTdisablePost = 'disableDIRECT'
    cam: NodePath
    camera: NodePath
    oobeCamera: NodePath

    def __init__(self):
        if False:
            print('Hello World!')
        global direct, base
        base = ShowBaseGlobal.base
        base.direct = self
        setattr(builtins, 'direct', self)
        ShowBaseGlobal.direct = self
        self.group = base.render.attachNewNode('DIRECT')
        self.font = TextNode.getDefaultFont()
        self.fEnabled = 0
        self.fEnabledLight = 0
        self.fScaleWidgetByCam = 0
        self.fIgnoreDirectOnlyKeyMap = 0
        self.drList = DisplayRegionList()
        self.iRayList = [x.iRay for x in self.drList]
        self.dr = self.drList[0]
        self.win: GraphicsWindow = base.win
        self.camera = base.camera
        self.cam = base.cam
        self.camNode = base.camNode
        self.trueCamera = self.camera
        self.iRay = self.dr.iRay
        self.coaMode = COA_ORIGIN
        self.cameraControl = DirectCameraControl()
        self.manipulationControl = DirectManipulationControl()
        self.useObjectHandles()
        self.grid = DirectGrid()
        self.grid.disable()
        self.lights = DirectLights(self.group)
        self.lights.createDefaultLights()
        self.lights.allOff()
        self.selected = SelectedNodePaths()
        self.ancestry = []
        self.ancestryIndex = 0
        self.activeParent = None
        self.selectedNPReadout = OnscreenText.OnscreenText(pos=(0.1, 0.1), bg=Vec4(0, 0, 0, 0.2), style=3, fg=(1, 1, 1, 1), scale=0.05, align=TextNode.ALeft, mayChange=1, font=self.font)
        useDirectRenderStyle(self.selectedNPReadout)
        self.selectedNPReadout.reparentTo(hidden)
        self.activeParentReadout = OnscreenText.OnscreenText(pos=(0.1, 0.16), bg=Vec4(0, 0, 0, 0.2), style=3, fg=(1, 1, 1, 1), scale=0.05, align=TextNode.ALeft, mayChange=1, font=self.font)
        useDirectRenderStyle(self.activeParentReadout)
        self.activeParentReadout.reparentTo(hidden)
        self.directMessageReadout = OnscreenText.OnscreenText(pos=(0.1, -0.1), bg=Vec4(0, 0, 0, 0.2), style=3, fg=(1, 1, 1, 1), scale=0.05, align=TextNode.ALeft, mayChange=1, font=self.font)
        useDirectRenderStyle(self.directMessageReadout)
        self.directMessageReadout.reparentTo(hidden)
        self.deviceManager = None
        self.joybox = None
        self.radamec = None
        self.fastrak = []
        if ConfigVariableBool('want-vrpn', False):
            from direct.directdevices import DirectDeviceManager
            self.deviceManager = DirectDeviceManager.DirectDeviceManager()
            joybox = ConfigVariableString('vrpn-joybox-device', '').value
            radamec = ConfigVariableString('vrpn-radamec-device', '').value
            fastrak = ConfigVariableString('vrpn-fastrak-device', '').value
            if joybox:
                from direct.directdevices import DirectJoybox
                self.joybox = DirectJoybox.DirectJoybox(joybox)
            if radamec:
                from direct.directdevices import DirectRadamec
                self.radamec = DirectRadamec.DirectRadamec(radamec)
            if fastrak:
                from direct.directdevices import DirectFastrak
                fastrak = fastrak.split()
                for i in range(len(fastrak))[1:]:
                    self.fastrak.append(DirectFastrak.DirectFastrak(fastrak[0] + ':' + fastrak[i]))
        self.fControl = 0
        self.fAlt = 0
        self.fShift = 0
        self.fMouse1 = 0
        self.fMouse2 = 0
        self.fMouse3 = 0
        self.pos = VBase3()
        self.hpr = VBase3()
        self.scale = VBase3()
        self.hitPt = Point3(0.0)
        self.undoList = []
        self.redoList = []
        self.drList.updateContext()
        for dr in self.drList:
            dr.camUpdate()
        self.actionEvents = [['select', self.select], ['DIRECT-select', self.selectCB], ['deselect', self.deselect], ['deselectAll', self.deselectAll], ['DIRECT-preDeselectAll', self.deselectAllCB], ['highlightAll', self.selected.highlightAll], ['preRemoveNodePath', self.deselect], ['SGE_Select', self.select], ['SGE_Deselect', self.deselect], ['SGE_Set Reparent Target', self.setActiveParent], ['SGE_Reparent', self.reparent], ['SGE_WRT Reparent', lambda np, s=self: s.reparent(np, fWrt=1)], ['SGE_Flash', self.flash], ['SGE_Isolate', self.isolate], ['SGE_Toggle Vis', self.toggleVis], ['SGE_Show All', self.showAllDescendants], ['SGE_Fit', self.fitOnNodePath], ['SGE_Delete', self.removeNodePath], ['SGE_Set Name', self.getAndSetName], ['DIRECT-delete', self.removeAllSelected], ['DIRECT-Undo', self.undo], ['DIRECT-Redo', self.redo], ['DIRECT-OOBE', self.oobe], ['DIRECT-toggleWidgetVis', self.toggleWidgetVis], ['DIRECT-toggleWireframe', base.toggleWireframe], ['DIRECT-toggleVisAll', self.selected.toggleVisAll], ['DIRECT-toggleTexture', base.toggleTexture], ['DIRECT-upAncestry', self.upAncestry], ['DIRECT-downAncestry', self.downAncestry], ['DIRECT-toggleBackface', base.toggleBackface], ['DIRECT-flash', self.flash], ['DIRECT-toggleLigths', self.lights.toggle], ['DIRECT-toggleCOALock', self.cameraControl.toggleCOALock], ['DIRECT-setActiveParent', self.doSetActiveParent], ['DIRECT-doWrtReparent', self.doWrtReparent], ['DIRECT-doReparent', self.doReparent], ['DIRECT-doSelect', self.doSelect]]
        if base.wantTk:
            from direct.tkpanels import Placer
            from direct.tkwidgets import Slider
            from direct.tkwidgets import SceneGraphExplorer
            self.actionEvents.extend([['SGE_Place', Placer.place], ['SGE_Set Color', Slider.rgbPanel], ['SGE_Explore', SceneGraphExplorer.explore]])
        self.modifierEvents = ['control', 'control-up', 'control-repeat', 'shift', 'shift-up', 'shift-repeat', 'alt', 'alt-up', 'alt-repeat']
        keyList = [chr(i) for i in range(97, 123)]
        keyList.extend([chr(i) for i in range(48, 58)])
        keyList.extend(['`', '-', '=', '[', ']', ';', "'", ',', '.', '/', '\\'])
        self.specialKeys = ['escape', 'delete', 'page_up', 'page_down', 'enter']

        def addCtrl(a):
            if False:
                while True:
                    i = 10
            return 'control-%s' % a

        def addShift(a):
            if False:
                print('Hello World!')
            return 'shift-%s' % a
        self.keyEvents = keyList[:]
        self.keyEvents.extend(list(map(addCtrl, keyList)))
        self.keyEvents.extend(list(map(addShift, keyList)))
        self.keyEvents.extend(self.specialKeys)
        self.mouseEvents = ['mouse1', 'mouse1-up', 'shift-mouse1', 'shift-mouse1-up', 'control-mouse1', 'control-mouse1-up', 'alt-mouse1', 'alt-mouse1-up', 'mouse2', 'mouse2-up', 'shift-mouse2', 'shift-mouse2-up', 'control-mouse2', 'control-mouse2-up', 'alt-mouse2', 'alt-mouse2-up', 'mouse3', 'mouse3-up', 'shift-mouse3', 'shift-mouse3-up', 'control-mouse3', 'control-mouse3-up', 'alt-mouse3', 'alt-mouse3-up']
        self.directOnlyKeyMap = {'u': ('Orbit Upright Camera', 'DIRECT-orbitUprightCam'), 'shift-u': ('Upright Camera', 'DIRECT-uprightCam'), '1': ('Move Camera to View 1', 'DIRECT-spwanMoveToView-1'), '2': ('Move Camera to View 2', 'DIRECT-spwanMoveToView-2'), '3': ('Move Camera to View 3', 'DIRECT-spwanMoveToView-3'), '4': ('Move Camera to View 4', 'DIRECT-spwanMoveToView-4'), '5': ('Move Camera to View 5', 'DIRECT-spwanMoveToView-5'), '6': ('Move Camera to View 6', 'DIRECT-spwanMoveToView-6'), '7': ('Move Camera to View 7', 'DIRECT-spwanMoveToView-7'), '8': ('Move Camera to View 8', 'DIRECT-spwanMoveToView-8'), '9': ('Rotate Camera About widget 90 degrees Counterclockwise', 'DIRECT-swingCamAboutWidget-0'), '0': ('Rotate Camera About widget 90 degrees Clockwise', 'DIRECT-swingCamAboutWidget-1'), '`': ('Remove ManipulateCameraTask', 'DIRECT-removeManipulateCameraTask'), '=': ('Zoom In', 'DIRECT-zoomInCam'), 'shift-=': ('Zoom In', 'DIRECT-zoomInCam'), 'shift-_': ('Zoom Out', 'DIRECT-zoomOutCam'), '-': ('Zoom Out', 'DIRECT-zoomOutCam'), 'o': ('Toggle OOBE', 'DIRECT-OOBE'), '[': ('DIRECT-Undo', 'DIRECT-Undo'), 'shift-[': ('DIRECT-Undo', 'DIRECT-Undo'), ']': ('DIRECT-Redo', 'DIRECT-Redo'), 'shift-]': ('DIRECT-Redo', 'DIRECT-Redo')}
        self.hotKeyMap = {'c': ('Center Camera', 'DIRECT-centerCamIn'), 'f': ('Fit on Widget', 'DIRECT-fitOnWidget'), 'h': ('Move Camera to ', 'DIRECT-homeCam'), 'shift-v': ('Toggle Marker', 'DIRECT-toggleMarkerVis'), 'm': ('Move to fit', 'DIRECT-moveToFit'), 'n': ('Pick Next COA', 'DIRECT-pickNextCOA'), 'delete': ('Delete', 'DIRECT-delete'), '.': ('Scale Up Widget', 'DIRECT-widgetScaleUp'), ',': ('Scale Down Widget', 'DIRECT-widgetScaleDown'), 'page_up': ('Up Ancestry', 'DIRECT-upAncestry'), 'page_down': ('Down Ancestry', 'DIRECT-downAncestry'), 'escape': ('Deselect All', 'deselectAll'), 'v': ('Toggle Manipulating Widget', 'DIRECT-toggleWidgetVis'), 'b': ('Toggle Backface', 'DIRECT-toggleBackface'), 'control-f': ('Flash', 'DIRECT-flash'), 'l': ('Toggle lights', 'DIRECT-toggleLigths'), 'shift-l': ('Toggle COA Lock', 'DIRECT-toggleCOALock'), 'p': ('Set Active Parent', 'DIRECT-setActiveParent'), 'r': ('Wrt Reparent', 'DIRECT-doWrtReparent'), 'shift-r': ('Reparent', 'DIRECT-doReparent'), 's': ('Select', 'DIRECT-doSelect'), 't': ('Toggle Textures', 'DIRECT-toggleTexture'), 'shift-a': ('Toggle Vis all', 'DIRECT-toggleVisAll'), 'w': ('Toggle Wireframe', 'DIRECT-toggleWireframe'), 'control-z': ('Undo', 'LE-Undo'), 'shift-z': ('Redo', 'LE-Redo'), 'control-d': ('Duplicate', 'LE-Duplicate'), 'control-l': ('Make Live', 'LE-MakeLive'), 'control-n': ('New Scene', 'LE-NewScene'), 'control-s': ('Save Scene', 'LE-SaveScene'), 'control-o': ('Open Scene', 'LE-OpenScene'), 'control-q': ('Quit', 'LE-Quit')}
        self.speicalKeyMap = {'enter': 'DIRECT-enter'}
        self.passThroughKeys = ['v', 'b', 'l', 'p', 'r', 'shift-r', 's', 't', 'shift-a', 'w']
        if base.wantTk:
            from direct.tkpanels import DirectSessionPanel
            self.panel = DirectSessionPanel.DirectSessionPanel(parent=base.tkRoot)
        clusterMode: str
        if hasattr(builtins, 'clusterMode'):
            clusterMode = builtins.clusterMode
        else:
            clusterMode = ConfigVariableString('cluster-mode', '').value
        self.clusterMode = clusterMode
        if self.clusterMode == 'client':
            from direct.cluster.ClusterClient import createClusterClient
            self.cluster = createClusterClient()
        elif self.clusterMode == 'server':
            from direct.cluster.ClusterServer import ClusterServer
            self.cluster = ClusterServer(base.camera, base.cam)
        else:
            from direct.cluster.ClusterClient import DummyClusterClient
            self.cluster = DummyClusterClient()
        setattr(builtins, 'cluster', self.cluster)

    def addPassThroughKey(self, key):
        if False:
            return 10
        self.passThroughKeys.append(key)

    def enable(self):
        if False:
            while True:
                i = 10
        if bboard.has(DirectSession.DIRECTdisablePost):
            return
        if self.fEnabled:
            return
        self.disable()
        self.drList.spawnContextTask()
        if not self.fEnabledLight:
            self.cameraControl.enableMouseFly()
        self.manipulationControl.enableManipulation()
        self.selected.reset()
        if not self.fEnabledLight:
            self.enableKeyEvents()
        self.enableMouseEvents()
        self.enableActionEvents()
        self.enableModifierEvents()
        self.fEnabled = 1

    def enableLight(self):
        if False:
            for i in range(10):
                print('nop')
        self.fEnabledLight = 1
        self.enable()

    def disable(self):
        if False:
            while True:
                i = 10
        self.drList.removeContextTask()
        self.cameraControl.disableMouseFly()
        self.deselectAll()
        self.manipulationControl.disableManipulation()
        self.disableKeyEvents()
        self.disableModifierEvents()
        self.disableMouseEvents()
        self.disableActionEvents()
        taskMgr.remove('flashNodePath')
        taskMgr.remove('hideDirectMessage')
        taskMgr.remove('hideDirectMessageLater')
        self.fEnabled = 0

    def toggleDirect(self):
        if False:
            i = 10
            return i + 15
        if self.fEnabled:
            self.disable()
        else:
            self.enable()

    def minimumConfiguration(self):
        if False:
            for i in range(10):
                print('nop')
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
        if not hasattr(self, 'oobeMode'):
            self.oobeMode = 0
            self.oobeCamera = hidden.attachNewNode('oobeCamera')
            self.oobeVis = base.loader.loadModel('models/misc/camera')
            if self.oobeVis:
                self.oobeVis.node().setFinal(1)
        if self.oobeMode:
            self.cameraControl.camManipRef.setPosHpr(self.trueCamera, 0, 0, 0, 0, 0, 0)
            ival = self.oobeCamera.posHprInterval(2.0, pos=Point3(0), hpr=Vec3(0), other=self.cameraControl.camManipRef, blendType='easeInOut')
            ival = Sequence(ival, Func(self.endOOBE), name='oobeTransition')
            ival.start()
        else:
            self.oobeVis.reparentTo(self.trueCamera)
            self.oobeVis.clearMat()
            cameraParent = self.camera.getParent()
            self.oobeCamera.reparentTo(cameraParent)
            self.oobeCamera.setPosHpr(self.trueCamera, 0, 0, 0, 0, 0, 0)
            self.cam.reparentTo(self.oobeCamera)
            self.cameraControl.camManipRef.setPos(self.trueCamera, Vec3(-2, -20, 5))
            self.cameraControl.camManipRef.lookAt(self.trueCamera)
            ival = self.oobeCamera.posHprInterval(2.0, pos=Point3(0), hpr=Vec3(0), other=self.cameraControl.camManipRef, blendType='easeInOut')
            ival = Sequence(ival, Func(self.beginOOBE), name='oobeTransition')
            ival.start()

    def beginOOBE(self):
        if False:
            for i in range(10):
                print('nop')
        self.oobeCamera.setPosHpr(self.cameraControl.camManipRef, 0, 0, 0, 0, 0, 0)
        self.camera = self.oobeCamera
        self.oobeMode = 1

    def endOOBE(self):
        if False:
            for i in range(10):
                print('nop')
        self.oobeCamera.setPosHpr(self.trueCamera, 0, 0, 0, 0, 0, 0)
        self.cam.reparentTo(self.trueCamera)
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
            while True:
                i = 10
        self.enable()

    def enableActionEvents(self):
        if False:
            i = 10
            return i + 15
        for event in self.actionEvents:
            self.accept(event[0], event[1], extraArgs=event[2:])

    def enableModifierEvents(self):
        if False:
            while True:
                i = 10
        for event in self.modifierEvents:
            self.accept(event, self.inputHandler, [event])

    def enableKeyEvents(self):
        if False:
            i = 10
            return i + 15
        for event in self.keyEvents:
            self.accept(event, self.inputHandler, [event])

    def enableMouseEvents(self):
        if False:
            i = 10
            return i + 15
        for event in self.mouseEvents:
            self.accept(event, self.inputHandler, [event])

    def disableActionEvents(self):
        if False:
            i = 10
            return i + 15
        for (event, method) in self.actionEvents:
            self.ignore(event)

    def disableModifierEvents(self):
        if False:
            print('Hello World!')
        for event in self.modifierEvents:
            self.ignore(event)

    def disableKeyEvents(self):
        if False:
            print('Hello World!')
        for event in self.keyEvents:
            self.ignore(event)

    def disableMouseEvents(self):
        if False:
            return 10
        for event in self.mouseEvents:
            self.ignore(event)

    def inputHandler(self, input):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, 'oobeMode') or self.oobeMode == 0:
            if self.manipulationControl.fMultiView:
                if self.fMouse1 and 'mouse1' not in input or (self.fMouse2 and 'mouse2' not in input) or (self.fMouse3 and 'mouse3' not in input):
                    if input.endswith('-up') or input not in self.modifierEvents:
                        return
                if self.fMouse1 == 0 and 'mouse1-up' in input or (self.fMouse2 == 0 and 'mouse2-up' in input) or (self.fMouse3 == 0 and 'mouse3-up' in input):
                    return
                if (self.fMouse1 or self.fMouse2 or self.fMouse3) and input[4:7] != self.camera.getName()[:3] and input.endswith('-up'):
                    return
                winCtrl = None
                possibleWinCtrls = []
                for cWinCtrl in base.winControls:
                    if cWinCtrl.mouseWatcher.node().hasMouse():
                        possibleWinCtrls.append(cWinCtrl)
                if len(possibleWinCtrls) == 1:
                    winCtrl = possibleWinCtrls[0]
                elif len(possibleWinCtrls) > 1:
                    for cWinCtrl in possibleWinCtrls:
                        if input.endswith('-up') and (not input in self.modifierEvents) and (not input in self.mouseEvents) or input in self.mouseEvents:
                            if input[4:7] == cWinCtrl.camera.getName()[:3]:
                                winCtrl = cWinCtrl
                        elif input[4:7] != cWinCtrl.camera.getName()[:3]:
                            winCtrl = cWinCtrl
                if winCtrl is None:
                    return
                if input not in self.modifierEvents:
                    self.win = winCtrl.win
                    self.camera = winCtrl.camera
                    self.trueCamera = self.camera
                    self.cam = NodePath(winCtrl.camNode)
                    self.camNode = winCtrl.camNode
                    if hasattr(winCtrl, 'grid'):
                        self.grid = winCtrl.grid
                    self.dr = self.drList[base.camList.index(NodePath(winCtrl.camNode))]
                    self.iRay = self.dr.iRay
                    base.mouseWatcher = winCtrl.mouseWatcher
                    base.mouseWatcherNode = winCtrl.mouseWatcher.node()
                    self.dr.mouseUpdate()
                    DG.LE_showInOneCam(self.selectedNPReadout, self.camera.getName())
                    self.widget = self.manipulationControl.widgetList[base.camList.index(NodePath(winCtrl.camNode))]
                input = input[8:]
                if self.fAlt and 'alt' not in input and (not input.endswith('-up')):
                    input = 'alt-' + input
                if input.endswith('-repeat'):
                    input = input[:-7]
        if input in self.hotKeyMap:
            keyDesc = self.hotKeyMap[input]
            messenger.send(keyDesc[1])
        elif input in self.speicalKeyMap:
            messenger.send(self.speicalKeyMap[input])
        elif input in self.directOnlyKeyMap:
            if self.fIgnoreDirectOnlyKeyMap:
                return
            keyDesc = self.directOnlyKeyMap[input]
            messenger.send(keyDesc[1])
        elif input == 'mouse1-up':
            self.fMouse1 = 0
            messenger.send('DIRECT-mouse1Up')
        elif input.find('mouse1') != -1:
            self.fMouse1 = 1
            modifiers = self.getModifiers(input, 'mouse1')
            messenger.send('DIRECT-mouse1', sentArgs=[modifiers])
        elif input == 'mouse2-up':
            self.fMouse2 = 0
            messenger.send('DIRECT-mouse2Up')
        elif input.find('mouse2') != -1:
            self.fMouse2 = 1
            modifiers = self.getModifiers(input, 'mouse2')
            messenger.send('DIRECT-mouse2', sentArgs=[modifiers])
        elif input == 'mouse3-up':
            self.fMouse3 = 0
            messenger.send('DIRECT-mouse3Up')
        elif input.find('mouse3') != -1:
            self.fMouse3 = 1
            modifiers = self.getModifiers(input, 'mouse3')
            messenger.send('DIRECT-mouse3', sentArgs=[modifiers])
        elif input == 'shift':
            self.fShift = 1
        elif input == 'shift-up':
            self.fShift = 0
        elif input == 'control':
            self.fControl = 1
            if self.fMouse1:
                modifiers = DG.DIRECT_NO_MOD
                modifiers |= DG.DIRECT_CONTROL_MOD
                messenger.send('DIRECT-mouse1', sentArgs=[modifiers])
        elif input == 'control-up':
            self.fControl = 0
        elif input == 'alt':
            if self.fAlt:
                return
            self.fAlt = 1
            if self.fMouse1:
                modifiers = DG.DIRECT_NO_MOD
                modifiers |= DG.DIRECT_ALT_MOD
                messenger.send('DIRECT-mouse1', sentArgs=[modifiers])
            elif self.fMouse2:
                modifiers = DG.DIRECT_NO_MOD
                modifiers |= DG.DIRECT_ALT_MOD
                messenger.send('DIRECT-mouse2', sentArgs=[modifiers])
            elif self.fMouse3:
                modifiers = DG.DIRECT_NO_MOD
                modifiers |= DG.DIRECT_ALT_MOD
                messenger.send('DIRECT-mouse3', sentArgs=[modifiers])
        elif input == 'alt-up':
            self.fAlt = 0
        if self.clusterMode == 'client':
            if input in self.passThroughKeys:
                self.cluster('messenger.send("%s")' % input, 0)

    def doSetActiveParent(self):
        if False:
            return 10
        if self.selected.last:
            self.setActiveParent(self.selected.last)

    def doReparent(self):
        if False:
            print('Hello World!')
        if self.selected.last:
            self.reparent(self.selected.last)

    def doWrtReparent(self):
        if False:
            print('Hello World!')
        if self.selected.last:
            self.reparent(self.selected.last, fWrt=1)

    def doSelect(self):
        if False:
            for i in range(10):
                print('nop')
        if self.selected.last:
            self.select(self.selected.last)

    def getModifiers(self, input, base):
        if False:
            i = 10
            return i + 15
        modifiers = DG.DIRECT_NO_MOD
        modifierString = input[:input.find(base)]
        if modifierString.find('shift') != -1:
            modifiers |= DG.DIRECT_SHIFT_MOD
        if modifierString.find('control') != -1:
            modifiers |= DG.DIRECT_CONTROL_MOD
        if modifierString.find('alt') != -1:
            modifiers |= DG.DIRECT_ALT_MOD
        return modifiers

    def gotShift(self, modifiers):
        if False:
            return 10
        return modifiers & DG.DIRECT_SHIFT_MOD

    def gotControl(self, modifiers):
        if False:
            for i in range(10):
                print('nop')
        return modifiers & DG.DIRECT_CONTROL_MOD

    def gotAlt(self, modifiers):
        if False:
            print('Hello World!')
        return modifiers & DG.DIRECT_ALT_MOD

    def setFScaleWidgetByCam(self, flag):
        if False:
            i = 10
            return i + 15
        self.fScaleWidgetByCam = flag
        if flag:
            taskMgr.add(self.widgetResizeTask, 'DIRECTWidgetResize')
        else:
            taskMgr.remove('DIRECTWidgetResize')

    def widgetResizeTask(self, state):
        if False:
            return 10
        if not taskMgr.hasTaskNamed('resizeObjectHandles'):
            dnp = self.selected.last
            if dnp:
                if self.manipulationControl.fMultiView:
                    for i in range(3):
                        sf = 30.0 * self.drList[i].orthoFactor
                        self.manipulationControl.widgetList[i].setDirectScalingFactor(sf)
                    nodeCamDist = Vec3(dnp.getPos(base.camList[3])).length()
                    sf = 0.075 * nodeCamDist * math.tan(deg2Rad(self.drList[3].fovV))
                    self.manipulationControl.widgetList[3].setDirectScalingFactor(sf)
                else:
                    nodeCamDist = Vec3(dnp.getPos(self.camera)).length()
                    sf = 0.075 * nodeCamDist * math.tan(deg2Rad(self.drList.getCurrentDr().fovV))
                    self.widget.setDirectScalingFactor(sf)
        return Task.cont

    def select(self, nodePath, fMultiSelect=0, fSelectTag=1, fResetAncestry=1, fLEPane=0, fUndo=1):
        if False:
            return 10
        messenger.send('DIRECT-select', [nodePath, fMultiSelect, fSelectTag, fResetAncestry, fLEPane, fUndo])

    def selectCB(self, nodePath, fMultiSelect=0, fSelectTag=1, fResetAncestry=1, fLEPane=0, fUndo=1):
        if False:
            return 10
        dnp = self.selected.select(nodePath, fMultiSelect, fSelectTag)
        if dnp:
            messenger.send('DIRECT_preSelectNodePath', [dnp])
            if fResetAncestry:
                self.ancestry = dnp.getAncestors()
                self.ancestryIndex = 0
            self.selectedNPReadout.reparentTo(base.a2dBottomLeft)
            self.selectedNPReadout.setText('Selected:' + dnp.getName())
            if self.manipulationControl.fMultiView:
                for widget in self.manipulationControl.widgetList:
                    widget.showWidget()
            else:
                self.widget.showWidget()
            editTypes = self.manipulationControl.getEditTypes([dnp])
            if editTypes & DG.EDIT_TYPE_UNEDITABLE == DG.EDIT_TYPE_UNEDITABLE:
                self.manipulationControl.disableWidgetMove()
            else:
                self.manipulationControl.enableWidgetMove()
            mCoa2Camera = dnp.mCoa2Dnp * dnp.getMat(self.camera)
            row = mCoa2Camera.getRow(3)
            coa = Vec3(row[0], row[1], row[2])
            self.cameraControl.updateCoa(coa)
            if not self.fScaleWidgetByCam:
                if self.manipulationControl.fMultiView:
                    for widget in self.manipulationControl.widgetList:
                        widget.setScalingFactor(dnp.getRadius())
                else:
                    self.widget.setScalingFactor(dnp.getRadius())
            taskMgr.remove('followSelectedNodePath')
            t = Task.Task(self.followSelectedNodePathTask)
            t.dnp = dnp
            taskMgr.add(t, 'followSelectedNodePath')
            messenger.send('DIRECT_selectedNodePath', [dnp])
            messenger.send('DIRECT_selectedNodePath_fMulti_fTag', [dnp, fMultiSelect, fSelectTag])
            messenger.send('DIRECT_selectedNodePath_fMulti_fTag_fLEPane', [dnp, fMultiSelect, fSelectTag, fLEPane])

    def followSelectedNodePathTask(self, state):
        if False:
            while True:
                i = 10
        mCoa2Render = state.dnp.mCoa2Dnp * state.dnp.getMat(base.render)
        decomposeMatrix(mCoa2Render, self.scale, self.hpr, self.pos, CSDefault)
        self.widget.setPosHpr(self.pos, self.hpr)
        return Task.cont

    def deselect(self, nodePath):
        if False:
            for i in range(10):
                print('nop')
        dnp = self.selected.deselect(nodePath)
        if dnp:
            if self.manipulationControl.fMultiView:
                for widget in self.manipulationControl.widgetList:
                    widget.hideWidget()
            else:
                self.widget.hideWidget()
            self.selectedNPReadout.reparentTo(hidden)
            self.selectedNPReadout.setText(' ')
            taskMgr.remove('followSelectedNodePath')
            self.ancestry = []
            messenger.send('DIRECT_deselectedNodePath', [dnp])

    def deselectAll(self):
        if False:
            while True:
                i = 10
        messenger.send('DIRECT-preDeselectAll')

    def deselectAllCB(self):
        if False:
            for i in range(10):
                print('nop')
        self.selected.deselectAll()
        if self.manipulationControl.fMultiView:
            for widget in self.manipulationControl.widgetList:
                widget.hideWidget()
        else:
            self.widget.hideWidget()
        self.selectedNPReadout.reparentTo(hidden)
        self.selectedNPReadout.setText(' ')
        taskMgr.remove('followSelectedNodePath')
        messenger.send('DIRECT_deselectAll')

    def setActiveParent(self, nodePath=None):
        if False:
            for i in range(10):
                print('nop')
        self.activeParent = nodePath
        self.activeParentReadout.reparentTo(base.a2dBottomLeft)
        self.activeParentReadout.setText('Active Reparent Target:' + nodePath.getName())
        messenger.send('DIRECT_activeParent', [self.activeParent])

    def reparent(self, nodePath=None, fWrt=0):
        if False:
            return 10
        if nodePath and self.activeParent and self.isNotCycle(nodePath, self.activeParent):
            oldParent = nodePath.getParent()
            if fWrt:
                nodePath.wrtReparentTo(self.activeParent)
            else:
                nodePath.reparentTo(self.activeParent)
            messenger.send('DIRECT_reparent', [nodePath, oldParent, self.activeParent])
            messenger.send('DIRECT_reparent_fWrt', [nodePath, oldParent, self.activeParent, fWrt])

    def isNotCycle(self, nodePath, parent):
        if False:
            while True:
                i = 10
        if nodePath == parent:
            print('DIRECT.reparent: Invalid parent')
            return 0
        elif parent.hasParent():
            return self.isNotCycle(nodePath, parent.getParent())
        else:
            return 1

    def flash(self, nodePath='None Given'):
        if False:
            i = 10
            return i + 15
        ' Highlight an object by setting it red for a few seconds '
        taskMgr.remove('flashNodePath')
        if nodePath == 'None Given':
            nodePath = self.selected.last
        if nodePath:
            if nodePath.hasColor():
                doneColor = nodePath.getColor()
                flashColor = VBase4(1) - doneColor
                flashColor.setW(1)
            else:
                doneColor = None
                flashColor = VBase4(1, 0, 0, 1)
            nodePath.setColor(flashColor)
            t = taskMgr.doMethodLater(DG.DIRECT_FLASH_DURATION, self.flashDummy, 'flashNodePath')
            t.nodePath = nodePath
            t.doneColor = doneColor
            t.setUponDeath(self.flashDone)

    def flashDummy(self, state):
        if False:
            i = 10
            return i + 15
        return Task.done

    def flashDone(self, state):
        if False:
            i = 10
            return i + 15
        if state.nodePath.isEmpty():
            return
        if state.doneColor:
            state.nodePath.setColor(state.doneColor)
        else:
            state.nodePath.clearColor()

    def fitOnNodePath(self, nodePath='None Given'):
        if False:
            return 10
        if nodePath == 'None Given':
            nodePath = self.selected.last
        self.select(nodePath)

        def fitTask(state, self=self):
            if False:
                print('Hello World!')
            self.cameraControl.fitOnWidget()
            return Task.done
        taskMgr.doMethodLater(0.1, fitTask, 'manipulateCamera')

    def isolate(self, nodePath='None Given'):
        if False:
            i = 10
            return i + 15
        ' Show a node path and hide its siblings '
        taskMgr.remove('flashNodePath')
        if nodePath == 'None Given':
            nodePath = self.selected.last
        if nodePath:
            self.showAllDescendants(nodePath.getParent())
            for sib in nodePath.getParent().getChildren():
                if sib.node() != nodePath.node():
                    sib.hide()

    def toggleVis(self, nodePath='None Given'):
        if False:
            for i in range(10):
                print('nop')
        ' Toggle visibility of node path '
        taskMgr.remove('flashNodePath')
        if nodePath == 'None Given':
            nodePath = self.selected.last
        if nodePath:
            if nodePath.isHidden():
                nodePath.show()
            else:
                nodePath.hide()

    def removeNodePath(self, nodePath='None Given'):
        if False:
            print('Hello World!')
        if nodePath == 'None Given':
            nodePath = self.selected.last
        if nodePath:
            nodePath.removeNode()

    def removeAllSelected(self):
        if False:
            print('Hello World!')
        self.selected.removeAll()

    def showAllDescendants(self, nodePath=None):
        if False:
            while True:
                i = 10
        ' Show the level and its descendants '
        if nodePath is None:
            nodePath = base.render
        if not isinstance(nodePath, CollisionNode):
            nodePath.show()
        for child in nodePath.getChildren():
            self.showAllDescendants(child)

    def upAncestry(self):
        if False:
            for i in range(10):
                print('nop')
        if self.ancestry:
            l = len(self.ancestry)
            i = self.ancestryIndex + 1
            if i < l:
                np = self.ancestry[i]
                name = np.getName()
                if name != 'render' and name != 'renderTop':
                    self.ancestryIndex = i
                    self.select(np, 0, 0)
                    self.flash(np)

    def downAncestry(self):
        if False:
            while True:
                i = 10
        if self.ancestry:
            l = len(self.ancestry)
            i = self.ancestryIndex - 1
            if i >= 0:
                np = self.ancestry[i]
                name = np.getName()
                if name != 'render' and name != 'renderTop':
                    self.ancestryIndex = i
                    self.select(np, 0, 0)
                    self.flash(np)

    def getAndSetName(self, nodePath):
        if False:
            return 10
        ' Prompt user for new node path name '
        from tkinter.simpledialog import askstring
        newName = askstring('Node Path: ' + nodePath.getName(), 'Enter new name:')
        if newName:
            nodePath.setName(newName)
            messenger.send('DIRECT_nodePathSetName', [nodePath, newName])

    def pushUndo(self, nodePathList, fResetRedo=1):
        if False:
            return 10
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
            while True:
                i = 10
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
            while True:
                i = 10
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
            nodePathList = [x[0] for x in undoGroup]
            self.pushRedo(nodePathList)
            for pose in undoGroup:
                pose[0].setTransform(pose[1])
            messenger.send('DIRECT_undo', [nodePathList])

    def redo(self):
        if False:
            i = 10
            return i + 15
        if self.redoList:
            redoGroup = self.popRedoGroup()
            nodePathList = [x[0] for x in redoGroup]
            self.pushUndo(nodePathList, fResetRedo=0)
            for pose in redoGroup:
                pose[0].setTransform(pose[1])
            messenger.send('DIRECT_redo', [nodePathList])

    def message(self, text):
        if False:
            while True:
                i = 10
        taskMgr.remove('hideDirectMessage')
        taskMgr.remove('hideDirectMessageLater')
        self.directMessageReadout.reparentTo(base.a2dTopLeft)
        self.directMessageReadout.setText(text)
        self.hideDirectMessageLater()

    def hideDirectMessageLater(self):
        if False:
            i = 10
            return i + 15
        taskMgr.doMethodLater(3.0, self.hideDirectMessage, 'hideDirectMessage')

    def hideDirectMessage(self, state):
        if False:
            return 10
        self.directMessageReadout.reparentTo(hidden)
        return Task.done

    def useObjectHandles(self):
        if False:
            while True:
                i = 10
        self.widget = self.manipulationControl.objectHandles
        self.widget.reparentTo(self.group)

    def hideSelectedNPReadout(self):
        if False:
            i = 10
            return i + 15
        self.selectedNPReadout.reparentTo(hidden)

    def hideActiveParentReadout(self):
        if False:
            print('Hello World!')
        self.activeParentReadout.reparentTo(hidden)

    def toggleWidgetVis(self):
        if False:
            return 10
        self.widget.toggleWidget()

    def setCOAMode(self, mode):
        if False:
            for i in range(10):
                print('nop')
        self.coaMode = mode

    def isEnabled(self):
        if False:
            while True:
                i = 10
        return self.fEnabled

    def addUnpickable(self, item):
        if False:
            for i in range(10):
                print('nop')
        for iRay in self.iRayList:
            iRay.addUnpickable(item)

    def removeUnpickable(self, item):
        if False:
            for i in range(10):
                print('nop')
        for iRay in self.iRayList:
            iRay.removeUnpickable(item)

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
        self.orthoFactor = 0.1
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
            while True:
                i = 10
        hpr = self.cam.getHpr()
        if hpr[2] < 135 and hpr[2] > 45 or (hpr[2] > 225 and hpr[2] < 315):
            self.isSideways = 1
        elif hpr[2] > -135 and hpr[2] < -45 or (hpr[2] < -225 and hpr[2] > -315):
            self.isSideways = 1
        else:
            self.isSideways = 0

    def getHfov(self):
        if False:
            i = 10
            return i + 15
        if self.isSideways:
            return self.camLens.getVfov()
        else:
            return self.camLens.getHfov()

    def getVfov(self):
        if False:
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        if self.isSideways:
            self.camLens.setFov(vfov, self.camLens.getVfov())
        else:
            self.camLens.setFov(self.camLens.getHfov(), vfov)

    def setFov(self, hfov, vfov):
        if False:
            i = 10
            return i + 15
        if self.isSideways:
            self.camLens.setFov(vfov, hfov)
        else:
            self.camLens.setFov(hfov, vfov)

    def getWidth(self):
        if False:
            return 10
        prop = ShowBaseGlobal.direct.win.getProperties()
        if prop.hasSize():
            return prop.getXSize()
        else:
            return 640

    def getHeight(self):
        if False:
            print('Hello World!')
        prop = ShowBaseGlobal.direct.win.getProperties()
        if prop.hasSize():
            return prop.getYSize()
        else:
            return 480

    def updateFilmSize(self, width, height):
        if False:
            i = 10
            return i + 15
        if self.camLens.__class__.__name__ == 'OrthographicLens':
            width *= self.orthoFactor
            height *= self.orthoFactor
        self.camLens.setFilmSize(width, height)

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
            while True:
                i = 10
        self.mouseLastX = self.mouseX
        self.mouseLastY = self.mouseY
        mouseWatcherNode = base.mouseWatcherNode
        if mouseWatcherNode and mouseWatcherNode.hasMouse():
            self.mouseX = mouseWatcherNode.getMouseX()
            self.mouseY = mouseWatcherNode.getMouseY()
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
        if getattr(base, 'oobeMode', False):
            drc = DisplayRegionContext(base.cam)
            self.displayRegionList.append(drc)
        else:
            for (camIndex, cam) in enumerate(base.camList):
                if cam.name == '<noname>':
                    cam.name = f'Camera{camIndex}'
                drc = DisplayRegionContext(cam)
                self.displayRegionList.append(drc)
        self.accept('DIRECT-mouse1', self.mouseUpdate)
        self.accept('DIRECT-mouse2', self.mouseUpdate)
        self.accept('DIRECT-mouse3', self.mouseUpdate)
        self.accept('DIRECT-mouse1Up', self.mouseUpdate)
        self.accept('DIRECT-mouse2Up', self.mouseUpdate)
        self.accept('DIRECT-mouse3Up', self.mouseUpdate)
        self.tryToGetCurrentDr = True

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        return self.displayRegionList[index]

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self.displayRegionList)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.displayRegionList)

    def updateContext(self):
        if False:
            while True:
                i = 10
        self.contextTask(None)

    def setNearFar(self, near, far):
        if False:
            for i in range(10):
                print('nop')
        for dr in self.displayRegionList:
            dr.camLens.setNearFar(near, far)

    def setNear(self, near):
        if False:
            print('Hello World!')
        for dr in self.displayRegionList:
            dr.camLens.setNear(near)

    def setFar(self, far):
        if False:
            print('Hello World!')
        for dr in self.displayRegionList:
            dr.camLens.setFar(far)

    def setFov(self, hfov, vfov):
        if False:
            print('Hello World!')
        for dr in self.displayRegionList:
            dr.setFov(hfov, vfov)

    def setHfov(self, fov):
        if False:
            for i in range(10):
                print('nop')
        for dr in self.displayRegionList:
            dr.setHfov(fov)

    def setVfov(self, fov):
        if False:
            for i in range(10):
                print('nop')
        for dr in self.displayRegionList:
            dr.setVfov(fov)

    def mouseUpdate(self, modifiers=DG.DIRECT_NO_MOD):
        if False:
            while True:
                i = 10
        for dr in self.displayRegionList:
            dr.mouseUpdate()

    def getCurrentDr(self):
        if False:
            i = 10
            return i + 15
        if not self.tryToGetCurrentDr:
            return ShowBaseGlobal.direct.dr
        for dr in self.displayRegionList:
            if dr.mouseX >= -1.0 and dr.mouseX <= 1.0 and (dr.mouseY >= -1.0) and (dr.mouseY <= 1.0):
                return dr
        return self.displayRegionList[0]

    def start(self):
        if False:
            i = 10
            return i + 15
        self.stop()
        self.spawnContextTask()

    def stop(self):
        if False:
            for i in range(10):
                print('nop')
        taskMgr.remove('DIRECTContextTask')

    def spawnContextTask(self):
        if False:
            i = 10
            return i + 15
        taskMgr.add(self.contextTask, 'DIRECTContextTask')

    def removeContextTask(self):
        if False:
            i = 10
            return i + 15
        taskMgr.remove('DIRECTContextTask')

    def contextTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.mouseUpdate()
        return Task.cont

    def addDisplayRegionContext(self, cam):
        if False:
            print('Hello World!')
        self.displayRegionList.append(DisplayRegionContext(cam))

    def removeDisplayRegionContext(self, cam):
        if False:
            while True:
                i = 10
        for drc in self.displayRegionList:
            if drc.cam == cam:
                self.displayRegionList.remove(drc)
                break