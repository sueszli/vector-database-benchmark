import sys
try:
    import _tkinter
except:
    sys.exit("Please install python module 'Tkinter'")
from direct.showbase.ShowBase import ShowBase
ShowBase()
from direct.showbase.TkGlobal import spawnTkLoop
from tkinter import *
from tkinter.filedialog import *
from direct.directtools.DirectGlobals import *
from direct.tkwidgets.AppShell import *
from SideWindow import *
from duplicateWindow import *
from lightingPanel import *
from seMopathRecorder import *
from seSession import *
from quad import *
from sePlacer import *
from seFileSaver import *
from propertyWindow import *
import seParticlePanel
from collisionWindow import *
from direct.gui.DirectGui import *
from MetadataPanel import *
from seBlendAnimPanel import *
from controllerWindow import *
from AlignTool import *
from direct.tkwidgets import Dial
from direct.tkwidgets import Floater
from direct.tkwidgets import Slider
from direct.actor import Actor
import seAnimPanel
from direct.task import Task
from dataHolder import *
AllScene = dataHolder()

class myLevelEditor(AppShell):
    appname = 'Scene Editor - New Scene'
    appversion = '1.0'
    copyright = 'Copyright 2004 E.T.C. Carnegie Mellon U.' + ' All Rights Reserved'
    contactname = 'Jesse Schell, Shalin Shodhan & YiHong Lin'
    contactphone = '(412) 268-5791'
    contactemail = 'etc-panda3d@lists.andrew.cmu.edu'
    frameWidth = 1024
    frameHeight = 80
    frameIniPosX = 0
    frameIniPosY = 0
    usecommandarea = 0
    usestatusarea = 0
    padx = 5
    pady = 5
    sideWindowCount = 0
    worldColor = [0, 0, 0, 0]
    lightEnable = 1
    ParticleEnable = 1
    basedriveEnable = 0
    collision = 1
    backface = 0
    texture = 1
    wireframe = 0
    grid = 0
    widgetVis = 0
    enableAutoCamera = 1
    enableControl = False
    controlType = 'Keyboard'
    keyboardMapDict = {}
    keyboardSpeedDict = {}
    Scene = None
    isSelect = False
    nodeSelected = None
    undoDic = {}
    redoDic = {}
    animPanel = {}
    animBlendPanel = {}
    propertyWindow = {}
    CurrentFileName = None
    CurrentDirName = None
    Dirty = 0

    def __init__(self, parent=None, **kw):
        if False:
            return 10
        base.setBackgroundColor(0, 0, 0)
        self.parent = parent
        self.wantTK = config.GetBool('want-tk', 0)
        if self.wantTK:
            pass
        else:
            taskMgr.remove('tkloop')
            spawnTkLoop()
        INITOPT = Pmw.INITOPT
        optiondefs = (('title', self.appname, None),)
        self.defineoptions(kw, optiondefs)
        AppShell.__init__(self, parent)
        self.parent.geometry('%dx%d+%d+%d' % (self.frameWidth, self.frameHeight, self.frameIniPosX, self.frameIniPosY))
        self.posLabel = DirectLabel(relief=None, pos=(-1.3, 0, 0.9), text='Position   : X: 00.00 Y: 00.00 Z: 00.00', color=Vec4(1, 1, 1, 1), text_scale=0.05, text_align=TextNode.ALeft)
        self.hprLabel = DirectLabel(relief=None, pos=(-1.3, 0, 0.8), text='Orientation: H: 00.00 P: 00.00 R: 00.00', color=Vec4(1, 1, 1, 1), text_scale=0.05, text_align=TextNode.ALeft)
        self.scaleLabel = DirectLabel(relief=None, pos=(-1.3, 0, 0.7), text='Scale      : X: 00.00 Y: 00.00 Z: 00.00', color=Vec4(1, 1, 1, 1), text_scale=0.05, text_align=TextNode.ALeft)
        self.initialiseoptions(myLevelEditor)
        self.parent.resizable(False, False)
        self.dataFlowEvents = [['SW_lightToggle', self.lightToggle], ['SW_collisionToggle', AllScene.toggleCollisionVisable], ['SW_particleToggle', self.toggleParticleVisable], ['SW_close', self.sideWindowClose], ['DW_duplicating', self.duplicationObj], ['AW_AnimationLoad', self.animationLoader], ['AW_removeAnim', self.animationRemove], ['AW_close', self.animPanelClose], ['BAW_saveBlendAnim', self.animBlendPanelSave], ['BAW_removeBlendAnim', self.animBlendPanelRemove], ['BAW_renameBlendAnim', self.animBlendPanelRename], ['BAW_close', self.animBlendPanelClose], ['LP_selectLight', self.lightSelect], ['LP_addLight', self.addLight], ['LP_rename', self.lightRename], ['LP_removeLight', self.removeLight], ['LP_close', self.lightingPanelClose], ['mPath_bindPathToNode', AllScene.bindCurveToNode], ['mPath_requestCurveList', self.requestCurveList], ['mPath_close', self.mopathClosed], ['PW_removeCurveFromNode', AllScene.removeCurveFromNode], ['PW_removeAnimFromNode', AllScene.removeAnimation], ['PW_toggleLight', AllScene.toggleLightNode], ['PW_close', self.closePropertyWindow], ['CW_addCollisionObj', AllScene.addCollisionObject], ['ALW_close', self.closeAlignPanel], ['ALW_align', self.alignObject], ['ControlW_close', self.closeInputPanel], ['ControlW_require', self.requestObjFromControlW], ['ControlW_controlSetting', self.setControlSet], ['ControlW_controlEnable', self.startControl], ['ControlW_controlDisable', self.stopControl], ['ControlW_saveSetting', AllScene.saveControlSetting], ['Placer_close', self.closePlacerPanel], ['ParticlePanle_close', self.closeParticlePanel], ['SEditor-ToggleWidgetVis', self.toggleWidgetVis], ['SEditor-ToggleBackface', self.toggleBackface], ['SEditor-ToggleTexture', self.toggleTexture], ['SEditor-ToggleWireframe', self.toggleWireframe], ['ParticlePanel_Added_Effect', self.addParticleEffect], ['f11', self.loadFromBam], ['f12', self.saveAsBam]]
        self.cTrav = CollisionTraverser()
        base.cTrav = self.cTrav
        for event in self.dataFlowEvents:
            self.accept(event[0], event[1], extraArgs=event[2:])
        self.actionEvents = [['SGE_changeName', self.changeName], ['SGE_Properties', self.openPropertyPanel], ['SGE_Duplicate', self.duplicate], ['SGE_Remove', self.remove], ['SGE_Add Dummy', self.addDummyNode], ['SGE_Add Collision Object', self.addCollisionObj], ['SGE_Metadata', self.openMetadataPanel], ['SGE_Set as Reparent Target', self.setAsReparentTarget], ['SGE_Reparent to Target', self.reparentToNode], ['SGE_Animation Panel', self.openAnimPanel], ['SGE_Blend Animation Panel', self.openBlendAnimPanel], ['SGE_MoPath Panel', self.openMoPathPanel], ['SGE_Align Tool', self.openAlignPanel], ['SGE_Flash', self.flash], ['SGE_madeSelection', self.selectNode], ['select', self.selectNode], ['deselect', self.deSelectNode], ['se_selectedNodePath', self.selectFromScene], ['se_deselectedAll', self.deselectFromScene]]
        ' All messages starting with "SGE_" are generated in seSceneGraphExplorer'
        for event in self.actionEvents:
            self.accept(event[0], event[1], extraArgs=event[2:])
        if camera.is_hidden():
            camera.show()
        else:
            camera.hide()
        self.selectNode(base.camera)

    def appInit(self):
        if False:
            return 10
        self.seSession = SeSession()
        self.seSession.enable()
        SEditor.camera.setPos(0, -50, 10)
        self.placer = None
        self.MopathPanel = None
        self.alignPanelDict = {}
        self.lightingPanel = None
        self.controllerPanel = None
        self.particlePanel = None
        self.sideWindow = sideWindow(worldColor=self.worldColor, lightEnable=self.lightEnable, ParticleEnable=self.ParticleEnable, basedriveEnable=self.basedriveEnable, collision=self.collision, backface=self.backface, texture=self.texture, wireframe=self.wireframe, grid=self.grid, widgetVis=self.widgetVis, enableAutoCamera=self.enableAutoCamera)
        self.sideWindowCount = 1
        self.sideWindow.selectPage()
        messenger.send('SGE_Update Explorer', [render])
        pass

    def getPhotoImage(self, name):
        if False:
            return 10
        modpath = ConfigVariableSearchPath('model-path')
        path = modpath.findFile(Filename(name))
        return PhotoImage(file=path.toOsSpecific())

    def createInterface(self):
        if False:
            while True:
                i = 10
        interior = self.interior()
        buttonFrame = Frame(interior)
        self.image = []
        self.image.append(self.getPhotoImage('models/icons/new.gif'))
        self.image.append(self.getPhotoImage('models/icons/open.gif'))
        self.image.append(self.getPhotoImage('models/icons/save.gif'))
        self.image.append(self.getPhotoImage('models/icons/model.gif'))
        self.image.append(self.getPhotoImage('models/icons/actor.gif'))
        self.image.append(self.getPhotoImage('models/icons/placer.gif'))
        self.image.append(self.getPhotoImage('models/icons/mopath.gif'))
        self.image.append(self.getPhotoImage('models/icons/lights.gif'))
        self.image.append(self.getPhotoImage('models/icons/particles.gif'))
        self.image.append(self.getPhotoImage('models/icons/control.gif'))
        self.image.append(self.getPhotoImage('models/icons/help.gif'))
        self.image.append(self.getPhotoImage('models/icons/blank.gif'))
        self.image.append(self.getPhotoImage('models/icons/blank.gif'))
        self.image.append(self.getPhotoImage('models/icons/blank.gif'))
        self.image.append(self.getPhotoImage('models/icons/blank.gif'))
        self.image.append(self.getPhotoImage('models/icons/blank.gif'))
        self.image.append(self.getPhotoImage('models/icons/blank.gif'))
        self.image.append(self.getPhotoImage('models/icons/blank.gif'))
        self.image.append(self.getPhotoImage('models/icons/blank.gif'))
        self.image.append(self.getPhotoImage('models/icons/blank.gif'))
        self.image.append(self.getPhotoImage('models/icons/blank.gif'))
        self.image.append(self.getPhotoImage('models/icons/blank.gif'))
        self.image.append(self.getPhotoImage('models/icons/blank.gif'))
        i = 0
        for element in self.image:
            i += 1
            button = Button(buttonFrame, image=element, command=lambda n=i: self.buttonPushed(n))
            button.pack(fill=X, side=LEFT)
        buttonFrame.pack(fill=X, side=LEFT, expand=True)

    def buttonPushed(self, buttonIndex):
        if False:
            return 10
        if buttonIndex == 1:
            self.newScene()
            return
        elif buttonIndex == 2:
            self.openScene()
            return
        elif buttonIndex == 3:
            self.saveScene()
            return
        elif buttonIndex == 4:
            self.loadModel()
            return
        elif buttonIndex == 5:
            self.loadActor()
            return
        elif buttonIndex == 6:
            self.openPlacerPanel()
            return
        elif buttonIndex == 7:
            self.openMoPathPanel()
            return
        elif buttonIndex == 8:
            self.openLightingPanel()
            return
        elif buttonIndex == 9:
            self.openParticlePanel()
            return
        elif buttonIndex == 10:
            self.openInputPanel()
            return
        elif buttonIndex == 11:
            self.showAbout()
            return
        elif buttonIndex == 12:
            print("You haven't defined the function for this Button, Number %d." % buttonIndex)
            return
        elif buttonIndex == 13:
            print("You haven't defined the function for this Button, Number %d." % buttonIndex)
            return
        elif buttonIndex == 14:
            print("You haven't defined the function for this Button, Number %d." % buttonIndex)
            return
        elif buttonIndex == 15:
            print("You haven't defined the function for this Button, Number %d." % buttonIndex)
            return
        elif buttonIndex == 16:
            print('Your scene will be eliminated within five seconds, Save your world!!!, Number %d.' % buttonIndex)
            return
        elif buttonIndex == 17:
            print("You haven't defined the function for this Button, Number %d." % buttonIndex)
            return
        elif buttonIndex == 18:
            print("You haven't defined the function for this Button, Number %d." % buttonIndex)
            return
        elif buttonIndex == 19:
            print("You haven't defined the function for this Button, Number %d." % buttonIndex)
            return
        elif buttonIndex == 20:
            print("You haven't defined the function for this Button, Number %d." % buttonIndex)
            return
        return

    def createMenuBar(self):
        if False:
            while True:
                i = 10
        self.menuBar.addmenuitem('Help', 'command', 'Get information on application', label='About...', command=self.showAbout)
        self.menuBar.addmenuitem('File', 'command', 'Creat New Scene', label='New Scene', command=self.newScene)
        self.menuBar.addmenuitem('File', 'command', 'Open a Scene', label='Open Scene', command=self.openScene)
        self.menuBar.addmenuitem('File', 'command', 'Save a Scene', label='Save Scene', command=self.saveScene)
        self.menuBar.addmenuitem('File', 'command', 'Save Scene as...', label='Save as...', command=self.saveAsScene)
        self.menuBar.addmenuitem('File', 'separator')
        self.menuBar.addmenuitem('File', 'command', 'Load Model', label='Load Model', command=self.loadModel)
        self.menuBar.addmenuitem('File', 'command', 'Load Actor', label='Load Actor', command=self.loadActor)
        self.menuBar.addmenuitem('File', 'separator')
        self.menuBar.addmenuitem('File', 'command', 'Import a Scene', label='Import...', command=self.importScene)
        self.menuBar.addmenuitem('File', 'separator')
        self.menuBar.addmenuitem('File', 'command', 'Quit this application', label='Exit', command=self.quit)
        self.menuBar.addmenu('Edit', 'Editting tools')
        self.menuBar.addmenuitem('Edit', 'command', 'Un-do', label='Undo...', command=self.unDo)
        self.menuBar.addmenuitem('Edit', 'command', 'Re-do', label='Redo...', command=self.reDo)
        self.menuBar.addmenuitem('Edit', 'separator')
        self.menuBar.addmenuitem('Edit', 'command', 'Deselect nodepath', label='Deselect', command=self.deSelectNode)
        self.menuBar.addmenuitem('Edit', 'separator')
        self.menuBar.addmenuitem('Edit', 'command', 'Add a Dummy', label='Add Dummy', command=self.addDummy)
        self.menuBar.addmenuitem('Edit', 'command', 'Duplicate nodepath', label='Duplicate', command=self.duplicateNode)
        self.menuBar.addmenuitem('Edit', 'command', 'Remove the nodepath', label='Remove', command=self.removeNode)
        self.menuBar.addmenuitem('Edit', 'command', 'Show the object properties', label='Object Properties', command=self.showObjProp)
        self.menuBar.addmenuitem('Edit', 'separator')
        self.menuBar.addmenuitem('Edit', 'command', 'Show the Camera setting', label='Camera Setting', command=self.showCameraSetting)
        self.menuBar.addmenuitem('Edit', 'command', 'Render setting', label='Render Setting', command=self.showRenderSetting)
        self.menuBar.addmenu('Panel', 'Panel tools')
        self.menuBar.addmenuitem('Panel', 'command', 'Open Side Window', label='Side Window', command=self.openSideWindow)
        self.menuBar.addmenuitem('Panel', 'command', 'Placer Panel', label='Placer Panel', command=self.openPlacerPanel)
        self.menuBar.addmenuitem('Panel', 'command', 'Animation Panel', label='Animation Panel', command=self.openAnimationPanel)
        self.menuBar.addmenuitem('Panel', 'command', 'Motion Path Panel', label='Mopath Panel', command=self.openMopathPanel)
        self.menuBar.addmenuitem('Panel', 'command', 'Lighting Panel', label='Lighting Panel', command=self.openLightingPanel)
        self.menuBar.addmenuitem('Panel', 'command', 'Particle Panel', label='Particle Panel', command=self.openParticlePanel)
        self.menuBar.addmenuitem('Panel', 'separator')
        self.menuBar.addmenuitem('Panel', 'command', 'Input control Panel', label='Input device panel', command=self.openInputPanel)
        self.menuBar.pack(fill=X, side=LEFT)
        self.menuFile = self.menuBar.component('File-menu')
        self.menuEdit = self.menuBar.component('Edit-menu')
        self.menuPanel = self.menuBar.component('Panel-menu')
        if not self.isSelect:
            self.menuEdit.entryconfig('Deselect', state=DISABLED)
            self.menuEdit.entryconfig('Add Dummy', state=DISABLED)
            self.menuEdit.entryconfig('Duplicate', state=DISABLED)
            self.menuEdit.entryconfig('Remove', state=DISABLED)
            self.menuEdit.entryconfig('Object Properties', state=DISABLED)
            self.menuPanel.entryconfig('Animation Panel', state=DISABLED)
            self.menuPanel.entryconfig('Side Window', state=DISABLED)

    def onDestroy(self, event):
        if False:
            while True:
                i = 10
        if taskMgr.hasTaskNamed('seMonitorSelectedNode'):
            taskMgr.remove('seMonitorSelectedNode')
        pass

    def closeAllSubWindows(self):
        if False:
            print('Hello World!')
        if self.lightingPanel != None:
            self.lightingPanel.quit()
        if self.placer != None:
            self.placer.quit()
        if self.MopathPanel != None:
            self.MopathPanel.quit()
        if self.particlePanel != None:
            self.particlePanel.quit()
        if self.controllerPanel != None:
            self.controllerPanel.quit()
        list = self.animPanel.keys()
        for index in list:
            self.animPanel[index].quit()
        list = self.animBlendPanel.keys()
        for index in list:
            self.animBlendPanel[index].quit()
        list = self.propertyWindow.keys()
        for index in list:
            self.propertyWindow[index].quit()
        list = self.alignPanelDict.keys()
        for index in list:
            self.alignPanelDict[index].quit()
        self.animPanel.clear()
        self.animBlendPanel.clear()
        self.propertyWindow.clear()
        self.alignPanelDict.clear()
        return

    def makeDirty(self):
        if False:
            while True:
                i = 10
        self.Dirty = 1

    def removeLight(self, lightNode):
        if False:
            print('Hello World!')
        list = AllScene.removeObj(lightNode)
        if self.lightingPanel != None:
            self.lightingPanel.updateList(list)
        return

    def lightRename(self, oName, nName):
        if False:
            print('Hello World!')
        (list, lightNode) = AllScene.rename(oName, nName)
        if self.lightingPanel != None:
            self.lightingPanel.updateList(list, lightNode)
        return

    def lightSelect(self, lightName):
        if False:
            print('Hello World!')
        lightNode = AllScene.getLightNode(lightName)
        if self.lightingPanel != None:
            self.lightingPanel.updateDisplay(lightNode)
        return

    def addLight(self, type):
        if False:
            while True:
                i = 10
        (list, lightNode) = AllScene.createLight(type=type)
        if self.lightingPanel != None:
            self.lightingPanel.updateList(list, lightNode)
        self.makeDirty()
        return

    def lightingPanelClose(self):
        if False:
            return 10
        self.menuPanel.entryconfig('Lighting Panel', state=NORMAL)
        self.lightingPanel = None
        return

    def openPropertyPanel(self, nodePath=None):
        if False:
            while True:
                i = 10
        (type, info) = AllScene.getInfoOfThisNode(nodePath)
        name = nodePath.getName()
        if name not in self.propertyWindow:
            self.propertyWindow[name] = propertyWindow(nodePath, type, info)
        pass

    def closePropertyWindow(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name in self.propertyWindow:
            del self.propertyWindow[name]
        return

    def openMetadataPanel(self, nodePath=None):
        if False:
            for i in range(10):
                print('nop')
        print(nodePath)
        self.MetadataPanel = MetadataPanel(nodePath)
        pass

    def duplicate(self, nodePath=None):
        if False:
            return 10
        print('----Duplication!!')
        if nodePath != None:
            self.duplicateWindow = duplicateWindow(nodePath=nodePath)
        pass

    def remove(self, nodePath=None):
        if False:
            for i in range(10):
                print('nop')
        if nodePath == None:
            if self.nodeSelected == None:
                return
            nodePath = self.nodeSelected
        self.deSelectNode()
        if AllScene.isLight(nodePath.getName()):
            self.removeLight(nodePath)
        else:
            AllScene.removeObj(nodePath)
        pass

    def addDummyNode(self, nodepath=None):
        if False:
            i = 10
            return i + 15
        AllScene.addDummyNode(nodepath)
        self.makeDirty()
        pass

    def addCollisionObj(self, nodepath=None):
        if False:
            i = 10
            return i + 15
        self.collisionWindow = collisionWindow(nodepath)
        pass

    def setAsReparentTarget(self, nodepath=None):
        if False:
            return 10
        SEditor.setActiveParent(nodepath)
        return

    def reparentToNode(self, nodepath=None):
        if False:
            i = 10
            return i + 15
        SEditor.reparent(nodepath, fWrt=1)
        return

    def openPlacerPanel(self, nodePath=None):
        if False:
            for i in range(10):
                print('nop')
        if self.placer == None:
            self.placer = Placer()
            self.menuPanel.entryconfig('Placer Panel', state=DISABLED)
        return

    def closePlacerPanel(self):
        if False:
            return 10
        self.placer = None
        self.menuPanel.entryconfig('Placer Panel', state=NORMAL)
        return

    def openAnimPanel(self, nodePath=None):
        if False:
            print('Hello World!')
        name = nodePath.getName()
        if AllScene.isActor(name):
            if name in self.animPanel:
                print('---- You already have an animation panel for this Actor!')
                return
            else:
                Actor = AllScene.getActor(name)
                self.animPanel[name] = seAnimPanel.AnimPanel(aNode=Actor)
                pass

    def openMoPathPanel(self, nodepath=None):
        if False:
            for i in range(10):
                print('nop')
        if self.MopathPanel == None:
            self.MopathPanel = MopathRecorder()
        pass

    def mopathClosed(self):
        if False:
            while True:
                i = 10
        self.MopathPanel = None
        return

    def changeName(self, nodePath, nName):
        if False:
            for i in range(10):
                print('nop')
        oName = nodePath.getName()
        AllScene.rename(nodePath, nName)
        if self.controllerPanel != None:
            list = AllScene.getAllObjNameAsList()
            self.controllerPanel.resetNameList(list=list, name=oName, nodePath=nodePath)
        return

    def newScene(self):
        if False:
            i = 10
            return i + 15
        self.closeAllSubWindows()
        if self.CurrentFileName:
            currentF = Filename(self.CurrentFileName)
            self.CurrentFileName = None
            AllScene.resetAll()
            currentModName = currentF.getBasenameWoExtension()
            if currentModName in sys.modules:
                del sys.modules[currentModName]
                print(sys.getrefcount(AllScene.theScene))
                del AllScene.theScene
        else:
            AllScene.resetAll()
        self.parent.title('Scene Editor - New Scene')
        pass

    def openScene(self):
        if False:
            for i in range(10):
                print('nop')
        if self.CurrentFileName or self.Dirty:
            saveScene = tkMessageBox._show('Load scene', 'Save the current scene?', icon=tkMessageBox.QUESTION, type=tkMessageBox.YESNOCANCEL)
            if saveScene == 'yes':
                self.saveScene()
            elif saveScene == 'cancel':
                return
        self.closeAllSubWindows()
        if self.CurrentFileName:
            currentF = Filename(self.CurrentFileName)
            AllScene.resetAll()
            currentModName = currentF.getBasenameWoExtension()
            if currentModName in sys.modules:
                del sys.modules[currentModName]
                print(sys.getrefcount(AllScene.theScene))
                del AllScene.theScene
        else:
            AllScene.resetAll()
        self.CurrentFileName = AllScene.loadScene()
        if self.CurrentFileName == None:
            return
        thefile = Filename(self.CurrentFileName)
        thedir = thefile.getFullpathWoExtension()
        print('SCENE EDITOR::' + thedir)
        self.CurrentDirName = thedir
        if self.CurrentFileName != None:
            self.parent.title('Scene Editor - ' + Filename.fromOsSpecific(self.CurrentFileName).getBasenameWoExtension())
        if self.lightingPanel != None:
            lightList = AllScene.getList()
            self.lightingPanel.updateList(lightList)
        messenger.send('SGE_Update Explorer', [render])
        self.sideWindow.quit()
        while self.sideWindow == None:
            wColor = base.getBackgroundColor()
            self.worldColor[0] = wColor.getX()
            self.worldColor[1] = wColor.getY()
            self.worldColor[2] = wColor.getZ()
            self.worldColor[3] = wColor.getW()
            self.lightEnable = 1
            self.ParticleEnable = 1
            self.collision = 1
            self.openSideWindow()

    def saveScene(self):
        if False:
            print('Hello World!')
        if self.CurrentFileName:
            f = FileSaver()
            f.SaveFile(AllScene, self.CurrentFileName, self.CurrentDirName, 1)
            self.Dirty = 0
        else:
            self.saveAsScene()
        pass

    def saveAsBam(self):
        if False:
            i = 10
            return i + 15
        fileName = tkFileDialog.asksaveasfilename(filetypes=[('BAM', '.bam')], title='Save Scenegraph as Bam file')
        theScene = render.find('**/Scene')
        if not theScene is None:
            theScene.writeBamFile(fileName)
        else:
            render.writeBamFile(fileName + '.bad')
        print(' Scenegraph saved as :' + str(fileName))

    def loadFromBam(self):
        if False:
            return 10
        fileName = tkFileDialog.askopenfilename(filetypes=[('BAM', '.bam')], title='Load Scenegraph from Bam file')
        if not fileName is None:
            d = path(fileName)
            scene = loader.loadModel(d.relpath())
            scene.reparentTo(render)

    def saveAsScene(self):
        if False:
            for i in range(10):
                print('nop')
        fileName = tkFileDialog.asksaveasfilename(filetypes=[('PY', 'py')], title='Save Scene')
        if not fileName:
            return
        fCheck = Filename(fileName)
        if fCheck.getBasenameWoExtension() in sys.modules:
            tkMessageBox.showwarning('Save file', 'Cannot save with this name because there is a system module with the same name. Please resave as something else.')
            return
        self.CurrentDirName = fileName
        fileName = fileName + '.py'
        f = FileSaver()
        self.CurrentFileName = fileName
        f.SaveFile(AllScene, fileName, self.CurrentDirName, 0)
        self.Dirty = 0
        self.parent.title('Scene Editor - ' + Filename.fromOsSpecific(self.CurrentFileName).getBasenameWoExtension())
        pass

    def loadModel(self):
        if False:
            i = 10
            return i + 15
        modelFilename = askopenfilename(defaultextension='.egg', filetypes=(('Egg Files', '*.egg'), ('Bam Files', '*.bam'), ('All files', '*')), initialdir='.', title='Load New Model', parent=self.parent)
        if modelFilename:
            self.makeDirty()
            if not AllScene.loadModel(modelFilename, Filename.fromOsSpecific(modelFilename)):
                print('----Error! No Such Model File!')
        pass

    def loadActor(self):
        if False:
            print('Hello World!')
        ActorFilename = askopenfilename(defaultextension='.egg', filetypes=(('Egg Files', '*.egg'), ('Bam Files', '*.bam'), ('All files', '*')), initialdir='.', title='Load New Actor', parent=self.parent)
        if ActorFilename:
            self.makeDirty()
            if not AllScene.loadActor(ActorFilename, Filename.fromOsSpecific(ActorFilename)):
                print('----Error! No Such Model File!')
        pass

    def importScene(self):
        if False:
            for i in range(10):
                print('nop')
        self.makeDirty()
        print('----God bless you Please Import!')
        pass

    def unDo(self):
        if False:
            print('Hello World!')
        pass

    def reDo(self):
        if False:
            return 10
        pass

    def deSelectNode(self, nodePath=None):
        if False:
            for i in range(10):
                print('nop')
        if nodePath != None:
            self.seSession.deselect(nodePath)
        if self.isSelect:
            self.isSelect = False
            self.nodeSelected = None
            self.menuEdit.entryconfig('Deselect', state=DISABLED)
            self.menuEdit.entryconfig('Add Dummy', state=DISABLED)
            self.menuEdit.entryconfig('Duplicate', state=DISABLED)
            self.menuEdit.entryconfig('Remove', state=DISABLED)
            self.menuEdit.entryconfig('Object Properties', state=DISABLED)
            if self.sideWindowCount == 1:
                self.sideWindow.SGE.deSelectTree()
            if taskMgr.hasTaskNamed('seMonitorSelectedNode'):
                taskMgr.remove('seMonitorSelectedNode')
            return
        pass

    def addDummy(self):
        if False:
            for i in range(10):
                print('nop')
        self.addDummyNode(self.nodeSelected)
        pass

    def duplicateNode(self):
        if False:
            while True:
                i = 10
        if self.nodeSelected != None:
            self.duplicate(self.nodeSelected)
        pass

    def removeNode(self):
        if False:
            i = 10
            return i + 15
        self.remove(self.nodeSelected)
        pass

    def showObjProp(self):
        if False:
            while True:
                i = 10
        self.openPropertyPanel(self.nodeSelected)
        pass

    def showCameraSetting(self):
        if False:
            return 10
        self.openPropertyPanel(camera)
        pass

    def showRenderSetting(self):
        if False:
            return 10
        'Currently, no idea what gonna pop-out here...'
        pass

    def openSideWindow(self):
        if False:
            while True:
                i = 10
        if self.sideWindowCount == 0:
            self.sideWindow = sideWindow(worldColor=self.worldColor, lightEnable=self.lightEnable, ParticleEnable=self.ParticleEnable, basedriveEnable=self.basedriveEnable, collision=self.collision, backface=self.backface, texture=self.texture, wireframe=self.wireframe, grid=self.grid, widgetVis=self.widgetVis, enableAutoCamera=self.enableAutoCamera)
            self.sideWindowCount = 1
            self.menuPanel.entryconfig('Side Window', state=DISABLED)
        return

    def openAnimationPanel(self):
        if False:
            return 10
        if AllScene.isActor(self.nodeSelected):
            self.openAnimPanel(self.nodeSelected)
        pass

    def openMopathPanel(self):
        if False:
            return 10
        MopathPanel = MopathRecorder()
        pass

    def toggleParticleVisable(self, visable):
        if False:
            i = 10
            return i + 15
        self.ParticleEnable = visable
        AllScene.toggleParticleVisable(visable)
        return

    def openLightingPanel(self):
        if False:
            print('Hello World!')
        if self.lightingPanel == None:
            self.lightingPanel = lightingPanel(AllScene.getLightList())
            self.menuPanel.entryconfig('Lighting Panel', state=DISABLED)
        return

    def addParticleEffect(self, effect_name, effect, node):
        if False:
            while True:
                i = 10
        AllScene.particleDict[effect_name] = effect
        AllScene.particleNodes[effect_name] = node
        if not self.ParticleEnable:
            AllScene.particleNodes[effect_name].setTransparency(True)
            AllScene.particleNodes[effect_name].setAlphaScale(0)
            AllScene.particleNodes[effect_name].setBin('fixed', 1)
        return

    def openParticlePanel(self):
        if False:
            print('Hello World!')
        if self.particlePanel != None:
            return
        if len(AllScene.particleDict) == 0:
            self.particlePanel = seParticlePanel.ParticlePanel()
        else:
            for effect in AllScene.particleDict:
                theeffect = AllScene.particleDict[effect]
            self.particlePanel = seParticlePanel.ParticlePanel(particleEffect=theeffect, effectsDict=AllScene.particleDict)
        pass

    def closeParticlePanel(self):
        if False:
            for i in range(10):
                print('nop')
        self.particlePanel = None
        return

    def openInputPanel(self):
        if False:
            print('Hello World!')
        if self.controllerPanel == None:
            list = AllScene.getAllObjNameAsList()
            (type, dataList) = AllScene.getControlSetting()
            self.controllerPanel = controllerWindow(listOfObj=list, controlType=type, dataList=dataList)
        pass

    def closeInputPanel(self):
        if False:
            i = 10
            return i + 15
        self.controllerPanel = None
        return

    def requestObjFromControlW(self, name):
        if False:
            return 10
        node = AllScene.getObjFromSceneByName(name)
        if self.controllerPanel != None and node != None:
            self.controllerPanel.setNodePathIn(node)
        return

    def setControlSet(self, controlType, dataList):
        if False:
            return 10
        if controlType == 'Keyboard':
            self.controlTarget = dataList[0]
            self.keyboardMapDict.clear()
            self.keyboardMapDict = dataList[1].copy()
            self.keyboardSpeedDict.clear()
            self.keyboardSpeedDict = dataList[2].copy()
        return

    def startControl(self, controlType, dataList):
        if False:
            print('Hello World!')
        if not self.enableControl:
            self.enableControl = True
        else:
            self.stopControl(controlType)
            self.enableControl = True
        self.setControlSet(controlType, dataList)
        self.lastContorlTimer = globalClock.getFrameTime()
        if controlType == 'Keyboard':
            self.controlType = 'Keyboard'
            self.keyControlEventDict = {}
            self.transNodeKeyboard = self.controlTarget.attachNewNode('transformNode')
            self.transNodeKeyboard.hide()
            for index in self.keyboardMapDict:
                self.keyControlEventDict[index] = 0
                self.accept(self.keyboardMapDict[index], lambda a=index: self.keyboardPushed(a))
                self.accept(self.keyboardMapDict[index] + '-up', lambda a=index: self.keyboardReleased(a))
        return

    def stopControl(self, controlType):
        if False:
            while True:
                i = 10
        if not self.enableControl:
            return
        if controlType == 'Keyboard':
            self.enableControl = False
            for index in self.keyboardMapDict:
                self.ignore(self.keyboardMapDict[index])
                self.ignore(self.keyboardMapDict[index] + '-up')
            taskMgr.remove('KeyboardControlTask')
            self.transNodeKeyboard.removeNode()
        return

    def keyboardPushed(self, key):
        if False:
            while True:
                i = 10
        self.keyControlEventDict[key] = 1
        if not taskMgr.hasTaskNamed('KeyboardControlTask'):
            self.keyboardLastTimer = globalClock.getFrameTime()
            taskMgr.add(self.keyboardControlTask, 'KeyboardControlTask')
        return

    def keyboardReleased(self, key):
        if False:
            while True:
                i = 10
        self.keyControlEventDict[key] = 0
        for index in self.keyControlEventDict:
            if self.keyControlEventDict[index] == 1:
                return
        if taskMgr.hasTaskNamed('KeyboardControlTask'):
            taskMgr.remove('KeyboardControlTask')
        return

    def keyboardControlTask(self, task):
        if False:
            while True:
                i = 10
        newTimer = globalClock.getFrameTime()
        delta = newTimer - self.keyboardLastTimer
        self.keyboardLastTimer = newTimer
        pos = self.controlTarget.getPos()
        hpr = self.controlTarget.getHpr()
        scale = self.controlTarget.getScale()
        self.transNodeKeyboard.setPosHpr((self.keyControlEventDict['KeyRight'] * self.keyboardSpeedDict['SpeedRight'] - self.keyControlEventDict['KeyLeft'] * self.keyboardSpeedDict['SpeedLeft']) * delta, (self.keyControlEventDict['KeyForward'] * self.keyboardSpeedDict['SpeedForward'] - self.keyControlEventDict['KeyBackward'] * self.keyboardSpeedDict['SpeedBackward']) * delta, (self.keyControlEventDict['KeyUp'] * self.keyboardSpeedDict['SpeedUp'] - self.keyControlEventDict['KeyDown'] * self.keyboardSpeedDict['SpeedDown']) * delta, (self.keyControlEventDict['KeyTurnLeft'] * self.keyboardSpeedDict['SpeedTurnLeft'] - self.keyControlEventDict['KeyTurnRight'] * self.keyboardSpeedDict['SpeedTurnRight']) * delta, (self.keyControlEventDict['KeyTurnUp'] * self.keyboardSpeedDict['SpeedTurnUp'] - self.keyControlEventDict['KeyTurnDown'] * self.keyboardSpeedDict['SpeedTurnDown']) * delta, (self.keyControlEventDict['KeyRollLeft'] * self.keyboardSpeedDict['SpeedRollLeft'] - self.keyControlEventDict['KeyRollRight'] * self.keyboardSpeedDict['SpeedRollRight']) * delta)
        newPos = self.transNodeKeyboard.getPos(self.controlTarget.getParent())
        newHpr = self.transNodeKeyboard.getHpr(self.controlTarget.getParent())
        overAllScale = self.keyControlEventDict['KeyScaleUp'] * self.keyboardSpeedDict['SpeedScaleUp'] - self.keyControlEventDict['KeyScaleDown'] * self.keyboardSpeedDict['SpeedScaleDown']
        newScale = Point3(scale.getX() + (overAllScale + self.keyControlEventDict['KeyScaleXUp'] * self.keyboardSpeedDict['SpeedScaleXUp'] - self.keyControlEventDict['KeyScaleXDown'] * self.keyboardSpeedDict['SpeedScaleXDown']) * delta, scale.getY() + (overAllScale + self.keyControlEventDict['KeyScaleYUp'] * self.keyboardSpeedDict['SpeedScaleYUp'] - self.keyControlEventDict['KeyScaleYDown'] * self.keyboardSpeedDict['SpeedScaleYDown']) * delta, scale.getZ() + (overAllScale + self.keyControlEventDict['KeyScaleZUp'] * self.keyboardSpeedDict['SpeedScaleZUp'] - self.keyControlEventDict['KeyScaleZDown'] * self.keyboardSpeedDict['SpeedScaleZDown']) * delta)
        self.controlTarget.setPos(newPos.getX(), newPos.getY(), newPos.getZ())
        self.controlTarget.setHpr(newHpr.getX(), newHpr.getY(), newHpr.getZ())
        self.controlTarget.setScale(newScale.getX(), newScale.getY(), newScale.getZ())
        self.transNodeKeyboard.setPosHpr(0, 0, 0, 0, 0, 0)
        return Task.cont

    def selectNode(self, nodePath=None, callBack=True):
        if False:
            while True:
                i = 10
        if nodePath == None:
            self.isSelect = False
            self.nodeSelected = None
            if taskMgr.hasTaskNamed('seMonitorSelectedNode'):
                taskMgr.remove('seMonitorSelectedNode')
            return
        else:
            self.isSelect = True
            self.nodeSelected = nodePath
            self.menuEdit.entryconfig('Deselect', state=NORMAL)
            self.menuEdit.entryconfig('Add Dummy', state=NORMAL)
            self.menuEdit.entryconfig('Duplicate', state=NORMAL)
            self.menuEdit.entryconfig('Remove', state=NORMAL)
            self.menuEdit.entryconfig('Object Properties', state=NORMAL)
            if callBack:
                self.seSession.select(nodePath, fResetAncestry=1)
            messenger.send('SGE_Update Explorer', [render])
            if not taskMgr.hasTaskNamed('seMonitorSelectedNode'):
                self.oPos = self.nodeSelected.getPos()
                self.oHpr = self.nodeSelected.getHpr()
                self.oScale = self.nodeSelected.getScale()
                taskMgr.add(self.monitorSelectedNodeTask, 'seMonitorSelectedNode')
            return
        pass

    def selectFromScene(self, nodePath=None, callBack=True):
        if False:
            i = 10
            return i + 15
        if nodePath == None:
            self.isSelect = False
            self.nodeSelected = None
            if taskMgr.hasTaskNamed('seMonitorSelectedNode'):
                taskMgr.remove('seMonitorSelectedNode')
            return
        else:
            self.isSelect = True
            self.nodeSelected = nodePath
            self.menuEdit.entryconfig('Deselect', state=NORMAL)
            self.menuEdit.entryconfig('Add Dummy', state=NORMAL)
            self.menuEdit.entryconfig('Duplicate', state=NORMAL)
            self.menuEdit.entryconfig('Remove', state=NORMAL)
            self.menuEdit.entryconfig('Object Properties', state=NORMAL)
            self.sideWindow.SGE.selectNodePath(nodePath, callBack)
            messenger.send('SGE_Update Explorer', [render])
            if not taskMgr.hasTaskNamed('seMonitorSelectedNode'):
                self.oPos = self.nodeSelected.getPos()
                self.oHpr = self.nodeSelected.getHpr()
                self.oScale = self.nodeSelected.getScale()
                taskMgr.add(self.monitorSelectedNodeTask, 'seMonitorSelectedNode')
            return
        pass

    def monitorSelectedNodeTask(self, task):
        if False:
            return 10
        if self.nodeSelected != None:
            pos = self.nodeSelected.getPos()
            hpr = self.nodeSelected.getHpr()
            scale = self.nodeSelected.getScale()
            if self.oPos != pos or self.oScale != scale or self.oHpr != hpr:
                messenger.send('forPorpertyWindow' + self.nodeSelected.getName(), [pos, hpr, scale])
                messenger.send('placerUpdate')
                self.oPos = pos
                self.oScale = scale
                self.oHpr = hpr
                self.posLabel['text'] = 'Position   : X: %2.2f Y: %2.2f Z: %2.2f' % (pos.getX(), pos.getY(), pos.getZ())
                self.hprLabel['text'] = 'Orientation: H: %2.2f P: %2.2f R: %2.2f' % (hpr.getX(), hpr.getY(), hpr.getZ())
                self.scaleLabel['text'] = 'Scale      : X: %2.2f Y: %2.2f Z: %2.2f' % (scale.getX(), scale.getY(), scale.getZ())
        return Task.cont

    def deselectFromScene(self):
        if False:
            i = 10
            return i + 15
        self.deSelectNode(self.nodeSelected)
        messenger.send('SGE_Update Explorer', [render])

    def lightToggle(self):
        if False:
            for i in range(10):
                print('nop')
        self.makeDirty()
        AllScene.toggleLight()
        return

    def sideWindowClose(self, worldColor, lightEnable, ParticleEnable, basedriveEnable, collision, backface, texture, wireframe, grid, widgetVis, enableAutoCamera):
        if False:
            return 10
        if self.sideWindowCount == 1:
            self.worldColor = worldColor
            self.lightEnable = lightEnable
            self.ParticleEnable = ParticleEnable
            self.basedriveEnable = basedriveEnable
            self.collision = collision
            self.backface = backface
            self.texture = texture
            self.wireframe = wireframe
            self.grid = grid
            self.enableAutoCamera = enableAutoCamera
            self.widgetVis = widgetVis
            self.sideWindowCount = 0
            self.sideWindow = None
            self.menuPanel.entryconfig('Side Window', state=NORMAL)
            return

    def duplicationObj(self, nodePath, pos, hpr, scale, num):
        if False:
            while True:
                i = 10
        AllScene.duplicateObj(nodePath, pos, hpr, scale, num)
        return

    def animationLoader(self, nodePath, Dic):
        if False:
            while True:
                i = 10
        name = nodePath.getName()
        AllScene.loadAnimation(name, Dic)
        return

    def animationRemove(self, nodePath, name):
        if False:
            print('Hello World!')
        AllScene.removeAnimation(nodePath.getName(), name)
        return

    def animPanelClose(self, name):
        if False:
            return 10
        if name in self.animPanel:
            del self.animPanel[name]
        return

    def openBlendAnimPanel(self, nodePath=None):
        if False:
            return 10
        name = nodePath.getName()
        if AllScene.isActor(name):
            if name in self.animBlendPanel:
                print('---- You already have an Blend Animation Panel for this Actor!')
                return
            else:
                Actor = AllScene.getActor(name)
                Dict = AllScene.getBlendAnimAsDict(name)
                self.animBlendPanel[name] = BlendAnimPanel(aNode=Actor, blendDict=Dict)
                pass
        return

    def animBlendPanelSave(self, actorName, blendName, animNameA, animNameB, effect):
        if False:
            return 10
        dict = AllScene.saveBlendAnim(actorName, blendName, animNameA, animNameB, effect)
        self.animBlendPanel[actorName].setBlendAnimList(dict)
        return

    def animBlendPanelRemove(self, actorName, blendName):
        if False:
            return 10
        dict = AllScene.removeBlendAnim(actorName, blendName)
        self.animBlendPanel[actorName].setBlendAnimList(dict, True)
        return

    def animBlendPanelRename(self, actorName, nName, oName, animNameA, animNameB, effect):
        if False:
            while True:
                i = 10
        dict = AllScene.renameBlendAnim(actorName, nName, oName, animNameA, animNameB, effect)
        self.animBlendPanel[actorName].setBlendAnimList(dict)
        return

    def animBlendPanelClose(self, name):
        if False:
            return 10
        if name in self.animBlendPanel:
            del self.animBlendPanel[name]
        return

    def toggleWidgetVis(self):
        if False:
            return 10
        if self.sideWindow != None:
            self.sideWindow.toggleWidgetVisFromMainW()
        else:
            self.widgetVis = (self.widgetVis + 1) % 2

    def toggleBackface(self):
        if False:
            print('Hello World!')
        if self.sideWindow != None:
            self.sideWindow.toggleBackfaceFromMainW()
        else:
            self.backface = (self.backface + 1) % 2

    def toggleTexture(self):
        if False:
            for i in range(10):
                print('nop')
        if self.sideWindow != None:
            self.sideWindow.toggleTextureFromMainW()
        else:
            self.texture = (self.texture + 1) % 2

    def toggleWireframe(self):
        if False:
            for i in range(10):
                print('nop')
        if self.sideWindow != None:
            self.sideWindow.toggleWireframeFromMainW()
        else:
            self.wireframe = (self.wireframe + 1) % 2

    def openAlignPanel(self, nodePath=None):
        if False:
            while True:
                i = 10
        name = nodePath.getName()
        if name not in self.alignPanelDict:
            list = AllScene.getAllObjNameAsList()
            if name in list:
                list.remove(name)
            else:
                return
            self.alignPanelDict[name] = AlignTool(nodePath=nodePath, list=list)
        return

    def closeAlignPanel(self, name=None):
        if False:
            while True:
                i = 10
        if name in self.alignPanelDict:
            del self.alignPanelDict[name]

    def alignObject(self, nodePath, name, list):
        if False:
            return 10
        target = AllScene.getObjFromSceneByName(name)
        pos = target.getPos()
        hpr = target.getHpr()
        scale = target.getScale()
        if list[0]:
            nodePath.setX(pos.getX())
        if list[1]:
            nodePath.setY(pos.getY())
        if list[2]:
            nodePath.setZ(pos.getZ())
        if list[3]:
            nodePath.setH(hpr.getX())
        if list[4]:
            nodePath.setP(hpr.getY())
        if list[5]:
            nodePath.setR(hpr.getZ())
        if list[6]:
            nodePath.setSx(scale.getX())
        if list[7]:
            nodePath.setSy(scale.getY())
        if list[8]:
            nodePath.setSz(scale.getZ())
        return

    def requestCurveList(self, nodePath, name):
        if False:
            return 10
        curveList = AllScene.getCurveList(nodePath)
        messenger.send('curveListFor' + name, [curveList])

    def flash(self, nodePath='None Given'):
        if False:
            return 10
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
            t = taskMgr.doMethodLater(1.5, self.flashDummy, 'flashNodePath')
            t.nodePath = nodePath
            t.doneColor = doneColor
            t.uponDeath = self.flashDone

    def flashDummy(self, state):
        if False:
            print('Hello World!')
        return Task.done

    def flashDone(self, state):
        if False:
            return 10
        if state.nodePath.isEmpty():
            return
        if state.doneColor:
            state.nodePath.setColor(state.doneColor)
        else:
            state.nodePath.clearColor()
editor = myLevelEditor(parent=base.tkRoot)
base.run()