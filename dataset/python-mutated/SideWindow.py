from direct.tkwidgets.AppShell import AppShell
from direct.tkwidgets.VectorWidgets import ColorEntry
from direct.showbase.TkGlobal import spawnTkLoop
import seSceneGraphExplorer
import Pmw
from tkinter import Frame, IntVar, Checkbutton, Toplevel
import tkinter

class sideWindow(AppShell):
    appversion = '1.0'
    appname = 'Navigation Window'
    frameWidth = 325
    frameHeight = 580
    frameIniPosX = 0
    frameIniPosY = 110
    padx = 0
    pady = 0
    lightEnable = 0
    ParticleEnable = 0
    basedriveEnable = 0
    collision = 0
    backface = 0
    texture = 1
    wireframe = 0
    enableBaseUseDrive = 0

    def __init__(self, worldColor, lightEnable, ParticleEnable, basedriveEnable, collision, backface, texture, wireframe, grid, widgetVis, enableAutoCamera, parent=None, nodePath=render, **kw):
        if False:
            while True:
                i = 10
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
        optiondefs = (('title', self.appname, None),)
        self.defineoptions(kw, optiondefs)
        if parent == None:
            self.parent = Toplevel()
        else:
            self.parent = parent
        AppShell.__init__(self, self.parent)
        self.parent.geometry('%dx%d+%d+%d' % (self.frameWidth, self.frameHeight, self.frameIniPosX, self.frameIniPosY))
        self.parent.resizable(False, False)

    def appInit(self):
        if False:
            for i in range(10):
                print('nop')
        print('----SideWindow is Initialized!!')

    def createInterface(self):
        if False:
            i = 10
            return i + 15
        interior = self.interior()
        mainFrame = Frame(interior)
        self.notebookFrame = Pmw.NoteBook(mainFrame)
        self.notebookFrame.pack(fill=tkinter.BOTH, expand=1)
        sgePage = self.notebookFrame.add('Tree Graph')
        envPage = self.notebookFrame.add('World Setting')
        self.notebookFrame['raisecommand'] = self.updateInfo
        self.SGE = seSceneGraphExplorer.seSceneGraphExplorer(sgePage, nodePath=render, scrolledCanvas_hull_width=270, scrolledCanvas_hull_height=570)
        self.SGE.pack(fill=tkinter.BOTH, expand=0)
        envPage = Frame(envPage)
        pageFrame = Frame(envPage)
        self.LightingVar = IntVar()
        self.LightingVar.set(self.lightEnable)
        self.LightingButton = Checkbutton(pageFrame, text='Enable Lighting', variable=self.LightingVar, command=self.toggleLights)
        self.LightingButton.pack(side=tkinter.LEFT, expand=False)
        pageFrame.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)
        pageFrame = Frame(envPage)
        self.CollisionVar = IntVar()
        self.CollisionVar.set(self.collision)
        self.CollisionButton = Checkbutton(pageFrame, text='Show Collision Object', variable=self.CollisionVar, command=self.showCollision)
        self.CollisionButton.pack(side=tkinter.LEFT, expand=False)
        pageFrame.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)
        pageFrame = Frame(envPage)
        self.ParticleVar = IntVar()
        self.ParticleVar.set(self.ParticleEnable)
        self.ParticleButton = Checkbutton(pageFrame, text='Show Particle Dummy', variable=self.ParticleVar, command=self.enableParticle)
        self.ParticleButton.pack(side=tkinter.LEFT, expand=False)
        pageFrame.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)
        pageFrame = Frame(envPage)
        self.baseUseDriveVar = IntVar()
        self.baseUseDriveVar.set(self.basedriveEnable)
        self.baseUseDriveButton = Checkbutton(pageFrame, text='Enable base.usedrive', variable=self.baseUseDriveVar, command=self.enablebaseUseDrive)
        self.baseUseDriveButton.pack(side=tkinter.LEFT, expand=False)
        pageFrame.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)
        pageFrame = Frame(envPage)
        self.backfaceVar = IntVar()
        self.backfaceVar.set(self.backface)
        self.backfaceButton = Checkbutton(pageFrame, text='Enable BackFace', variable=self.backfaceVar, command=self.toggleBackface)
        self.backfaceButton.pack(side=tkinter.LEFT, expand=False)
        pageFrame.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)
        pageFrame = Frame(envPage)
        self.textureVar = IntVar()
        self.textureVar.set(self.texture)
        self.textureButton = Checkbutton(pageFrame, text='Enable Texture', variable=self.textureVar, command=self.toggleTexture)
        self.textureButton.pack(side=tkinter.LEFT, expand=False)
        pageFrame.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)
        pageFrame = Frame(envPage)
        self.wireframeVar = IntVar()
        self.wireframeVar.set(self.wireframe)
        self.wireframeButton = Checkbutton(pageFrame, text='Enable Wireframe', variable=self.wireframeVar, command=self.toggleWireframe)
        self.wireframeButton.pack(side=tkinter.LEFT, expand=False)
        pageFrame.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)
        pageFrame = Frame(envPage)
        self.gridVar = IntVar()
        self.gridVar.set(self.grid)
        self.gridButton = Checkbutton(pageFrame, text='Enable Grid', variable=self.gridVar, command=self.toggleGrid)
        self.gridButton.pack(side=tkinter.LEFT, expand=False)
        pageFrame.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)
        pageFrame = Frame(envPage)
        self.widgetVisVar = IntVar()
        self.widgetVisVar.set(self.widgetVis)
        self.widgetVisButton = Checkbutton(pageFrame, text='Enable WidgetVisible', variable=self.widgetVisVar, command=self.togglewidgetVis)
        self.widgetVisButton.pack(side=tkinter.LEFT, expand=False)
        pageFrame.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)
        pageFrame = Frame(envPage)
        self.enableAutoCameraVar = IntVar()
        self.enableAutoCameraVar.set(self.enableAutoCamera)
        self.enableAutoCameraButton = Checkbutton(pageFrame, text='Enable Auto Camera Movement for Loading Objects', variable=self.enableAutoCameraVar, command=self.toggleAutoCamera)
        self.enableAutoCameraButton.pack(side=tkinter.LEFT, expand=False)
        pageFrame.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)
        pageFrame = Frame(envPage)
        self.backgroundColor = ColorEntry(pageFrame, text='BG Color', value=self.worldColor)
        self.backgroundColor['command'] = self.setBackgroundColorVec
        self.backgroundColor['resetValue'] = [0, 0, 0, 0]
        self.backgroundColor.pack(side=tkinter.LEFT, expand=False)
        self.bind(self.backgroundColor, 'Set background color')
        pageFrame.pack(side=tkinter.TOP, fill=tkinter.X, expand=True)
        envPage.pack(expand=False)
        self.notebookFrame.setnaturalsize()
        mainFrame.pack(fill='both', expand=1)

    def createMenuBar(self):
        if False:
            for i in range(10):
                print('nop')
        self.menuBar.destroy()

    def onDestroy(self, event):
        if False:
            while True:
                i = 10
        messenger.send('SW_close', [self.worldColor, self.lightEnable, self.ParticleEnable, self.basedriveEnable, self.collision, self.backface, self.texture, self.wireframe, self.grid, self.widgetVis, self.enableAutoCamera])
        '\n        If you have open any thing, please rewrite here!\n        '
        pass

    def updateInfo(self, page='Tree Graph'):
        if False:
            for i in range(10):
                print('nop')
        if page == 'Tree Graph':
            self.updateTreeGraph()
        elif page == 'World Setting':
            self.updateWorldSetting()

    def updateTreeGraph(self):
        if False:
            while True:
                i = 10
        self.SGE.update()
        pass

    def updateWorldSetting(self):
        if False:
            return 10
        self.LightingVar.set(self.lightEnable)
        self.CollisionVar.set(self.collision)
        self.ParticleVar.set(self.ParticleEnable)
        self.baseUseDriveVar.set(self.basedriveEnable)
        self.backgroundColor.set(value=self.worldColor)
        pass

    def toggleLights(self):
        if False:
            return 10
        self.lightEnable = (self.lightEnable + 1) % 2
        messenger.send('SW_lightToggle')
        pass

    def showCollision(self):
        if False:
            while True:
                i = 10
        self.collision = (self.collision + 1) % 2
        messenger.send('SW_collisionToggle', [self.collision])
        pass

    def enableParticle(self):
        if False:
            return 10
        self.ParticleEnable = (self.ParticleEnable + 1) % 2
        messenger.send('SW_particleToggle', [self.ParticleEnable])
        pass

    def enablebaseUseDrive(self):
        if False:
            print('Hello World!')
        if self.enableBaseUseDrive == 0:
            print('Enabled')
            base.useDrive()
            self.enableBaseUseDrive = 1
        else:
            print('disabled')
            base.disableMouse()
            self.enableBaseUseDrive = 0
        self.basedriveEnable = (self.basedriveEnable + 1) % 2
        pass

    def toggleBackface(self):
        if False:
            while True:
                i = 10
        base.toggleBackface()
        self.backface = (self.backface + 1) % 2
        return

    def toggleBackfaceFromMainW(self):
        if False:
            for i in range(10):
                print('nop')
        self.backface = (self.backface + 1) % 2
        self.backfaceButton.toggle()
        return

    def toggleTexture(self):
        if False:
            return 10
        base.toggleTexture()
        self.texture = (self.texture + 1) % 2
        return

    def toggleTextureFromMainW(self):
        if False:
            i = 10
            return i + 15
        self.texture = (self.texture + 1) % 2
        self.textureButton.toggle()
        return

    def toggleWireframe(self):
        if False:
            print('Hello World!')
        base.toggleWireframe()
        self.wireframe = (self.wireframe + 1) % 2
        return

    def toggleWireframeFromMainW(self):
        if False:
            for i in range(10):
                print('nop')
        self.wireframe = (self.wireframe + 1) % 2
        self.wireframeButton.toggle()
        return

    def toggleGrid(self):
        if False:
            return 10
        self.grid = (self.grid + 1) % 2
        if self.grid == 1:
            SEditor.grid.enable()
        else:
            SEditor.grid.disable()

    def togglewidgetVis(self):
        if False:
            print('Hello World!')
        self.widgetVis = (self.widgetVis + 1) % 2
        SEditor.toggleWidgetVis()
        if SEditor.widget.fActive:
            messenger.send('shift-f')
        return

    def toggleWidgetVisFromMainW(self):
        if False:
            i = 10
            return i + 15
        self.widgetVis = (self.widgetVis + 1) % 2
        self.widgetVisButton.toggle()
        return

    def setBackgroundColorVec(self, color):
        if False:
            for i in range(10):
                print('nop')
        base.setBackgroundColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        self.worldColor = [color[0], color[1], color[2], 0]

    def toggleAutoCamera(self):
        if False:
            for i in range(10):
                print('nop')
        self.enableAutoCamera = (self.enableAutoCamera + 1) % 2
        SEditor.toggleAutoCamera()
        return

    def selectPage(self, page='Tree Graph'):
        if False:
            return 10
        self.notebookFrame.selectpage(page)