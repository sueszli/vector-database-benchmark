from direct.tkwidgets.AppShell import AppShell
from seColorEntry import *
from direct.tkwidgets.VectorWidgets import Vector3Entry
from direct.tkwidgets.Slider import Slider
import Pmw
from panda3d.core import *
from tkinter import Frame, Button, Menubutton, Menu
import tkinter

class lightingPanel(AppShell):
    appname = 'Lighting Panel'
    frameWidth = 400
    frameHeight = 400
    currentLight = None

    def __init__(self, lightList, parent=None, **kw):
        if False:
            i = 10
            return i + 15
        self.lightList = lightList
        self.lightColor = [0.3 * 255, 0.3 * 255, 0.3 * 255]
        self.type = ''
        INITOPT = Pmw.INITOPT
        optiondefs = (('title', self.appname, None),)
        self.defineoptions(kw, optiondefs)
        AppShell.__init__(self)
        self.initialiseoptions(lightingPanel)
        self.parent.resizable(False, False)

    def createInterface(self):
        if False:
            for i in range(10):
                print('nop')
        interior = self.interior()
        menuBar = self.menuBar
        self.menuBar.destroy()
        mainFrame = Frame(interior)
        self.listZone = Pmw.Group(mainFrame, tag_pyclass=None)
        self.listZone.pack(expand=0, fill=tkinter.X, padx=3, pady=3)
        listFrame = self.listZone.interior()
        self.lightEntry = self.createcomponent('Lights List', (), None, Pmw.ComboBox, (listFrame,), label_text='Light :', labelpos=tkinter.W, entry_width=25, selectioncommand=self.selectLight, scrolledlist_items=self.lightList)
        self.lightEntry.pack(side=tkinter.LEFT)
        self.renameButton = self.createcomponent('Rename Light', (), None, Button, (listFrame,), text=' Rename ', command=self.renameLight)
        self.renameButton.pack(side=tkinter.LEFT)
        self.addLighZone = Pmw.Group(listFrame, tag_pyclass=None)
        self.addLighZone.pack(side=tkinter.LEFT)
        insideFrame = self.addLighZone.interior()
        self.lightsButton = Menubutton(insideFrame, text='Add light', borderwidth=3, activebackground='#909090')
        lightsMenu = Menu(self.lightsButton)
        lightsMenu.add_command(label='Add Ambient Light', command=self.addAmbient)
        lightsMenu.add_command(label='Add Directional Light', command=self.addDirectional)
        lightsMenu.add_command(label='Add Point Light', command=self.addPoint)
        lightsMenu.add_command(label='Add Spotlight', command=self.addSpot)
        self.lightsButton.pack(expand=0)
        self.lightsButton['menu'] = lightsMenu
        self.deleteButton = self.createcomponent('delete Light', (), None, Button, (listFrame,), text='  Delete  ', command=self.deleteLight)
        self.deleteButton.pack(side=tkinter.LEFT)
        self.lightColor = seColorEntry(mainFrame, text='Light Color', value=self.lightColor)
        self.lightColor['command'] = self.setLightingColorVec
        self.lightColor['resetValue'] = [0.3 * 255, 0.3 * 255, 0.3 * 255, 0]
        self.lightColor.pack(fill=tkinter.X, expand=0)
        self.bind(self.lightColor, 'Set light color')
        self.lightNotebook = Pmw.NoteBook(mainFrame, tabpos=None, borderwidth=0)
        ambientPage = self.lightNotebook.add('Ambient')
        directionalPage = self.lightNotebook.add('Directional')
        pointPage = self.lightNotebook.add('Point')
        spotPage = self.lightNotebook.add('Spot')
        self.lightNotebook['raisecommand'] = self.updateLightInfo
        self.dSpecularColor = seColorEntry(directionalPage, text='Specular Color')
        self.dSpecularColor['command'] = self.setSpecularColor
        self.dSpecularColor.pack(fill=tkinter.X, expand=0)
        self.bind(self.dSpecularColor, 'Set directional light specular color')
        self.dPosition = Vector3Entry(directionalPage, text='Position')
        self.dPosition['command'] = self.setPosition
        self.dPosition['resetValue'] = [0, 0, 0, 0]
        self.dPosition.pack(fill=tkinter.X, expand=0)
        self.bind(self.dPosition, 'Set directional light position')
        self.dOrientation = Vector3Entry(directionalPage, text='Orientation')
        self.dOrientation['command'] = self.setOrientation
        self.dOrientation['resetValue'] = [0, 0, 0, 0]
        self.dOrientation.pack(fill=tkinter.X, expand=0)
        self.bind(self.dOrientation, 'Set directional light orientation')
        self.pSpecularColor = seColorEntry(pointPage, text='Specular Color')
        self.pSpecularColor['command'] = self.setSpecularColor
        self.pSpecularColor.pack(fill=tkinter.X, expand=0)
        self.bind(self.pSpecularColor, 'Set point light specular color')
        self.pPosition = Vector3Entry(pointPage, text='Position')
        self.pPosition['command'] = self.setPosition
        self.pPosition['resetValue'] = [0, 0, 0, 0]
        self.pPosition.pack(fill=tkinter.X, expand=0)
        self.bind(self.pPosition, 'Set point light position')
        self.pConstantAttenuation = Slider(pointPage, text='Constant Attenuation', max=1.0, resolution=0.01, value=1.0)
        self.pConstantAttenuation['command'] = self.setConstantAttenuation
        self.pConstantAttenuation.pack(fill=tkinter.X, expand=0)
        self.bind(self.pConstantAttenuation, 'Set point light constant attenuation')
        self.pLinearAttenuation = Slider(pointPage, text='Linear Attenuation', max=1.0, resolution=0.01, value=0.0)
        self.pLinearAttenuation['command'] = self.setLinearAttenuation
        self.pLinearAttenuation.pack(fill=tkinter.X, expand=0)
        self.bind(self.pLinearAttenuation, 'Set point light linear attenuation')
        self.pQuadraticAttenuation = Slider(pointPage, text='Quadratic Attenuation', max=1.0, resolution=0.01, value=0.0)
        self.pQuadraticAttenuation['command'] = self.setQuadraticAttenuation
        self.pQuadraticAttenuation.pack(fill=tkinter.X, expand=0)
        self.bind(self.pQuadraticAttenuation, 'Set point light quadratic attenuation')
        self.sSpecularColor = seColorEntry(spotPage, text='Specular Color')
        self.sSpecularColor['command'] = self.setSpecularColor
        self.sSpecularColor.pack(fill=tkinter.X, expand=0)
        self.bind(self.sSpecularColor, 'Set spot light specular color')
        self.sConstantAttenuation = Slider(spotPage, text='Constant Attenuation', max=1.0, resolution=0.01, value=1.0)
        self.sConstantAttenuation['command'] = self.setConstantAttenuation
        self.sConstantAttenuation.pack(fill=tkinter.X, expand=0)
        self.bind(self.sConstantAttenuation, 'Set spot light constant attenuation')
        self.sLinearAttenuation = Slider(spotPage, text='Linear Attenuation', max=1.0, resolution=0.01, value=0.0)
        self.sLinearAttenuation['command'] = self.setLinearAttenuation
        self.sLinearAttenuation.pack(fill=tkinter.X, expand=0)
        self.bind(self.sLinearAttenuation, 'Set spot light linear attenuation')
        self.sQuadraticAttenuation = Slider(spotPage, text='Quadratic Attenuation', max=1.0, resolution=0.01, value=0.0)
        self.sQuadraticAttenuation['command'] = self.setQuadraticAttenuation
        self.sQuadraticAttenuation.pack(fill=tkinter.X, expand=0)
        self.bind(self.sQuadraticAttenuation, 'Set spot light quadratic attenuation')
        self.sExponent = Slider(spotPage, text='Exponent', max=1.0, resolution=0.01, value=0.0)
        self.sExponent['command'] = self.setExponent
        self.sExponent.pack(fill=tkinter.X, expand=0)
        self.bind(self.sExponent, 'Set spot light exponent')
        self.lightNotebook.setnaturalsize()
        self.lightNotebook.pack(expand=1, fill=tkinter.BOTH)
        mainFrame.pack(expand=1, fill=tkinter.BOTH)

    def onDestroy(self, event):
        if False:
            for i in range(10):
                print('nop')
        messenger.send('LP_close')
        '\n        If you have open any thing, please rewrite here!\n        '
        pass

    def renameLight(self):
        if False:
            return 10
        oName = self.currentLight
        nName = self.lightEntry.get()
        messenger.send('LP_rename', [oName, nName])
        return

    def deleteLight(self):
        if False:
            while True:
                i = 10
        messenger.send('LP_removeLight', [self.currentLight])
        return

    def updateList(self, list, node=None):
        if False:
            while True:
                i = 10
        self.lightList = list
        self.lightEntry.setlist(list)
        if node != None:
            self.lightEntry.selectitem(index=node.getName(), setentry=True)
            self.updateDisplay(node)
        elif len(list) > 0:
            self.lightEntry.selectitem(index=0, setentry=True)
            self.selectLight(list[0])
        else:
            self.lightEntry.clear()
        return

    def selectLight(self, lightName):
        if False:
            i = 10
            return i + 15
        if lightName in self.lightList:
            messenger.send('LP_selectLight', [lightName])
        return

    def updateDisplay(self, lightNode):
        if False:
            return 10
        self.currentLight = lightNode
        if self.currentLight != None:
            color = lightNode.getLightColor()
            self.lightColor.set([255 * color.getX(), 255 * color.getY(), 255 * color.getZ()])
            oldType = self.type
            self.type = lightNode.getType()
        else:
            self.lightColor.set([255 * 0.3, 255 * 0.3, 255 * 0.3])
            oldType = self.type
            self.type = 'ambient'
        if self.type == 'ambient':
            self.lightNotebook.selectpage('Ambient')
        elif self.type == 'directional':
            self.lightNotebook.selectpage('Directional')
        elif self.type == 'point':
            self.lightNotebook.selectpage('Point')
        elif self.type == 'spot':
            self.lightNotebook.selectpage('Spot')
        if oldType == self.type:
            self.updateLightInfo()
        return

    def updateLightInfo(self, page=None):
        if False:
            i = 10
            return i + 15
        if self.currentLight != None:
            light = self.currentLight.getLight()
        if self.type != 'ambient':
            specColor = light.getSpecularColor()
        if self.type == 'directional':
            point = self.currentLight.getPosition()
            dir = self.currentLight.getOrientation()
            self.dSpecularColor.set([specColor.getX() * 255, specColor.getY() * 255, specColor.getZ() * 255])
            self.dPosition.set([point.getX(), point.getY(), point.getZ()])
            self.dOrientation.set([dir.getX(), dir.getY(), dir.getZ()])
        elif self.type == 'point':
            point = self.currentLight.getPosition()
            attenuation = light.getAttenuation()
            self.pSpecularColor.set([specColor.getX() * 255, specColor.getY() * 255, specColor.getZ() * 255])
            self.pPosition.set([point.getX(), point.getY(), point.getZ()])
            self.pConstantAttenuation.set(attenuation.getX())
            self.pLinearAttenuation.set(attenuation.getY())
            self.pQuadraticAttenuation.set(attenuation.getZ())
        elif self.type == 'spot':
            attenuation = light.getAttenuation()
            expo = light.getExponent()
            self.sSpecularColor.set([specColor.getX() * 255, specColor.getY() * 255, specColor.getZ() * 255])
            self.sConstantAttenuation.set(attenuation.getX())
            self.sLinearAttenuation.set(attenuation.getY())
            self.sQuadraticAttenuation.set(attenuation.getZ())
            self.sExponent.set(expo)
        return

    def addAmbient(self):
        if False:
            i = 10
            return i + 15
        messenger.send('LP_addLight', ['ambient'])
        return

    def addDirectional(self):
        if False:
            for i in range(10):
                print('nop')
        messenger.send('LP_addLight', ['directional'])
        return

    def addPoint(self):
        if False:
            while True:
                i = 10
        messenger.send('LP_addLight', ['point'])
        return

    def addSpot(self):
        if False:
            print('Hello World!')
        messenger.send('LP_addLight', ['spot'])
        return

    def setLightingColorVec(self, color):
        if False:
            for i in range(10):
                print('nop')
        if self.currentLight == None:
            return
        self.currentLight.setColor(VBase4(color[0] / 255, color[1] / 255, color[2] / 255, 1))
        return

    def setSpecularColor(self, color):
        if False:
            print('Hello World!')
        if self.currentLight == None:
            return
        self.currentLight.setSpecColor(VBase4(color[0] / 255, color[1] / 255, color[2] / 255, 1))
        return

    def setPosition(self, position):
        if False:
            print('Hello World!')
        if self.currentLight == None:
            return
        self.currentLight.setPosition(Point3(position[0], position[1], position[2]))
        return

    def setOrientation(self, orient):
        if False:
            for i in range(10):
                print('nop')
        if self.currentLight == None:
            return
        self.currentLight.setOrientation(Vec3(orient[0], orient[1], orient[2]))
        return

    def setConstantAttenuation(self, value):
        if False:
            i = 10
            return i + 15
        self.currentLight.setConstantAttenuation(value)
        return

    def setLinearAttenuation(self, value):
        if False:
            i = 10
            return i + 15
        self.currentLight.setLinearAttenuation(value)
        return

    def setQuadraticAttenuation(self, value):
        if False:
            i = 10
            return i + 15
        self.currentLight.setQuadraticAttenuation(value)
        return

    def setExponent(self, value):
        if False:
            i = 10
            return i + 15
        self.currentLight.setExponent(value)
        return