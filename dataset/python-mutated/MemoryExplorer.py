from direct.showbase.DirectObject import DirectObject
import Pmw
import tkinter as tk
DEFAULT_BT_WIDTH = 50.0

class MemoryExplorer(Pmw.MegaWidget, DirectObject):

    def __init__(self, parent=None, nodePath=None, **kw):
        if False:
            print('Hello World!')
        if nodePath is None:
            nodePath = render
        optiondefs = (('menuItems', [], Pmw.INITOPT),)
        self.defineoptions(kw, optiondefs)
        Pmw.MegaWidget.__init__(self, parent)
        self.nodePath = nodePath
        self.renderItem = None
        self.render2dItem = None
        self.buttons = []
        self.labels = []
        self.rootItem = None
        self.btWidth = DEFAULT_BT_WIDTH
        self.createScrolledFrame()
        self.createScale()
        self.createRefreshBT()
        self.balloon = Pmw.Balloon(self.interior())

    def createScrolledFrame(self):
        if False:
            print('Hello World!')
        self.frame = Pmw.ScrolledFrame(self.interior(), labelpos='n', label_text='ScrolledFrame', usehullsize=1, hull_width=200, hull_height=220)
        self.frame.pack(padx=3, pady=3, fill=tk.BOTH, expand=1)

    def createScale(self):
        if False:
            print('Hello World!')
        self.scaleCtrl = tk.Scale(self.interior(), label='Graph Scale', from_=0.0, to=20.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.onScaleUpdate)
        self.scaleCtrl.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.scaleCtrl.set(0.0)

    def createRefreshBT(self):
        if False:
            print('Hello World!')
        self.refreshBT = tk.Button(self.interior(), text='Refresh', command=self.refresh)
        self.refreshBT.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    def createDefaultCtrls(self):
        if False:
            return 10
        if self.renderItem is None or self.render2dItem is None:
            return
        totalBytes = self.renderItem.getVertexBytes() + self.render2dItem.getVertexBytes()
        self.addChildCtrl(self.renderItem, totalBytes)
        self.addChildCtrl(self.render2dItem, totalBytes)
        self.setTitle('ALL', totalBytes)

    def setTitle(self, parent, bytes):
        if False:
            while True:
                i = 10
        self.frame['label_text'] = '[%s] - %s bytes' % (parent, bytes)

    def resetCtrls(self):
        if False:
            return 10
        for button in self.buttons:
            self.balloon.unbind(button)
            button.destroy()
        self.buttons = []
        for label in self.labels:
            label.destroy()
        self.labels = []

    def getNewButton(self, width, ratio):
        if False:
            i = 10
            return i + 15
        newBT = tk.Button(self.frame.interior(), anchor=tk.W, width=width)
        if ratio == 0.0:
            newBT['bg'] = 'grey'
            newBT['text'] = '.'
        else:
            newBT['bg'] = Pmw.Color.hue2name(0.0, 1.0 - ratio)
            newBT['text'] = '%0.2f%%' % (ratio * 100.0)
        return newBT

    def addSelfCtrl(self, item, totalBytes):
        if False:
            return 10
        self.addLabel('[self] : %s bytes' % item.getSelfVertexBytes())
        bt = self.addButton(item.getSelfVertexBytes(), totalBytes, self.onSelfButtonLClick, self.onSelfButtonRClick, item)

    def addChildCtrl(self, item, totalBytes):
        if False:
            i = 10
            return i + 15
        self.addLabel('%s [+%s] : %s bytes' % (item.getName(), item.getNumChildren(), item.getVertexBytes()))
        bt = self.addButton(item.getVertexBytes(), totalBytes, self.onChildButtonLClick, self.onChildButtonRClick, item)

    def addButton(self, vertexBytes, totalBytes, funcLClick, funcRClick, item):
        if False:
            print('Hello World!')
        width = self.getBTWidth(vertexBytes, totalBytes)
        if totalBytes == 0:
            ratio = 0.0
        else:
            ratio = vertexBytes / float(totalBytes)
        bt = self.getNewButton(width, ratio)

        def callbackL(event):
            if False:
                for i in range(10):
                    print('nop')
            funcLClick(item)

        def callbackR(event):
            if False:
                i = 10
                return i + 15
            funcRClick(item)
        bt.bind('<Button-1>', callbackL)
        bt.bind('<Button-3>', callbackR)
        bt.pack(side=tk.TOP, anchor=tk.NW)
        self.buttons.append(bt)
        self.balloon.bind(bt, item.getPathName())
        return bt

    def addLabel(self, label):
        if False:
            print('Hello World!')
        label = tk.Label(self.frame.interior(), text=label)
        label.pack(side=tk.TOP, anchor=tk.NW, expand=0)
        self.labels.append(label)

    def getBTWidth(self, vertexBytes, totalBytes):
        if False:
            for i in range(10):
                print('nop')
        if totalBytes == 0:
            return 1
        width = int(self.btWidth * vertexBytes / totalBytes)
        if width == 0:
            width = 1
        return width

    def onScaleUpdate(self, arg):
        if False:
            for i in range(10):
                print('nop')
        self.btWidth = DEFAULT_BT_WIDTH + DEFAULT_BT_WIDTH * float(arg)
        if self.rootItem:
            self.updateBTWidth()
        else:
            self.updateDefaultBTWidth()

    def updateBTWidth(self):
        if False:
            for i in range(10):
                print('nop')
        self.buttons[0]['width'] = self.getBTWidth(self.rootItem.getSelfVertexBytes(), self.rootItem.getVertexBytes())
        btIndex = 1
        for item in self.rootItem.getChildren():
            self.buttons[btIndex]['width'] = self.getBTWidth(item.getVertexBytes(), self.rootItem.getVertexBytes())
            btIndex += 1

    def updateDefaultBTWidth(self):
        if False:
            return 10
        if self.renderItem is None or self.render2dItem is None:
            return
        totalBytes = self.renderItem.getVertexBytes() + self.render2dItem.getVertexBytes()
        self.buttons[0]['width'] = self.getBTWidth(self.renderItem.getVertexBytes(), totalBytes)
        self.buttons[1]['width'] = self.getBTWidth(self.render2dItem.getVertexBytes(), totalBytes)

    def onSelfButtonLClick(self, item):
        if False:
            while True:
                i = 10
        pass

    def onSelfButtonRClick(self, item):
        if False:
            i = 10
            return i + 15
        parentItem = item.getParent()
        self.resetCtrls()
        self.addItemCtrls(parentItem)

    def onChildButtonLClick(self, item):
        if False:
            while True:
                i = 10
        if item.getNumChildren() == 0:
            return
        self.resetCtrls()
        self.addItemCtrls(item)

    def onChildButtonRClick(self, item):
        if False:
            print('Hello World!')
        parentItem = item.getParent()
        if parentItem:
            self.resetCtrls()
            self.addItemCtrls(parentItem.getParent())

    def addItemCtrls(self, item):
        if False:
            while True:
                i = 10
        self.rootItem = item
        if item is None:
            self.createDefaultCtrls()
        else:
            self.addSelfCtrl(item, item.getVertexBytes())
            for child in item.getChildren():
                self.addChildCtrl(child, item.getVertexBytes())
            self.setTitle(item.getPathName(), item.getVertexBytes())

    def makeList(self):
        if False:
            i = 10
            return i + 15
        self.renderItem = MemoryExplorerItem(None, base.render)
        self.buildList(self.renderItem)
        self.render2dItem = MemoryExplorerItem(None, base.render2d)
        self.buildList(self.render2dItem)

    def buildList(self, parentItem):
        if False:
            for i in range(10):
                print('nop')
        for nodePath in parentItem.nodePath.getChildren():
            item = MemoryExplorerItem(parentItem, nodePath)
            parentItem.addChild(item)
            self.buildList(item)

    def analyze(self):
        if False:
            i = 10
            return i + 15
        self.renderItem.analyze()
        self.render2dItem.analyze()

    def refresh(self):
        if False:
            for i in range(10):
                print('nop')
        self.makeList()
        self.analyze()
        self.resetCtrls()
        self.createDefaultCtrls()

class MemoryExplorerItem:

    def __init__(self, parent, nodePath):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.nodePath = nodePath
        self.children = []
        self.selfVertexBytes = 0
        self.childrenVertexBytes = 0
        self.numFaces = 0
        self.textureBytes = 0
        if parent:
            self.pathName = parent.pathName + '/' + nodePath.getName()
        else:
            self.pathName = nodePath.getName()

    def getParent(self):
        if False:
            while True:
                i = 10
        return self.parent

    def addChild(self, child):
        if False:
            while True:
                i = 10
        self.children.append(child)

    def getNumChildren(self):
        if False:
            i = 10
            return i + 15
        return len(self.children)

    def getChildren(self):
        if False:
            for i in range(10):
                print('nop')
        return self.children

    def getName(self):
        if False:
            while True:
                i = 10
        return self.nodePath.getName()

    def getPathName(self):
        if False:
            print('Hello World!')
        return self.pathName

    def getVertexBytes(self):
        if False:
            for i in range(10):
                print('nop')
        return self.selfVertexBytes + self.childrenVertexBytes

    def getSelfVertexBytes(self):
        if False:
            for i in range(10):
                print('nop')
        return self.selfVertexBytes

    def analyze(self):
        if False:
            return 10
        self.selfVertexBytes = 0
        self.childrenVertexBytes = 0
        self.numFaces = 0
        self.textureBytes = 0
        self.calcTextureBytes()
        if self.nodePath.node().isGeomNode():
            geomNode = self.nodePath.node()
            for i in range(geomNode.getNumGeoms()):
                geom = geomNode.getGeom(i)
                self.calcVertexBytes(geom)
                self.calcNumFaces(geom)
        self.analyzeChildren()

    def calcVertexBytes(self, geom):
        if False:
            return 10
        vData = geom.getVertexData()
        for j in range(vData.getNumArrays()):
            array = vData.getArray(j)
            self.selfVertexBytes += array.getDataSizeBytes()

    def calcTextureBytes(self):
        if False:
            print('Hello World!')
        texCol = self.nodePath.findAllTextures()
        for i in range(texCol.getNumTextures()):
            tex = texCol.getTexture(i)
            self.textureBytes += tex.estimateTextureMemory()

    def calcNumFaces(self, geom):
        if False:
            while True:
                i = 10
        for k in range(geom.getNumPrimitives()):
            primitive = geom.getPrimitive(k)
            self.numFaces += primitive.getNumFaces()

    def analyzeChildren(self):
        if False:
            i = 10
            return i + 15
        for child in self.children:
            child.analyze()
            self.childrenVertexBytes += child.getVertexBytes()
            self.numFaces += child.numFaces

    def ls(self, indent=''):
        if False:
            while True:
                i = 10
        print(indent + self.nodePath.getName() + ' ' + str(self.getVertexBytes()) + ' ' + str(self.numFaces) + ' ' + str(self.textureBytes))
        indent = indent + ' '
        for child in self.children:
            child.ls(indent)