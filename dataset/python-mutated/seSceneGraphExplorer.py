from direct.showbase.DirectObject import DirectObject
from seTree import TreeNode, TreeItem
import Pmw
from tkinter import IntVar, Frame, Label
import tkinter
DEFAULT_MENU_ITEMS = ['Update Explorer', 'Separator', 'Properties', 'Separator', 'Duplicate', 'Remove', 'Add Dummy', 'Add Collision Object', 'Metadata', 'Separator', 'Set as Reparent Target', 'Reparent to Target', 'Separator', 'Animation Panel', 'Blend Animation Panel', 'MoPath Panel', 'Align Tool', 'Separator']

class seSceneGraphExplorer(Pmw.MegaWidget, DirectObject):
    """Graphical display of a scene graph"""

    def __init__(self, parent=None, nodePath=render, **kw):
        if False:
            i = 10
            return i + 15
        optiondefs = (('menuItems', [], Pmw.INITOPT),)
        self.defineoptions(kw, optiondefs)
        Pmw.MegaWidget.__init__(self, parent)
        self.nodePath = nodePath
        interior = self.interior()
        interior.configure(relief=tkinter.GROOVE, borderwidth=2)
        self._scrolledCanvas = self.createcomponent('scrolledCanvas', (), None, Pmw.ScrolledCanvas, (interior,), hull_width=200, hull_height=300, usehullsize=1)
        self._canvas = self._scrolledCanvas.component('canvas')
        self._canvas['scrollregion'] = ('0i', '0i', '2i', '4i')
        self._scrolledCanvas.resizescrollregion()
        self._scrolledCanvas.pack(padx=3, pady=3, expand=1, fill=tkinter.BOTH)
        self._canvas.bind('<ButtonPress-2>', self.mouse2Down)
        self._canvas.bind('<B2-Motion>', self.mouse2Motion)
        self._canvas.bind('<Configure>', lambda e, sc=self._scrolledCanvas: sc.resizescrollregion())
        self.interior().bind('<Destroy>', self.onDestroy)
        self._treeItem = SceneGraphExplorerItem(self.nodePath)
        self._node = TreeNode(self._canvas, None, self._treeItem, DEFAULT_MENU_ITEMS + self['menuItems'])
        self._node.expand()
        self._parentFrame = Frame(interior)
        self._label = self.createcomponent('parentLabel', (), None, Label, (interior,), text='Active Reparent Target: ', anchor=tkinter.W, justify=tkinter.LEFT)
        self._label.pack(fill=tkinter.X)

        def updateLabel(nodePath=None, s=self):
            if False:
                print('Hello World!')
            s._label['text'] = 'Active Reparent Target: ' + nodePath.getName()
        self.accept('DIRECT_activeParent', updateLabel)
        self.accept('SGE_Update Explorer', lambda np, s=self: s.update())
        self.initialiseoptions(seSceneGraphExplorer)

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        ' Refresh scene graph explorer '
        self._node.update()

    def mouse2Down(self, event):
        if False:
            for i in range(10):
                print('nop')
        self._width = 1.0 * self._canvas.winfo_width()
        self._height = 1.0 * self._canvas.winfo_height()
        xview = self._canvas.xview()
        yview = self._canvas.yview()
        self._left = xview[0]
        self._top = yview[0]
        self._dxview = xview[1] - xview[0]
        self._dyview = yview[1] - yview[0]
        self._2lx = event.x
        self._2ly = event.y

    def mouse2Motion(self, event):
        if False:
            i = 10
            return i + 15
        newx = self._left - (event.x - self._2lx) / self._width * self._dxview
        self._canvas.xview_moveto(newx)
        newy = self._top - (event.y - self._2ly) / self._height * self._dyview
        self._canvas.yview_moveto(newy)
        self._2lx = event.x
        self._2ly = event.y
        self._left = self._canvas.xview()[0]
        self._top = self._canvas.yview()[0]

    def onDestroy(self, event):
        if False:
            while True:
                i = 10
        self.ignore('DIRECT_activeParent')
        self.ignore('SGE_Update Explorer')

    def deSelectTree(self):
        if False:
            for i in range(10):
                print('nop')
        self._node.deselecttree()

    def selectNodePath(self, nodePath, callBack=True):
        if False:
            for i in range(10):
                print('nop')
        item = self._node.find(nodePath.get_key())
        if item != None:
            item.select(callBack)
        else:
            print('----SGE: Error Selection')

class SceneGraphExplorerItem(TreeItem):
    """Example TreeItem subclass -- browse the file system."""

    def __init__(self, nodePath):
        if False:
            print('Hello World!')
        self.nodePath = nodePath

    def GetText(self):
        if False:
            i = 10
            return i + 15
        type = self.nodePath.node().getType().getName()
        name = self.nodePath.getName()
        return type + '  ' + name

    def GetTextForEdit(self):
        if False:
            i = 10
            return i + 15
        name = self.nodePath.getName()
        return name

    def GetKey(self):
        if False:
            for i in range(10):
                print('nop')
        return self.nodePath.get_key()

    def IsEditable(self):
        if False:
            return 10
        return 1

    def SetText(self, text):
        if False:
            return 10
        try:
            messenger.send('SGE_changeName', [self.nodePath, text])
        except AttributeError:
            pass

    def GetIconName(self):
        if False:
            i = 10
            return i + 15
        return 'sphere2'

    def IsExpandable(self):
        if False:
            while True:
                i = 10
        return self.nodePath.getNumChildren() != 0

    def GetSubList(self):
        if False:
            print('Hello World!')
        sublist = []
        for nodePath in self.nodePath.getChildren():
            item = SceneGraphExplorerItem(nodePath)
            sublist.append(item)
        return sublist

    def OnSelect(self, callback):
        if False:
            while True:
                i = 10
        messenger.send('SGE_Flash', [self.nodePath])
        if not callback:
            messenger.send('SGE_madeSelection', [self.nodePath, callback])
        else:
            messenger.send('SGE_madeSelection', [self.nodePath])

    def MenuCommand(self, command):
        if False:
            print('Hello World!')
        messenger.send('SGE_' + command, [self.nodePath])

def explore(nodePath=render):
    if False:
        for i in range(10):
            print('nop')
    tl = Toplevel()
    tl.title('Explore: ' + nodePath.getName())
    sge = seSceneGraphExplorer(parent=tl, nodePath=nodePath)
    sge.pack(expand=1, fill='both')
    return sge