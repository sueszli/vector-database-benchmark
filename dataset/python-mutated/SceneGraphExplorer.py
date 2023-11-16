"""This module defines a widget used to display a graphical overview of the
scene graph using the tkinter GUI system.

Requires Pmw."""
__all__ = ['SceneGraphExplorer', 'SceneGraphExplorerItem', 'explore']
from direct.showbase.DirectObject import DirectObject
from direct.showbase.MessengerGlobal import messenger
from .Tree import TreeItem, TreeNode
import Pmw
import tkinter as tk
DEFAULT_MENU_ITEMS = ['Update Explorer', 'Expand All', 'Collapse All', 'Separator', 'Select', 'Deselect', 'Separator', 'Delete', 'Separator', 'Fit', 'Flash', 'Isolate', 'Toggle Vis', 'Show All', 'Separator', 'Set Reparent Target', 'Reparent', 'WRT Reparent', 'Separator', 'Place', 'Set Name', 'Set Color', 'Explore', 'Separator']

class SceneGraphExplorer(Pmw.MegaWidget, DirectObject):
    """Graphical display of a scene graph"""

    def __init__(self, parent=None, nodePath=None, isItemEditable=True, **kw):
        if False:
            while True:
                i = 10
        if nodePath is None:
            nodePath = base.render
        optiondefs = (('menuItems', [], Pmw.INITOPT),)
        self.defineoptions(kw, optiondefs)
        Pmw.MegaWidget.__init__(self, parent)
        self.nodePath = nodePath
        interior = self.interior()
        interior.configure(relief=tk.GROOVE, borderwidth=2)
        self._scrolledCanvas = self.createcomponent('scrolledCanvas', (), None, Pmw.ScrolledCanvas, (interior,), hull_width=200, hull_height=300, usehullsize=1)
        self._canvas = self._scrolledCanvas.component('canvas')
        self._canvas['scrollregion'] = ('0i', '0i', '2i', '4i')
        self._scrolledCanvas.resizescrollregion()
        self._scrolledCanvas.pack(padx=3, pady=3, expand=1, fill=tk.BOTH)
        self._canvas.bind('<ButtonPress-2>', self.mouse2Down)
        self._canvas.bind('<B2-Motion>', self.mouse2Motion)
        self._canvas.bind('<Configure>', lambda e, sc=self._scrolledCanvas: sc.resizescrollregion())
        self.interior().bind('<Destroy>', self.onDestroy)
        self._treeItem = SceneGraphExplorerItem(self.nodePath, isItemEditable)
        self._node = TreeNode(self._canvas, None, self._treeItem, DEFAULT_MENU_ITEMS + self['menuItems'])
        self._node.expand()
        self._parentFrame = tk.Frame(interior)
        self._label = self.createcomponent('parentLabel', (), None, tk.Label, (interior,), text='Active Reparent Target: ', anchor=tk.W, justify=tk.LEFT)
        self._label.pack(fill=tk.X)

        def updateLabel(nodePath=None, s=self):
            if False:
                for i in range(10):
                    print('nop')
            s._label['text'] = 'Active Reparent Target: ' + nodePath.getName()
        self.accept('DIRECT_activeParent', updateLabel)
        self.accept('SGE_Update Explorer', lambda np, s=self: s.update())
        self.initialiseoptions(SceneGraphExplorer)

    def setChildrenTag(self, tag, fModeChildrenTag):
        if False:
            print('Hello World!')
        self._node.setChildrenTag(tag, fModeChildrenTag)
        self._node.update()

    def setFSortChildren(self, fSortChildren):
        if False:
            while True:
                i = 10
        self._node.setFSortChildren(fSortChildren)
        self._node.update()

    def update(self, fUseCachedChildren=1):
        if False:
            while True:
                i = 10
        ' Refresh scene graph explorer '
        self._node.update(fUseCachedChildren)

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
            print('Hello World!')
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
            return 10
        self.ignore('DIRECT_activeParent')
        self.ignore('SGE_Update Explorer')

    def updateSelection(self, searchKey):
        if False:
            i = 10
            return i + 15
        sceneGraphItem = self._node.find(searchKey)
        if sceneGraphItem:
            sceneGraphItem.reveal()
            sceneGraphItem.select()

class SceneGraphExplorerItem(TreeItem):
    """Example TreeItem subclass -- browse the file system."""

    def __init__(self, nodePath, isItemEditable=True):
        if False:
            for i in range(10):
                print('nop')
        self.nodePath = nodePath
        self.isItemEditable = isItemEditable

    def GetText(self):
        if False:
            while True:
                i = 10
        type = self.nodePath.node().getType().getName()
        name = self.nodePath.getName()
        return type + '  ' + name

    def GetKey(self):
        if False:
            return 10
        return hash(self.nodePath)

    def IsEditable(self):
        if False:
            i = 10
            return i + 15
        return self.isItemEditable

    def SetText(self, text):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.nodePath.setName(text)
        except AttributeError:
            pass

    def GetIconName(self):
        if False:
            while True:
                i = 10
        return 'sphere2'

    def IsExpandable(self):
        if False:
            return 10
        return self.nodePath.getNumChildren() != 0

    def GetSubList(self):
        if False:
            for i in range(10):
                print('nop')
        sublist = []
        for nodePath in self.nodePath.getChildren():
            item = SceneGraphExplorerItem(nodePath, self.isItemEditable)
            sublist.append(item)
        return sublist

    def OnSelect(self):
        if False:
            for i in range(10):
                print('nop')
        messenger.send('SGE_Flash', [self.nodePath])

    def MenuCommand(self, command):
        if False:
            i = 10
            return i + 15
        messenger.send('SGE_' + command, [self.nodePath])

def explore(nodePath=None):
    if False:
        for i in range(10):
            print('nop')
    if nodePath is None:
        nodePath = base.render
    tl = tk.Toplevel()
    tl.title('Explore: ' + nodePath.getName())
    sge = SceneGraphExplorer(parent=tl, nodePath=nodePath)
    sge.pack(expand=1, fill='both')
    return sge