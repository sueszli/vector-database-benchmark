"""Implements a pop-up menu containing multiple clickable options.

See the :ref:`directoptionmenu` page in the programming manual for a more
in-depth explanation and an example of how to use this class.
"""
__all__ = ['DirectOptionMenu']
from panda3d.core import OmniBoundingVolume, TextNode, VBase3
from direct.showbase import ShowBaseGlobal
from . import DirectGuiGlobals as DGG
from .DirectButton import DirectButton
from .DirectFrame import DirectFrame

class DirectOptionMenu(DirectButton):
    """
    DirectOptionMenu(parent) - Create a DirectButton which pops up a
    menu which can be used to select from a list of items.
    Execute button command (passing the selected item through) if defined
    To cancel the popup menu click anywhere on the screen outside of the
    popup menu.  No command is executed in this case.
    """

    def __init__(self, parent=None, **kw):
        if False:
            while True:
                i = 10
        optiondefs = (('items', [], self.setItems), ('initialitem', None, DGG.INITOPT), ('popupMarkerBorder', (0.1, 0.1), None), ('popupMarker_pos', None, None), ('highlightColor', (0.5, 0.5, 0.5, 1), None), ('highlightScale', (1, 1), None), ('text_align', TextNode.ALeft, None), ('pressEffect', 0, DGG.INITOPT))
        self.defineoptions(kw, optiondefs)
        DirectButton.__init__(self, parent)
        self.initFrameSize = self['frameSize']
        self.popupMarker = self.createcomponent('popupMarker', (), None, DirectFrame, (self,), frameSize=(-0.5, 0.5, -0.2, 0.2), scale=0.4, relief=DGG.RAISED)
        self.initPopupMarkerPos = self['popupMarker_pos']
        self.popupMarker.bind(DGG.B1PRESS, self.showPopupMenu)
        self.popupMarker.bind(DGG.B1RELEASE, self.selectHighlightedIndex)
        if self['clickSound']:
            self.popupMarker.guiItem.setSound(DGG.B1PRESS + self.popupMarker.guiId, self['clickSound'])
        else:
            self.popupMarker.guiItem.clearSound(DGG.B1PRESS + self.popupMarker.guiId)
        self.popupMenu = None
        self.selectedIndex = None
        self.highlightedIndex = None
        if 'item_text_scale' in kw:
            self._prevItemTextScale = kw['item_text_scale']
        else:
            self._prevItemTextScale = (1, 1)
        self.cancelFrame = self.createcomponent('cancelframe', (), None, DirectFrame, (self,), frameSize=(-1, 1, -1, 1), relief=None, state='normal')
        self.cancelFrame.setBin('gui-popup', 0)
        self.cancelFrame.node().setBounds(OmniBoundingVolume())
        self.cancelFrame.bind(DGG.B1PRESS, self.hidePopupMenu)
        self.bind(DGG.B1PRESS, self.showPopupMenu)
        self.bind(DGG.B1RELEASE, self.selectHighlightedIndex)
        self.initialiseoptions(DirectOptionMenu)
        self.resetFrameSize()

    def setItems(self):
        if False:
            return 10
        "\n        self['items'] = itemList\n        Create new popup menu to reflect specified set of items\n        "
        if self.popupMenu is not None:
            self.destroycomponent('popupMenu')
        self.popupMenu = self.createcomponent('popupMenu', (), None, DirectFrame, (self,), relief='raised')
        self.popupMenu.setBin('gui-popup', 0)
        self.highlightedIndex = None
        if not self['items']:
            return
        itemIndex = 0
        self.minX = self.maxX = self.minZ = self.maxZ = None
        for item in self['items']:
            c = self.createcomponent('item%d' % itemIndex, (), 'item', DirectButton, (self.popupMenu,), text=item, text_align=TextNode.ALeft, command=lambda i=itemIndex: self.set(i))
            bounds = c.getBounds()
            if self.minX is None:
                self.minX = bounds[0]
            elif bounds[0] < self.minX:
                self.minX = bounds[0]
            if self.maxX is None:
                self.maxX = bounds[1]
            elif bounds[1] > self.maxX:
                self.maxX = bounds[1]
            if self.minZ is None:
                self.minZ = bounds[2]
            elif bounds[2] < self.minZ:
                self.minZ = bounds[2]
            if self.maxZ is None:
                self.maxZ = bounds[3]
            elif bounds[3] > self.maxZ:
                self.maxZ = bounds[3]
            itemIndex += 1
        self.maxWidth = self.maxX - self.minX
        self.maxHeight = self.maxZ - self.minZ
        for i in range(itemIndex):
            item = self.component('item%d' % i)
            item['frameSize'] = (self.minX, self.maxX, self.minZ, self.maxZ)
            item.setPos(-self.minX, 0, -self.maxZ - i * self.maxHeight)
            item.bind(DGG.B1RELEASE, self.hidePopupMenu)
            item.bind(DGG.WITHIN, lambda x, i=i, item=item: self._highlightItem(item, i))
            fc = item['frameColor']
            item.bind(DGG.WITHOUT, lambda x, item=item, fc=fc: self._unhighlightItem(item, fc))
        f = self.component('popupMenu')
        f['frameSize'] = (0, self.maxWidth, -self.maxHeight * itemIndex, 0)
        if self['initialitem']:
            self.set(self['initialitem'], fCommand=0)
        else:
            self.set(0, fCommand=0)
        pm = self.popupMarker
        pmw = pm.getWidth() * pm.getScale()[0] + 2 * self['popupMarkerBorder'][0]
        if self.initFrameSize:
            bounds = list(self.initFrameSize)
        else:
            bounds = [self.minX, self.maxX, self.minZ, self.maxZ]
        if self.initPopupMarkerPos:
            pmPos = list(self.initPopupMarkerPos)
        else:
            pmPos = [bounds[1] + pmw / 2.0, 0, bounds[2] + (bounds[3] - bounds[2]) / 2.0]
        pm.setPos(pmPos[0], pmPos[1], pmPos[2])
        bounds[1] += pmw
        self['frameSize'] = (bounds[0], bounds[1], bounds[2], bounds[3])
        self.hidePopupMenu()

    def showPopupMenu(self, event=None):
        if False:
            while True:
                i = 10
        '\n        Make popup visible and try to position it just to right of\n        mouse click with currently selected item aligned with button.\n        Adjust popup position if default position puts it outside of\n        visible screen region\n        '
        items = self['items']
        assert items and len(items) > 0, 'Cannot show an empty popup menu! You must add items!'
        self.popupMenu.show()
        self.popupMenu.setScale(self, VBase3(1))
        b = self.getBounds()
        fb = self.popupMenu.getBounds()
        xPos = (b[1] - b[0]) / 2.0 - fb[0]
        self.popupMenu.setX(self, xPos)
        self.popupMenu.setZ(self, self.minZ + (self.selectedIndex + 1) * self.maxHeight)
        pos = self.popupMenu.getPos(ShowBaseGlobal.render2d)
        scale = self.popupMenu.getScale(ShowBaseGlobal.render2d)
        maxX = pos[0] + fb[1] * scale[0]
        if maxX > 1.0:
            self.popupMenu.setX(ShowBaseGlobal.render2d, pos[0] + (1.0 - maxX))
        minZ = pos[2] + fb[2] * scale[2]
        maxZ = pos[2] + fb[3] * scale[2]
        if minZ < -1.0:
            self.popupMenu.setZ(ShowBaseGlobal.render2d, pos[2] + (-1.0 - minZ))
        elif maxZ > 1.0:
            self.popupMenu.setZ(ShowBaseGlobal.render2d, pos[2] + (1.0 - maxZ))
        self.cancelFrame.show()
        self.cancelFrame.setPos(ShowBaseGlobal.render2d, 0, 0, 0)
        self.cancelFrame.setScale(ShowBaseGlobal.render2d, 1, 1, 1)

    def hidePopupMenu(self, event=None):
        if False:
            while True:
                i = 10
        ' Put away popup and cancel frame '
        self.popupMenu.hide()
        self.cancelFrame.hide()

    def _highlightItem(self, item, index):
        if False:
            print('Hello World!')
        ' Set frame color of highlighted item, record index '
        self._prevItemTextScale = item['text_scale']
        item['frameColor'] = self['highlightColor']
        item['frameSize'] = (self['highlightScale'][0] * self.minX, self['highlightScale'][0] * self.maxX, self['highlightScale'][1] * self.minZ, self['highlightScale'][1] * self.maxZ)
        item['text_scale'] = self['highlightScale']
        self.highlightedIndex = index

    def _unhighlightItem(self, item, frameColor):
        if False:
            for i in range(10):
                print('nop')
        ' Clear frame color, clear highlightedIndex '
        item['frameColor'] = frameColor
        item['frameSize'] = (self.minX, self.maxX, self.minZ, self.maxZ)
        item['text_scale'] = self._prevItemTextScale
        self.highlightedIndex = None

    def selectHighlightedIndex(self, event=None):
        if False:
            print('Hello World!')
        '\n        Check to see if item is highlighted (by cursor being within\n        that item).  If so, selected it.  If not, do nothing\n        '
        if self.highlightedIndex is not None:
            self.set(self.highlightedIndex)
            self.hidePopupMenu()

    def index(self, index):
        if False:
            print('Hello World!')
        intIndex = None
        if isinstance(index, int):
            intIndex = index
        elif index in self['items']:
            i = 0
            for item in self['items']:
                if item == index:
                    intIndex = i
                    break
                i += 1
        return intIndex

    def set(self, index, fCommand=1):
        if False:
            i = 10
            return i + 15
        newIndex = self.index(index)
        if newIndex is not None:
            self.selectedIndex = newIndex
            item = self['items'][self.selectedIndex]
            self['text'] = item
            if fCommand and self['command']:
                self['command'](*[item] + self['extraArgs'])

    def get(self):
        if False:
            while True:
                i = 10
        ' Get currently selected item '
        return self['items'][self.selectedIndex]

    def commandFunc(self, event):
        if False:
            i = 10
            return i + 15
        "\n        Override popup menu button's command func\n        Command is executed in response to selecting menu items\n        "