"""Contains the DirectScrolledFrame class.

See the :ref:`directscrolledframe` page in the programming manual for a more
in-depth explanation and an example of how to use this class.
"""
__all__ = ['DirectScrolledFrame']
from panda3d.core import NodePath, PGScrollFrame
from . import DirectGuiGlobals as DGG
from .DirectFrame import DirectFrame
from .DirectScrollBar import DirectScrollBar

class DirectScrolledFrame(DirectFrame):
    """
    DirectScrolledFrame -- a special frame that uses DirectScrollBar to
    implement a small window (the frameSize) into a potentially much
    larger virtual canvas (the canvasSize, scrolledFrame.getCanvas()).

    Unless specified otherwise, scroll bars are automatically created
    and managed as needed, based on the relative sizes od the
    frameSize and the canvasSize.  You can also set manageScrollBars =
    0 and explicitly position and hide or show the scroll bars
    yourself.
    """

    def __init__(self, parent=None, **kw):
        if False:
            print('Hello World!')
        optiondefs = (('pgFunc', PGScrollFrame, None), ('frameSize', (-0.5, 0.5, -0.5, 0.5), None), ('canvasSize', (-1, 1, -1, 1), self.setCanvasSize), ('manageScrollBars', 1, self.setManageScrollBars), ('autoHideScrollBars', 1, self.setAutoHideScrollBars), ('scrollBarWidth', 0.08, self.setScrollBarWidth), ('borderWidth', (0.01, 0.01), self.setBorderWidth))
        self.defineoptions(kw, optiondefs)
        DirectFrame.__init__(self, parent)
        w = self['scrollBarWidth']
        self.verticalScroll = self.createcomponent('verticalScroll', (), None, DirectScrollBar, (self,), borderWidth=self['borderWidth'], frameSize=(-w / 2.0, w / 2.0, -1, 1), orientation=DGG.VERTICAL)
        self.horizontalScroll = self.createcomponent('horizontalScroll', (), None, DirectScrollBar, (self,), borderWidth=self['borderWidth'], frameSize=(-1, 1, -w / 2.0, w / 2.0), orientation=DGG.HORIZONTAL)
        self.guiItem.setVerticalSlider(self.verticalScroll.guiItem)
        self.guiItem.setHorizontalSlider(self.horizontalScroll.guiItem)
        self.canvas = NodePath(self.guiItem.getCanvasNode())
        self.initialiseoptions(DirectScrolledFrame)

    def setScrollBarWidth(self):
        if False:
            while True:
                i = 10
        if self.fInit:
            return
        w = self['scrollBarWidth']
        self.verticalScroll['frameSize'] = (-w / 2.0, w / 2.0, self.verticalScroll['frameSize'][2], self.verticalScroll['frameSize'][3])
        self.horizontalScroll['frameSize'] = (self.horizontalScroll['frameSize'][0], self.horizontalScroll['frameSize'][1], -w / 2.0, w / 2.0)

    def setCanvasSize(self):
        if False:
            i = 10
            return i + 15
        f = self['canvasSize']
        self.guiItem.setVirtualFrame(f[0], f[1], f[2], f[3])

    def getCanvas(self):
        if False:
            i = 10
            return i + 15
        'Returns the NodePath of the virtual canvas.  Nodes parented to this\n        canvas will show inside the scrolled area.\n        '
        return self.canvas

    def setManageScrollBars(self):
        if False:
            for i in range(10):
                print('nop')
        self.guiItem.setManagePieces(self['manageScrollBars'])

    def setAutoHideScrollBars(self):
        if False:
            for i in range(10):
                print('nop')
        self.guiItem.setAutoHide(self['autoHideScrollBars'])

    def commandFunc(self):
        if False:
            print('Hello World!')
        if self['command']:
            self['command'](*self['extraArgs'])

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        for child in self.canvas.getChildren():
            childGui = self.guiDict.get(child.getName())
            if childGui:
                childGui.destroy()
            else:
                parts = child.getName().split('-')
                simpleChildGui = self.guiDict.get(parts[-1])
                if simpleChildGui:
                    simpleChildGui.destroy()
        if self.verticalScroll:
            self.verticalScroll.destroy()
        if self.horizontalScroll:
            self.horizontalScroll.destroy()
        self.verticalScroll = None
        self.horizontalScroll = None
        DirectFrame.destroy(self)