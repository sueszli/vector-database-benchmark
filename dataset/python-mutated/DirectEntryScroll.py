__all__ = ['DirectEntryScroll']
from panda3d.core import NodePath, OmniBoundingVolume, PGVirtualFrame
from . import DirectGuiGlobals as DGG
from .DirectFrame import DirectFrame

class DirectEntryScroll(DirectFrame):

    def __init__(self, entry, parent=None, **kw):
        if False:
            print('Hello World!')
        optiondefs = (('pgFunc', PGVirtualFrame, None), ('relief', None, None), ('clipSize', (-1, 1, -1, 1), self.setClipSize))
        self.defineoptions(kw, optiondefs)
        DirectFrame.__init__(self, parent, **kw)
        self.canvas = None
        self.visXMin = 0.0
        self.visXMax = 0.0
        self.clipXMin = 0.0
        self.clipXMax = 0.0
        self.initialiseoptions(DirectEntryScroll)
        self.canvas = NodePath(self.guiItem.getCanvasNode())
        self.canvas.setPos(0, 0, 0)
        self.entry = None
        if entry is not None:
            self.entry = entry
            self.entry.reparentTo(self.canvas)
            self.entry.bind(DGG.CURSORMOVE, self.cursorMove)
        self.canvas.node().setBounds(OmniBoundingVolume())
        self.canvas.node().setFinal(1)
        self.resetCanvas()

    def setEntry(self, entry):
        if False:
            while True:
                i = 10
        '\n        Sets a DirectEntry element for this scroll frame. A DirectEntryScroll\n        can only hold one entry at a time, so make sure to not call this\n        function twice or call clearEntry before to make sure no entry\n        is already set.\n        '
        assert self.entry is None, 'An entry was already set for this DirectEntryScroll element'
        self.entry = entry
        self.entry.reparentTo(self.canvas)
        self.entry.bind(DGG.CURSORMOVE, self.cursorMove)

    def clearEntry(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        detaches and unbinds the entry from the scroll frame and its\n        events. You'll be responsible for destroying it.\n        "
        if self.entry is None:
            return
        self.entry.unbind(DGG.CURSORMOVE)
        self.entry.detachNode()
        self.entry = None

    def cursorMove(self, cursorX, cursorY):
        if False:
            i = 10
            return i + 15
        cursorX = self.entry.guiItem.getCursorX() * self.entry['text_scale'][0]
        canvasX = self.canvas.getX()
        visXMin = self.clipXMin - canvasX
        visXMax = self.clipXMax - canvasX
        visXCenter = (visXMin + visXMax) * 0.5
        distanceToCenter = visXCenter - cursorX
        clipExtent = self.clipXMax - self.clipXMin
        entryExtent = self.entry['text_scale'][0] * self.entry['width']
        entryWiggle = entryExtent - clipExtent
        if abs(distanceToCenter) > clipExtent * 0.5:
            self.moveToCenterCursor()

    def moveToCenterCursor(self):
        if False:
            for i in range(10):
                print('nop')
        cursorX = self.entry.guiItem.getCursorX() * self.entry['text_scale'][0]
        canvasX = self.canvas.getX()
        visXMin = self.clipXMin - canvasX
        visXMax = self.clipXMax - canvasX
        visXCenter = (visXMin + visXMax) * 0.5
        distanceToCenter = visXCenter - cursorX
        newX = canvasX + distanceToCenter
        clipExtent = self.clipXMax - self.clipXMin
        entryExtent = self.entry['text_scale'][0] * self.entry['width']
        entryWiggle = entryExtent - clipExtent
        if self.entry.guiItem.getCursorPosition() <= 0:
            newX = 0.0
        elif newX > 0.0:
            newX = 0.0
        elif newX < -entryWiggle:
            newX = -entryWiggle
        self.canvas.setX(newX)

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
        self.entry.destroy()
        self.entry = None
        DirectFrame.destroy(self)

    def getCanvas(self):
        if False:
            for i in range(10):
                print('nop')
        return self.canvas

    def setClipSize(self):
        if False:
            print('Hello World!')
        self.guiItem.setClipFrame(self['clipSize'])
        self.clipXMin = self['clipSize'][0]
        self.clipXMax = self['clipSize'][1]
        self.visXMin = self.clipXMin
        self.visXMax = self.clipXMax
        if self.canvas:
            self.resetCanvas()

    def resetCanvas(self):
        if False:
            return 10
        self.canvas.setPos(0, 0, 0)