"""Defines the DirectScrollBar class.

See the :ref:`directscrollbar` page in the programming manual for a more
in-depth explanation and an example of how to use this class.
"""
__all__ = ['DirectScrollBar']
from panda3d.core import PGSliderBar, Vec3
from . import DirectGuiGlobals as DGG
from .DirectFrame import DirectFrame
from .DirectButton import DirectButton

class DirectScrollBar(DirectFrame):
    """
    DirectScrollBar -- a widget which represents a scroll bar the user can
    use for paging through a large document or panel.
    """

    def __init__(self, parent=None, **kw):
        if False:
            return 10
        optiondefs = (('pgFunc', PGSliderBar, None), ('state', DGG.NORMAL, None), ('frameColor', (0.6, 0.6, 0.6, 1), None), ('range', (0, 1), self.setRange), ('value', 0, self.__setValue), ('scrollSize', 0.01, self.setScrollSize), ('pageSize', 0.1, self.setPageSize), ('orientation', DGG.HORIZONTAL, self.setOrientation), ('manageButtons', 1, self.setManageButtons), ('resizeThumb', 1, self.setResizeThumb), ('command', None, None), ('extraArgs', [], None))
        if kw.get('orientation') in (DGG.VERTICAL, DGG.VERTICAL_INVERTED):
            optiondefs += (('frameSize', (-0.04, 0.04, -0.5, 0.5), None),)
        else:
            optiondefs += (('frameSize', (-0.5, 0.5, -0.04, 0.04), None),)
        self.defineoptions(kw, optiondefs)
        DirectFrame.__init__(self, parent)
        self.thumb = self.createcomponent('thumb', (), None, DirectButton, (self,), borderWidth=self['borderWidth'])
        self.incButton = self.createcomponent('incButton', (), None, DirectButton, (self,), borderWidth=self['borderWidth'])
        self.decButton = self.createcomponent('decButton', (), None, DirectButton, (self,), borderWidth=self['borderWidth'])
        if self.decButton['frameSize'] is None and self.decButton.bounds == [0.0, 0.0, 0.0, 0.0]:
            f = self['frameSize']
            if self['orientation'] == DGG.HORIZONTAL:
                self.decButton['frameSize'] = (f[0] * 0.05, f[1] * 0.05, f[2], f[3])
            else:
                self.decButton['frameSize'] = (f[0], f[1], f[2] * 0.05, f[3] * 0.05)
        if self.incButton['frameSize'] is None and self.incButton.bounds == [0.0, 0.0, 0.0, 0.0]:
            f = self['frameSize']
            if self['orientation'] == DGG.HORIZONTAL:
                self.incButton['frameSize'] = (f[0] * 0.05, f[1] * 0.05, f[2], f[3])
            else:
                self.incButton['frameSize'] = (f[0], f[1], f[2] * 0.05, f[3] * 0.05)
        self._lastOrientation = self['orientation']
        self.guiItem.setThumbButton(self.thumb.guiItem)
        self.guiItem.setLeftButton(self.decButton.guiItem)
        self.guiItem.setRightButton(self.incButton.guiItem)
        self.bind(DGG.ADJUST, self.commandFunc)
        self.initialiseoptions(DirectScrollBar)

    def setRange(self):
        if False:
            for i in range(10):
                print('nop')
        v = self['value']
        r = self['range']
        self.guiItem.setRange(r[0], r[1])
        self['value'] = v

    def __setValue(self):
        if False:
            for i in range(10):
                print('nop')
        self.guiItem.setValue(self['value'])

    def setValue(self, value):
        if False:
            while True:
                i = 10
        self['value'] = value

    def getValue(self):
        if False:
            print('Hello World!')
        return self.guiItem.getValue()

    def getRatio(self):
        if False:
            i = 10
            return i + 15
        return self.guiItem.getRatio()

    def setScrollSize(self):
        if False:
            i = 10
            return i + 15
        self.guiItem.setScrollSize(self['scrollSize'])

    def setPageSize(self):
        if False:
            return 10
        self.guiItem.setPageSize(self['pageSize'])

    def scrollStep(self, stepCount):
        if False:
            print('Hello World!')
        'Scrolls the indicated number of steps forward.  If\n        stepCount is negative, scrolls backward.'
        self['value'] = self.guiItem.getValue() + self.guiItem.getScrollSize() * stepCount

    def scrollPage(self, pageCount):
        if False:
            while True:
                i = 10
        'Scrolls the indicated number of pages forward.  If\n        pageCount is negative, scrolls backward.'
        self['value'] = self.guiItem.getValue() + self.guiItem.getPageSize() * pageCount

    def setOrientation(self):
        if False:
            return 10
        if self['orientation'] == DGG.HORIZONTAL:
            if self._lastOrientation in (DGG.VERTICAL, DGG.VERTICAL_INVERTED):
                fpre = self['frameSize']
                self['frameSize'] = (fpre[2], fpre[3], fpre[0], fpre[1])
                f = self.decButton['frameSize']
                self.decButton['frameSize'] = (f[2], f[3], f[0], f[1])
                f = self.incButton['frameSize']
                self.incButton['frameSize'] = (f[2], f[3], f[0], f[1])
            self.guiItem.setAxis(Vec3(1, 0, 0))
        elif self['orientation'] == DGG.VERTICAL:
            if self._lastOrientation == DGG.HORIZONTAL:
                fpre = self['frameSize']
                self['frameSize'] = (fpre[2], fpre[3], fpre[0], fpre[1])
                f = self.decButton['frameSize']
                self.decButton['frameSize'] = (f[2], f[3], f[0], f[1])
                f = self.incButton['frameSize']
                self.incButton['frameSize'] = (f[2], f[3], f[0], f[1])
            self.guiItem.setAxis(Vec3(0, 0, -1))
        elif self['orientation'] == DGG.VERTICAL_INVERTED:
            if self._lastOrientation == DGG.HORIZONTAL:
                fpre = self['frameSize']
                self['frameSize'] = (fpre[2], fpre[3], fpre[0], fpre[1])
                f = self.decButton['frameSize']
                self.decButton['frameSize'] = (f[2], f[3], f[0], f[1])
                f = self.incButton['frameSize']
                self.incButton['frameSize'] = (f[2], f[3], f[0], f[1])
            self.guiItem.setAxis(Vec3(0, 0, 1))
        else:
            raise ValueError('Invalid value for orientation: %s' % self['orientation'])
        self._lastOrientation = self['orientation']

    def setManageButtons(self):
        if False:
            i = 10
            return i + 15
        self.guiItem.setManagePieces(self['manageButtons'])

    def setResizeThumb(self):
        if False:
            for i in range(10):
                print('nop')
        self.guiItem.setResizeThumb(self['resizeThumb'])

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        self.thumb.destroy()
        del self.thumb
        self.incButton.destroy()
        del self.incButton
        self.decButton.destroy()
        del self.decButton
        DirectFrame.destroy(self)

    def commandFunc(self):
        if False:
            print('Hello World!')
        self._optionInfo['value'][DGG._OPT_VALUE] = self.guiItem.getValue()
        if self['command']:
            self['command'](*self['extraArgs'])