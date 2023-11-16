"""Defines the DirectSlider class.

See the :ref:`directslider` page in the programming manual for a more
in-depth explanation and an example of how to use this class.
"""
__all__ = ['DirectSlider']
from panda3d.core import PGSliderBar, Vec3
from . import DirectGuiGlobals as DGG
from .DirectFrame import DirectFrame
from .DirectButton import DirectButton
from math import isnan

class DirectSlider(DirectFrame):
    """
    DirectSlider -- a widget which represents a slider that the
    user can pull left and right to represent a continuous value.
    """

    def __init__(self, parent=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        optiondefs = (('pgFunc', PGSliderBar, None), ('state', DGG.NORMAL, None), ('frameColor', (0.6, 0.6, 0.6, 1), None), ('range', (0, 1), self.setRange), ('value', 0, self.__setValue), ('scrollSize', 0.01, self.setScrollSize), ('pageSize', 0.1, self.setPageSize), ('orientation', DGG.HORIZONTAL, self.setOrientation), ('command', None, None), ('extraArgs', [], None))
        if kw.get('orientation') in (DGG.VERTICAL, DGG.VERTICAL_INVERTED):
            optiondefs += (('frameSize', (-0.08, 0.08, -1, 1), None), ('frameVisibleScale', (0.25, 1), None))
        else:
            optiondefs += (('frameSize', (-1, 1, -0.08, 0.08), None), ('frameVisibleScale', (1, 0.25), None))
        self.defineoptions(kw, optiondefs)
        DirectFrame.__init__(self, parent)
        self.thumb = self.createcomponent('thumb', (), None, DirectButton, (self,), borderWidth=self['borderWidth'])
        if self.thumb['frameSize'] is None:
            f = self['frameSize']
            if self['orientation'] == DGG.HORIZONTAL:
                self.thumb['frameSize'] = (f[0] * 0.05, f[1] * 0.05, f[2], f[3])
            else:
                self.thumb['frameSize'] = (f[0], f[1], f[2] * 0.05, f[3] * 0.05)
        self._lastOrientation = self['orientation']
        self.guiItem.setThumbButton(self.thumb.guiItem)
        self.bind(DGG.ADJUST, self.commandFunc)
        self.initialiseoptions(DirectSlider)

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
            print('Hello World!')
        value = self['value']
        assert not isnan(value)
        self.guiItem.setValue(value)

    def setValue(self, value):
        if False:
            while True:
                i = 10
        assert not isnan(value)
        self['value'] = value

    def getValue(self):
        if False:
            i = 10
            return i + 15
        return self.guiItem.getValue()

    def getRatio(self):
        if False:
            i = 10
            return i + 15
        return self.guiItem.getRatio()

    def setScrollSize(self):
        if False:
            print('Hello World!')
        self.guiItem.setScrollSize(self['scrollSize'])

    def setPageSize(self):
        if False:
            return 10
        self.guiItem.setPageSize(self['pageSize'])

    def setOrientation(self):
        if False:
            i = 10
            return i + 15
        if self['orientation'] == DGG.HORIZONTAL:
            if self._lastOrientation in (DGG.VERTICAL, DGG.VERTICAL_INVERTED):
                fpre = self['frameSize']
                self['frameSize'] = (fpre[2], fpre[3], fpre[0], fpre[1])
                tf = self.thumb['frameSize']
                self.thumb['frameSize'] = (tf[2], tf[3], tf[0], tf[1])
            self.guiItem.setAxis(Vec3(1, 0, 0))
            self['frameVisibleScale'] = (1, 0.25)
        elif self['orientation'] == DGG.VERTICAL:
            if self._lastOrientation == DGG.HORIZONTAL:
                fpre = self['frameSize']
                self['frameSize'] = (fpre[2], fpre[3], fpre[0], fpre[1])
                tf = self.thumb['frameSize']
                self.thumb['frameSize'] = (tf[2], tf[3], tf[0], tf[1])
            self.guiItem.setAxis(Vec3(0, 0, 1))
            self['frameVisibleScale'] = (0.25, 1)
        elif self['orientation'] == DGG.VERTICAL_INVERTED:
            if self._lastOrientation == DGG.HORIZONTAL:
                fpre = self['frameSize']
                self['frameSize'] = (fpre[2], fpre[3], fpre[0], fpre[1])
                tf = self.thumb['frameSize']
                self.thumb['frameSize'] = (tf[2], tf[3], tf[0], tf[1])
            self.guiItem.setAxis(Vec3(0, 0, -1))
            self['frameVisibleScale'] = (0.25, 1)
        else:
            raise ValueError('Invalid value for orientation: %s' % self['orientation'])
        self._lastOrientation = self['orientation']

    def destroy(self):
        if False:
            return 10
        if hasattr(self, 'thumb'):
            self.thumb.destroy()
            del self.thumb
        DirectFrame.destroy(self)

    def commandFunc(self):
        if False:
            for i in range(10):
                print('nop')
        self._optionInfo['value'][DGG._OPT_VALUE] = self.guiItem.getValue()
        if self['command']:
            self['command'](*self['extraArgs'])