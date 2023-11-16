"""This module contains the DirectButton class.

See the :ref:`directbutton` page in the programming manual for a more
in-depth explanation and an example of how to use this class.
"""
__all__ = ['DirectButton']
from panda3d.core import Mat4, MouseButton, PGButton
from . import DirectGuiGlobals as DGG
from .DirectFrame import DirectFrame

class DirectButton(DirectFrame):
    """
    DirectButton(parent) - Create a DirectGuiWidget which responds
    to mouse clicks and execute a callback function if defined
    """

    def __init__(self, parent=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        optiondefs = (('pgFunc', PGButton, None), ('numStates', 4, None), ('state', DGG.NORMAL, None), ('relief', DGG.RAISED, None), ('invertedFrames', (1,), None), ('command', None, None), ('extraArgs', [], None), ('commandButtons', (DGG.LMB,), self.setCommandButtons), ('rolloverSound', DGG.getDefaultRolloverSound(), self.setRolloverSound), ('clickSound', DGG.getDefaultClickSound(), self.setClickSound), ('pressEffect', 1, DGG.INITOPT))
        self.defineoptions(kw, optiondefs)
        DirectFrame.__init__(self, parent)
        pressEffectNP = None
        if self['pressEffect']:
            pressEffectNP = self.stateNodePath[1].attachNewNode('pressEffect', 1)
            self.stateNodePath[1] = pressEffectNP
        self.initialiseoptions(DirectButton)
        if pressEffectNP:
            bounds = self.getBounds()
            centerX = (bounds[0] + bounds[1]) / 2
            centerY = (bounds[2] + bounds[3]) / 2
            mat = Mat4.translateMat(-centerX, 0, -centerY) * Mat4.scaleMat(0.98) * Mat4.translateMat(centerX, 0, centerY)
            pressEffectNP.setMat(mat)

    def setCommandButtons(self):
        if False:
            while True:
                i = 10
        if DGG.LMB in self['commandButtons']:
            self.guiItem.addClickButton(MouseButton.one())
            self.bind(DGG.B1CLICK, self.commandFunc)
        else:
            self.unbind(DGG.B1CLICK)
            self.guiItem.removeClickButton(MouseButton.one())
        if DGG.MMB in self['commandButtons']:
            self.guiItem.addClickButton(MouseButton.two())
            self.bind(DGG.B2CLICK, self.commandFunc)
        else:
            self.unbind(DGG.B2CLICK)
            self.guiItem.removeClickButton(MouseButton.two())
        if DGG.RMB in self['commandButtons']:
            self.guiItem.addClickButton(MouseButton.three())
            self.bind(DGG.B3CLICK, self.commandFunc)
        else:
            self.unbind(DGG.B3CLICK)
            self.guiItem.removeClickButton(MouseButton.three())

    def commandFunc(self, event):
        if False:
            print('Hello World!')
        if self['command']:
            self['command'](*self['extraArgs'])

    def setClickSound(self):
        if False:
            print('Hello World!')
        clickSound = self['clickSound']
        self.guiItem.clearSound(DGG.B1PRESS + self.guiId)
        self.guiItem.clearSound(DGG.B2PRESS + self.guiId)
        self.guiItem.clearSound(DGG.B3PRESS + self.guiId)
        if clickSound:
            if DGG.LMB in self['commandButtons']:
                self.guiItem.setSound(DGG.B1PRESS + self.guiId, clickSound)
            if DGG.MMB in self['commandButtons']:
                self.guiItem.setSound(DGG.B2PRESS + self.guiId, clickSound)
            if DGG.RMB in self['commandButtons']:
                self.guiItem.setSound(DGG.B3PRESS + self.guiId, clickSound)

    def setRolloverSound(self):
        if False:
            i = 10
            return i + 15
        rolloverSound = self['rolloverSound']
        if rolloverSound:
            self.guiItem.setSound(DGG.ENTER + self.guiId, rolloverSound)
        else:
            self.guiItem.clearSound(DGG.ENTER + self.guiId)