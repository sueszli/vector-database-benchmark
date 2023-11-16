from direct.gui.DirectGui import DGG, DirectButton
from panda3d.core import PGButton

class DirectCheckBox(DirectButton):
    """
    DirectCheckBox(parent) - Create a DirectGuiWidget which responds
    to mouse clicks by setting a state of True or False and executes
    a callback function if defined.

    Uses an image swap rather than a text change to indicate state.
    """

    def __init__(self, parent=None, **kw):
        if False:
            print('Hello World!')
        optiondefs = (('pgFunc', PGButton, None), ('numStates', 4, None), ('state', DGG.NORMAL, None), ('relief', DGG.RAISED, None), ('invertedFrames', (1,), None), ('command', None, None), ('extraArgs', [], None), ('commandButtons', (DGG.LMB,), self.setCommandButtons), ('rolloverSound', DGG.getDefaultRolloverSound(), self.setRolloverSound), ('clickSound', DGG.getDefaultClickSound(), self.setClickSound), ('pressEffect', 1, DGG.INITOPT), ('uncheckedImage', None, None), ('checkedImage', None, None), ('isChecked', False, None))
        self.defineoptions(kw, optiondefs)
        DirectButton.__init__(self, parent)
        self.initialiseoptions(DirectCheckBox)

    def commandFunc(self, event):
        if False:
            while True:
                i = 10
        self['isChecked'] = not self['isChecked']
        if self['isChecked']:
            self['image'] = self['checkedImage']
        else:
            self['image'] = self['uncheckedImage']
        self.setImage()
        if self['command']:
            self['command'](*[self['isChecked']] + self['extraArgs'])