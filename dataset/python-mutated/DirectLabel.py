"""Contains the DirectLabel class.

See the :ref:`directlabel` page in the programming manual for a more in-depth
explanation and an example of how to use this class.
"""
__all__ = ['DirectLabel']
from panda3d.core import PGItem
from .DirectFrame import DirectFrame

class DirectLabel(DirectFrame):
    """
    DirectLabel(parent) - Create a DirectGuiWidget which has multiple
    states.  User explicitly chooses a state to display
    """

    def __init__(self, parent=None, **kw):
        if False:
            print('Hello World!')
        optiondefs = (('pgFunc', PGItem, None), ('numStates', 1, None), ('state', self.inactiveInitState, None), ('activeState', 0, self.setActiveState))
        self.defineoptions(kw, optiondefs)
        DirectFrame.__init__(self, parent)
        self.initialiseoptions(DirectLabel)

    def setActiveState(self):
        if False:
            print('Hello World!')
        ' setActiveState - change label to specifed state '
        self.guiItem.setState(self['activeState'])