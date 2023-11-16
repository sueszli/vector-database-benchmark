from direct.tkwidgets import Valuator
from direct.tkwidgets import Floater
from direct.tkwidgets import Slider
import Pmw
from direct.tkwidgets.VectorWidgets import VectorEntry
from tkinter.colorchooser import askcolor

class seColorEntry(VectorEntry):

    def __init__(self, parent=None, **kw):
        if False:
            i = 10
            return i + 15
        optiondefs = (('dim', 3, Pmw.INITOPT), ('type', 'slider', Pmw.INITOPT), ('fGroup_labels', ('R', 'G', 'B'), None), ('min', 0.0, None), ('max', 255.0, None), ('nuDigits', 0, None), ('valuator_resolution', 1.0, None))
        self.defineoptions(kw, optiondefs)
        VectorEntry.__init__(self, parent, dim=self['dim'])
        self.addMenuItem('Popup color picker', command=lambda s=self: s.popupColorPicker())
        self.initialiseoptions(seColorEntry)

    def popupColorPicker(self):
        if False:
            while True:
                i = 10
        color = askcolor(parent=self.interior(), initialcolor=tuple(self.get()[:3]))[0]
        if color:
            self.set((color[0], color[1], color[2]))