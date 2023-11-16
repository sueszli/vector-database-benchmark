"""Undocumented Module"""
__all__ = ['VectorEntry', 'Vector2Entry', 'Vector3Entry', 'Vector4Entry', 'ColorEntry']
from . import Valuator
import Pmw
import tkinter as tk
from tkinter.colorchooser import askcolor

class VectorEntry(Pmw.MegaWidget):

    def __init__(self, parent=None, **kw):
        if False:
            while True:
                i = 10
        DEFAULT_DIM = 3
        DEFAULT_VALUE = [0.0] * kw.get('dim', DEFAULT_DIM)
        DEFAULT_LABELS = ['v[%d]' % x for x in range(kw.get('dim', DEFAULT_DIM))]
        INITOPT = Pmw.INITOPT
        optiondefs = (('dim', DEFAULT_DIM, INITOPT), ('value', DEFAULT_VALUE, INITOPT), ('resetValue', DEFAULT_VALUE, None), ('label_width', 12, None), ('labelIpadx', 2, None), ('command', None, None), ('entryWidth', 8, self._updateEntryWidth), ('relief', tk.GROOVE, self._updateRelief), ('bd', 2, self._updateBorderWidth), ('text', 'Vector:', self._updateText), ('min', None, self._updateValidate), ('max', None, self._updateValidate), ('numDigits', 2, self._setSigDigits), ('type', 'floater', None), ('state', 'normal', self._setState))
        self.defineoptions(kw, optiondefs)
        Pmw.MegaWidget.__init__(self, parent)
        self._value = list(self['value'])
        self['resetValue'] = self['value']
        self._floaters = None
        self.entryFormat = '%.2f'
        interior = self.interior()
        self._label = self.createcomponent('label', (), None, tk.Menubutton, (interior,), text=self['text'], activebackground='#909090')
        self.menu = self._label['menu'] = tk.Menu(self._label)
        self.menu.add_command(label='Reset', command=self.reset)
        self.menu.add_command(label='Popup sliders', command=self.popupSliders)
        self._label.pack(side=tk.LEFT, fill=tk.X, ipadx=self['labelIpadx'])
        self.variableList = []
        self.entryList = []
        for index in range(self['dim']):
            var = tk.StringVar()
            self.variableList.append(var)
            entry = self.createcomponent('entryField%d' % index, (('entry%d' % index, 'entryField%d_entry' % index),), 'Entry', Pmw.EntryField, (interior,), entry_justify=tk.RIGHT, entry_textvariable=var, command=lambda s=self, i=index: s._entryUpdateAt(i))
            entry.pack(side=tk.LEFT, expand=1, fill=tk.X)
            self.entryList.append(entry)
        self._floaters = self.createcomponent('floaterGroup', (('fGroup', 'floaterGroup'), ('valuator', 'floaterGroup_valuator')), None, Valuator.ValuatorGroupPanel, (self.interior(),), dim=self['dim'], type=self['type'], command=self.set)
        self._floaters.userdeletefunc(self._floaters.withdraw)
        self._floaters.withdraw()
        self.set(self['value'])
        self.entryBackground = self.cget('Entry_entry_background')
        self.initialiseoptions(VectorEntry)

    def label(self):
        if False:
            return 10
        return self._label

    def entry(self, index):
        if False:
            return 10
        return self.entryList[index]

    def floaters(self):
        if False:
            i = 10
            return i + 15
        return self._floaters

    def _clearFloaters(self):
        if False:
            while True:
                i = 10
        self._floaters.withdraw()

    def _updateText(self):
        if False:
            for i in range(10):
                print('nop')
        self._label['text'] = self['text']

    def _updateRelief(self):
        if False:
            i = 10
            return i + 15
        self.interior()['relief'] = self['relief']

    def _updateBorderWidth(self):
        if False:
            i = 10
            return i + 15
        self.interior()['bd'] = self['bd']

    def _updateEntryWidth(self):
        if False:
            print('Hello World!')
        self['Entry_entry_width'] = self['entryWidth']

    def _setSigDigits(self):
        if False:
            i = 10
            return i + 15
        sd = self['numDigits']
        self.entryFormat = '%.' + '%d' % sd + 'f'
        self.configure(valuator_numDigits=sd)
        for index in range(self['dim']):
            self._refreshEntry(index)

    def _updateValidate(self):
        if False:
            print('Hello World!')
        self.configure(Entry_validate={'validator': 'real', 'min': self['min'], 'max': self['max'], 'minstrict': 0, 'maxstrict': 0})
        self.configure(valuator_min=self['min'], valuator_max=self['max'])

    def get(self):
        if False:
            while True:
                i = 10
        return self._value

    def getAt(self, index):
        if False:
            i = 10
            return i + 15
        return self._value[index]

    def set(self, value, fCommand=1):
        if False:
            return 10
        if type(value) in (float, int):
            value = [value] * self['dim']
        for i in range(self['dim']):
            self._value[i] = value[i]
            self.variableList[i].set(self.entryFormat % value[i])
        self.action(fCommand)

    def setAt(self, index, value, fCommand=1):
        if False:
            i = 10
            return i + 15
        self.variableList[index].set(self.entryFormat % value)
        self._value[index] = value
        self.action(fCommand)

    def _entryUpdateAt(self, index):
        if False:
            return 10
        entryVar = self.variableList[index]
        try:
            newVal = float(entryVar.get())
        except ValueError:
            return
        if self['min'] is not None:
            if newVal < self['min']:
                newVal = self['min']
        if self['max'] is not None:
            if newVal > self['max']:
                newVal = self['max']
        self._value[index] = newVal
        self._refreshEntry(index)
        self.action()

    def _refreshEntry(self, index):
        if False:
            return 10
        self.variableList[index].set(self.entryFormat % self._value[index])
        self.entryList[index].checkentry()

    def _refreshFloaters(self):
        if False:
            i = 10
            return i + 15
        if self._floaters:
            self._floaters.set(self._value, 0)

    def action(self, fCommand=1):
        if False:
            i = 10
            return i + 15
        self._refreshFloaters()
        if fCommand and self['command'] is not None:
            self['command'](self._value)

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.set(self['resetValue'])

    def addMenuItem(self, label='', command=None):
        if False:
            i = 10
            return i + 15
        self.menu.add_command(label=label, command=command)

    def popupSliders(self):
        if False:
            while True:
                i = 10
        self._floaters.set(self.get()[:])
        self._floaters.show()

    def _setState(self):
        if False:
            return 10
        if self['state'] == 'disabled':
            self.configure(Entry_entry_state='disabled')
            self.configure(Entry_entry_background='#C0C0C0')
            self.component('fGroup').configure(valuator_state='disabled')
            self.component('fGroup').configure(valuator_entry_state='disabled')
            self.component('fGroup').configure(valuator_entry_background='#C0C0C0')
        else:
            self.configure(Entry_entry_state='normal')
            self.configure(Entry_entry_background=self.entryBackground)
            self.component('fGroup').configure(valuator_state='normal')
            self.component('fGroup').configure(valuator_entry_state='normal')
            self.component('fGroup').configure(valuator_entry_background=self.entryBackground)

class Vector2Entry(VectorEntry):

    def __init__(self, parent=None, **kw):
        if False:
            while True:
                i = 10
        optiondefs = (('dim', 2, Pmw.INITOPT), ('fGroup_labels', ('X', 'Y', 'Z'), None))
        self.defineoptions(kw, optiondefs)
        VectorEntry.__init__(self, parent, dim=self['dim'])
        self.initialiseoptions(Vector2Entry)

class Vector3Entry(VectorEntry):

    def __init__(self, parent=None, **kw):
        if False:
            i = 10
            return i + 15
        optiondefs = (('dim', 3, Pmw.INITOPT), ('fGroup_labels', ('X', 'Y', 'Z'), None))
        self.defineoptions(kw, optiondefs)
        VectorEntry.__init__(self, parent, dim=self['dim'])
        self.initialiseoptions(Vector3Entry)

class Vector4Entry(VectorEntry):

    def __init__(self, parent=None, **kw):
        if False:
            i = 10
            return i + 15
        optiondefs = (('dim', 4, Pmw.INITOPT), ('fGroup_labels', ('X', 'Y', 'Z', 'W'), None))
        self.defineoptions(kw, optiondefs)
        VectorEntry.__init__(self, parent, dim=self['dim'])
        self.initialiseoptions(Vector4Entry)

class ColorEntry(VectorEntry):

    def __init__(self, parent=None, **kw):
        if False:
            while True:
                i = 10
        optiondefs = (('dim', 4, Pmw.INITOPT), ('type', 'slider', Pmw.INITOPT), ('fGroup_labels', ('R', 'G', 'B', 'A'), None), ('min', 0.0, None), ('max', 255.0, None), ('nuDigits', 0, None), ('valuator_resolution', 1.0, None))
        self.defineoptions(kw, optiondefs)
        VectorEntry.__init__(self, parent, dim=self['dim'])
        self.addMenuItem('Popup color picker', command=lambda s=self: s.popupColorPicker())
        self.initialiseoptions(ColorEntry)

    def popupColorPicker(self):
        if False:
            print('Hello World!')
        color = askcolor(parent=self.interior(), initialcolor=tuple(self.get()[:3]))[0]
        if color:
            self.set((color[0], color[1], color[2], self.getAt(3)))