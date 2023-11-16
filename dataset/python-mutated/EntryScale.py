"""
EntryScale Class: Scale with a label, and a linked and validated entry
"""
__all__ = ['EntryScale', 'EntryScaleGroup']
from panda3d.core import Vec4
import Pmw
import tkinter as tk
from tkinter.simpledialog import askfloat, askstring
from tkinter.colorchooser import askcolor

class EntryScale(Pmw.MegaWidget):
    """Scale with linked and validated entry"""

    def __init__(self, parent=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        optiondefs = (('state', None, None), ('value', 0.0, Pmw.INITOPT), ('resolution', 0.001, None), ('command', None, None), ('preCallback', None, None), ('postCallback', None, None), ('callbackData', [], None), ('min', 0.0, self._updateValidate), ('max', 100.0, self._updateValidate), ('text', 'EntryScale', self._updateLabelText), ('numDigits', 2, self._setSigDigits))
        self.defineoptions(kw, optiondefs)
        Pmw.MegaWidget.__init__(self, parent)
        self.value = self['value']
        self.entryFormat = '%.2f'
        self.fScaleCommand = 0
        interior = self.interior()
        interior.configure(relief=tk.GROOVE, borderwidth=2)
        self.labelFrame = self.createcomponent('frame', (), None, tk.Frame, interior)
        self.entryValue = tk.StringVar()
        self.entryValue.set(self['value'])
        self.entry = self.createcomponent('entryField', (('entry', 'entryField_entry'),), None, Pmw.EntryField, self.labelFrame, entry_width=10, validate={'validator': 'real', 'min': self['min'], 'max': self['max'], 'minstrict': 0, 'maxstrict': 0}, entry_justify='right', entry_textvar=self.entryValue, command=self._entryCommand)
        self.entry.pack(side='left', padx=4)
        self.label = self.createcomponent('label', (), None, tk.Label, self.labelFrame, text=self['text'], width=12, anchor='center', font='Arial 12 bold')
        self.label.pack(side='left', expand=1, fill='x')
        self.label.bind('<Button-3>', self.askForLabel)
        self.labelFrame.pack(expand=1, fill='both')
        self.minMaxFrame = self.createcomponent('mmFrame', (), None, tk.Frame, interior)
        self.minLabel = self.createcomponent('minLabel', (), None, tk.Label, self.minMaxFrame, text=repr(self['min']), relief=tk.FLAT, width=5, anchor=tk.W, font='Arial 8')
        self.minLabel.pack(side='left', fill='x')
        self.minLabel.bind('<Button-3>', self.askForMin)
        self.scale = self.createcomponent('scale', (), None, tk.Scale, self.minMaxFrame, command=self._scaleCommand, orient='horizontal', length=150, from_=self['min'], to=self['max'], resolution=self['resolution'], showvalue=0)
        self.scale.pack(side='left', expand=1, fill='x')
        self.scale.set(self['value'])
        self.scale.bind('<Button-1>', self.__onPress)
        self.scale.bind('<ButtonRelease-1>', self.__onRelease)
        self.scale.bind('<Button-3>', self.askForResolution)
        self.maxLabel = self.createcomponent('maxLabel', (), None, tk.Label, self.minMaxFrame, text=repr(self['max']), relief=tk.FLAT, width=5, anchor=tk.E, font='Arial 8')
        self.maxLabel.bind('<Button-3>', self.askForMax)
        self.maxLabel.pack(side='left', fill='x')
        self.minMaxFrame.pack(expand=1, fill='both')
        self.initialiseoptions(EntryScale)

    def askForLabel(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        newLabel = askstring(title=self['text'], prompt='New label:', initialvalue=repr(self['text']), parent=self.interior())
        if newLabel:
            self['text'] = newLabel

    def askForMin(self, event=None):
        if False:
            i = 10
            return i + 15
        newMin = askfloat(title=self['text'], prompt='New min val:', initialvalue=repr(self['min']), parent=self.interior())
        if newMin:
            self.setMin(newMin)

    def setMin(self, newMin):
        if False:
            for i in range(10):
                print('nop')
        self['min'] = newMin
        self.scale['from_'] = newMin
        self.minLabel['text'] = newMin
        self.entry.checkentry()

    def askForMax(self, event=None):
        if False:
            while True:
                i = 10
        newMax = askfloat(title=self['text'], parent=self.interior(), initialvalue=self['max'], prompt='New max val:')
        if newMax:
            self.setMax(newMax)

    def setMax(self, newMax):
        if False:
            while True:
                i = 10
        self['max'] = newMax
        self.scale['to'] = newMax
        self.maxLabel['text'] = newMax
        self.entry.checkentry()

    def askForResolution(self, event=None):
        if False:
            i = 10
            return i + 15
        newResolution = askfloat(title=self['text'], parent=self.interior(), initialvalue=self['resolution'], prompt='New resolution:')
        if newResolution:
            self.setResolution(newResolution)

    def setResolution(self, newResolution):
        if False:
            while True:
                i = 10
        self['resolution'] = newResolution
        self.scale['resolution'] = newResolution
        self.entry.checkentry()

    def _updateLabelText(self):
        if False:
            while True:
                i = 10
        self.label['text'] = self['text']

    def _updateValidate(self):
        if False:
            i = 10
            return i + 15
        self.configure(entryField_validate={'validator': 'real', 'min': self['min'], 'max': self['max'], 'minstrict': 0, 'maxstrict': 0})
        self.minLabel['text'] = self['min']
        self.scale['from_'] = self['min']
        self.scale['to'] = self['max']
        self.maxLabel['text'] = self['max']

    def _scaleCommand(self, strVal):
        if False:
            print('Hello World!')
        if not self.fScaleCommand:
            return
        self.set(float(strVal))

    def _entryCommand(self, event=None):
        if False:
            while True:
                i = 10
        try:
            val = float(self.entryValue.get())
            self.onReturn(*self['callbackData'])
            self.set(val)
            self.onReturnRelease(*self['callbackData'])
        except ValueError:
            pass

    def _setSigDigits(self):
        if False:
            print('Hello World!')
        sd = self['numDigits']
        self.entryFormat = '%.' + '%d' % sd + 'f'
        self.entryValue.set(self.entryFormat % self.value)

    def get(self):
        if False:
            i = 10
            return i + 15
        return self.value

    def set(self, newVal, fCommand=1):
        if False:
            for i in range(10):
                print('nop')
        if self['min'] is not None:
            if newVal < self['min']:
                newVal = self['min']
        if self['max'] is not None:
            if newVal > self['max']:
                newVal = self['max']
        if self['resolution'] is not None:
            newVal = round(newVal / self['resolution']) * self['resolution']
        self.value = newVal
        self.scale.set(newVal)
        self.entryValue.set(self.entryFormat % self.value)
        self.entry.checkentry()
        if fCommand and self['command'] is not None:
            self['command'](newVal)

    def onReturn(self, *args):
        if False:
            print('Hello World!')
        ' User redefinable callback executed on <Return> in entry '

    def onReturnRelease(self, *args):
        if False:
            i = 10
            return i + 15
        ' User redefinable callback executed on <Return> release in entry '

    def __onPress(self, event):
        if False:
            for i in range(10):
                print('nop')
        if self['preCallback']:
            self['preCallback'](*self['callbackData'])
        self.fScaleCommand = 1

    def onPress(self, *args):
        if False:
            while True:
                i = 10
        ' User redefinable callback executed on button press '

    def __onRelease(self, event):
        if False:
            i = 10
            return i + 15
        self.fScaleCommand = 0
        if self['postCallback']:
            self['postCallback'](*self['callbackData'])

    def onRelease(self, *args):
        if False:
            while True:
                i = 10
        ' User redefinable callback executed on button release '

class EntryScaleGroup(Pmw.MegaToplevel):

    def __init__(self, parent=None, **kw):
        if False:
            print('Hello World!')
        DEFAULT_DIM = 1
        DEFAULT_VALUE = [0.0] * kw.get('dim', DEFAULT_DIM)
        DEFAULT_LABELS = ['v[%d]' % x for x in range(kw.get('dim', DEFAULT_DIM))]
        INITOPT = Pmw.INITOPT
        optiondefs = (('dim', DEFAULT_DIM, INITOPT), ('side', tk.TOP, INITOPT), ('title', 'Group', None), ('value', DEFAULT_VALUE, INITOPT), ('command', None, None), ('preCallback', None, None), ('postCallback', None, None), ('labels', DEFAULT_LABELS, self._updateLabels), ('fDestroy', 0, INITOPT))
        self.defineoptions(kw, optiondefs)
        Pmw.MegaToplevel.__init__(self, parent)
        interior = self.interior()
        self._value = list(self['value'])
        self.balloon = Pmw.Balloon()
        menubar = self.createcomponent('menubar', (), None, Pmw.MenuBar, (interior,), balloon=self.balloon)
        menubar.pack(fill=tk.X)
        menubar.addmenu('EntryScale Group', 'EntryScale Group Operations')
        menubar.addmenuitem('EntryScale Group', 'command', 'Reset the EntryScale Group panel', label='Reset', command=lambda s=self: s.reset())
        if self['fDestroy']:
            dismissCommand = self.destroy
        else:
            dismissCommand = self.withdraw
        menubar.addmenuitem('EntryScale Group', 'command', 'Dismiss EntryScale Group panel', label='Dismiss', command=dismissCommand)
        menubar.addmenu('Help', 'EntryScale Group Help Operations')
        self.toggleBalloonVar = tk.IntVar()
        self.toggleBalloonVar.set(0)
        menubar.addmenuitem('Help', 'checkbutton', 'Toggle balloon help', label='Balloon Help', variable=self.toggleBalloonVar, command=self.toggleBalloon)
        self.entryScaleList = []
        for index in range(self['dim']):
            f = self.createcomponent('entryScale%d' % index, (), 'Valuator', EntryScale, (interior,), value=self._value[index], text=self['labels'][index])
            f['command'] = lambda val, s=self, i=index: s._entryScaleSetAt(i, val)
            f['callbackData'] = [self]
            f.onReturn = self.__onReturn
            f.onReturnRelease = self.__onReturnRelease
            f['preCallback'] = self.__onPress
            f['postCallback'] = self.__onRelease
            f.pack(side=self['side'], expand=1, fill=tk.X)
            self.entryScaleList.append(f)
        self.set(self['value'])
        self.initialiseoptions(EntryScaleGroup)

    def _updateLabels(self):
        if False:
            print('Hello World!')
        if self['labels']:
            for index in range(self['dim']):
                self.entryScaleList[index]['text'] = self['labels'][index]

    def toggleBalloon(self):
        if False:
            i = 10
            return i + 15
        if self.toggleBalloonVar.get():
            self.balloon.configure(state='balloon')
        else:
            self.balloon.configure(state='none')

    def get(self):
        if False:
            i = 10
            return i + 15
        return self._value

    def getAt(self, index):
        if False:
            while True:
                i = 10
        return self._value[index]

    def set(self, value, fCommand=1):
        if False:
            for i in range(10):
                print('nop')
        for i in range(self['dim']):
            self._value[i] = value[i]
            self.entryScaleList[i].set(value[i], 0)
        if fCommand and self['command'] is not None:
            self['command'](self._value)

    def setAt(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self.entryScaleList[index].set(value)

    def _entryScaleSetAt(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        self._value[index] = value
        if self['command']:
            self['command'](self._value)

    def reset(self):
        if False:
            while True:
                i = 10
        self.set(self['value'])

    def __onReturn(self, esg):
        if False:
            for i in range(10):
                print('nop')
        self.onReturn(*esg.get())

    def onReturn(self, *args):
        if False:
            return 10
        ' User redefinable callback executed on button press '

    def __onReturnRelease(self, esg):
        if False:
            print('Hello World!')
        self.onReturnRelease(*esg.get())

    def onReturnRelease(self, *args):
        if False:
            return 10
        ' User redefinable callback executed on button press '

    def __onPress(self, esg):
        if False:
            for i in range(10):
                print('nop')
        if self['preCallback']:
            self['preCallback'](*esg.get())

    def onPress(self, *args):
        if False:
            for i in range(10):
                print('nop')
        ' User redefinable callback executed on button press '

    def __onRelease(self, esg):
        if False:
            print('Hello World!')
        if self['postCallback']:
            self['postCallback'](*esg.get())

    def onRelease(self, *args):
        if False:
            print('Hello World!')
        ' User redefinable callback executed on button release '

def rgbPanel(nodePath, callback=None):
    if False:
        for i in range(10):
            print('nop')
    from direct.showbase.MessengerGlobal import messenger

    def setNodePathColor(color, np=nodePath, cb=callback):
        if False:
            i = 10
            return i + 15
        np.setColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0)
        if cb:
            cb(color)
    if nodePath.hasColor():
        initColor = nodePath.getColor() * 255.0
    else:
        initColor = Vec4(255)
    esg = EntryScaleGroup(title='RGBA Panel: ' + nodePath.getName(), dim=4, labels=['R', 'G', 'B', 'A'], value=[int(initColor[0]), int(initColor[1]), int(initColor[2]), int(initColor[3])], Valuator_max=255, Valuator_resolution=1, fDestroy=1, command=setNodePathColor)
    esg.component('menubar').component('EntryScale Group-button')['text'] = 'RGBA Panel'
    menubar = esg.component('menubar')
    menubar.deletemenuitems('EntryScale Group', 1, 1)
    menubar.addmenuitem('EntryScale Group', 'command', label='Clear Color', command=lambda np=nodePath: np.clearColor())
    menubar.addmenuitem('EntryScale Group', 'command', label='Set Transparency', command=lambda np=nodePath: np.setTransparency(1))
    menubar.addmenuitem('EntryScale Group', 'command', label='Clear Transparency', command=lambda np=nodePath: np.clearTransparency())

    def popupColorPicker(esg=esg):
        if False:
            for i in range(10):
                print('nop')
        color = askcolor(parent=esg.interior(), initialcolor=tuple(esg.get()[:3]))[0]
        if color:
            esg.set((color[0], color[1], color[2], esg.getAt(3)))
    menubar.addmenuitem('EntryScale Group', 'command', label='Popup Color Picker', command=popupColorPicker)

    def printToLog(nodePath=nodePath):
        if False:
            for i in range(10):
                print('nop')
        c = nodePath.getColor()
        print('Vec4(%.3f, %.3f, %.3f, %.3f)' % (c[0], c[1], c[2], c[3]))
    menubar.addmenuitem('EntryScale Group', 'command', label='Print to log', command=printToLog)
    if esg['fDestroy']:
        dismissCommand = esg.destroy
    else:
        dismissCommand = esg.withdraw
    menubar.addmenuitem('EntryScale Group', 'command', 'Dismiss EntryScale Group panel', label='Dismiss', command=dismissCommand)

    def onRelease(r, g, b, a, nodePath=nodePath):
        if False:
            print('Hello World!')
        messenger.send('RGBPanel_setColor', [nodePath, r, g, b, a])
    esg['postCallback'] = onRelease
    return esg