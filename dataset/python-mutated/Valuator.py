"""Undocumented Module"""
__all__ = ['Valuator', 'ValuatorGroup', 'ValuatorGroupPanel']
from panda3d.core import Vec4
from direct.directtools.DirectUtil import getTkColorString
from . import WidgetPropertiesDialog
import tkinter as tk
from tkinter.colorchooser import askcolor
import Pmw
VALUATOR_MINI = 'mini'
VALUATOR_FULL = 'full'

class Valuator(Pmw.MegaWidget):
    sfBase = 3.0
    sfDist = 7
    deadband = 5
    ' Base class for widgets used to interactively adjust numeric values '

    def __init__(self, parent=None, **kw):
        if False:
            print('Hello World!')
        INITOPT = Pmw.INITOPT
        optiondefs = (('state', tk.NORMAL, self.setState), ('relief', tk.GROOVE, None), ('borderwidth', 2, None), ('text', 'Valuator', self.setLabel), ('value', 0.0, INITOPT), ('resetValue', 0.0, None), ('min', None, None), ('max', None, None), ('resolution', None, None), ('numDigits', 2, self.setEntryFormat), ('fAdjustable', 1, None), ('command', None, None), ('commandData', [], None), ('fCommandOnInit', 0, INITOPT), ('preCallback', None, None), ('postCallback', None, None), ('callbackData', [], None))
        self.defineoptions(kw, optiondefs)
        Pmw.MegaWidget.__init__(self, parent)
        self.adjustedValue = self['value']
        interior = self.interior()
        interior.configure(relief=self['relief'], bd=self['borderwidth'])
        self.createValuator()
        self._valuator['preCallback'] = self._mouseDown
        self._valuator['postCallback'] = self._mouseUp
        if self['text'] is not None:
            self._label = self.createcomponent('label', (), None, tk.Label, (interior,), text=self['text'], font=('MS Sans Serif', 12), anchor=tk.CENTER)
        else:
            self._label = None
        self._entryVal = tk.StringVar()
        self._entry = self.createcomponent('entry', (), None, tk.Entry, (interior,), justify=tk.RIGHT, width=12, textvariable=self._entryVal)
        self._entry.bind('<Return>', self.validateEntryInput)
        self._entryBackground = self._entry.cget('background')
        self.packValuator()
        if 'resetValue' not in kw:
            self['resetValue'] = self['value']
        if self['fAdjustable']:
            self._popupMenu = tk.Menu(interior, tearoff=0)
            self.addValuatorMenuEntries()
            self._popupMenu.add_command(label='Reset', command=self.reset)
            self._popupMenu.add_command(label='Set to Zero', command=self.zero)
            self._popupMenu.add_command(label='Properties...', command=self._popupPropertiesDialog)
            if self._label:
                self._label.bind('<ButtonPress-3>', self._popupValuatorMenu)
            self._entry.bind('<ButtonPress-3>', self._popupValuatorMenu)
            self._valuator._widget.bind('<ButtonPress-3>', self._popupValuatorMenu)
            self.propertyDict = {'state': {'widget': self, 'type': 'string', 'help': 'Enter state: normal or disabled.'}, 'text': {'widget': self, 'type': 'string', 'help': 'Enter label text.'}, 'min': {'widget': self, 'type': 'real', 'fNone': 1, 'help': 'Minimum allowable value. Enter None for no minimum.'}, 'max': {'widget': self, 'type': 'real', 'fNone': 1, 'help': 'Maximum allowable value. Enter None for no maximum.'}, 'numDigits': {'widget': self, 'type': 'integer', 'help': 'Number of digits after decimal point.'}, 'resolution': {'widget': self, 'type': 'real', 'fNone': 1, 'help': 'Widget resolution. Enter None for no resolution .'}, 'resetValue': {'widget': self, 'type': 'real', 'help': 'Enter value to set widget to on reset.'}}
            self.propertyList = ['state', 'text', 'min', 'max', 'numDigits', 'resolution', 'resetValue']
            self.addValuatorPropertiesToDialog()
        self.fInit = self['fCommandOnInit']
        self.initialiseoptions(Valuator)

    def set(self, value, fCommand=1):
        if False:
            while True:
                i = 10
        "\n        Update widget's value by setting valuator, which will in\n        turn update the entry.  fCommand flag (which is passed to the\n        valuator as commandData, which is then passed in turn to\n        self.setEntry) controls command execution.\n        "
        self._valuator['commandData'] = [fCommand]
        self._valuator.set(value)
        self._valuator['commandData'] = [1]

    def get(self):
        if False:
            i = 10
            return i + 15
        ' Return current widget value '
        return self.adjustedValue

    def setEntry(self, value, fCommand=1):
        if False:
            i = 10
            return i + 15
        '\n        Update value displayed in entry, fCommand flag controls\n        command execution\n        '
        if self['min'] is not None:
            if value < self['min']:
                value = self['min']
        if self['max'] is not None:
            if value > self['max']:
                value = self['max']
        if self['resolution'] is not None:
            value = round(value / self['resolution']) * self['resolution']
        self._entryVal.set(self.entryFormat % value)
        self._valuator.updateIndicator(value)
        if fCommand and self.fInit and (self['command'] is not None):
            self['command'](*[value] + self['commandData'])
        self.adjustedValue = value
        self.fInit = 1

    def setEntryFormat(self):
        if False:
            return 10
        '\n        Change the number of significant digits in entry\n        '
        self.entryFormat = '%.' + '%df' % self['numDigits']
        self.setEntry(self.get())
        self._valuator['numDigits'] = self['numDigits']

    def validateEntryInput(self, event):
        if False:
            i = 10
            return i + 15
        ' Check validity of entry and if valid pass along to valuator '
        input = self._entryVal.get()
        try:
            self._entry.configure(background=self._entryBackground)
            newValue = float(input)
            self._preCallback()
            self.set(newValue)
            self._postCallback()
            self._valuator.set(self.adjustedValue, 0)
        except ValueError:
            self._entry.configure(background='Pink')

    def _mouseDown(self):
        if False:
            print('Hello World!')
        ' Function to execute at start of mouse interaction '
        self._preCallback()

    def _mouseUp(self):
        if False:
            while True:
                i = 10
        ' Function to execute at end of mouse interaction '
        self._postCallback()
        self._valuator.set(self.adjustedValue, 0)

    def _preCallback(self):
        if False:
            while True:
                i = 10
        if self['preCallback']:
            self['preCallback'](*self['callbackData'])

    def _postCallback(self):
        if False:
            print('Hello World!')
        if self['postCallback']:
            self['postCallback'](*self['callbackData'])

    def setState(self):
        if False:
            i = 10
            return i + 15
        ' Enable/disable widget '
        if self['state'] == tk.NORMAL:
            self._entry['state'] = tk.NORMAL
            self._entry['background'] = self._entryBackground
            self._valuator._widget['state'] = tk.NORMAL
        elif self['state'] == tk.DISABLED:
            self._entry['background'] = 'grey75'
            self._entry['state'] = tk.DISABLED
            self._valuator._widget['state'] = tk.DISABLED

    def setLabel(self):
        if False:
            for i in range(10):
                print('nop')
        " Update label's text "
        if self._label:
            self._label['text'] = self['text']

    def zero(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        self.zero()\n        Set valuator to zero\n        '
        self.set(0.0)

    def reset(self):
        if False:
            while True:
                i = 10
        '\n        self.reset()\n        Reset valuator to reset value\n        '
        self.set(self['resetValue'])

    def mouseReset(self, event):
        if False:
            print('Hello World!')
        '\n        Reset valuator to resetValue\n        '
        self.reset()

    def _popupValuatorMenu(self, event):
        if False:
            i = 10
            return i + 15
        self._popupMenu.post(event.widget.winfo_pointerx(), event.widget.winfo_pointery())

    def _popupPropertiesDialog(self):
        if False:
            return 10
        WidgetPropertiesDialog.WidgetPropertiesDialog(self.propertyDict, propertyList=self.propertyList, title='Widget Properties', parent=self.interior())

    def addPropertyToDialog(self, property, pDict):
        if False:
            print('Hello World!')
        self.propertyDict[property] = pDict
        self.propertyList.append(property)

    def createValuator(self):
        if False:
            i = 10
            return i + 15
        ' Function used by subclass to create valuator geometry '

    def packValuator(self):
        if False:
            for i in range(10):
                print('nop')
        ' Function used by subclass to pack widget '

    def addValuatorMenuEntries(self):
        if False:
            print('Hello World!')
        ' Function used by subclass to add menu entries to popup menu '

    def addValuatorPropertiesToDialog(self):
        if False:
            print('Hello World!')
        ' Function used by subclass to add properties to property dialog '
FLOATER = 'floater'
DIAL = 'dial'
ANGLEDIAL = 'angledial'
SLIDER = 'slider'

class ValuatorGroup(Pmw.MegaWidget):

    def __init__(self, parent=None, **kw):
        if False:
            return 10
        DEFAULT_DIM = 1
        DEFAULT_VALUE = [0.0] * kw.get('dim', DEFAULT_DIM)
        DEFAULT_LABELS = ['v[%d]' % x for x in range(kw.get('dim', DEFAULT_DIM))]
        INITOPT = Pmw.INITOPT
        optiondefs = (('type', FLOATER, INITOPT), ('dim', DEFAULT_DIM, INITOPT), ('side', tk.TOP, INITOPT), ('value', DEFAULT_VALUE, INITOPT), ('min', None, INITOPT), ('max', None, INITOPT), ('resolution', None, INITOPT), ('numDigits', 2, self._setNumDigits), ('labels', DEFAULT_LABELS, self._updateLabels), ('command', None, None), ('preCallback', None, None), ('postCallback', None, None), ('callbackData', [], None))
        self.defineoptions(kw, optiondefs)
        Pmw.MegaWidget.__init__(self, parent)
        interior = self.interior()
        self._value = list(self['value'])
        self._valuatorList = []
        for index in range(self['dim']):
            if self['type'] == DIAL:
                from . import Dial
                valuatorType = Dial.Dial
            elif self['type'] == ANGLEDIAL:
                from . import Dial
                valuatorType = Dial.AngleDial
            elif self['type'] == SLIDER:
                from . import Slider
                valuatorType = Slider.Slider
            else:
                from . import Floater
                valuatorType = Floater.Floater
            f = self.createcomponent('valuator%d' % index, (), 'valuator', valuatorType, (interior,), value=self._value[index], min=self['min'], max=self['max'], resolution=self['resolution'], text=self['labels'][index], command=lambda val, i=index: self._valuatorSetAt(i, val), preCallback=self._preCallback, postCallback=self._postCallback, callbackData=[self])
            f.pack(side=self['side'], expand=1, fill=tk.X)
            self._valuatorList.append(f)
        self.set(self['value'], fCommand=0)
        self.initialiseoptions(ValuatorGroup)

    def set(self, value, fCommand=1):
        if False:
            while True:
                i = 10
        for i in range(self['dim']):
            self._value[i] = value[i]
            self._valuatorList[i].set(value[i], 0)
        if fCommand and self['command'] is not None:
            self['command'](self._value)

    def setAt(self, index, value):
        if False:
            i = 10
            return i + 15
        self._valuatorList[index].set(value)

    def _valuatorSetAt(self, index, value):
        if False:
            print('Hello World!')
        self._value[index] = value
        if self['command']:
            self['command'](self._value)

    def get(self):
        if False:
            return 10
        return self._value

    def getAt(self, index):
        if False:
            print('Hello World!')
        return self._value[index]

    def _setNumDigits(self):
        if False:
            for i in range(10):
                print('nop')
        self['valuator_numDigits'] = self['numDigits']
        self.formatString = '%0.' + '%df' % self['numDigits']

    def _updateLabels(self):
        if False:
            i = 10
            return i + 15
        if self['labels']:
            for index in range(self['dim']):
                self._valuatorList[index]['text'] = self['labels'][index]

    def _preCallback(self, valGroup):
        if False:
            i = 10
            return i + 15
        if self['preCallback']:
            self['preCallback'](*valGroup.get())

    def _postCallback(self, valGroup):
        if False:
            return 10
        if self['postCallback']:
            self['postCallback'](*valGroup.get())

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self['dim']

    def __repr__(self):
        if False:
            while True:
                i = 10
        str = '[' + self.formatString % self._value[0]
        for val in self._value[1:]:
            str += ', ' + self.formatString % val
        str += ']'
        return str

class ValuatorGroupPanel(Pmw.MegaToplevel):

    def __init__(self, parent=None, **kw):
        if False:
            while True:
                i = 10
        DEFAULT_DIM = 1
        DEFAULT_VALUE = [0.0] * kw.get('dim', DEFAULT_DIM)
        DEFAULT_LABELS = ['v[%d]' % x for x in range(kw.get('dim', DEFAULT_DIM))]
        INITOPT = Pmw.INITOPT
        optiondefs = (('type', FLOATER, INITOPT), ('dim', DEFAULT_DIM, INITOPT), ('side', tk.TOP, INITOPT), ('title', 'Valuator Group', None), ('value', DEFAULT_VALUE, INITOPT), ('min', None, INITOPT), ('max', None, INITOPT), ('resolution', None, INITOPT), ('labels', DEFAULT_LABELS, self._updateLabels), ('numDigits', 2, self._setNumDigits), ('command', None, self._setCommand), ('preCallback', None, self._setPreCallback), ('postCallback', None, self._setPostCallback), ('callbackData', [], self._setCallbackData), ('fDestroy', 0, INITOPT))
        self.defineoptions(kw, optiondefs)
        Pmw.MegaToplevel.__init__(self, parent)
        interior = self.interior()
        self.balloon = Pmw.Balloon()
        menubar = self.createcomponent('menubar', (), None, Pmw.MenuBar, (interior,), balloon=self.balloon)
        menubar.pack(fill=tk.X)
        menubar.addmenu('Valuator Group', 'Valuator Group Operations')
        menubar.addmenuitem('Valuator Group', 'command', 'Reset the Valuator Group panel', label='Reset', command=lambda s=self: s.reset())
        if self['fDestroy']:
            dismissCommand = self.destroy
        else:
            dismissCommand = self.withdraw
        menubar.addmenuitem('Valuator Group', 'command', 'Dismiss Valuator Group panel', label='Dismiss', command=dismissCommand)
        menubar.addmenu('Help', 'Valuator Group Help Operations')
        self.toggleBalloonVar = tk.IntVar()
        self.toggleBalloonVar.set(0)
        menubar.addmenuitem('Help', 'checkbutton', 'Toggle balloon help', label='Balloon Help', variable=self.toggleBalloonVar, command=self.toggleBalloon)
        self.valuatorGroup = self.createcomponent('valuatorGroup', (('valuator', 'valuatorGroup_valuator'),), None, ValuatorGroup, (interior,), type=self['type'], dim=self['dim'], value=self['value'], min=self['min'], max=self['max'], resolution=self['resolution'], labels=self['labels'], command=self['command'])
        self.valuatorGroup.pack(expand=1, fill=tk.X)
        self.initialiseoptions(ValuatorGroupPanel)

    def toggleBalloon(self):
        if False:
            return 10
        if self.toggleBalloonVar.get():
            self.balloon.configure(state='balloon')
        else:
            self.balloon.configure(state='none')

    def _updateLabels(self):
        if False:
            i = 10
            return i + 15
        self.valuatorGroup['labels'] = self['labels']

    def _setNumDigits(self):
        if False:
            i = 10
            return i + 15
        self.valuatorGroup['numDigits'] = self['numDigits']

    def _setCommand(self):
        if False:
            i = 10
            return i + 15
        self.valuatorGroup['command'] = self['command']

    def _setPreCallback(self):
        if False:
            for i in range(10):
                print('nop')
        self.valuatorGroup['preCallback'] = self['preCallback']

    def _setPostCallback(self):
        if False:
            return 10
        self.valuatorGroup['postCallback'] = self['postCallback']

    def _setCallbackData(self):
        if False:
            print('Hello World!')
        self.valuatorGroup['callbackData'] = self['callbackData']

    def reset(self):
        if False:
            while True:
                i = 10
        self.set(self['value'])
Pmw.forwardmethods(ValuatorGroupPanel, ValuatorGroup, 'valuatorGroup')

def rgbPanel(nodePath, callback=None, style='mini'):
    if False:
        return 10
    from direct.showbase.MessengerGlobal import messenger

    def onRelease(r, g, b, a, nodePath=nodePath):
        if False:
            i = 10
            return i + 15
        messenger.send('RGBPanel_setColor', [nodePath, r, g, b, a])

    def popupColorPicker():
        if False:
            i = 10
            return i + 15
        color = askcolor(parent=vgp.interior(), initialcolor=tuple(vgp.get()[:3]))[0]
        if color:
            vgp.set((color[0], color[1], color[2], vgp.getAt(3)))

    def printToLog():
        if False:
            for i in range(10):
                print('nop')
        c = nodePath.getColor()
        print('Vec4(%.3f, %.3f, %.3f, %.3f)' % (c[0], c[1], c[2], c[3]))
    if nodePath.hasColor():
        initColor = nodePath.getColor() * 255.0
    else:
        initColor = Vec4(255)
    vgp = ValuatorGroupPanel(title='RGBA Panel: ' + nodePath.getName(), dim=4, labels=['R', 'G', 'B', 'A'], value=[int(initColor[0]), int(initColor[1]), int(initColor[2]), int(initColor[3])], type='slider', valuator_style=style, valuator_min=0, valuator_max=255, valuator_resolution=1, fDestroy=1)
    vgp.component('menubar').component('Valuator Group-button')['text'] = 'RGBA Panel'
    vgp['postCallback'] = onRelease
    pButton = tk.Button(vgp.interior(), text='Print to Log', bg=getTkColorString(initColor), command=printToLog)
    pButton.pack(expand=1, fill=tk.BOTH)
    menubar = vgp.component('menubar')
    menubar.deletemenuitems('Valuator Group', 1, 1)
    menubar.addmenuitem('Valuator Group', 'command', label='Clear Color', command=lambda : nodePath.clearColor())
    menubar.addmenuitem('Valuator Group', 'command', label='Set Transparency', command=lambda : nodePath.setTransparency(1))
    menubar.addmenuitem('Valuator Group', 'command', label='Clear Transparency', command=lambda : nodePath.clearTransparency())
    menubar.addmenuitem('Valuator Group', 'command', label='Popup Color Picker', command=popupColorPicker)
    menubar.addmenuitem('Valuator Group', 'command', label='Print to log', command=printToLog)
    menubar.addmenuitem('Valuator Group', 'command', 'Dismiss Valuator Group panel', label='Dismiss', command=vgp.destroy)

    def setNodePathColor(color):
        if False:
            print('Hello World!')
        nodePath.setColor(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0)
        pButton['bg'] = getTkColorString(color)
        if callback:
            callback(color)
    vgp['command'] = setNodePathColor
    return vgp

def lightRGBPanel(light, style='mini'):
    if False:
        return 10

    def popupColorPicker():
        if False:
            print('Hello World!')
        color = askcolor(parent=vgp.interior(), initialcolor=tuple(vgp.get()[:3]))[0]
        if color:
            vgp.set((color[0], color[1], color[2], vgp.getAt(3)))

    def printToLog():
        if False:
            for i in range(10):
                print('nop')
        n = light.getName()
        c = light.getColor()
        print(n + '.setColor(Vec4(%.3f, %.3f, %.3f, %.3f))' % (c[0], c[1], c[2], c[3]))
    initColor = light.getColor() * 255.0
    vgp = ValuatorGroupPanel(title='RGBA Panel: ' + light.getName(), dim=4, labels=['R', 'G', 'B', 'A'], value=[int(initColor[0]), int(initColor[1]), int(initColor[2]), int(initColor[3])], type='slider', valuator_style=style, valuator_min=0, valuator_max=255, valuator_resolution=1, fDestroy=1)
    vgp.component('menubar').component('Valuator Group-button')['text'] = 'Light Control Panel'
    pButton = tk.Button(vgp.interior(), text='Print to Log', bg=getTkColorString(initColor), command=printToLog)
    pButton.pack(expand=1, fill=tk.BOTH)
    menubar = vgp.component('menubar')
    menubar.addmenuitem('Valuator Group', 'command', label='Popup Color Picker', command=popupColorPicker)
    menubar.addmenuitem('Valuator Group', 'command', label='Print to log', command=printToLog)

    def setLightColor(color):
        if False:
            for i in range(10):
                print('nop')
        light.setColor(Vec4(color[0] / 255.0, color[1] / 255.0, color[2] / 255.0, color[3] / 255.0))
        pButton['bg'] = getTkColorString(color)
    vgp['command'] = setLightColor
    return vgp