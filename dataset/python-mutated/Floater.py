"""
Floater Class: Velocity style controller for floating point values with
                a label, entry (validated), and scale
"""
__all__ = ['Floater', 'FloaterWidget', 'FloaterGroup']
from .Valuator import Valuator, VALUATOR_MINI, VALUATOR_FULL
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import ClockObject
import math
import Pmw
import tkinter as tk
FLOATER_WIDTH = 22
FLOATER_HEIGHT = 18

class Floater(Valuator):

    def __init__(self, parent=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        INITOPT = Pmw.INITOPT
        optiondefs = (('style', VALUATOR_MINI, INITOPT),)
        self.defineoptions(kw, optiondefs)
        Valuator.__init__(self, parent)
        self.initialiseoptions(Floater)

    def createValuator(self):
        if False:
            return 10
        self._valuator = self.createcomponent('valuator', (('floater', 'valuator'),), None, FloaterWidget, (self.interior(),), command=self.setEntry, value=self['value'])
        self._valuator._widget.bind('<Double-ButtonPress-1>', self.mouseReset)

    def packValuator(self):
        if False:
            print('Hello World!')
        if self._label:
            self._label.grid(row=0, column=0, sticky=tk.EW)
        self._entry.grid(row=0, column=1, sticky=tk.EW)
        self._valuator.grid(row=0, column=2, padx=2, pady=2)
        self.interior().columnconfigure(0, weight=1)

class FloaterWidget(Pmw.MegaWidget):

    def __init__(self, parent=None, **kw):
        if False:
            return 10
        INITOPT = Pmw.INITOPT
        optiondefs = (('width', FLOATER_WIDTH, INITOPT), ('height', FLOATER_HEIGHT, INITOPT), ('relief', tk.RAISED, self.setRelief), ('borderwidth', 2, self.setBorderwidth), ('background', 'grey75', self.setBackground), ('value', 0.0, INITOPT), ('numDigits', 2, self.setNumDigits), ('command', None, None), ('commandData', [], None), ('preCallback', None, None), ('postCallback', None, None), ('callbackData', [], None))
        self.defineoptions(kw, optiondefs)
        Pmw.MegaWidget.__init__(self, parent)
        interior = self.interior()
        self.value = self['value']
        width = self['width']
        height = self['height']
        self._widget = self.createcomponent('canvas', (), None, tk.Canvas, (interior,), width=width, height=height, background=self['background'], highlightthickness=0, scrollregion=(-width / 2.0, -height / 2.0, width / 2.0, height / 2.0))
        self._widget.pack(expand=1, fill=tk.BOTH)
        self._widget.create_polygon(-width / 2.0, 0, -2.0, -height / 2.0, -2.0, height / 2.0, fill='grey50', tags=('floater',))
        self._widget.create_polygon(width / 2.0, 0, 2.0, height / 2.0, 2.0, -height / 2.0, fill='grey50', tags=('floater',))
        self._widget.bind('<ButtonPress-1>', self.mouseDown)
        self._widget.bind('<B1-Motion>', self.updateFloaterSF)
        self._widget.bind('<ButtonRelease-1>', self.mouseUp)
        self._widget.bind('<Enter>', self.highlightWidget)
        self._widget.bind('<Leave>', self.restoreWidget)
        self.initialiseoptions(FloaterWidget)

    def set(self, value, fCommand=1):
        if False:
            while True:
                i = 10
        '\n        self.set(value, fCommand = 1)\n        Set floater to new value, execute command if fCommand == 1\n        '
        if fCommand and self['command'] is not None:
            self['command'](*[value] + self['commandData'])
        self.value = value

    def updateIndicator(self, value):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        self.get()\n        Get current floater value\n        '
        return self.value

    def mouseDown(self, event):
        if False:
            for i in range(10):
                print('nop')
        ' Begin mouse interaction '
        self['relief'] = tk.SUNKEN
        if self['preCallback']:
            self['preCallback'](*self['callbackData'])
        self.velocitySF = 0.0
        self.updateTask = taskMgr.add(self.updateFloaterTask, 'updateFloater')
        self.updateTask.lastTime = ClockObject.getGlobalClock().getFrameTime()

    def updateFloaterTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update floaterWidget value based on current scaleFactor\n        Adjust for time to compensate for fluctuating frame rates\n        '
        currT = ClockObject.getGlobalClock().getFrameTime()
        dt = currT - state.lastTime
        self.set(self.value + self.velocitySF * dt)
        state.lastTime = currT
        return Task.cont

    def updateFloaterSF(self, event):
        if False:
            return 10
        '\n        Update velocity scale factor based of mouse distance from origin\n        '
        x = self._widget.canvasx(event.x)
        y = self._widget.canvasy(event.y)
        offset = max(0, abs(x) - Valuator.deadband)
        if offset == 0:
            return 0
        sf = math.pow(Valuator.sfBase, self.minExp + offset / Valuator.sfDist)
        if x > 0:
            self.velocitySF = sf
        else:
            self.velocitySF = -sf

    def mouseUp(self, event):
        if False:
            for i in range(10):
                print('nop')
        taskMgr.remove(self.updateTask)
        self.velocitySF = 0.0
        if self['postCallback']:
            self['postCallback'](*self['callbackData'])
        self['relief'] = tk.RAISED

    def setNumDigits(self):
        if False:
            return 10
        '\n        Adjust minimum exponent to use in velocity task based\n        upon the number of digits to be displayed in the result\n        '
        self.minExp = math.floor(-self['numDigits'] / math.log10(Valuator.sfBase))

    def setRelief(self):
        if False:
            i = 10
            return i + 15
        self.interior()['relief'] = self['relief']

    def setBorderwidth(self):
        if False:
            return 10
        self.interior()['borderwidth'] = self['borderwidth']

    def setBackground(self):
        if False:
            return 10
        self._widget['background'] = self['background']

    def highlightWidget(self, event):
        if False:
            while True:
                i = 10
        self._widget.itemconfigure('floater', fill='black')

    def restoreWidget(self, event):
        if False:
            return 10
        self._widget.itemconfigure('floater', fill='grey50')

class FloaterGroup(Pmw.MegaToplevel):

    def __init__(self, parent=None, **kw):
        if False:
            while True:
                i = 10
        DEFAULT_DIM = 1
        DEFAULT_VALUE = [0.0] * kw.get('dim', DEFAULT_DIM)
        DEFAULT_LABELS = ['v[%d]' % x for x in range(kw.get('dim', DEFAULT_DIM))]
        INITOPT = Pmw.INITOPT
        optiondefs = (('dim', DEFAULT_DIM, INITOPT), ('side', tk.TOP, INITOPT), ('title', 'Floater Group', None), ('value', DEFAULT_VALUE, INITOPT), ('command', None, None), ('labels', DEFAULT_LABELS, self._updateLabels))
        self.defineoptions(kw, optiondefs)
        Pmw.MegaToplevel.__init__(self, parent)
        interior = self.interior()
        self._value = list(self['value'])
        self.balloon = Pmw.Balloon()
        menubar = self.createcomponent('menubar', (), None, Pmw.MenuBar, (interior,), balloon=self.balloon)
        menubar.pack(fill=tk.X)
        menubar.addmenu('Floater Group', 'Floater Group Operations')
        menubar.addmenuitem('Floater Group', 'command', 'Reset the Floater Group panel', label='Reset', command=lambda s=self: s.reset())
        menubar.addmenuitem('Floater Group', 'command', 'Dismiss Floater Group panel', label='Dismiss', command=self.withdraw)
        menubar.addmenu('Help', 'Floater Group Help Operations')
        self.toggleBalloonVar = tk.IntVar()
        self.toggleBalloonVar.set(0)
        menubar.addmenuitem('Help', 'checkbutton', 'Toggle balloon help', label='Balloon Help', variable=self.toggleBalloonVar, command=self.toggleBalloon)
        self.floaterList = []
        for index in range(self['dim']):
            f = self.createcomponent('floater%d' % index, (), 'Valuator', Floater, (interior,), value=self._value[index], text=self['labels'][index])
            f['command'] = lambda val, s=self, i=index: s._floaterSetAt(i, val)
            f.pack(side=self['side'], expand=1, fill=tk.X)
            self.floaterList.append(f)
        self.set(self['value'])
        self.initialiseoptions(FloaterGroup)

    def _updateLabels(self):
        if False:
            i = 10
            return i + 15
        if self['labels']:
            for index in range(self['dim']):
                self.floaterList[index]['text'] = self['labels'][index]

    def toggleBalloon(self):
        if False:
            return 10
        if self.toggleBalloonVar.get():
            self.balloon.configure(state='balloon')
        else:
            self.balloon.configure(state='none')

    def get(self):
        if False:
            while True:
                i = 10
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
            self.floaterList[i].set(value[i], 0)
        if fCommand and self['command'] is not None:
            self['command'](self._value)

    def setAt(self, index, value):
        if False:
            while True:
                i = 10
        self.floaterList[index].set(value)

    def _floaterSetAt(self, index, value):
        if False:
            i = 10
            return i + 15
        self._value[index] = value
        if self['command']:
            self['command'](self._value)

    def reset(self):
        if False:
            for i in range(10):
                print('nop')
        self.set(self['value'])