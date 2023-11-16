"""
Dial Class: Velocity style controller for floating point values with
             a label, entry (validated), and scale
"""
__all__ = ['Dial', 'AngleDial', 'DialWidget']
from .Valuator import Valuator, VALUATOR_MINI, VALUATOR_FULL
from direct.task import Task
from direct.task.TaskManagerGlobal import taskMgr
from panda3d.core import ClockObject
import math
import operator
import Pmw
import tkinter as tk
TWO_PI = 2.0 * math.pi
ONEPOINTFIVE_PI = 1.5 * math.pi
POINTFIVE_PI = 0.5 * math.pi
INNER_SF = 0.2
DIAL_FULL_SIZE = 45
DIAL_MINI_SIZE = 30

class Dial(Valuator):
    """
    Valuator widget which includes an angle dial and an entry for setting
    floating point values
    """

    def __init__(self, parent=None, **kw):
        if False:
            return 10
        INITOPT = Pmw.INITOPT
        optiondefs = (('style', VALUATOR_FULL, INITOPT), ('base', 0.0, self.setBase), ('delta', 1.0, self.setDelta), ('fSnap', 0, self.setSnap), ('fRollover', 1, self.setRollover))
        self.defineoptions(kw, optiondefs)
        Valuator.__init__(self, parent)
        self.initialiseoptions(Dial)

    def createValuator(self):
        if False:
            return 10
        self._valuator = self.createcomponent('valuator', (('dial', 'valuator'),), None, DialWidget, (self.interior(),), style=self['style'], command=self.setEntry, value=self['value'])
        self._valuator._widget.bind('<Double-ButtonPress-1>', self.mouseReset)

    def packValuator(self):
        if False:
            return 10
        if self['style'] == VALUATOR_FULL:
            self._valuator.grid(rowspan=2, columnspan=2, padx=2, pady=2)
            if self._label:
                self._label.grid(row=0, column=2, sticky=tk.EW)
            self._entry.grid(row=1, column=2, sticky=tk.EW)
            self.interior().columnconfigure(2, weight=1)
        else:
            if self._label:
                self._label.grid(row=0, column=0, sticky=tk.EW)
            self._entry.grid(row=0, column=1, sticky=tk.EW)
            self._valuator.grid(row=0, column=2, padx=2, pady=2)
            self.interior().columnconfigure(0, weight=1)

    def addValuatorPropertiesToDialog(self):
        if False:
            for i in range(10):
                print('nop')
        self.addPropertyToDialog('base', {'widget': self._valuator, 'type': 'real', 'help': 'Dial value = base + delta * numRevs'})
        self.addPropertyToDialog('delta', {'widget': self._valuator, 'type': 'real', 'help': 'Dial value = base + delta * numRevs'})
        self.addPropertyToDialog('numSegments', {'widget': self._valuator, 'type': 'integer', 'help': 'Number of segments to divide dial into.'})

    def addValuatorMenuEntries(self):
        if False:
            while True:
                i = 10
        self._fSnap = tk.IntVar()
        self._fSnap.set(self['fSnap'])
        self._popupMenu.add_checkbutton(label='Snap', variable=self._fSnap, command=self._setSnap)
        self._fRollover = tk.IntVar()
        self._fRollover.set(self['fRollover'])
        if self['fAdjustable']:
            self._popupMenu.add_checkbutton(label='Rollover', variable=self._fRollover, command=self._setRollover)

    def setBase(self):
        if False:
            i = 10
            return i + 15
        ' Set Dial base value: value = base + delta * numRevs '
        self._valuator['base'] = self['base']

    def setDelta(self):
        if False:
            i = 10
            return i + 15
        ' Set Dial delta value: value = base + delta * numRevs '
        self._valuator['delta'] = self['delta']

    def _setSnap(self):
        if False:
            while True:
                i = 10
        ' Menu command to turn Dial angle snap on/off '
        self._valuator['fSnap'] = self._fSnap.get()

    def setSnap(self):
        if False:
            for i in range(10):
                print('nop')
        ' Turn Dial angle snap on/off '
        self._fSnap.set(self['fSnap'])
        self._setSnap()

    def _setRollover(self):
        if False:
            print('Hello World!')
        '\n        Menu command to turn Dial rollover on/off (i.e. does value accumulate\n        every time you complete a revolution of the dial?)\n        '
        self._valuator['fRollover'] = self._fRollover.get()

    def setRollover(self):
        if False:
            while True:
                i = 10
        ' Turn Dial rollover (accumulation of a sum) on/off '
        self._fRollover.set(self['fRollover'])
        self._setRollover()

class AngleDial(Dial):

    def __init__(self, parent=None, **kw):
        if False:
            while True:
                i = 10
        optiondefs = (('delta', 360.0, None), ('fRollover', 0, None), ('dial_numSegments', 12, None))
        self.defineoptions(kw, optiondefs)
        Dial.__init__(self, parent)
        self.initialiseoptions(AngleDial)

class DialWidget(Pmw.MegaWidget):

    def __init__(self, parent=None, **kw):
        if False:
            print('Hello World!')
        INITOPT = Pmw.INITOPT
        optiondefs = (('style', VALUATOR_FULL, INITOPT), ('size', None, INITOPT), ('relief', tk.SUNKEN, self.setRelief), ('borderwidth', 2, self.setBorderwidth), ('background', 'white', self.setBackground), ('numSegments', 10, self.setNumSegments), ('value', 0.0, INITOPT), ('numDigits', 2, self.setNumDigits), ('base', 0.0, None), ('delta', 1.0, None), ('fSnap', 0, None), ('fRollover', 1, None), ('command', None, None), ('commandData', [], None), ('preCallback', None, None), ('postCallback', None, None), ('callbackData', [], None))
        self.defineoptions(kw, optiondefs)
        Pmw.MegaWidget.__init__(self, parent)
        interior = self.interior()
        self.value = self['value']
        self.rollCount = 0
        if not self['size']:
            if self['style'] == VALUATOR_FULL:
                size = DIAL_FULL_SIZE
            else:
                size = DIAL_MINI_SIZE
        else:
            size = self['size']
        radius = self.radius = int(size / 2.0)
        inner_radius = max(3, radius * INNER_SF)
        self._widget = self.createcomponent('canvas', (), None, tk.Canvas, (interior,), width=size, height=size, background=self['background'], highlightthickness=0, scrollregion=(-radius, -radius, radius, radius))
        self._widget.pack(expand=1, fill=tk.BOTH)
        self._widget.create_oval(-radius, -radius, radius, radius, outline='', tags=('dial',))
        self._widget.create_line(0, 0, 0, -radius, width=2, tags=('indicator', 'dial'))
        self._widget.create_oval(-inner_radius, -inner_radius, inner_radius, inner_radius, fill='grey50', tags=('knob',))
        self._widget.tag_bind('dial', '<ButtonPress-1>', self.mouseDown)
        self._widget.tag_bind('dial', '<B1-Motion>', self.mouseMotion)
        self._widget.tag_bind('dial', '<Shift-B1-Motion>', self.shiftMouseMotion)
        self._widget.tag_bind('dial', '<ButtonRelease-1>', self.mouseUp)
        self._widget.tag_bind('knob', '<ButtonPress-1>', self.knobMouseDown)
        self._widget.tag_bind('knob', '<B1-Motion>', self.updateDialSF)
        self._widget.tag_bind('knob', '<ButtonRelease-1>', self.knobMouseUp)
        self._widget.tag_bind('knob', '<Enter>', self.highlightKnob)
        self._widget.tag_bind('knob', '<Leave>', self.restoreKnob)
        self.initialiseoptions(DialWidget)

    def set(self, value, fCommand=1):
        if False:
            while True:
                i = 10
        '\n        self.set(value, fCommand = 1)\n        Set dial to new value, execute command if fCommand == 1\n        '
        if not self['fRollover']:
            if value > self['delta']:
                self.rollCount = 0
            value = self['base'] + (value - self['base']) % self['delta']
        if fCommand and self['command'] is not None:
            self['command'](*[value] + self['commandData'])
        self.value = value

    def get(self):
        if False:
            i = 10
            return i + 15
        '\n        self.get()\n        Get current dial value\n        '
        return self.value

    def mouseDown(self, event):
        if False:
            for i in range(10):
                print('nop')
        self._onButtonPress()
        self.lastAngle = dialAngle = self.computeDialAngle(event)
        self.computeValueFromAngle(dialAngle)

    def mouseUp(self, event):
        if False:
            i = 10
            return i + 15
        self._onButtonRelease()

    def shiftMouseMotion(self, event):
        if False:
            return 10
        self.mouseMotion(event, 1)

    def mouseMotion(self, event, fShift=0):
        if False:
            return 10
        dialAngle = self.computeDialAngle(event, fShift)
        self.computeValueFromAngle(dialAngle)

    def computeDialAngle(self, event, fShift=0):
        if False:
            print('Hello World!')
        x = self._widget.canvasx(event.x)
        y = self._widget.canvasy(event.y)
        rawAngle = math.atan2(y, x)
        dialAngle = rawAngle + POINTFIVE_PI
        if operator.xor(self['fSnap'], fShift):
            dialAngle = round(dialAngle / self.snapAngle) * self.snapAngle
        return dialAngle

    def computeValueFromAngle(self, dialAngle):
        if False:
            print('Hello World!')
        delta = self['delta']
        dialAngle = dialAngle % TWO_PI
        if self.lastAngle > ONEPOINTFIVE_PI and dialAngle < POINTFIVE_PI:
            self.rollCount += 1
        elif self.lastAngle < POINTFIVE_PI and dialAngle > ONEPOINTFIVE_PI:
            self.rollCount -= 1
        self.lastAngle = dialAngle
        newValue = self['base'] + (self.rollCount + dialAngle / TWO_PI) * delta
        self.set(newValue)

    def updateIndicator(self, value):
        if False:
            i = 10
            return i + 15
        delta = self['delta']
        factors = divmod(value - self['base'], delta)
        self.rollCount = factors[0]
        self.updateIndicatorRadians(factors[1] / delta * TWO_PI)

    def updateIndicatorDegrees(self, degAngle):
        if False:
            print('Hello World!')
        self.updateIndicatorRadians(degAngle * (math.pi / 180.0))

    def updateIndicatorRadians(self, dialAngle):
        if False:
            return 10
        rawAngle = dialAngle - POINTFIVE_PI
        endx = math.cos(rawAngle) * self.radius
        endy = math.sin(rawAngle) * self.radius
        self._widget.coords('indicator', endx * INNER_SF, endy * INNER_SF, endx, endy)

    def knobMouseDown(self, event):
        if False:
            i = 10
            return i + 15
        self._onButtonPress()
        self.knobSF = 0.0
        self.updateTask = taskMgr.add(self.updateDialTask, 'updateDial')
        self.updateTask.lastTime = ClockObject.getGlobalClock().getFrameTime()

    def updateDialTask(self, state):
        if False:
            for i in range(10):
                print('nop')
        currT = ClockObject.getGlobalClock().getFrameTime()
        dt = currT - state.lastTime
        self.set(self.value + self.knobSF * dt)
        state.lastTime = currT
        return Task.cont

    def updateDialSF(self, event):
        if False:
            while True:
                i = 10
        x = self._widget.canvasx(event.x)
        y = self._widget.canvasy(event.y)
        offset = max(0, abs(x) - Valuator.deadband)
        if offset == 0:
            return 0
        sf = math.pow(Valuator.sfBase, self.minExp + offset / Valuator.sfDist)
        if x > 0:
            self.knobSF = sf
        else:
            self.knobSF = -sf

    def knobMouseUp(self, event):
        if False:
            i = 10
            return i + 15
        taskMgr.remove(self.updateTask)
        self.knobSF = 0.0
        self._onButtonRelease()

    def setNumDigits(self):
        if False:
            while True:
                i = 10
        self.minExp = math.floor(-self['numDigits'] / math.log10(Valuator.sfBase))

    def setRelief(self):
        if False:
            print('Hello World!')
        self.interior()['relief'] = self['relief']

    def setBorderwidth(self):
        if False:
            print('Hello World!')
        self.interior()['borderwidth'] = self['borderwidth']

    def setBackground(self):
        if False:
            print('Hello World!')
        self._widget['background'] = self['background']

    def setNumSegments(self):
        if False:
            return 10
        self._widget.delete('ticks')
        numSegments = self['numSegments']
        self.snapAngle = snapAngle = TWO_PI / numSegments
        for ticknum in range(numSegments):
            angle = snapAngle * ticknum
            angle = angle - POINTFIVE_PI
            startx = math.cos(angle) * self.radius
            starty = math.sin(angle) * self.radius
            if angle % POINTFIVE_PI == 0.0:
                sf = 0.6
            else:
                sf = 0.8
            endx = startx * sf
            endy = starty * sf
            self._widget.create_line(startx, starty, endx, endy, tags=('ticks', 'dial'))

    def highlightKnob(self, event):
        if False:
            print('Hello World!')
        self._widget.itemconfigure('knob', fill='black')

    def restoreKnob(self, event):
        if False:
            for i in range(10):
                print('nop')
        self._widget.itemconfigure('knob', fill='grey50')

    def _onButtonPress(self, *args):
        if False:
            while True:
                i = 10
        ' User redefinable callback executed on button press '
        if self['preCallback']:
            self['preCallback'](*self['callbackData'])

    def _onButtonRelease(self, *args):
        if False:
            return 10
        ' User redefinable callback executed on button release '
        if self['postCallback']:
            self['postCallback'](*self['callbackData'])