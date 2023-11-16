"""
Slider Class: Velocity style controller for floating point values with
               a label, entry (validated), and min/max slider
"""
__all__ = ['Slider', 'SliderWidget', 'rgbPanel']
from .Valuator import Valuator, rgbPanel, VALUATOR_MINI, VALUATOR_FULL
import Pmw
import tkinter as tk

class Slider(Valuator):
    """
    Valuator widget which includes an min/max slider and an entry for setting
    floating point values in a range
    """

    def __init__(self, parent=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        INITOPT = Pmw.INITOPT
        optiondefs = (('min', 0.0, self.setMin), ('max', 100.0, self.setMax), ('style', VALUATOR_MINI, INITOPT))
        self.defineoptions(kw, optiondefs)
        Valuator.__init__(self, parent)
        self.propertyDict['min']['fNone'] = 0
        self.propertyDict['min']['help'] = 'Minimum allowable value.'
        self.propertyDict['max']['fNone'] = 0
        self.propertyDict['max']['help'] = 'Maximum allowable value.'
        self.initialiseoptions(Slider)

    def createValuator(self):
        if False:
            while True:
                i = 10
        self._valuator = self.createcomponent('valuator', (('slider', 'valuator'),), None, SliderWidget, (self.interior(),), style=self['style'], command=self.setEntry, value=self['value'])
        try:
            self._valuator._arrowBtn.bind('<ButtonPress-3>', self._popupValuatorMenu)
        except AttributeError:
            pass
        self._valuator._minLabel.bind('<ButtonPress-3>', self._popupValuatorMenu)
        self._valuator._maxLabel.bind('<ButtonPress-3>', self._popupValuatorMenu)

    def packValuator(self):
        if False:
            for i in range(10):
                print('nop')
        if self['style'] == VALUATOR_FULL:
            if self._label:
                self._label.grid(row=0, column=0, sticky=tk.EW)
            self._entry.grid(row=0, column=1, sticky=tk.EW)
            self._valuator.grid(row=1, columnspan=2, padx=2, pady=2, sticky='ew')
            self.interior().columnconfigure(0, weight=1)
        else:
            if self._label:
                self._label.grid(row=0, column=0, sticky=tk.EW)
            self._entry.grid(row=0, column=1, sticky=tk.EW)
            self._valuator.grid(row=0, column=2, padx=2, pady=2)
            self.interior().columnconfigure(0, weight=1)

    def setMin(self):
        if False:
            print('Hello World!')
        if self['min'] is not None:
            self._valuator['min'] = self['min']

    def setMax(self):
        if False:
            print('Hello World!')
        if self['max'] is not None:
            self._valuator['max'] = self['max']

class SliderWidget(Pmw.MegaWidget):

    def __init__(self, parent=None, **kw):
        if False:
            return 10
        INITOPT = Pmw.INITOPT
        optiondefs = (('style', VALUATOR_MINI, INITOPT), ('relief', tk.RAISED, self.setRelief), ('borderwidth', 2, self.setBorderwidth), ('background', 'grey75', self.setBackground), ('fliparrow', 0, INITOPT), ('min', 0.0, self.setMin), ('max', 100.0, self.setMax), ('value', 0.0, INITOPT), ('numDigits', 2, self.setNumDigits), ('command', None, None), ('commandData', [], None), ('preCallback', None, None), ('postCallback', None, None), ('callbackData', [], None))
        self.defineoptions(kw, optiondefs)
        Pmw.MegaWidget.__init__(self, parent)
        interior = self.interior()
        self.value = self['value']
        self.formatString = '%2f'
        self.increment = 0.01
        self._isPosted = 0
        self._fUnpost = 0
        self._fUpdate = 0
        self._firstPress = 1
        self._fPressInsde = 0
        width = 100
        self.xPad = xPad = 10
        sliderWidth = width + 2 * xPad
        height = 20
        self.left = left = -(width / 2.0)
        self.right = right = width / 2.0
        top = -5
        bottom = top + height

        def createSlider(parent):
            if False:
                i = 10
                return i + 15
            self._minLabel = tk.Label(parent, text=self['min'], width=8, anchor=tk.W)
            self._minLabel.pack(side=tk.LEFT)
            if self['style'] == VALUATOR_FULL:
                self._widgetVar = tk.DoubleVar()
                self._widgetVar.set(self['value'])
                self._widget = self.createcomponent('slider', (), None, tk.Scale, (interior,), variable=self._widgetVar, from_=self['min'], to=self['max'], resolution=0.0, width=10, orient='horizontal', showvalue=0, length=sliderWidth, relief=tk.FLAT, bd=2, highlightthickness=0)
            else:
                self._widget = self.createcomponent('slider', (), None, tk.Canvas, (parent,), width=sliderWidth, height=height, bd=2, highlightthickness=0, scrollregion=(left - xPad, top, right + xPad, bottom))
                xShift = 1
                self._marker = self._widget.create_polygon(-7 + xShift, 12, 7 + xShift, 12, xShift, 0, fill='black', tags=('marker',))
                self._widget.create_polygon(-6.0, 10, 6.0, 10, 0, 0, fill='grey85', outline='black', tags=('marker',))
                self._widget.create_line(left, 0, right, 0, width=2, tags=('line',))
            self._widget.pack(side=tk.LEFT, expand=1, fill=tk.X)
            self._maxLabel = tk.Label(parent, text=self['max'], width=8, anchor=tk.W)
            self._maxLabel.pack(side=tk.LEFT)
        if self['style'] == VALUATOR_MINI:
            self._arrowBtn = self.createcomponent('arrowbutton', (), None, tk.Canvas, (interior,), borderwidth=0, relief=tk.FLAT, width=14, height=14, scrollregion=(-7, -7, 7, 7))
            self._arrowBtn.pack(expand=1, fill=tk.BOTH)
            self._arrowBtn.create_polygon(-5, -5, 5, -5, 0, 5, fill='grey50', tags='arrow')
            self._arrowBtn.create_line(-5, 5, 5, 5, fill='grey50', tags='arrow')
            self._popup = self.createcomponent('popup', (), None, tk.Toplevel, (interior,), relief=tk.RAISED, borderwidth=2)
            self._popup.withdraw()
            self._popup.overrideredirect(1)
            createSlider(self._popup)
            self._arrowBtn.bind('<1>', self._postSlider)
            self._arrowBtn.bind('<Enter>', self.highlightWidget)
            self._arrowBtn.bind('<Leave>', self.restoreWidget)
            self._arrowBtn.bind('<Unmap>', self._unpostSlider)
            self._popup.bind('<Escape>', self._unpostSlider)
            self._popup.bind('<ButtonRelease-1>', self._widgetBtnRelease)
            self._popup.bind('<ButtonPress-1>', self._widgetBtnPress)
            self._popup.bind('<Motion>', self._widgetMove)
            self._widget.bind('<Left>', self._decrementValue)
            self._widget.bind('<Right>', self._incrementValue)
            self._widget.bind('<Shift-Left>', self._bigDecrementValue)
            self._widget.bind('<Shift-Right>', self._bigIncrementValue)
            self._widget.bind('<Home>', self._goToMin)
            self._widget.bind('<End>', self._goToMax)
        else:
            createSlider(interior)
            self._widget['command'] = self._firstScaleCommand
            self._widget.bind('<ButtonRelease-1>', self._scaleBtnRelease)
            self._widget.bind('<ButtonPress-1>', self._scaleBtnPress)
        self.initialiseoptions(SliderWidget)
        if 'relief' not in kw:
            if self['style'] == VALUATOR_FULL:
                self['relief'] = tk.FLAT
        self.updateIndicator(self['value'])

    def destroy(self):
        if False:
            return 10
        if self['style'] == VALUATOR_MINI and self._isPosted:
            Pmw.popgrab(self._popup)
        Pmw.MegaWidget.destroy(self)

    def set(self, value, fCommand=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        self.set(value, fCommand = 1)\n        Set slider to new value, execute command if fCommand == 1\n        '
        if fCommand and self['command'] is not None:
            self['command'](*[value] + self['commandData'])
        self.value = value

    def get(self):
        if False:
            while True:
                i = 10
        '\n        self.get()\n        Get current slider value\n        '
        return self.value

    def updateIndicator(self, value):
        if False:
            print('Hello World!')
        if self['style'] == VALUATOR_MINI:
            percentX = (value - self['min']) / float(self['max'] - self['min'])
            newX = percentX * (self.right - self.left) + self.left
            markerX = self._getMarkerX()
            dx = newX - markerX
            self._widget.move('marker', dx, 0)
        else:
            self._widgetVar.set(value)

    def _postSlider(self, event=None):
        if False:
            return 10
        self._isPosted = 1
        self._fUpdate = 0
        self.interior()['relief'] = tk.SUNKEN
        self.update_idletasks()
        x = self._arrowBtn.winfo_rootx() + self._arrowBtn.winfo_width() / 2.0 - self.interior()['bd']
        y = self._arrowBtn.winfo_rooty() + self._arrowBtn.winfo_height()
        bd = self._popup['bd']
        minW = self._minLabel.winfo_width()
        cw = self._getMarkerX() - self.left + self.xPad
        popupOffset = bd + minW + cw
        ch = self._widget.winfo_height()
        sh = self.winfo_screenheight()
        if y + ch > sh and y > sh / 2:
            y = self._arrowBtn.winfo_rooty() - ch
        Pmw.setgeometryanddeiconify(self._popup, '+%d+%d' % (x - popupOffset, y))
        Pmw.pushgrab(self._popup, 1, self._unpostSlider)
        self._widget.focus_set()
        self._fUpdate = 0
        self._fUnpost = 0
        self._firstPress = 1
        self._fPressInsde = 0

    def _updateValue(self, event):
        if False:
            while True:
                i = 10
        mouseX = self._widget.canvasx(event.x_root - self._widget.winfo_rootx())
        if mouseX < self.left:
            mouseX = self.left
        if mouseX > self.right:
            mouseX = self.right
        sf = (mouseX - self.left) / (self.right - self.left)
        newVal = sf * (self['max'] - self['min']) + self['min']
        self.set(newVal)

    def _widgetBtnPress(self, event):
        if False:
            while True:
                i = 10
        widget = self._popup
        xPos = event.x_root - widget.winfo_rootx()
        yPos = event.y_root - widget.winfo_rooty()
        fInside = xPos > 0 and xPos < widget.winfo_width() and (yPos > 0) and (yPos < widget.winfo_height())
        if fInside:
            self._fPressInside = 1
            self._fUpdate = 1
            if self['preCallback']:
                self['preCallback'](*self['callbackData'])
            self._updateValue(event)
        else:
            self._fPressInside = 0
            self._fUpdate = 0

    def _widgetMove(self, event):
        if False:
            return 10
        if self._firstPress and (not self._fUpdate):
            canvasY = self._widget.canvasy(event.y_root - self._widget.winfo_rooty())
            if canvasY > 0:
                self._fUpdate = 1
                if self['preCallback']:
                    self['preCallback'](*self['callbackData'])
                self._unpostOnNextRelease()
        elif self._fUpdate:
            self._updateValue(event)

    def _scaleBtnPress(self, event):
        if False:
            i = 10
            return i + 15
        if self['preCallback']:
            self['preCallback'](*self['callbackData'])

    def _scaleBtnRelease(self, event):
        if False:
            for i in range(10):
                print('nop')
        if self['postCallback']:
            self['postCallback'](*self['callbackData'])

    def _widgetBtnRelease(self, event):
        if False:
            return 10
        if self._fUpdate and self['postCallback']:
            self['postCallback'](*self['callbackData'])
        if self._fUnpost or not (self._firstPress or self._fPressInside):
            self._unpostSlider()
        self._fUpdate = 0
        self._firstPress = 0
        self._fPressInside = 0

    def _unpostOnNextRelease(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        self._fUnpost = 1

    def _unpostSlider(self, event=None):
        if False:
            while True:
                i = 10
        if not self._isPosted:
            return
        Pmw.popgrab(self._popup)
        self._popup.withdraw()
        self._isPosted = 0
        self.interior()['relief'] = tk.RAISED

    def _incrementValue(self, event):
        if False:
            return 10
        self.set(self.value + self.increment)

    def _bigIncrementValue(self, event):
        if False:
            return 10
        self.set(self.value + self.increment * 10.0)

    def _decrementValue(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.set(self.value - self.increment)

    def _bigDecrementValue(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.set(self.value - self.increment * 10.0)

    def _goToMin(self, event):
        if False:
            while True:
                i = 10
        self.set(self['min'])

    def _goToMax(self, event):
        if False:
            return 10
        self.set(self['max'])

    def _firstScaleCommand(self, val):
        if False:
            print('Hello World!')
        ' Hack to avoid calling command on instantiation of Scale '
        self._widget['command'] = self._scaleCommand

    def _scaleCommand(self, val):
        if False:
            while True:
                i = 10
        self.set(float(val))

    def setMin(self):
        if False:
            while True:
                i = 10
        self._minLabel['text'] = self.formatString % self['min']
        if self['style'] == VALUATOR_FULL:
            self._widget['from_'] = self['min']
        self.updateIndicator(self.value)

    def setMax(self):
        if False:
            while True:
                i = 10
        self._maxLabel['text'] = self.formatString % self['max']
        if self['style'] == VALUATOR_FULL:
            self._widget['to'] = self['max']
        self.updateIndicator(self.value)

    def setNumDigits(self):
        if False:
            i = 10
            return i + 15
        self.formatString = '%0.' + '%d' % self['numDigits'] + 'f'
        self._minLabel['text'] = self.formatString % self['min']
        self._maxLabel['text'] = self.formatString % self['max']
        self.updateIndicator(self.value)
        self.increment = pow(10, -self['numDigits'])

    def _getMarkerX(self):
        if False:
            while True:
                i = 10
        c = self._widget.coords(self._marker)
        return c[4]

    def setRelief(self):
        if False:
            while True:
                i = 10
        self.interior()['relief'] = self['relief']

    def setBorderwidth(self):
        if False:
            return 10
        self.interior()['borderwidth'] = self['borderwidth']

    def setBackground(self):
        if False:
            for i in range(10):
                print('nop')
        self._widget['background'] = self['background']

    def highlightWidget(self, event):
        if False:
            for i in range(10):
                print('nop')
        self._arrowBtn.itemconfigure('arrow', fill='black')

    def restoreWidget(self, event):
        if False:
            for i in range(10):
                print('nop')
        self._arrowBtn.itemconfigure('arrow', fill='grey50')