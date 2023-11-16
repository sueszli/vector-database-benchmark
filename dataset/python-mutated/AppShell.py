"""
AppShell provides a GUI application framework.
This is an adaption of AppShell.py found in Python and Tkinter Programming
by John E. Grayson which is a streamlined adaptation of GuiAppD.py, originally
created by Doug Hellmann (doughellmann@mindspring.com).
"""
__all__ = ['AppShell']
from direct.showbase.DirectObject import DirectObject
from . import Dial
from . import Floater
from . import Slider
from . import EntryScale
from . import VectorWidgets
from . import ProgressBar
import Pmw
import tkinter as tk
import builtins
if not hasattr(builtins, 'widgetDict'):
    builtins.widgetDict = {}
if not hasattr(builtins, 'variableDict'):
    builtins.variableDict = {}

def resetWidgetDict():
    if False:
        while True:
            i = 10
    builtins.widgetDict = {}

def resetVariableDict():
    if False:
        for i in range(10):
            print('nop')
    builtins.variableDict = {}

class AppShell(Pmw.MegaWidget, DirectObject):
    appversion = '1.0'
    appname = 'Generic Application Frame'
    copyright = 'Copyright 2004 Walt Disney Imagineering.' + ' All Rights Reserved'
    contactname = 'Mark R. Mine'
    contactphone = '(818) 544-2921'
    contactemail = 'Mark.Mine@disney.com'
    frameWidth = 450
    frameHeight = 320
    padx = 5
    pady = 5
    usecommandarea = 0
    usestatusarea = 0
    balloonState = 'none'
    panelCount = 0

    def __init__(self, parent=None, **kw):
        if False:
            print('Hello World!')
        optiondefs = (('title', self.appname, None), ('padx', 1, Pmw.INITOPT), ('pady', 1, Pmw.INITOPT), ('framewidth', self.frameWidth, Pmw.INITOPT), ('frameheight', self.frameHeight, Pmw.INITOPT), ('usecommandarea', self.usecommandarea, Pmw.INITOPT), ('usestatusarea', self.usestatusarea, Pmw.INITOPT))
        self.defineoptions(kw, optiondefs)
        if parent is None:
            self.parent = tk.Toplevel()
        else:
            self.parent = parent
        Pmw.MegaWidget.__init__(self, self.parent)
        self.parent.geometry('%dx%d' % (self.frameWidth, self.frameHeight))
        self.parent.title(self['title'])
        AppShell.panelCount += 1
        self.id = self.appname + '-' + repr(AppShell.panelCount)
        self.widgetDict = builtins.widgetDict[self.id] = {}
        self.variableDict = builtins.variableDict[self.id] = {}
        self._hull = self.component('hull')
        self.appInit()
        self.__createInterface()
        self.focus_set()
        self.initialiseoptions(AppShell)
        self.pack(fill=tk.BOTH, expand=1)

    def __createInterface(self):
        if False:
            for i in range(10):
                print('nop')
        self.__createBalloon()
        self.__createMenuBar()
        self.__createDataArea()
        self.__createCommandArea()
        self.__createMessageBar()
        self.__createAboutBox()
        self.interior().bind('<Destroy>', self.onDestroy)
        self.createMenuBar()
        self.createInterface()

    def __createBalloon(self):
        if False:
            for i in range(10):
                print('nop')
        self.__balloon = self.createcomponent('balloon', (), None, Pmw.Balloon, (self._hull,))
        self.__balloon.configure(state=self.balloonState)

    def __createMenuBar(self):
        if False:
            for i in range(10):
                print('nop')
        self.menuFrame = tk.Frame(self._hull)
        self.menuBar = self.createcomponent('menubar', (), None, Pmw.MenuBar, (self.menuFrame,), hull_relief=tk.FLAT, hull_borderwidth=0, balloon=self.balloon())
        self.menuBar.addmenu('Help', 'About %s' % self.appname, side='right')
        self.menuBar.addmenu('File', 'File commands and Quit')
        self.menuBar.pack(fill=tk.X, side=tk.LEFT)
        spacer = tk.Label(self.menuFrame, text='   ')
        spacer.pack(side=tk.LEFT, expand=0)
        self.menuFrame.pack(fill=tk.X)

    def __createDataArea(self):
        if False:
            while True:
                i = 10
        self.dataArea = self.createcomponent('dataarea', (), None, tk.Frame, (self._hull,), relief=tk.GROOVE, bd=1)
        self.dataArea.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES, padx=self['padx'], pady=self['pady'])

    def __createCommandArea(self):
        if False:
            for i in range(10):
                print('nop')
        self.__commandFrame = self.createcomponent('commandframe', (), None, tk.Frame, (self._hull,), relief=tk.SUNKEN, bd=1)
        self.__buttonBox = self.createcomponent('buttonbox', (), None, Pmw.ButtonBox, (self.__commandFrame,), padx=0, pady=0)
        self.__buttonBox.pack(side=tk.TOP, expand=tk.NO, fill=tk.X)
        if self['usecommandarea']:
            self.__commandFrame.pack(side=tk.TOP, expand=tk.NO, fill=tk.X, padx=self['padx'], pady=self['pady'])

    def __createMessageBar(self):
        if False:
            print('Hello World!')
        frame = self.createcomponent('bottomtray', (), None, tk.Frame, (self._hull,), relief=tk.SUNKEN)
        self.__messageBar = self.createcomponent('messagebar', (), None, Pmw.MessageBar, (frame,), entry_relief=tk.SUNKEN, entry_bd=1, labelpos=None)
        self.__messageBar.pack(side=tk.LEFT, expand=tk.YES, fill=tk.X)
        self.__progressBar = ProgressBar.ProgressBar(frame, fillColor='slateblue', doLabel=1, width=150)
        self.__progressBar.frame.pack(side=tk.LEFT, expand=tk.NO, fill=tk.NONE)
        self.updateProgress(0)
        if self['usestatusarea']:
            frame.pack(side=tk.BOTTOM, expand=tk.NO, fill=tk.X)
        self.__balloon.configure(statuscommand=self.__messageBar.helpmessage)

    def __createAboutBox(self):
        if False:
            i = 10
            return i + 15
        Pmw.aboutversion(self.appversion)
        Pmw.aboutcopyright(self.copyright)
        Pmw.aboutcontact('For more information, contact:\n %s\n Phone: %s\n Email: %s' % (self.contactname, self.contactphone, self.contactemail))
        self.about = Pmw.AboutDialog(self._hull, applicationname=self.appname)
        self.about.withdraw()

    def toggleBalloon(self):
        if False:
            print('Hello World!')
        if self.toggleBalloonVar.get():
            self.__balloon.configure(state='both')
        else:
            self.__balloon.configure(state='status')

    def showAbout(self):
        if False:
            for i in range(10):
                print('nop')
        self.about.show()
        self.about.focus_set()

    def quit(self):
        if False:
            while True:
                i = 10
        self.parent.destroy()

    def appInit(self):
        if False:
            print('Hello World!')
        pass

    def createInterface(self):
        if False:
            while True:
                i = 10
        pass

    def onDestroy(self, event):
        if False:
            print('Hello World!')
        pass

    def createMenuBar(self):
        if False:
            return 10
        self.menuBar.addmenuitem('Help', 'command', 'Get information on application', label='About...', command=self.showAbout)
        self.toggleBalloonVar = tk.IntVar()
        if self.balloonState == 'none':
            self.toggleBalloonVar.set(0)
        else:
            self.toggleBalloonVar.set(1)
        self.menuBar.addmenuitem('Help', 'checkbutton', 'Toggle balloon help', label='Balloon help', variable=self.toggleBalloonVar, command=self.toggleBalloon)
        self.menuBar.addmenuitem('File', 'command', 'Quit this application', label='Quit', command=self.quit)

    def interior(self):
        if False:
            return 10
        return self.dataArea

    def balloon(self):
        if False:
            print('Hello World!')
        return self.__balloon

    def buttonBox(self):
        if False:
            print('Hello World!')
        return self.__buttonBox

    def messageBar(self):
        if False:
            print('Hello World!')
        return self.__messageBar

    def buttonAdd(self, buttonName, helpMessage=None, statusMessage=None, **kw):
        if False:
            while True:
                i = 10
        newBtn = self.__buttonBox.add(buttonName)
        newBtn.configure(kw)
        if helpMessage:
            self.bind(newBtn, helpMessage, statusMessage)
        return newBtn

    def alignbuttons(self):
        if False:
            i = 10
            return i + 15
        ' Make all buttons wide as widest '
        self.__buttonBox.alignbuttons()

    def bind(self, child, balloonHelpMsg, statusHelpMsg=None):
        if False:
            i = 10
            return i + 15
        self.__balloon.bind(child, balloonHelpMsg, statusHelpMsg)

    def updateProgress(self, newValue=0, newMax=0):
        if False:
            for i in range(10):
                print('nop')
        self.__progressBar.updateProgress(newValue, newMax)

    def addWidget(self, category, text, widget):
        if False:
            i = 10
            return i + 15
        self.widgetDict[category + '-' + text] = widget

    def getWidget(self, category, text):
        if False:
            for i in range(10):
                print('nop')
        return self.widgetDict.get(category + '-' + text, None)

    def addVariable(self, category, text, variable):
        if False:
            return 10
        self.variableDict[category + '-' + text] = variable

    def getVariable(self, category, text):
        if False:
            return 10
        return self.variableDict.get(category + '-' + text, None)

    def createWidget(self, parent, category, text, widgetClass, help, command, side, fill, expand, kw):
        if False:
            for i in range(10):
                print('nop')
        kw['text'] = text
        widget = widgetClass(parent, **kw)
        widget['command'] = command
        widget.pack(side=side, fill=fill, expand=expand)
        self.bind(widget, help)
        self.addWidget(category, text, widget)
        return widget

    def newCreateLabeledEntry(self, parent, category, text, help='', command=None, value='', width=12, relief=tk.SUNKEN, side=tk.LEFT, fill=tk.X, expand=0):
        if False:
            for i in range(10):
                print('nop')
        ' createLabeledEntry(parent, category, text, [options]) '
        frame = tk.Frame(parent)
        variable = tk.StringVar()
        variable.set(value)
        label = tk.Label(frame, text=text)
        label.pack(side=tk.LEFT, fill=tk.X, expand=0)
        entry = tk.Entry(frame, width=width, relief=relief, textvariable=variable)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=1)
        frame.pack(side=side, fill=tk.X, expand=expand)
        if command:
            entry.bind('<Return>', command)
        self.bind(label, help)
        self.bind(entry, help)
        self.addWidget(category, text, entry)
        self.addWidget(category, text + '-Label', label)
        self.addVariable(category, text, variable)
        return entry

    def newCreateButton(self, parent, category, text, help='', command=None, side=tk.LEFT, fill=tk.X, expand=0, **kw):
        if False:
            return 10
        ' createButton(parent, category, text, [options]) '
        widget = self.createWidget(parent, category, text, tk.Button, help, command, side, fill, expand, kw)
        return widget

    def newCreateCheckbutton(self, parent, category, text, help='', command=None, initialState=0, anchor=tk.W, side=tk.LEFT, fill=tk.X, expand=0, **kw):
        if False:
            return 10
        ' createCheckbutton(parent, category, text, [options]) '
        widget = self.createWidget(parent, category, text, tk.Checkbutton, help, command, side, fill, expand, kw)
        widget['anchor'] = anchor
        variable = tk.BooleanVar()
        variable.set(initialState)
        self.addVariable(category, text, variable)
        widget['variable'] = variable
        return widget

    def newCreateRadiobutton(self, parent, category, text, variable, value, command=None, help='', anchor=tk.W, side=tk.LEFT, fill=tk.X, expand=0, **kw):
        if False:
            for i in range(10):
                print('nop')
        '\n        createRadiobutton(parent, category, text, variable, value, [options])\n        '
        widget = self.createWidget(parent, category, text, tk.Radiobutton, help, command, side, fill, expand, kw)
        widget['anchor'] = anchor
        widget['value'] = value
        widget['variable'] = variable
        return widget

    def newCreateFloater(self, parent, category, text, help='', command=None, side=tk.LEFT, fill=tk.X, expand=0, **kw):
        if False:
            i = 10
            return i + 15
        widget = self.createWidget(parent, category, text, Floater.Floater, help, command, side, fill, expand, kw)
        return widget

    def newCreateDial(self, parent, category, text, help='', command=None, side=tk.LEFT, fill=tk.X, expand=0, **kw):
        if False:
            return 10
        widget = self.createWidget(parent, category, text, Dial.Dial, help, command, side, fill, expand, kw)
        return widget

    def newCreateSider(self, parent, category, text, help='', command=None, side=tk.LEFT, fill=tk.X, expand=0, **kw):
        if False:
            print('Hello World!')
        widget = self.createWidget(parent, category, text, Slider.Slider, help, command, side, fill, expand, kw)
        return widget

    def newCreateEntryScale(self, parent, category, text, help='', command=None, side=tk.LEFT, fill=tk.X, expand=0, **kw):
        if False:
            while True:
                i = 10
        widget = self.createWidget(parent, category, text, EntryScale.EntryScale, help, command, side, fill, expand, kw)
        return widget

    def newCreateVector2Entry(self, parent, category, text, help='', command=None, side=tk.LEFT, fill=tk.X, expand=0, **kw):
        if False:
            print('Hello World!')
        widget = self.createWidget(parent, category, text, VectorWidgets.Vector2Entry, help, command, side, fill, expand, kw)

    def newCreateVector3Entry(self, parent, category, text, help='', command=None, side=tk.LEFT, fill=tk.X, expand=0, **kw):
        if False:
            print('Hello World!')
        widget = self.createWidget(parent, category, text, VectorWidgets.Vector3Entry, help, command, side, fill, expand, kw)
        return widget

    def newCreateColorEntry(self, parent, category, text, help='', command=None, side=tk.LEFT, fill=tk.X, expand=0, **kw):
        if False:
            while True:
                i = 10
        widget = self.createWidget(parent, category, text, VectorWidgets.ColorEntry, help, command, side, fill, expand, kw)
        return widget

    def newCreateOptionMenu(self, parent, category, text, help='', command=None, items=[], labelpos=tk.W, label_anchor=tk.W, label_width=16, menu_tearoff=1, side=tk.LEFT, fill=tk.X, expand=0, **kw):
        if False:
            i = 10
            return i + 15
        variable = tk.StringVar()
        if len(items) > 0:
            variable.set(items[0])
        kw['items'] = items
        kw['label_text'] = text
        kw['labelpos'] = labelpos
        kw['label_anchor'] = label_anchor
        kw['label_width'] = label_width
        kw['menu_tearoff'] = menu_tearoff
        kw['menubutton_textvariable'] = variable
        widget = Pmw.OptionMenu(parent, **kw)
        widget['command'] = command
        widget.pack(side=side, fill=fill, expand=expand)
        self.bind(widget.component('menubutton'), help)
        self.addWidget(category, text, widget)
        self.addVariable(category, text, variable)
        return widget

    def newCreateComboBox(self, parent, category, text, help='', command=None, items=[], state=tk.DISABLED, history=0, labelpos=tk.W, label_anchor=tk.W, label_width=16, entry_width=16, side=tk.LEFT, fill=tk.X, expand=0, **kw):
        if False:
            i = 10
            return i + 15
        kw['label_text'] = text
        kw['labelpos'] = labelpos
        kw['label_anchor'] = label_anchor
        kw['label_width'] = label_width
        kw['entry_width'] = entry_width
        kw['scrolledlist_items'] = items
        kw['entryfield_entry_state'] = state
        widget = Pmw.ComboBox(parent, **kw)
        widget['selectioncommand'] = command
        if len(items) > 0:
            widget.selectitem(items[0])
        widget.pack(side=side, fill=fill, expand=expand)
        self.bind(widget, help)
        self.addWidget(category, text, widget)
        return widget

    def transformRGB(self, rgb, max=1.0):
        if False:
            while True:
                i = 10
        retval = '#'
        for v in [rgb[0], rgb[1], rgb[2]]:
            v = v / max * 255
            if v > 255:
                v = 255
            if v < 0:
                v = 0
            retval = '%s%02x' % (retval, int(v))
        return retval

class TestAppShell(AppShell):
    appname = 'Test Application Shell'
    usecommandarea = 1
    usestatusarea = 1

    def __init__(self, parent=None, **kw):
        if False:
            return 10
        AppShell.__init__(self)
        self.initialiseoptions(TestAppShell)

    def createButtons(self):
        if False:
            return 10
        self.buttonAdd('Ok', helpMessage='Exit', statusMessage='Exit', command=self.quit)

    def createMain(self):
        if False:
            i = 10
            return i + 15
        self.label = self.createcomponent('label', (), None, tk.Label, (self.interior(),), text='Data Area')
        self.label.pack()
        self.bind(self.label, 'Space taker')

    def createInterface(self):
        if False:
            for i in range(10):
                print('nop')
        self.createButtons()
        self.createMain()