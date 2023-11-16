from panda3d.direct import CInterval
from .extension_native_helpers import Dtool_funcToMethod
from direct.directnotify.DirectNotifyGlobal import directNotify
import warnings
CInterval.DtoolClassDict['notify'] = directNotify.newCategory('Interval')

def setT(self, t):
    if False:
        while True:
            i = 10
    self.setT_Old(t)
    self.privPostEvent()
CInterval.DtoolClassDict['setT_Old'] = CInterval.setT
Dtool_funcToMethod(setT, CInterval)
del setT

def play(self, t0=0.0, duration=None, scale=1.0):
    if False:
        while True:
            i = 10
    if __debug__:
        warnings.warn('CInterval.play() is deprecated, use start() instead', DeprecationWarning, stacklevel=2)
    if duration:
        self.start(t0, t0 + duration, scale)
    else:
        self.start(t0, -1, scale)
Dtool_funcToMethod(play, CInterval)
del play

def stop(self):
    if False:
        for i in range(10):
            print('nop')
    if __debug__:
        warnings.warn('CInterval.stop() is deprecated, use finish() instead', DeprecationWarning, stacklevel=2)
    self.finish()
Dtool_funcToMethod(stop, CInterval)
del stop

def setFinalT(self):
    if False:
        for i in range(10):
            print('nop')
    if __debug__:
        warnings.warn('CInterval.setFinalT() is deprecated, use finish() instead', DeprecationWarning, stacklevel=2)
    self.finish()
Dtool_funcToMethod(setFinalT, CInterval)
del setFinalT

def privPostEvent(self):
    if False:
        while True:
            i = 10
    t = self.getT()
    if hasattr(self, 'setTHooks'):
        for func in self.setTHooks:
            func(t)
Dtool_funcToMethod(privPostEvent, CInterval)
del privPostEvent

def popupControls(self, tl=None):
    if False:
        print('Hello World!')
    '\n    Popup control panel for interval.\n    '
    import math
    import importlib
    EntryScale = importlib.import_module('direct.tkwidgets.EntryScale')
    tkinter = importlib.import_module('tkinter')
    if tl is None:
        tl = tkinter.Toplevel()
        tl.title('Interval Controls')
    outerFrame = tkinter.Frame(tl)

    def entryScaleCommand(t, s=self):
        if False:
            while True:
                i = 10
        s.setT(t)
        s.pause()
    self.es = es = EntryScale.EntryScale(outerFrame, text=self.getName(), min=0, max=math.floor(self.getDuration() * 100) / 100, command=entryScaleCommand)
    es.set(self.getT(), fCommand=0)
    es.pack(expand=1, fill=tkinter.X)
    bf = tkinter.Frame(outerFrame)

    def toStart(s=self, es=es):
        if False:
            i = 10
            return i + 15
        s.setT(0.0)
        s.pause()

    def toEnd(s=self):
        if False:
            while True:
                i = 10
        s.setT(s.getDuration())
        s.pause()
    jumpToStart = tkinter.Button(bf, text='<<', command=toStart)

    def doPlay(s=self, es=es):
        if False:
            return 10
        s.resume(es.get())
    stop = tkinter.Button(bf, text='Stop', command=lambda s=self: s.pause())
    play = tkinter.Button(bf, text='Play', command=doPlay)
    jumpToEnd = tkinter.Button(bf, text='>>', command=toEnd)
    jumpToStart.pack(side=tkinter.LEFT, expand=1, fill=tkinter.X)
    play.pack(side=tkinter.LEFT, expand=1, fill=tkinter.X)
    stop.pack(side=tkinter.LEFT, expand=1, fill=tkinter.X)
    jumpToEnd.pack(side=tkinter.LEFT, expand=1, fill=tkinter.X)
    bf.pack(expand=1, fill=tkinter.X)
    outerFrame.pack(expand=1, fill=tkinter.X)

    def update(t, es=es):
        if False:
            for i in range(10):
                print('nop')
        es.set(t, fCommand=0)
    if not hasattr(self, 'setTHooks'):
        self.setTHooks = []
    self.setTHooks.append(update)
    self.setWantsTCallback(1)

    def onDestroy(e, s=self, u=update):
        if False:
            return 10
        if u in s.setTHooks:
            s.setTHooks.remove(u)
    tl.bind('<Destroy>', onDestroy)
Dtool_funcToMethod(popupControls, CInterval)
del popupControls