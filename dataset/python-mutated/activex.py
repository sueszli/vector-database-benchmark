"""Support for ActiveX control hosting in Pythonwin.
"""
import win32ui
import win32uiole
from . import window

class Control(window.Wnd):
    """An ActiveX control base class.  A new class must be derived from both
    this class and the Events class.  See the demos for more details.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.__dict__['_dispobj_'] = None
        window.Wnd.__init__(self)

    def _GetControlCLSID(self):
        if False:
            return 10
        return self.CLSID

    def _GetDispatchClass(self):
        if False:
            i = 10
            return i + 15
        return self.default_interface

    def _GetEventMap(self):
        if False:
            i = 10
            return i + 15
        return self.default_source._dispid_to_func_

    def CreateControl(self, windowTitle, style, rect, parent, id, lic_string=None):
        if False:
            return 10
        clsid = str(self._GetControlCLSID())
        self.__dict__['_obj_'] = win32ui.CreateControl(clsid, windowTitle, style, rect, parent, id, None, False, lic_string)
        klass = self._GetDispatchClass()
        dispobj = klass(win32uiole.GetIDispatchForWindow(self._obj_))
        self.HookOleEvents()
        self.__dict__['_dispobj_'] = dispobj

    def HookOleEvents(self):
        if False:
            for i in range(10):
                print('nop')
        dict = self._GetEventMap()
        for (dispid, methodName) in dict.items():
            if hasattr(self, methodName):
                self._obj_.HookOleEvent(getattr(self, methodName), dispid)

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        try:
            return window.Wnd.__getattr__(self, attr)
        except AttributeError:
            pass
        return getattr(self._dispobj_, attr)

    def __setattr__(self, attr, value):
        if False:
            print('Hello World!')
        if hasattr(self.__dict__, attr):
            self.__dict__[attr] = value
            return
        try:
            if self._dispobj_:
                self._dispobj_.__setattr__(attr, value)
                return
        except AttributeError:
            pass
        self.__dict__[attr] = value

def MakeControlClass(controlClass, name=None):
    if False:
        i = 10
        return i + 15
    'Given a CoClass in a generated .py file, this function will return a Class\n    object which can be used as an OCX control.\n\n    This function is used when you do not want to handle any events from the OCX\n    control.  If you need events, then you should derive a class from both the\n    activex.Control class and the CoClass\n    '
    if name is None:
        name = controlClass.__name__
    return type('OCX' + name, (Control, controlClass), {})

def MakeControlInstance(controlClass, name=None):
    if False:
        while True:
            i = 10
    'As for MakeControlClass(), but returns an instance of the class.'
    return MakeControlClass(controlClass, name)()