"""dynamic dispatch objects for AX Script.

 This is an IDispatch object that a scripting host may use to
 query and invoke methods on the main script.  Not may hosts use
 this yet, so it is not well tested!
"""
import types
import pythoncom
import win32com.server.policy
import win32com.server.util
import winerror
from win32com.axscript import axscript
from win32com.client import Dispatch
from win32com.server.exception import COMException
debugging = 0
PyIDispatchType = pythoncom.TypeIIDs[pythoncom.IID_IDispatch]

def _is_callable(obj):
    if False:
        return 10
    return isinstance(obj, (types.FunctionType, types.MethodType))

class ScriptDispatch:
    _public_methods_ = []

    def __init__(self, engine, scriptNamespace):
        if False:
            i = 10
            return i + 15
        self.engine = engine
        self.scriptNamespace = scriptNamespace

    def _dynamic_(self, name, lcid, wFlags, args):
        if False:
            return 10
        self.engine.RegisterNewNamedItems()
        self.engine.ProcessNewNamedItemsConnections()
        if wFlags & pythoncom.INVOKE_FUNC:
            try:
                func = getattr(self.scriptNamespace, name)
                if not _is_callable(func):
                    raise AttributeError(name)
                realArgs = []
                for arg in args:
                    if isinstance(arg, PyIDispatchType):
                        realArgs.append(Dispatch(arg))
                    else:
                        realArgs.append(arg)
                return self.engine.ApplyInScriptedSection(None, func, tuple(realArgs))
            except AttributeError:
                if not wFlags & pythoncom.DISPATCH_PROPERTYGET:
                    raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)
        if wFlags & pythoncom.DISPATCH_PROPERTYGET:
            try:
                ret = getattr(self.scriptNamespace, name)
                if _is_callable(ret):
                    raise AttributeError(name)
            except AttributeError:
                raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)
            except COMException as instance:
                raise
            except:
                ret = self.engine.HandleException()
            return ret
        raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)

class StrictDynamicPolicy(win32com.server.policy.DynamicPolicy):

    def _wrap_(self, object):
        if False:
            for i in range(10):
                print('nop')
        win32com.server.policy.DynamicPolicy._wrap_(self, object)
        if hasattr(self._obj_, 'scriptNamespace'):
            for name in dir(self._obj_.scriptNamespace):
                self._dyn_dispid_to_name_[self._getdispid_(name, 0)] = name

    def _getmembername_(self, dispid):
        if False:
            return 10
        try:
            return str(self._dyn_dispid_to_name_[dispid])
        except KeyError:
            raise COMException(scode=winerror.DISP_E_UNKNOWNNAME, desc='Name not found')

    def _getdispid_(self, name, fdex):
        if False:
            return 10
        try:
            func = getattr(self._obj_.scriptNamespace, str(name))
        except AttributeError:
            raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)
        return win32com.server.policy.DynamicPolicy._getdispid_(self, name, fdex)

def _wrap_debug(obj):
    if False:
        while True:
            i = 10
    return win32com.server.util.wrap(obj, usePolicy=StrictDynamicPolicy, useDispatcher=win32com.server.policy.DispatcherWin32trace)

def _wrap_nodebug(obj):
    if False:
        i = 10
        return i + 15
    return win32com.server.util.wrap(obj, usePolicy=StrictDynamicPolicy)
if debugging:
    _wrap = _wrap_debug
else:
    _wrap = _wrap_nodebug

def MakeScriptDispatch(engine, namespace):
    if False:
        for i in range(10):
            print('nop')
    return _wrap(ScriptDispatch(engine, namespace))