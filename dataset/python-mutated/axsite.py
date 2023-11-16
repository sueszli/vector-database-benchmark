import pythoncom
import win32com.axscript.axscript
import winerror
from win32com.axscript import axscript
from win32com.server import exception, util

class AXEngine:

    def __init__(self, site, engine):
        if False:
            for i in range(10):
                print('nop')
        self.eScript = self.eParse = self.eSafety = None
        if isinstance(engine, str):
            engine = pythoncom.CoCreateInstance(engine, None, pythoncom.CLSCTX_SERVER, pythoncom.IID_IUnknown)
        self.eScript = engine.QueryInterface(axscript.IID_IActiveScript)
        self.eParse = engine.QueryInterface(axscript.IID_IActiveScriptParse)
        self.eSafety = engine.QueryInterface(axscript.IID_IObjectSafety)
        self.eScript.SetScriptSite(site)
        self.eParse.InitNew()

    def __del__(self):
        if False:
            i = 10
            return i + 15
        self.Close()

    def GetScriptDispatch(self, name=None):
        if False:
            i = 10
            return i + 15
        return self.eScript.GetScriptDispatch(name)

    def AddNamedItem(self, item, flags):
        if False:
            return 10
        return self.eScript.AddNamedItem(item, flags)

    def AddCode(self, code, flags=0):
        if False:
            for i in range(10):
                print('nop')
        self.eParse.ParseScriptText(code, None, None, None, 0, 0, flags)

    def EvalCode(self, code):
        if False:
            for i in range(10):
                print('nop')
        return self.eParse.ParseScriptText(code, None, None, None, 0, 0, axscript.SCRIPTTEXT_ISEXPRESSION)

    def Start(self):
        if False:
            while True:
                i = 10
        self.eScript.SetScriptState(axscript.SCRIPTSTATE_STARTED)

    def Close(self):
        if False:
            i = 10
            return i + 15
        if self.eScript:
            self.eScript.Close()
        self.eScript = self.eParse = self.eSafety = None

    def SetScriptState(self, state):
        if False:
            print('Hello World!')
        self.eScript.SetScriptState(state)
IActiveScriptSite_methods = ['GetLCID', 'GetItemInfo', 'GetDocVersionString', 'OnScriptTerminate', 'OnStateChange', 'OnScriptError', 'OnEnterScript', 'OnLeaveScript']

class AXSite:
    """An Active Scripting site.  A Site can have exactly one engine."""
    _public_methods_ = IActiveScriptSite_methods
    _com_interfaces_ = [axscript.IID_IActiveScriptSite]

    def __init__(self, objModel={}, engine=None, lcid=0):
        if False:
            while True:
                i = 10
        self.lcid = lcid
        self.objModel = {}
        for (name, object) in objModel.items():
            self.objModel[name] = object
        self.engine = None
        if engine:
            self._AddEngine(engine)

    def AddEngine(self, engine):
        if False:
            i = 10
            return i + 15
        'Adds a new engine to the site.\n        engine can be a string, or a fully wrapped engine object.\n        '
        if isinstance(engine, str):
            newEngine = AXEngine(util.wrap(self), engine)
        else:
            newEngine = engine
        self.engine = newEngine
        flags = axscript.SCRIPTITEM_ISVISIBLE | axscript.SCRIPTITEM_NOCODE | axscript.SCRIPTITEM_GLOBALMEMBERS | axscript.SCRIPTITEM_ISPERSISTENT
        for name in self.objModel.keys():
            newEngine.AddNamedItem(name, flags)
            newEngine.SetScriptState(axscript.SCRIPTSTATE_INITIALIZED)
        return newEngine
    _AddEngine = AddEngine

    def _Close(self):
        if False:
            print('Hello World!')
        self.engine.Close()
        self.objModel = {}

    def GetLCID(self):
        if False:
            for i in range(10):
                print('nop')
        return self.lcid

    def GetItemInfo(self, name, returnMask):
        if False:
            for i in range(10):
                print('nop')
        if name not in self.objModel:
            raise exception.Exception(scode=winerror.TYPE_E_ELEMENTNOTFOUND, desc='item not found')
        if returnMask & axscript.SCRIPTINFO_IUNKNOWN:
            return (self.objModel[name], None)
        return (None, None)

    def GetDocVersionString(self):
        if False:
            return 10
        return 'Python AXHost version 1.0'

    def OnScriptTerminate(self, result, excepInfo):
        if False:
            while True:
                i = 10
        pass

    def OnStateChange(self, state):
        if False:
            while True:
                i = 10
        pass

    def OnScriptError(self, errorInterface):
        if False:
            for i in range(10):
                print('nop')
        return winerror.S_FALSE

    def OnEnterScript(self):
        if False:
            return 10
        pass

    def OnLeaveScript(self):
        if False:
            print('Hello World!')
        pass