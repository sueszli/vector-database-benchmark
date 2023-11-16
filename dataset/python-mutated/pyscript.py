"""Python ActiveX Scripting Implementation

This module implements the Python ActiveX Scripting client.

To register the implementation, simply "run" this Python program - ie
either double-click on it, or run "python.exe pyscript.py" from the
command line.
"""
import re
import types
import pythoncom
import win32api
import win32com
import win32com.client.dynamic
import win32com.server.register
import winerror
from win32com.axscript import axscript
from win32com.axscript.client import framework, scriptdispatch
from win32com.axscript.client.framework import SCRIPTTEXT_FORCEEXECUTION, SCRIPTTEXT_ISEXPRESSION, SCRIPTTEXT_ISPERSISTENT, Exception, RaiseAssert, trace
PyScript_CLSID = '{DF630910-1C1D-11d0-AE36-8C0F5E000000}'
debugging_attr = 0

def debug_attr_print(*args):
    if False:
        while True:
            i = 10
    if debugging_attr:
        trace(*args)

def ExpandTabs(text):
    if False:
        return 10
    return re.sub('\\t', '    ', text)

def AddCR(text):
    if False:
        for i in range(10):
            print('nop')
    return re.sub('\\n', '\r\n', text)

class AXScriptCodeBlock(framework.AXScriptCodeBlock):

    def GetDisplayName(self):
        if False:
            for i in range(10):
                print('nop')
        return 'PyScript - ' + framework.AXScriptCodeBlock.GetDisplayName(self)

class AXScriptAttribute:
    """An attribute in a scripts namespace."""

    def __init__(self, engine):
        if False:
            print('Hello World!')
        self.__dict__['_scriptEngine_'] = engine

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        if attr[1] == '_' and attr[:-1] == '_':
            raise AttributeError(attr)
        rc = self._FindAttribute_(attr)
        if rc is None:
            raise AttributeError(attr)
        return rc

    def _Close_(self):
        if False:
            i = 10
            return i + 15
        self.__dict__['_scriptEngine_'] = None

    def _DoFindAttribute_(self, obj, attr):
        if False:
            while True:
                i = 10
        try:
            return obj.subItems[attr.lower()].attributeObject
        except KeyError:
            pass
        for item in obj.subItems.values():
            try:
                return self._DoFindAttribute_(item, attr)
            except AttributeError:
                pass
        raise AttributeError(attr)

    def _FindAttribute_(self, attr):
        if False:
            while True:
                i = 10
        for item in self._scriptEngine_.subItems.values():
            try:
                return self._DoFindAttribute_(item, attr)
            except AttributeError:
                pass
        return getattr(self._scriptEngine_.globalNameSpaceModule, attr)

class NamedScriptAttribute:
    """An explicitely named object in an objects namespace"""

    def __init__(self, scriptItem):
        if False:
            return 10
        self.__dict__['_scriptItem_'] = scriptItem

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<NamedItemAttribute' + repr(self._scriptItem_) + '>'

    def __getattr__(self, attr):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._scriptItem_.subItems[attr.lower()].attributeObject
        except KeyError:
            if self._scriptItem_.dispatchContainer:
                return getattr(self._scriptItem_.dispatchContainer, attr)
        raise AttributeError(attr)

    def __setattr__(self, attr, value):
        if False:
            i = 10
            return i + 15
        attr = attr.lower()
        if self._scriptItem_.dispatchContainer:
            try:
                return setattr(self._scriptItem_.dispatchContainer, attr, value)
            except AttributeError:
                pass
        raise AttributeError(attr)

    def _Close_(self):
        if False:
            print('Hello World!')
        self.__dict__['_scriptItem_'] = None

class ScriptItem(framework.ScriptItem):

    def __init__(self, parentItem, name, dispatch, flags):
        if False:
            for i in range(10):
                print('nop')
        framework.ScriptItem.__init__(self, parentItem, name, dispatch, flags)
        self.scriptlets = {}
        self.attributeObject = None

    def Reset(self):
        if False:
            print('Hello World!')
        framework.ScriptItem.Reset(self)
        if self.attributeObject:
            self.attributeObject._Close_()
        self.attributeObject = None

    def Close(self):
        if False:
            return 10
        framework.ScriptItem.Close(self)
        self.dispatchContainer = None
        self.scriptlets = {}

    def Register(self):
        if False:
            for i in range(10):
                print('nop')
        framework.ScriptItem.Register(self)
        self.attributeObject = NamedScriptAttribute(self)
        if self.dispatch:
            try:
                engine = self.GetEngine()
                olerepr = clsid = None
                typeinfo = self.dispatch.GetTypeInfo()
                clsid = typeinfo.GetTypeAttr()[0]
                try:
                    olerepr = engine.mapKnownCOMTypes[clsid]
                except KeyError:
                    pass
            except pythoncom.com_error:
                typeinfo = None
            if olerepr is None:
                olerepr = win32com.client.dynamic.MakeOleRepr(self.dispatch, typeinfo, None)
                if clsid is not None:
                    engine.mapKnownCOMTypes[clsid] = olerepr
            self.dispatchContainer = win32com.client.dynamic.CDispatch(self.dispatch, olerepr, self.name)

class PyScript(framework.COMScript):
    _reg_verprogid_ = 'Python.AXScript.2'
    _reg_progid_ = 'Python'
    _reg_catids_ = [axscript.CATID_ActiveScript, axscript.CATID_ActiveScriptParse]
    _reg_desc_ = 'Python ActiveX Scripting Engine'
    _reg_clsid_ = PyScript_CLSID
    _reg_class_spec_ = 'win32com.axscript.client.pyscript.PyScript'
    _reg_remove_keys_ = [('.pys',), ('pysFile',)]
    _reg_threading_ = 'both'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        framework.COMScript.__init__(self)
        self.globalNameSpaceModule = None
        self.codeBlocks = []
        self.scriptDispatch = None

    def InitNew(self):
        if False:
            i = 10
            return i + 15
        framework.COMScript.InitNew(self)
        self.scriptDispatch = None
        self.globalNameSpaceModule = types.ModuleType('__ax_main__')
        self.globalNameSpaceModule.__dict__['ax'] = AXScriptAttribute(self)
        self.codeBlocks = []
        self.persistedCodeBlocks = []
        self.mapKnownCOMTypes = {}
        self.codeBlockCounter = 0

    def Stop(self):
        if False:
            while True:
                i = 10
        for b in self.codeBlocks:
            b.beenExecuted = 1
        return framework.COMScript.Stop(self)

    def Reset(self):
        if False:
            while True:
                i = 10
        oldCodeBlocks = self.codeBlocks[:]
        self.codeBlocks = []
        for b in oldCodeBlocks:
            if b.flags & SCRIPTTEXT_ISPERSISTENT:
                b.beenExecuted = 0
                self.codeBlocks.append(b)
        return framework.COMScript.Reset(self)

    def _GetNextCodeBlockNumber(self):
        if False:
            for i in range(10):
                print('nop')
        self.codeBlockCounter = self.codeBlockCounter + 1
        return self.codeBlockCounter

    def RegisterNamedItem(self, item):
        if False:
            i = 10
            return i + 15
        wasReg = item.isRegistered
        framework.COMScript.RegisterNamedItem(self, item)
        if not wasReg:
            if item.IsVisible():
                self.globalNameSpaceModule.__dict__[item.name] = item.attributeObject
            if item.IsGlobal():
                for subitem in item.subItems.values():
                    self.globalNameSpaceModule.__dict__[subitem.name] = subitem.attributeObject
                for (name, entry) in item.dispatchContainer._olerepr_.mapFuncs.items():
                    if not entry.hidden:
                        self.globalNameSpaceModule.__dict__[name] = getattr(item.dispatchContainer, name)

    def DoExecutePendingScripts(self):
        if False:
            while True:
                i = 10
        try:
            globs = self.globalNameSpaceModule.__dict__
            for codeBlock in self.codeBlocks:
                if not codeBlock.beenExecuted:
                    if self.CompileInScriptedSection(codeBlock, 'exec'):
                        self.ExecInScriptedSection(codeBlock, globs)
        finally:
            pass

    def DoRun(self):
        if False:
            print('Hello World!')
        pass

    def Close(self):
        if False:
            return 10
        self.ResetNamespace()
        self.globalNameSpaceModule = None
        self.codeBlocks = []
        self.scriptDispatch = None
        framework.COMScript.Close(self)

    def GetScriptDispatch(self, name):
        if False:
            print('Hello World!')
        if self.scriptDispatch is None:
            self.scriptDispatch = scriptdispatch.MakeScriptDispatch(self, self.globalNameSpaceModule)
        return self.scriptDispatch

    def MakeEventMethodName(self, subItemName, eventName):
        if False:
            print('Hello World!')
        return subItemName[0].upper() + subItemName[1:] + '_' + eventName[0].upper() + eventName[1:]

    def DoAddScriptlet(self, defaultName, code, itemName, subItemName, eventName, delimiter, sourceContextCookie, startLineNumber):
        if False:
            while True:
                i = 10
        item = self.GetNamedItem(itemName)
        if itemName == subItemName:
            subItem = item
        else:
            subItem = item.GetCreateSubItem(item, subItemName, None, None)
        funcName = self.MakeEventMethodName(subItemName, eventName)
        codeBlock = AXScriptCodeBlock('Script Event %s' % funcName, code, sourceContextCookie, startLineNumber, 0)
        self._AddScriptCodeBlock(codeBlock)
        subItem.scriptlets[funcName] = codeBlock

    def DoProcessScriptItemEvent(self, item, event, lcid, wFlags, args):
        if False:
            return 10
        funcName = self.MakeEventMethodName(item.name, event.name)
        codeBlock = function = None
        try:
            function = item.scriptlets[funcName]
            if isinstance(function, PyScript):
                codeBlock = function
                function = None
        except KeyError:
            pass
        if codeBlock is not None:
            realCode = 'def %s():\n' % funcName
            for line in framework.RemoveCR(codeBlock.codeText).split('\n'):
                realCode = realCode + '\t' + line + '\n'
            realCode = realCode + '\n'
            if not self.CompileInScriptedSection(codeBlock, 'exec', realCode):
                return
            dict = {}
            self.ExecInScriptedSection(codeBlock, self.globalNameSpaceModule.__dict__, dict)
            function = dict[funcName]
            item.scriptlets[funcName] = function
        if function is None:
            try:
                function = self.globalNameSpaceModule.__dict__[funcName]
            except KeyError:
                funcNameLook = funcName.lower()
                for attr in self.globalNameSpaceModule.__dict__.keys():
                    if funcNameLook == attr.lower():
                        function = self.globalNameSpaceModule.__dict__[attr]
                        item.scriptlets[funcName] = function
        if function is None:
            raise Exception(scode=winerror.DISP_E_MEMBERNOTFOUND)
        return self.ApplyInScriptedSection(codeBlock, function, args)

    def DoParseScriptText(self, code, sourceContextCookie, startLineNumber, bWantResult, flags):
        if False:
            print('Hello World!')
        code = framework.RemoveCR(code) + '\n'
        if flags & SCRIPTTEXT_ISEXPRESSION:
            name = 'Script Expression'
            exec_type = 'eval'
        else:
            name = 'Script Block'
            exec_type = 'exec'
        num = self._GetNextCodeBlockNumber()
        if num == 1:
            num = ''
        name = f'{name} {num}'
        codeBlock = AXScriptCodeBlock(name, code, sourceContextCookie, startLineNumber, flags)
        self._AddScriptCodeBlock(codeBlock)
        globs = self.globalNameSpaceModule.__dict__
        if bWantResult:
            if self.CompileInScriptedSection(codeBlock, exec_type):
                if flags & SCRIPTTEXT_ISEXPRESSION:
                    return self.EvalInScriptedSection(codeBlock, globs)
                else:
                    return self.ExecInScriptedSection(codeBlock, globs)
        elif flags & SCRIPTTEXT_FORCEEXECUTION:
            if self.CompileInScriptedSection(codeBlock, exec_type):
                self.ExecInScriptedSection(codeBlock, globs)
        else:
            self.codeBlocks.append(codeBlock)

    def GetNamedItemClass(self):
        if False:
            for i in range(10):
                print('nop')
        return ScriptItem

    def ResetNamespace(self):
        if False:
            print('Hello World!')
        if self.globalNameSpaceModule is not None:
            try:
                self.globalNameSpaceModule.ax._Reset_()
            except AttributeError:
                pass
            globalNameSpaceModule = None

def DllRegisterServer():
    if False:
        return 10
    klass = PyScript
    win32com.server.register._set_subkeys(klass._reg_progid_ + '\\OLEScript', {})
    win32com.server.register._set_string('.pys', 'pysFile')
    win32com.server.register._set_string('pysFile\\ScriptEngine', klass._reg_progid_)
    guid_wsh_shellex = '{60254CA5-953B-11CF-8C96-00AA00B8708C}'
    win32com.server.register._set_string('pysFile\\ShellEx\\DropHandler', guid_wsh_shellex)
    win32com.server.register._set_string('pysFile\\ShellEx\\PropertySheetHandlers\\WSHProps', guid_wsh_shellex)

def Register(klass=PyScript):
    if False:
        while True:
            i = 10
    ret = win32com.server.register.UseCommandLine(klass, finalize_register=DllRegisterServer)
    return ret
if __name__ == '__main__':
    Register()