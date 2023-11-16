"""Support for stack-frames.

Provides Implements a nearly complete wrapper for a stack frame.
"""
import pythoncom
from . import axdebug, expressions, gateways
from .util import RaiseNotImpl, _wrap, trace

class EnumDebugStackFrames(gateways.EnumDebugStackFrames):
    """A class that given a debugger object, can return an enumerator
    of DebugStackFrame objects.
    """

    def __init__(self, debugger):
        if False:
            print('Hello World!')
        infos = []
        frame = debugger.currentframe
        while frame:
            cc = debugger.codeContainerProvider.FromFileName(frame.f_code.co_filename)
            if cc is not None:
                try:
                    address = frame.f_locals['__axstack_address__']
                except KeyError:
                    address = axdebug.GetStackAddress()
                frameInfo = (DebugStackFrame(frame, frame.f_lineno - 1, cc), address, address + 1, 0, None)
                infos.append(frameInfo)
            frame = frame.f_back
        gateways.EnumDebugStackFrames.__init__(self, infos, 0)

    def Next(self, count):
        if False:
            for i in range(10):
                print('nop')
        return gateways.EnumDebugStackFrames.Next(self, count)

    def _wrap(self, obj):
        if False:
            i = 10
            return i + 15
        (obFrame, min, lim, fFinal, obFinal) = obj
        obFrame = _wrap(obFrame, axdebug.IID_IDebugStackFrame)
        if obFinal:
            obFinal = _wrap(obFinal, pythoncom.IID_IUnknown)
        return (obFrame, min, lim, fFinal, obFinal)

class DebugStackFrame(gateways.DebugStackFrame):

    def __init__(self, frame, lineno, codeContainer):
        if False:
            print('Hello World!')
        self.frame = frame
        self.lineno = lineno
        self.codeContainer = codeContainer
        self.expressionContext = None

    def _query_interface_(self, iid):
        if False:
            while True:
                i = 10
        if iid == axdebug.IID_IDebugExpressionContext:
            if self.expressionContext is None:
                self.expressionContext = _wrap(expressions.ExpressionContext(self.frame), axdebug.IID_IDebugExpressionContext)
            return self.expressionContext
        return 0

    def GetThread(self):
        if False:
            print('Hello World!')
        'Returns the thread associated with this stack frame.\n\n        Result must be a IDebugApplicationThread\n        '
        RaiseNotImpl('GetThread')

    def GetCodeContext(self):
        if False:
            print('Hello World!')
        offset = self.codeContainer.GetPositionOfLine(self.lineno)
        return self.codeContainer.GetCodeContextAtPosition(offset)

    def GetDescriptionString(self, fLong):
        if False:
            print('Hello World!')
        filename = self.frame.f_code.co_filename
        s = ''
        if 0:
            s = s + filename
        if self.frame.f_code.co_name:
            s = s + self.frame.f_code.co_name
        else:
            s = s + '<lambda>'
        return s

    def GetLanguageString(self, fLong):
        if False:
            i = 10
            return i + 15
        if fLong:
            return 'Python ActiveX Scripting Engine'
        else:
            return 'Python'

    def GetDebugProperty(self):
        if False:
            i = 10
            return i + 15
        return _wrap(StackFrameDebugProperty(self.frame), axdebug.IID_IDebugProperty)

class DebugStackFrameSniffer:
    _public_methods_ = ['EnumStackFrames']
    _com_interfaces_ = [axdebug.IID_IDebugStackFrameSniffer]

    def __init__(self, debugger):
        if False:
            for i in range(10):
                print('nop')
        self.debugger = debugger
        trace('DebugStackFrameSniffer instantiated')

    def EnumStackFrames(self):
        if False:
            while True:
                i = 10
        trace('DebugStackFrameSniffer.EnumStackFrames called')
        return _wrap(EnumDebugStackFrames(self.debugger), axdebug.IID_IEnumDebugStackFrames)

class StackFrameDebugProperty:
    _com_interfaces_ = [axdebug.IID_IDebugProperty]
    _public_methods_ = ['GetPropertyInfo', 'GetExtendedInfo', 'SetValueAsString', 'EnumMembers', 'GetParent']

    def __init__(self, frame):
        if False:
            print('Hello World!')
        self.frame = frame

    def GetPropertyInfo(self, dwFieldSpec, nRadix):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('StackFrameDebugProperty::GetPropertyInfo')

    def GetExtendedInfo(self):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('StackFrameDebugProperty::GetExtendedInfo')

    def SetValueAsString(self, value, radix):
        if False:
            i = 10
            return i + 15
        RaiseNotImpl('DebugProperty::SetValueAsString')

    def EnumMembers(self, dwFieldSpec, nRadix, iid):
        if False:
            return 10
        print('EnumMembers', dwFieldSpec, nRadix, iid)
        from . import expressions
        return expressions.MakeEnumDebugProperty(self.frame.f_locals, dwFieldSpec, nRadix, iid, self.frame)

    def GetParent(self):
        if False:
            while True:
                i = 10
        RaiseNotImpl('DebugProperty::GetParent')