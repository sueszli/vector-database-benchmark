import pythoncom
import win32com.server.connect
import winerror
from win32com.axdebug import axdebug
from win32com.axdebug.util import RaiseNotImpl, _wrap
from win32com.server.exception import Exception
from win32com.server.util import ListEnumeratorGateway

class EnumDebugCodeContexts(ListEnumeratorGateway):
    """A class to expose a Python sequence as an EnumDebugCodeContexts

    Create an instance of this class passing a sequence (list, tuple, or
    any sequence protocol supporting object) and it will automatically
    support the EnumDebugCodeContexts interface for the object.

    """
    _com_interfaces_ = [axdebug.IID_IEnumDebugCodeContexts]

class EnumDebugStackFrames(ListEnumeratorGateway):
    """A class to expose a Python sequence as an EnumDebugStackFrames

    Create an instance of this class passing a sequence (list, tuple, or
    any sequence protocol supporting object) and it will automatically
    support the EnumDebugStackFrames interface for the object.

    """
    _com_interfaces_ = [axdebug.IID_IEnumDebugStackFrames]

class EnumDebugApplicationNodes(ListEnumeratorGateway):
    """A class to expose a Python sequence as an EnumDebugStackFrames

    Create an instance of this class passing a sequence (list, tuple, or
    any sequence protocol supporting object) and it will automatically
    support the EnumDebugApplicationNodes interface for the object.

    """
    _com_interfaces_ = [axdebug.IID_IEnumDebugApplicationNodes]

class EnumRemoteDebugApplications(ListEnumeratorGateway):
    _com_interfaces_ = [axdebug.IID_IEnumRemoteDebugApplications]

class EnumRemoteDebugApplicationThreads(ListEnumeratorGateway):
    _com_interfaces_ = [axdebug.IID_IEnumRemoteDebugApplicationThreads]

class DebugDocumentInfo:
    _public_methods_ = ['GetName', 'GetDocumentClassId']
    _com_interfaces_ = [axdebug.IID_IDebugDocumentInfo]

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def GetName(self, dnt):
        if False:
            print('Hello World!')
        'Get the one of the name of the document\n        dnt -- int DOCUMENTNAMETYPE\n        '
        RaiseNotImpl('GetName')

    def GetDocumentClassId(self):
        if False:
            i = 10
            return i + 15
        '\n        Result must be an IID object (or string representing one).\n        '
        RaiseNotImpl('GetDocumentClassId')

class DebugDocumentProvider(DebugDocumentInfo):
    _public_methods_ = DebugDocumentInfo._public_methods_ + ['GetDocument']
    _com_interfaces_ = DebugDocumentInfo._com_interfaces_ + [axdebug.IID_IDebugDocumentProvider]

    def GetDocument(self):
        if False:
            print('Hello World!')
        RaiseNotImpl('GetDocument')

class DebugApplicationNode(DebugDocumentProvider):
    """Provides the functionality of IDebugDocumentProvider, plus a context within a project tree."""
    _public_methods_ = 'EnumChildren GetParent SetDocumentProvider\n                    Close Attach Detach'.split() + DebugDocumentProvider._public_methods_
    _com_interfaces_ = [axdebug.IID_IDebugDocumentProvider] + DebugDocumentProvider._com_interfaces_

    def __init__(self):
        if False:
            while True:
                i = 10
        DebugDocumentProvider.__init__(self)

    def EnumChildren(self):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('EnumChildren')

    def GetParent(self):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('GetParent')

    def SetDocumentProvider(self, pddp):
        if False:
            print('Hello World!')
        RaiseNotImpl('SetDocumentProvider')

    def Close(self):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('Close')

    def Attach(self, parent):
        if False:
            i = 10
            return i + 15
        RaiseNotImpl('Attach')

    def Detach(self):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('Detach')

class DebugApplicationNodeEvents:
    """Event interface for DebugApplicationNode object."""
    _public_methods_ = 'onAddChild onRemoveChild onDetach'.split()
    _com_interfaces_ = [axdebug.IID_IDebugApplicationNodeEvents]

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def onAddChild(self, child):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('onAddChild')

    def onRemoveChild(self, child):
        if False:
            while True:
                i = 10
        RaiseNotImpl('onRemoveChild')

    def onDetach(self):
        if False:
            while True:
                i = 10
        RaiseNotImpl('onDetach')

    def onAttach(self, parent):
        if False:
            while True:
                i = 10
        RaiseNotImpl('onAttach')

class DebugDocument(DebugDocumentInfo):
    """The base interface to all debug documents."""
    _public_methods_ = DebugDocumentInfo._public_methods_
    _com_interfaces_ = [axdebug.IID_IDebugDocument] + DebugDocumentInfo._com_interfaces_

class DebugDocumentText(DebugDocument):
    """The interface to a text only debug document."""
    _com_interfaces_ = [axdebug.IID_IDebugDocumentText] + DebugDocument._com_interfaces_
    _public_methods_ = ['GetDocumentAttributes', 'GetSize', 'GetPositionOfLine', 'GetLineOfPosition', 'GetText', 'GetPositionOfContext', 'GetContextOfPosition'] + DebugDocument._public_methods_

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def GetDocumentAttributes(self):
        if False:
            while True:
                i = 10
        RaiseNotImpl('GetDocumentAttributes')

    def GetSize(self):
        if False:
            i = 10
            return i + 15
        RaiseNotImpl('GetSize')

    def GetPositionOfLine(self, cLineNumber):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('GetPositionOfLine')

    def GetLineOfPosition(self, charPos):
        if False:
            while True:
                i = 10
        RaiseNotImpl('GetLineOfPosition')

    def GetText(self, charPos, maxChars, wantAttr):
        if False:
            print('Hello World!')
        'Params\n        charPos -- integer\n        maxChars -- integer\n        wantAttr -- Should the function compute attributes.\n\n        Return value must be (string, attribtues).  attributes may be\n        None if(not wantAttr)\n        '
        RaiseNotImpl('GetText')

    def GetPositionOfContext(self, debugDocumentContext):
        if False:
            print('Hello World!')
        'Params\n        debugDocumentContext -- a PyIDebugDocumentContext object.\n\n        Return value must be (charPos, numChars)\n        '
        RaiseNotImpl('GetPositionOfContext')

    def GetContextOfPosition(self, charPos, maxChars):
        if False:
            return 10
        'Params are integers.\n        Return value must be PyIDebugDocumentContext object\n        '
        print(self)
        RaiseNotImpl('GetContextOfPosition')

class DebugDocumentTextExternalAuthor:
    """Allow external editors to edit file-based debugger documents, and to notify the document when the source file has been changed."""
    _public_methods_ = ['GetPathName', 'GetFileName', 'NotifyChanged']
    _com_interfaces_ = [axdebug.IID_IDebugDocumentTextExternalAuthor]

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def GetPathName(self):
        if False:
            print('Hello World!')
        "Return the full path (including file name) to the document's source file.\n\n        Result must be (filename, fIsOriginal), where\n        - if fIsOriginalPath is TRUE if the path refers to the original file for the document.\n        - if fIsOriginalPath is FALSE if the path refers to a newly created temporary file.\n\n        raise Exception(winerror.E_FAIL) if no source file can be created/determined.\n        "
        RaiseNotImpl('GetPathName')

    def GetFileName(self):
        if False:
            while True:
                i = 10
        'Return just the name of the document, with no path information.  (Used for "Save As...")\n\n        Result is a string\n        '
        RaiseNotImpl('GetFileName')

    def NotifyChanged(self):
        if False:
            return 10
        "Notify the host that the document's source file has been saved and\n        that its contents should be refreshed.\n        "
        RaiseNotImpl('NotifyChanged')

class DebugDocumentTextEvents:
    _public_methods_ = 'onDestroy onInsertText onRemoveText\n              onReplaceText onUpdateTextAttributes\n              onUpdateDocumentAttributes'.split()
    _com_interfaces_ = [axdebug.IID_IDebugDocumentTextEvents]

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def onDestroy(self):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('onDestroy')

    def onInsertText(self, cCharacterPosition, cNumToInsert):
        if False:
            return 10
        RaiseNotImpl('onInsertText')

    def onRemoveText(self, cCharacterPosition, cNumToRemove):
        if False:
            while True:
                i = 10
        RaiseNotImpl('onRemoveText')

    def onReplaceText(self, cCharacterPosition, cNumToReplace):
        if False:
            return 10
        RaiseNotImpl('onReplaceText')

    def onUpdateTextAttributes(self, cCharacterPosition, cNumToUpdate):
        if False:
            return 10
        RaiseNotImpl('onUpdateTextAttributes')

    def onUpdateDocumentAttributes(self, textdocattr):
        if False:
            while True:
                i = 10
        RaiseNotImpl('onUpdateDocumentAttributes')

class DebugDocumentContext:
    _public_methods_ = ['GetDocument', 'EnumCodeContexts']
    _com_interfaces_ = [axdebug.IID_IDebugDocumentContext]

    def __init__(self):
        if False:
            return 10
        pass

    def GetDocument(self):
        if False:
            return 10
        'Return value must be a PyIDebugDocument object'
        RaiseNotImpl('GetDocument')

    def EnumCodeContexts(self):
        if False:
            i = 10
            return i + 15
        'Return value must be a PyIEnumDebugCodeContexts object'
        RaiseNotImpl('EnumCodeContexts')

class DebugCodeContext:
    _public_methods_ = ['GetDocumentContext', 'SetBreakPoint']
    _com_interfaces_ = [axdebug.IID_IDebugCodeContext]

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def GetDocumentContext(self):
        if False:
            for i in range(10):
                print('nop')
        'Return value must be a PyIDebugDocumentContext object'
        RaiseNotImpl('GetDocumentContext')

    def SetBreakPoint(self, bps):
        if False:
            i = 10
            return i + 15
        'bps -- an integer with flags.'
        RaiseNotImpl('SetBreakPoint')

class DebugStackFrame:
    """Abstraction representing a logical stack frame on the stack of a thread."""
    _public_methods_ = ['GetCodeContext', 'GetDescriptionString', 'GetLanguageString', 'GetThread', 'GetDebugProperty']
    _com_interfaces_ = [axdebug.IID_IDebugStackFrame]

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def GetCodeContext(self):
        if False:
            return 10
        'Returns the current code context associated with the stack frame.\n\n        Return value must be a IDebugCodeContext object\n        '
        RaiseNotImpl('GetCodeContext')

    def GetDescriptionString(self, fLong):
        if False:
            return 10
        'Returns a textual description of the stack frame.\n\n        fLong -- A flag indicating if the long name is requested.\n        '
        RaiseNotImpl('GetDescriptionString')

    def GetLanguageString(self):
        if False:
            i = 10
            return i + 15
        'Returns a short or long textual description of the language.\n\n        fLong -- A flag indicating if the long name is requested.\n        '
        RaiseNotImpl('GetLanguageString')

    def GetThread(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the thread associated with this stack frame.\n\n        Result must be a IDebugApplicationThread\n        '
        RaiseNotImpl('GetThread')

    def GetDebugProperty(self):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('GetDebugProperty')

class DebugDocumentHost:
    """The interface from the IDebugDocumentHelper back to
    the smart host or language engine.  This interface
    exposes host specific functionality such as syntax coloring.
    """
    _public_methods_ = ['GetDeferredText', 'GetScriptTextAttributes', 'OnCreateDocumentContext', 'GetPathName', 'GetFileName', 'NotifyChanged']
    _com_interfaces_ = [axdebug.IID_IDebugDocumentHost]

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def GetDeferredText(self, dwTextStartCookie, maxChars, bWantAttr):
        if False:
            return 10
        RaiseNotImpl('GetDeferredText')

    def GetScriptTextAttributes(self, codeText, delimterText, flags):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('GetScriptTextAttributes')

    def OnCreateDocumentContext(self):
        if False:
            return 10
        RaiseNotImpl('OnCreateDocumentContext')

    def GetPathName(self):
        if False:
            while True:
                i = 10
        RaiseNotImpl('GetPathName')

    def GetFileName(self):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('GetFileName')

    def NotifyChanged(self):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('NotifyChanged')

class DebugDocumentTextConnectServer:
    _public_methods_ = win32com.server.connect.IConnectionPointContainer_methods + win32com.server.connect.IConnectionPoint_methods
    _com_interfaces_ = [pythoncom.IID_IConnectionPoint, pythoncom.IID_IConnectionPointContainer]

    def __init__(self):
        if False:
            print('Hello World!')
        self.cookieNo = -1
        self.connections = {}

    def EnumConnections(self):
        if False:
            print('Hello World!')
        RaiseNotImpl('EnumConnections')

    def GetConnectionInterface(self):
        if False:
            while True:
                i = 10
        RaiseNotImpl('GetConnectionInterface')

    def GetConnectionPointContainer(self):
        if False:
            print('Hello World!')
        return _wrap(self)

    def Advise(self, pUnk):
        if False:
            for i in range(10):
                print('nop')
        interface = pUnk.QueryInterface(axdebug.IID_IDebugDocumentTextEvents, 1)
        self.cookieNo = self.cookieNo + 1
        self.connections[self.cookieNo] = interface
        return self.cookieNo

    def Unadvise(self, cookie):
        if False:
            i = 10
            return i + 15
        try:
            del self.connections[cookie]
        except KeyError:
            return Exception(scode=winerror.E_UNEXPECTED)

    def EnumConnectionPoints(self):
        if False:
            while True:
                i = 10
        RaiseNotImpl('EnumConnectionPoints')

    def FindConnectionPoint(self, iid):
        if False:
            for i in range(10):
                print('nop')
        if iid == axdebug.IID_IDebugDocumentTextEvents:
            return _wrap(self)
        raise Exception(scode=winerror.E_NOINTERFACE)

class RemoteDebugApplicationEvents:
    _public_methods_ = ['OnConnectDebugger', 'OnDisconnectDebugger', 'OnSetName', 'OnDebugOutput', 'OnClose', 'OnEnterBreakPoint', 'OnLeaveBreakPoint', 'OnCreateThread', 'OnDestroyThread', 'OnBreakFlagChange']
    _com_interfaces_ = [axdebug.IID_IRemoteDebugApplicationEvents]

    def OnConnectDebugger(self, appDebugger):
        if False:
            i = 10
            return i + 15
        'appDebugger -- a PyIApplicationDebugger'
        RaiseNotImpl('OnConnectDebugger')

    def OnDisconnectDebugger(self):
        if False:
            return 10
        RaiseNotImpl('OnDisconnectDebugger')

    def OnSetName(self, name):
        if False:
            print('Hello World!')
        RaiseNotImpl('OnSetName')

    def OnDebugOutput(self, string):
        if False:
            for i in range(10):
                print('nop')
        RaiseNotImpl('OnDebugOutput')

    def OnClose(self):
        if False:
            i = 10
            return i + 15
        RaiseNotImpl('OnClose')

    def OnEnterBreakPoint(self, rdat):
        if False:
            while True:
                i = 10
        'rdat -- PyIRemoteDebugApplicationThread'
        RaiseNotImpl('OnEnterBreakPoint')

    def OnLeaveBreakPoint(self, rdat):
        if False:
            while True:
                i = 10
        'rdat -- PyIRemoteDebugApplicationThread'
        RaiseNotImpl('OnLeaveBreakPoint')

    def OnCreateThread(self, rdat):
        if False:
            print('Hello World!')
        'rdat -- PyIRemoteDebugApplicationThread'
        RaiseNotImpl('OnCreateThread')

    def OnDestroyThread(self, rdat):
        if False:
            print('Hello World!')
        'rdat -- PyIRemoteDebugApplicationThread'
        RaiseNotImpl('OnDestroyThread')

    def OnBreakFlagChange(self, abf, rdat):
        if False:
            print('Hello World!')
        'abf -- int - one of the axdebug.APPBREAKFLAGS constants\n        rdat -- PyIRemoteDebugApplicationThread\n        RaiseNotImpl("OnBreakFlagChange")\n        '

class DebugExpressionContext:
    _public_methods_ = ['ParseLanguageText', 'GetLanguageInfo']
    _com_interfaces_ = [axdebug.IID_IDebugExpressionContext]

    def __init__(self):
        if False:
            return 10
        pass

    def ParseLanguageText(self, code, radix, delim, flags):
        if False:
            while True:
                i = 10
        '\n        result is IDebugExpression\n        '
        RaiseNotImpl('ParseLanguageText')

    def GetLanguageInfo(self):
        if False:
            print('Hello World!')
        '\n        result is (string langName, iid langId)\n        '
        RaiseNotImpl('GetLanguageInfo')

class DebugExpression:
    _public_methods_ = ['Start', 'Abort', 'QueryIsComplete', 'GetResultAsString', 'GetResultAsDebugProperty']
    _com_interfaces_ = [axdebug.IID_IDebugExpression]

    def Start(self, callback):
        if False:
            for i in range(10):
                print('nop')
        '\n        callback -- an IDebugExpressionCallback\n\n        result - void\n        '
        RaiseNotImpl('Start')

    def Abort(self):
        if False:
            return 10
        '\n        no params\n        result -- void\n        '
        RaiseNotImpl('Abort')

    def QueryIsComplete(self):
        if False:
            print('Hello World!')
        '\n        no params\n        result -- void\n        '
        RaiseNotImpl('QueryIsComplete')

    def GetResultAsString(self):
        if False:
            print('Hello World!')
        RaiseNotImpl('GetResultAsString')

    def GetResultAsDebugProperty(self):
        if False:
            while True:
                i = 10
        RaiseNotImpl('GetResultAsDebugProperty')

class ProvideExpressionContexts:
    _public_methods_ = ['EnumExpressionContexts']
    _com_interfaces_ = [axdebug.IID_IProvideExpressionContexts]

    def EnumExpressionContexts(self):
        if False:
            while True:
                i = 10
        RaiseNotImpl('EnumExpressionContexts')