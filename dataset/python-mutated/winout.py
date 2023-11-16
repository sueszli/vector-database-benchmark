import queue
import re
import pywin.scintilla.document
import win32api
import win32con
import win32ui
from pywin.framework import app, window
from pywin.mfc import docview
from pywin.scintilla import scintillacon
debug = lambda msg: None
WindowOutputDocumentParent = pywin.scintilla.document.CScintillaDocument

class flags:
    WQ_NONE = 0
    WQ_LINE = 1
    WQ_IDLE = 2

class WindowOutputDocument(WindowOutputDocumentParent):

    def SaveModified(self):
        if False:
            i = 10
            return i + 15
        return 1

    def OnSaveDocument(self, fileName):
        if False:
            i = 10
            return i + 15
        win32ui.SetStatusText('Saving file...', 1)
        try:
            self.SaveFile(fileName)
        except OSError as details:
            win32ui.MessageBox('Error - could not save file\r\n\r\n%s' % details)
            return 0
        win32ui.SetStatusText('Ready')
        return 1

class WindowOutputFrame(window.MDIChildWnd):

    def __init__(self, wnd=None):
        if False:
            for i in range(10):
                print('nop')
        window.MDIChildWnd.__init__(self, wnd)
        self.HookMessage(self.OnSizeMove, win32con.WM_SIZE)
        self.HookMessage(self.OnSizeMove, win32con.WM_MOVE)

    def LoadFrame(self, idResource, style, wndParent, context):
        if False:
            return 10
        self.template = context.template
        return self._obj_.LoadFrame(idResource, style, wndParent, context)

    def PreCreateWindow(self, cc):
        if False:
            print('Hello World!')
        cc = self._obj_.PreCreateWindow(cc)
        if self.template.defSize and self.template.defSize[0] != self.template.defSize[1]:
            rect = app.RectToCreateStructRect(self.template.defSize)
            cc = (cc[0], cc[1], cc[2], cc[3], rect, cc[5], cc[6], cc[7], cc[8])
        return cc

    def OnSizeMove(self, msg):
        if False:
            for i in range(10):
                print('nop')
        mdiClient = self.GetParent()
        self.template.defSize = mdiClient.ScreenToClient(self.GetWindowRect())

    def OnDestroy(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.template.OnFrameDestroy(self)
        return 1

class WindowOutputViewImpl:

    def __init__(self):
        if False:
            print('Hello World!')
        self.patErrorMessage = re.compile('\\W*File "(.*)", line ([0-9]+)')
        self.template = self.GetDocument().GetDocTemplate()

    def HookHandlers(self):
        if False:
            return 10
        self.HookMessage(self.OnRClick, win32con.WM_RBUTTONDOWN)

    def OnDestroy(self, msg):
        if False:
            while True:
                i = 10
        self.template.OnViewDestroy(self)

    def OnInitialUpdate(self):
        if False:
            while True:
                i = 10
        self.RestoreKillBuffer()
        self.SetSel(-2)

    def GetRightMenuItems(self):
        if False:
            print('Hello World!')
        ret = []
        flags = win32con.MF_STRING | win32con.MF_ENABLED
        ret.append((flags, win32ui.ID_EDIT_COPY, '&Copy'))
        ret.append((flags, win32ui.ID_EDIT_SELECT_ALL, '&Select all'))
        return ret

    def OnRClick(self, params):
        if False:
            print('Hello World!')
        paramsList = self.GetRightMenuItems()
        menu = win32ui.CreatePopupMenu()
        for appendParams in paramsList:
            if not isinstance(appendParams, tuple):
                appendParams = (appendParams,)
            menu.AppendMenu(*appendParams)
        menu.TrackPopupMenu(params[5])
        return 0

    def HandleSpecialLine(self):
        if False:
            return 10
        from . import scriptutils
        line = self.GetLine()
        if line[:11] == 'com_error: ':
            try:
                import win32api
                import win32con
                det = eval(line[line.find(':') + 1:].strip())
                win32ui.SetStatusText('Opening help file on OLE error...')
                from . import help
                help.OpenHelpFile(det[2][3], win32con.HELP_CONTEXT, det[2][4])
                return 1
            except win32api.error as details:
                win32ui.SetStatusText('The help file could not be opened - %s' % details.strerror)
                return 1
            except:
                win32ui.SetStatusText('Line is a COM error, but no WinHelp details can be parsed')
        matchResult = self.patErrorMessage.match(line)
        if matchResult is None:
            lineNo = self.LineFromChar()
            if lineNo > 0:
                line = self.GetLine(lineNo - 1)
                matchResult = self.patErrorMessage.match(line)
        if matchResult is not None:
            fileName = matchResult.group(1)
            if fileName[0] == '<':
                win32ui.SetStatusText('Can not load this file')
                return 1
            else:
                lineNoString = matchResult.group(2)
                fileNameSpec = fileName
                fileName = scriptutils.LocatePythonFile(fileName)
                if fileName is None:
                    win32ui.SetStatusText("Cant locate the file '%s'" % fileNameSpec, 0)
                    return 1
                win32ui.SetStatusText('Jumping to line ' + lineNoString + ' of file ' + fileName, 1)
                if not scriptutils.JumpToDocument(fileName, int(lineNoString)):
                    win32ui.SetStatusText('Could not open %s' % fileName)
                    return 1
                return 1
        return 0

    def write(self, msg):
        if False:
            return 10
        return self.template.write(msg)

    def writelines(self, lines):
        if False:
            while True:
                i = 10
        for line in lines:
            self.write(line)

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        self.template.flush()

class WindowOutputViewRTF(docview.RichEditView, WindowOutputViewImpl):

    def __init__(self, doc):
        if False:
            for i in range(10):
                print('nop')
        docview.RichEditView.__init__(self, doc)
        WindowOutputViewImpl.__init__(self)

    def OnInitialUpdate(self):
        if False:
            while True:
                i = 10
        WindowOutputViewImpl.OnInitialUpdate(self)
        return docview.RichEditView.OnInitialUpdate(self)

    def OnDestroy(self, msg):
        if False:
            while True:
                i = 10
        WindowOutputViewImpl.OnDestroy(self, msg)
        docview.RichEditView.OnDestroy(self, msg)

    def HookHandlers(self):
        if False:
            print('Hello World!')
        WindowOutputViewImpl.HookHandlers(self)
        self.HookMessage(self.OnLDoubleClick, win32con.WM_LBUTTONDBLCLK)

    def OnLDoubleClick(self, params):
        if False:
            i = 10
            return i + 15
        if self.HandleSpecialLine():
            return 0
        return 1

    def RestoreKillBuffer(self):
        if False:
            while True:
                i = 10
        if len(self.template.killBuffer):
            self.StreamIn(win32con.SF_RTF, self._StreamRTFIn)
            self.template.killBuffer = []

    def SaveKillBuffer(self):
        if False:
            for i in range(10):
                print('nop')
        self.StreamOut(win32con.SF_RTFNOOBJS, self._StreamRTFOut)

    def _StreamRTFOut(self, data):
        if False:
            i = 10
            return i + 15
        self.template.killBuffer.append(data)
        return 1

    def _StreamRTFIn(self, bytes):
        if False:
            while True:
                i = 10
        try:
            item = self.template.killBuffer[0]
            self.template.killBuffer.remove(item)
            if bytes < len(item):
                print('Warning - output buffer not big enough!')
            return item
        except IndexError:
            return None

    def dowrite(self, str):
        if False:
            print('Hello World!')
        self.SetSel(-2)
        self.ReplaceSel(str)
import pywin.scintilla.view

class WindowOutputViewScintilla(pywin.scintilla.view.CScintillaView, WindowOutputViewImpl):

    def __init__(self, doc):
        if False:
            i = 10
            return i + 15
        pywin.scintilla.view.CScintillaView.__init__(self, doc)
        WindowOutputViewImpl.__init__(self)

    def OnInitialUpdate(self):
        if False:
            return 10
        pywin.scintilla.view.CScintillaView.OnInitialUpdate(self)
        self.SCISetMarginWidth(3)
        WindowOutputViewImpl.OnInitialUpdate(self)

    def OnDestroy(self, msg):
        if False:
            while True:
                i = 10
        WindowOutputViewImpl.OnDestroy(self, msg)
        pywin.scintilla.view.CScintillaView.OnDestroy(self, msg)

    def HookHandlers(self):
        if False:
            i = 10
            return i + 15
        WindowOutputViewImpl.HookHandlers(self)
        pywin.scintilla.view.CScintillaView.HookHandlers(self)
        self.GetParent().HookNotify(self.OnScintillaDoubleClick, scintillacon.SCN_DOUBLECLICK)

    def OnScintillaDoubleClick(self, std, extra):
        if False:
            while True:
                i = 10
        self.HandleSpecialLine()

    def RestoreKillBuffer(self):
        if False:
            print('Hello World!')
        assert len(self.template.killBuffer) in (0, 1), 'Unexpected killbuffer contents'
        if self.template.killBuffer:
            self.SCIAddText(self.template.killBuffer[0])
        self.template.killBuffer = []

    def SaveKillBuffer(self):
        if False:
            i = 10
            return i + 15
        self.template.killBuffer = [self.GetTextRange(0, -1)]

    def dowrite(self, str):
        if False:
            print('Hello World!')
        end = self.GetTextLength()
        atEnd = end == self.GetSel()[0]
        self.SCIInsertText(str, end)
        if atEnd:
            self.SetSel(self.GetTextLength())

    def SetWordWrap(self, bWrapOn=1):
        if False:
            return 10
        if bWrapOn:
            wrap_mode = scintillacon.SC_WRAP_WORD
        else:
            wrap_mode = scintillacon.SC_WRAP_NONE
        self.SCISetWrapMode(wrap_mode)

    def _MakeColorizer(self):
        if False:
            while True:
                i = 10
        return None
WindowOutputView = WindowOutputViewScintilla

class WindowOutput(docview.DocTemplate):
    """Looks like a general Output Window - text can be written by the 'write' method.
    Will auto-create itself on first write, and also on next write after being closed"""
    softspace = 1

    def __init__(self, title=None, defSize=None, queueing=flags.WQ_LINE, bAutoRestore=1, style=None, makeDoc=None, makeFrame=None, makeView=None):
        if False:
            for i in range(10):
                print('nop')
        'init the output window -\n        Params\n        title=None -- What is the title of the window\n        defSize=None -- What is the default size for the window - if this\n                        is a string, the size will be loaded from the ini file.\n        queueing = flags.WQ_LINE -- When should output be written\n        bAutoRestore=1 -- Should a minimized window be restored.\n        style -- Style for Window, or None for default.\n        makeDoc, makeFrame, makeView -- Classes for frame, view and window respectively.\n        '
        if makeDoc is None:
            makeDoc = WindowOutputDocument
        if makeFrame is None:
            makeFrame = WindowOutputFrame
        if makeView is None:
            makeView = WindowOutputViewScintilla
        docview.DocTemplate.__init__(self, win32ui.IDR_PYTHONTYPE, makeDoc, makeFrame, makeView)
        self.SetDocStrings('\nOutput\n\nText Documents (*.txt)\n.txt\n\n\n')
        win32ui.GetApp().AddDocTemplate(self)
        self.writeQueueing = queueing
        self.errorCantRecreate = 0
        self.killBuffer = []
        self.style = style
        self.bAutoRestore = bAutoRestore
        self.title = title
        self.bCreating = 0
        self.interruptCount = 0
        if isinstance(defSize, str):
            self.iniSizeSection = defSize
            self.defSize = app.LoadWindowSize(defSize)
            self.loadedSize = self.defSize
        else:
            self.iniSizeSection = None
            self.defSize = defSize
        self.currentView = None
        self.outputQueue = queue.Queue(-1)
        self.mainThreadId = win32api.GetCurrentThreadId()
        self.idleHandlerSet = 0
        self.SetIdleHandler()

    def __del__(self):
        if False:
            while True:
                i = 10
        self.Close()

    def Create(self, title=None, style=None):
        if False:
            print('Hello World!')
        self.bCreating = 1
        if title:
            self.title = title
        if style:
            self.style = style
        doc = self.OpenDocumentFile()
        if doc is None:
            return
        self.currentView = doc.GetFirstView()
        self.bCreating = 0
        if self.title:
            doc.SetTitle(self.title)

    def Close(self):
        if False:
            while True:
                i = 10
        self.RemoveIdleHandler()
        try:
            parent = self.currentView.GetParent()
        except (AttributeError, win32ui.error):
            return
        parent.DestroyWindow()

    def SetTitle(self, title):
        if False:
            return 10
        self.title = title
        if self.currentView:
            self.currentView.GetDocument().SetTitle(self.title)

    def OnViewDestroy(self, view):
        if False:
            print('Hello World!')
        self.currentView.SaveKillBuffer()
        self.currentView = None

    def OnFrameDestroy(self, frame):
        if False:
            i = 10
            return i + 15
        if self.iniSizeSection:
            newSize = frame.GetWindowPlacement()[4]
            if self.loadedSize != newSize:
                app.SaveWindowSize(self.iniSizeSection, newSize)

    def SetIdleHandler(self):
        if False:
            while True:
                i = 10
        if not self.idleHandlerSet:
            debug('Idle handler set\n')
            win32ui.GetApp().AddIdleHandler(self.QueueIdleHandler)
            self.idleHandlerSet = 1

    def RemoveIdleHandler(self):
        if False:
            i = 10
            return i + 15
        if self.idleHandlerSet:
            debug('Idle handler reset\n')
            if win32ui.GetApp().DeleteIdleHandler(self.QueueIdleHandler) == 0:
                debug('Error deleting idle handler\n')
            self.idleHandlerSet = 0

    def RecreateWindow(self):
        if False:
            i = 10
            return i + 15
        if self.errorCantRecreate:
            debug('Error = not trying again')
            return 0
        try:
            win32ui.GetMainFrame().GetSafeHwnd()
            self.Create()
            return 1
        except (win32ui.error, AttributeError):
            self.errorCantRecreate = 1
            debug('Winout can not recreate the Window!\n')
            return 0

    def QueueIdleHandler(self, handler, count):
        if False:
            i = 10
            return i + 15
        try:
            bEmpty = self.QueueFlush(20)
            if bEmpty:
                self.interruptCount = 0
        except KeyboardInterrupt:
            self.interruptCount = self.interruptCount + 1
            if self.interruptCount > 1:
                self.outputQueue = queue.Queue(-1)
                print('Interrupted.')
                bEmpty = 1
            else:
                raise
        return not bEmpty

    def NeedRecreateWindow(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            if self.currentView is not None and self.currentView.IsWindow():
                return 0
        except (win32ui.error, AttributeError):
            pass
        return 1

    def CheckRecreateWindow(self):
        if False:
            i = 10
            return i + 15
        if self.bCreating:
            return 1
        if not self.NeedRecreateWindow():
            return 1
        if self.bAutoRestore:
            if self.RecreateWindow():
                return 1
        return 0

    def QueueFlush(self, max=None):
        if False:
            return 10
        if self.bCreating:
            return 1
        items = []
        rc = 0
        while max is None or max > 0:
            try:
                item = self.outputQueue.get_nowait()
                items.append(item)
            except queue.Empty:
                rc = 1
                break
            if max is not None:
                max = max - 1
        if len(items) != 0:
            if not self.CheckRecreateWindow():
                debug(':Recreate failed!\n')
                return 1
            win32ui.PumpWaitingMessages()
            self.currentView.dowrite(''.join(items))
        return rc

    def HandleOutput(self, message):
        if False:
            while True:
                i = 10
        self.outputQueue.put(message)
        if win32api.GetCurrentThreadId() != self.mainThreadId:
            pass
        elif self.writeQueueing == flags.WQ_LINE:
            pos = message.rfind('\n')
            if pos >= 0:
                self.QueueFlush()
                return
        elif self.writeQueueing == flags.WQ_NONE:
            self.QueueFlush()
            return
        try:
            win32ui.GetMainFrame().PostMessage(win32con.WM_USER)
        except win32ui.error:
            win32api.OutputDebugString(message)

    def writelines(self, lines):
        if False:
            i = 10
            return i + 15
        for line in lines:
            self.write(line)

    def write(self, message):
        if False:
            return 10
        self.HandleOutput(message)

    def flush(self):
        if False:
            while True:
                i = 10
        self.QueueFlush()

    def HandleSpecialLine(self):
        if False:
            for i in range(10):
                print('nop')
        self.currentView.HandleSpecialLine()

def RTFWindowOutput(*args, **kw):
    if False:
        i = 10
        return i + 15
    kw['makeView'] = WindowOutputViewRTF
    return WindowOutput(*args, **kw)

def thread_test(o):
    if False:
        return 10
    for i in range(5):
        o.write('Hi from thread %d\n' % win32api.GetCurrentThreadId())
        win32api.Sleep(100)

def test():
    if False:
        for i in range(10):
            print('nop')
    w = WindowOutput(queueing=flags.WQ_IDLE)
    w.write('First bit of text\n')
    import _thread
    for i in range(5):
        w.write('Hello from the main thread\n')
        _thread.start_new(thread_test, (w,))
    for i in range(2):
        w.write('Hello from the main thread\n')
        win32api.Sleep(50)
    return w
if __name__ == '__main__':
    test()