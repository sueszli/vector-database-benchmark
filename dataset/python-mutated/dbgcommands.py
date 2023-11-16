import warnings
import win32ui
from pywin.scintilla.control import CScintillaEditInterface
from . import scriptutils
IdToBarNames = {win32ui.IDC_DBG_STACK: ('Stack', 0), win32ui.IDC_DBG_BREAKPOINTS: ('Breakpoints', 0), win32ui.IDC_DBG_WATCH: ('Watch', 1)}

class DebuggerCommandHandler:

    def HookCommands(self):
        if False:
            return 10
        commands = ((self.OnStep, None, win32ui.IDC_DBG_STEP), (self.OnStepOut, self.OnUpdateOnlyBreak, win32ui.IDC_DBG_STEPOUT), (self.OnStepOver, None, win32ui.IDC_DBG_STEPOVER), (self.OnGo, None, win32ui.IDC_DBG_GO), (self.OnClose, self.OnUpdateClose, win32ui.IDC_DBG_CLOSE), (self.OnAdd, self.OnUpdateAddBreakpoints, win32ui.IDC_DBG_ADD), (self.OnClearAll, self.OnUpdateClearAllBreakpoints, win32ui.IDC_DBG_CLEAR))
        frame = win32ui.GetMainFrame()
        for (methHandler, methUpdate, id) in commands:
            frame.HookCommand(methHandler, id)
            if not methUpdate is None:
                frame.HookCommandUpdate(methUpdate, id)
        for id in list(IdToBarNames.keys()):
            frame.HookCommand(self.OnDebuggerBar, id)
            frame.HookCommandUpdate(self.OnUpdateDebuggerBar, id)

    def OnDebuggerToolbar(self, id, code):
        if False:
            return 10
        if code == 0:
            return not win32ui.GetMainFrame().OnBarCheck(id)

    def OnUpdateDebuggerToolbar(self, cmdui):
        if False:
            return 10
        win32ui.GetMainFrame().OnUpdateControlBarMenu(cmdui)
        cmdui.Enable(1)

    def _GetDebugger(self):
        if False:
            i = 10
            return i + 15
        try:
            import pywin.debugger
            return pywin.debugger.currentDebugger
        except ImportError:
            return None

    def _DoOrStart(self, doMethod, startFlag):
        if False:
            i = 10
            return i + 15
        d = self._GetDebugger()
        if d is not None and d.IsDebugging():
            method = getattr(d, doMethod)
            method()
        else:
            scriptutils.RunScript(defName=None, defArgs=None, bShowDialog=0, debuggingType=startFlag)

    def OnStep(self, msg, code):
        if False:
            print('Hello World!')
        self._DoOrStart('do_set_step', scriptutils.RS_DEBUGGER_STEP)

    def OnStepOver(self, msg, code):
        if False:
            i = 10
            return i + 15
        self._DoOrStart('do_set_next', scriptutils.RS_DEBUGGER_STEP)

    def OnStepOut(self, msg, code):
        if False:
            print('Hello World!')
        d = self._GetDebugger()
        if d is not None and d.IsDebugging():
            d.do_set_return()

    def OnGo(self, msg, code):
        if False:
            for i in range(10):
                print('nop')
        self._DoOrStart('do_set_continue', scriptutils.RS_DEBUGGER_GO)

    def OnClose(self, msg, code):
        if False:
            while True:
                i = 10
        d = self._GetDebugger()
        if d is not None:
            if d.IsDebugging():
                d.set_quit()
            else:
                d.close()

    def OnUpdateClose(self, cmdui):
        if False:
            while True:
                i = 10
        d = self._GetDebugger()
        if d is not None and d.inited:
            cmdui.Enable(1)
        else:
            cmdui.Enable(0)

    def OnAdd(self, msg, code):
        if False:
            print('Hello World!')
        (doc, view) = scriptutils.GetActiveEditorDocument()
        if doc is None:
            warnings.warn('There is no active window - no breakpoint can be added')
            return None
        pathName = doc.GetPathName()
        lineNo = view.LineFromChar(view.GetSel()[0]) + 1
        d = self._GetDebugger()
        if d is None:
            import pywin.framework.editor.color.coloreditor
            doc.MarkerToggle(lineNo, pywin.framework.editor.color.coloreditor.MARKER_BREAKPOINT)
        else:
            if d.get_break(pathName, lineNo):
                win32ui.SetStatusText('Clearing breakpoint', 1)
                rc = d.clear_break(pathName, lineNo)
            else:
                win32ui.SetStatusText('Setting breakpoint', 1)
                rc = d.set_break(pathName, lineNo)
            if rc:
                win32ui.MessageBox(rc)
            d.GUIRespondDebuggerData()

    def OnClearAll(self, msg, code):
        if False:
            print('Hello World!')
        win32ui.SetStatusText('Clearing all breakpoints')
        d = self._GetDebugger()
        if d is None:
            import pywin.framework.editor
            import pywin.framework.editor.color.coloreditor
            for doc in pywin.framework.editor.editorTemplate.GetDocumentList():
                doc.MarkerDeleteAll(pywin.framework.editor.color.coloreditor.MARKER_BREAKPOINT)
        else:
            d.clear_all_breaks()
            d.UpdateAllLineStates()
            d.GUIRespondDebuggerData()

    def OnUpdateOnlyBreak(self, cmdui):
        if False:
            for i in range(10):
                print('nop')
        d = self._GetDebugger()
        ok = d is not None and d.IsBreak()
        cmdui.Enable(ok)

    def OnUpdateAddBreakpoints(self, cmdui):
        if False:
            return 10
        (doc, view) = scriptutils.GetActiveEditorDocument()
        if doc is None or not isinstance(view, CScintillaEditInterface):
            enabled = 0
        else:
            enabled = 1
            lineNo = view.LineFromChar(view.GetSel()[0]) + 1
            import pywin.framework.editor.color.coloreditor
            cmdui.SetCheck(doc.MarkerAtLine(lineNo, pywin.framework.editor.color.coloreditor.MARKER_BREAKPOINT) != 0)
        cmdui.Enable(enabled)

    def OnUpdateClearAllBreakpoints(self, cmdui):
        if False:
            while True:
                i = 10
        d = self._GetDebugger()
        cmdui.Enable(d is None or len(d.breaks) != 0)

    def OnUpdateDebuggerBar(self, cmdui):
        if False:
            i = 10
            return i + 15
        (name, always) = IdToBarNames.get(cmdui.m_nID)
        enabled = always
        d = self._GetDebugger()
        if d is not None and d.IsDebugging() and (name is not None):
            enabled = 1
            bar = d.GetDebuggerBar(name)
            cmdui.SetCheck(bar.IsWindowVisible())
        cmdui.Enable(enabled)

    def OnDebuggerBar(self, id, code):
        if False:
            print('Hello World!')
        name = IdToBarNames.get(id)[0]
        d = self._GetDebugger()
        if d is not None and name is not None:
            bar = d.GetDebuggerBar(name)
            newState = not bar.IsWindowVisible()
            win32ui.GetMainFrame().ShowControlBar(bar, newState, 1)