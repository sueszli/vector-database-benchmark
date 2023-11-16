import pywin.scintilla.keycodes
import pywin.scintilla.view
import win32api
import win32con
import win32ui
from pywin.debugger import dbgcon
from pywin.framework.editor import GetEditorOption
from pywin.framework.editor.document import EditorDocumentBase
from pywin.scintilla import bindings, scintillacon
MSG_CHECK_EXTERNAL_FILE = win32con.WM_USER + 1999
MARKER_BOOKMARK = 0
MARKER_BREAKPOINT = 1
MARKER_CURRENT = 2

class SyntEditDocument(EditorDocumentBase):
    """A SyntEdit document."""

    def OnDebuggerStateChange(self, state):
        if False:
            print('Hello World!')
        self._ApplyOptionalToViews('OnDebuggerStateChange', state)

    def HookViewNotifications(self, view):
        if False:
            print('Hello World!')
        EditorDocumentBase.HookViewNotifications(self, view)
        view.SCISetUndoCollection(1)

    def FinalizeViewCreation(self, view):
        if False:
            while True:
                i = 10
        EditorDocumentBase.FinalizeViewCreation(self, view)
        if view == self.GetFirstView():
            self.GetDocTemplate().CheckIDLEMenus(view.idle)
SyntEditViewParent = pywin.scintilla.view.CScintillaView

class SyntEditView(SyntEditViewParent):
    """A view of a SyntEdit.  Obtains data from document."""

    def __init__(self, doc):
        if False:
            i = 10
            return i + 15
        SyntEditViewParent.__init__(self, doc)
        self.bCheckingFile = 0

    def OnInitialUpdate(self):
        if False:
            for i in range(10):
                print('nop')
        SyntEditViewParent.OnInitialUpdate(self)
        self.HookMessage(self.OnRClick, win32con.WM_RBUTTONDOWN)
        for id in (win32ui.ID_VIEW_FOLD_COLLAPSE, win32ui.ID_VIEW_FOLD_COLLAPSE_ALL, win32ui.ID_VIEW_FOLD_EXPAND, win32ui.ID_VIEW_FOLD_EXPAND_ALL):
            self.HookCommand(self.OnCmdViewFold, id)
            self.HookCommandUpdate(self.OnUpdateViewFold, id)
        self.HookCommand(self.OnCmdViewFoldTopLevel, win32ui.ID_VIEW_FOLD_TOPLEVEL)
        self.SCIMarkerDefineAll(MARKER_BOOKMARK, scintillacon.SC_MARK_ROUNDRECT, win32api.RGB(0, 0, 0), win32api.RGB(0, 255, 255))
        self.SCIMarkerDefine(MARKER_CURRENT, scintillacon.SC_MARK_ARROW)
        self.SCIMarkerSetBack(MARKER_CURRENT, win32api.RGB(255, 255, 0))
        if 1:
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDEROPEN, scintillacon.SC_MARK_MINUS, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDER, scintillacon.SC_MARK_PLUS, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDERSUB, scintillacon.SC_MARK_EMPTY, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDERTAIL, scintillacon.SC_MARK_EMPTY, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDEREND, scintillacon.SC_MARK_EMPTY, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDEROPENMID, scintillacon.SC_MARK_EMPTY, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDERMIDTAIL, scintillacon.SC_MARK_EMPTY, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
        else:
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDEROPEN, scintillacon.SC_MARK_CIRCLEMINUS, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDER, scintillacon.SC_MARK_CIRCLEPLUS, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDERSUB, scintillacon.SC_MARK_VLINE, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDERTAIL, scintillacon.SC_MARK_LCORNERCURVE, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDEREND, scintillacon.SC_MARK_CIRCLEPLUSCONNECTED, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDEROPENMID, scintillacon.SC_MARK_CIRCLEMINUSCONNECTED, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
            self.SCIMarkerDefineAll(scintillacon.SC_MARKNUM_FOLDERMIDTAIL, scintillacon.SC_MARK_TCORNERCURVE, win32api.RGB(255, 255, 255), win32api.RGB(0, 0, 0))
        self.SCIMarkerDefine(MARKER_BREAKPOINT, scintillacon.SC_MARK_CIRCLE)
        self.SCIMarkerSetFore(MARKER_BREAKPOINT, win32api.RGB(0, 0, 0))
        try:
            import pywin.debugger
            if pywin.debugger.currentDebugger is None:
                state = dbgcon.DBGSTATE_NOT_DEBUGGING
            else:
                state = pywin.debugger.currentDebugger.debuggerState
        except ImportError:
            state = dbgcon.DBGSTATE_NOT_DEBUGGING
        self.OnDebuggerStateChange(state)

    def _GetSubConfigNames(self):
        if False:
            return 10
        return ['editor']

    def DoConfigChange(self):
        if False:
            return 10
        SyntEditViewParent.DoConfigChange(self)
        tabSize = GetEditorOption('Tab Size', 4, 2)
        indentSize = GetEditorOption('Indent Size', 4, 2)
        bUseTabs = GetEditorOption('Use Tabs', 0)
        bSmartTabs = GetEditorOption('Smart Tabs', 1)
        ext = self.idle.IDLEExtension('AutoIndent')
        self.SCISetViewWS(GetEditorOption('View Whitespace', 0))
        self.SCISetViewEOL(GetEditorOption('View EOL', 0))
        self.SCISetIndentationGuides(GetEditorOption('View Indentation Guides', 0))
        if GetEditorOption('Right Edge Enabled', 0):
            mode = scintillacon.EDGE_BACKGROUND
        else:
            mode = scintillacon.EDGE_NONE
        self.SCISetEdgeMode(mode)
        self.SCISetEdgeColumn(GetEditorOption('Right Edge Column', 75))
        self.SCISetEdgeColor(GetEditorOption('Right Edge Color', win32api.RGB(239, 239, 239)))
        width = GetEditorOption('Marker Margin Width', 16)
        self.SCISetMarginWidthN(1, width)
        width = GetEditorOption('Fold Margin Width', 12)
        self.SCISetMarginWidthN(2, width)
        width = GetEditorOption('Line Number Margin Width', 0)
        self.SCISetMarginWidthN(0, width)
        self.bFolding = GetEditorOption('Enable Folding', 1)
        fold_flags = 0
        self.SendScintilla(scintillacon.SCI_SETMODEVENTMASK, scintillacon.SC_MOD_CHANGEFOLD)
        if self.bFolding:
            if GetEditorOption('Fold Lines', 1):
                fold_flags = 16
        self.SCISetProperty('fold', self.bFolding)
        self.SCISetFoldFlags(fold_flags)
        tt_color = GetEditorOption('Tab Timmy Color', win32api.RGB(255, 0, 0))
        self.SendScintilla(scintillacon.SCI_INDICSETFORE, 1, tt_color)
        tt_use = GetEditorOption('Use Tab Timmy', 1)
        if tt_use:
            self.SCISetProperty('tab.timmy.whinge.level', '1')
        if bSmartTabs:
            ext.config(usetabs=1, tabwidth=5, indentwidth=4)
            ext.set_indentation_params(1)
            if ext.indentwidth == 5:
                usetabs = 1
                indentwidth = tabSize
            elif self.GetTextLength() == 0:
                usetabs = bUseTabs
                indentwidth = indentSize
            else:
                indentwidth = ext.indentwidth
                usetabs = 0
            ext.config(usetabs=usetabs, indentwidth=indentwidth, tabwidth=tabSize)
        else:
            ext.config(usetabs=bUseTabs, tabwidth=tabSize, indentwidth=indentSize)
        self.SCISetIndent(indentSize)
        self.SCISetTabWidth(tabSize)

    def OnDebuggerStateChange(self, state):
        if False:
            i = 10
            return i + 15
        if state == dbgcon.DBGSTATE_NOT_DEBUGGING:
            self.SCIMarkerSetBack(MARKER_BREAKPOINT, win32api.RGB(239, 239, 239))
        else:
            self.SCIMarkerSetBack(MARKER_BREAKPOINT, win32api.RGB(255, 128, 128))

    def HookDocumentHandlers(self):
        if False:
            print('Hello World!')
        SyntEditViewParent.HookDocumentHandlers(self)
        self.HookMessage(self.OnCheckExternalDocumentUpdated, MSG_CHECK_EXTERNAL_FILE)

    def HookHandlers(self):
        if False:
            i = 10
            return i + 15
        SyntEditViewParent.HookHandlers(self)
        self.HookMessage(self.OnSetFocus, win32con.WM_SETFOCUS)

    def _PrepareUserStateChange(self):
        if False:
            return 10
        return (self.GetSel(), self.GetFirstVisibleLine())

    def _EndUserStateChange(self, info):
        if False:
            print('Hello World!')
        scrollOff = info[1] - self.GetFirstVisibleLine()
        if scrollOff:
            self.LineScroll(scrollOff)
        max = self.GetTextLength()
        newPos = (min(info[0][0], max), min(info[0][1], max))
        self.SetSel(newPos)

    def OnMarginClick(self, std, extra):
        if False:
            print('Hello World!')
        notify = self.SCIUnpackNotifyMessage(extra)
        if notify.margin == 2:
            line_click = self.LineFromChar(notify.position)
            if self.SCIGetFoldLevel(line_click) & scintillacon.SC_FOLDLEVELHEADERFLAG:
                self.SCIToggleFold(line_click)
        return 1

    def OnSetFocus(self, msg):
        if False:
            i = 10
            return i + 15
        self.OnCheckExternalDocumentUpdated(msg)
        return 1

    def OnCheckExternalDocumentUpdated(self, msg):
        if False:
            i = 10
            return i + 15
        if self.bCheckingFile:
            return
        self.bCheckingFile = 1
        self.GetDocument().CheckExternalDocumentUpdated()
        self.bCheckingFile = 0

    def OnRClick(self, params):
        if False:
            return 10
        menu = win32ui.CreatePopupMenu()
        self.AppendMenu(menu, '&Locate module', 'LocateModule')
        self.AppendMenu(menu, flags=win32con.MF_SEPARATOR)
        self.AppendMenu(menu, '&Undo', 'EditUndo')
        self.AppendMenu(menu, '&Redo', 'EditRedo')
        self.AppendMenu(menu, flags=win32con.MF_SEPARATOR)
        self.AppendMenu(menu, 'Cu&t', 'EditCut')
        self.AppendMenu(menu, '&Copy', 'EditCopy')
        self.AppendMenu(menu, '&Paste', 'EditPaste')
        self.AppendMenu(menu, flags=win32con.MF_SEPARATOR)
        self.AppendMenu(menu, '&Select all', 'EditSelectAll')
        self.AppendMenu(menu, 'View &Whitespace', 'ViewWhitespace', checked=self.SCIGetViewWS())
        self.AppendMenu(menu, '&Fixed Font', 'ViewFixedFont', checked=self._GetColorizer().bUseFixed)
        self.AppendMenu(menu, flags=win32con.MF_SEPARATOR)
        self.AppendMenu(menu, '&Goto line...', 'GotoLine')
        submenu = win32ui.CreatePopupMenu()
        newitems = self.idle.GetMenuItems('edit')
        for (text, event) in newitems:
            self.AppendMenu(submenu, text, event)
        flags = win32con.MF_STRING | win32con.MF_ENABLED | win32con.MF_POPUP
        menu.AppendMenu(flags, submenu.GetHandle(), '&Source code')
        flags = win32con.TPM_LEFTALIGN | win32con.TPM_LEFTBUTTON | win32con.TPM_RIGHTBUTTON
        menu.TrackPopupMenu(params[5], flags, self)
        return 0

    def OnCmdViewFold(self, cid, code):
        if False:
            while True:
                i = 10
        if cid == win32ui.ID_VIEW_FOLD_EXPAND_ALL:
            self.FoldExpandAllEvent(None)
        elif cid == win32ui.ID_VIEW_FOLD_EXPAND:
            self.FoldExpandEvent(None)
        elif cid == win32ui.ID_VIEW_FOLD_COLLAPSE_ALL:
            self.FoldCollapseAllEvent(None)
        elif cid == win32ui.ID_VIEW_FOLD_COLLAPSE:
            self.FoldCollapseEvent(None)
        else:
            print('Unknown collapse/expand ID')

    def OnUpdateViewFold(self, cmdui):
        if False:
            print('Hello World!')
        if not self.bFolding:
            cmdui.Enable(0)
            return
        id = cmdui.m_nID
        if id in (win32ui.ID_VIEW_FOLD_EXPAND_ALL, win32ui.ID_VIEW_FOLD_COLLAPSE_ALL):
            cmdui.Enable()
        else:
            enable = 0
            lineno = self.LineFromChar(self.GetSel()[0])
            foldable = self.SCIGetFoldLevel(lineno) & scintillacon.SC_FOLDLEVELHEADERFLAG
            is_expanded = self.SCIGetFoldExpanded(lineno)
            if id == win32ui.ID_VIEW_FOLD_EXPAND:
                if foldable and (not is_expanded):
                    enable = 1
            elif id == win32ui.ID_VIEW_FOLD_COLLAPSE:
                if foldable and is_expanded:
                    enable = 1
            cmdui.Enable(enable)

    def OnCmdViewFoldTopLevel(self, cid, code):
        if False:
            return 10
        self.FoldTopLevelEvent(None)

    def ToggleBookmarkEvent(self, event, pos=-1):
        if False:
            while True:
                i = 10
        'Toggle a bookmark at the specified or current position'
        if pos == -1:
            (pos, end) = self.GetSel()
        startLine = self.LineFromChar(pos)
        self.GetDocument().MarkerToggle(startLine + 1, MARKER_BOOKMARK)
        return 0

    def GotoNextBookmarkEvent(self, event, fromPos=-1):
        if False:
            print('Hello World!')
        'Move to the next bookmark'
        if fromPos == -1:
            (fromPos, end) = self.GetSel()
        startLine = self.LineFromChar(fromPos) + 1
        nextLine = self.GetDocument().MarkerGetNext(startLine + 1, MARKER_BOOKMARK) - 1
        if nextLine < 0:
            nextLine = self.GetDocument().MarkerGetNext(0, MARKER_BOOKMARK) - 1
        if nextLine < 0 or nextLine == startLine - 1:
            win32api.MessageBeep()
        else:
            self.SCIEnsureVisible(nextLine)
            self.SCIGotoLine(nextLine)
        return 0

    def TabKeyEvent(self, event):
        if False:
            while True:
                i = 10
        'Insert an indent.  If no selection, a single indent, otherwise a block indent'
        if self.SCIAutoCActive():
            self.SCIAutoCComplete()
            return 0
        return self.bindings.fire('<<smart-indent>>', event)

    def EnterKeyEvent(self, event):
        if False:
            print('Hello World!')
        'Handle the enter key with special handling for auto-complete'
        if self.SCIAutoCActive():
            self.SCIAutoCComplete()
            self.SCIAutoCCancel()
        return self.bindings.fire('<<newline-and-indent>>', event)

    def ShowInteractiveWindowEvent(self, event):
        if False:
            print('Hello World!')
        import pywin.framework.interact
        pywin.framework.interact.ShowInteractiveWindow()

    def FoldTopLevelEvent(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        try:
            self.Colorize()
            maxLine = self.GetLineCount()
            for lineSeek in range(maxLine):
                if self.SCIGetFoldLevel(lineSeek) & scintillacon.SC_FOLDLEVELHEADERFLAG:
                    expanding = not self.SCIGetFoldExpanded(lineSeek)
                    break
            else:
                return
            for lineSeek in range(lineSeek, maxLine):
                level = self.SCIGetFoldLevel(lineSeek)
                level_no = level & scintillacon.SC_FOLDLEVELNUMBERMASK - scintillacon.SC_FOLDLEVELBASE
                is_header = level & scintillacon.SC_FOLDLEVELHEADERFLAG
                if level_no == 0 and is_header:
                    if expanding and (not self.SCIGetFoldExpanded(lineSeek)) or (not expanding and self.SCIGetFoldExpanded(lineSeek)):
                        self.SCIToggleFold(lineSeek)
        finally:
            win32ui.DoWaitCursor(-1)

    def FoldExpandSecondLevelEvent(self, event):
        if False:
            print('Hello World!')
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        self.Colorize()
        levels = [scintillacon.SC_FOLDLEVELBASE]
        for lineno in range(self.GetLineCount()):
            level = self.SCIGetFoldLevel(lineno)
            if not level & scintillacon.SC_FOLDLEVELHEADERFLAG:
                continue
            curr_level = level & scintillacon.SC_FOLDLEVELNUMBERMASK
            if curr_level > levels[-1]:
                levels.append(curr_level)
            try:
                level_ind = levels.index(curr_level)
            except ValueError:
                break
            levels = levels[:level_ind + 1]
            if level_ind == 1 and (not self.SCIGetFoldExpanded(lineno)):
                self.SCIToggleFold(lineno)
        win32ui.DoWaitCursor(-1)

    def FoldCollapseSecondLevelEvent(self, event):
        if False:
            i = 10
            return i + 15
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        self.Colorize()
        levels = [scintillacon.SC_FOLDLEVELBASE]
        for lineno in range(self.GetLineCount()):
            level = self.SCIGetFoldLevel(lineno)
            if not level & scintillacon.SC_FOLDLEVELHEADERFLAG:
                continue
            curr_level = level & scintillacon.SC_FOLDLEVELNUMBERMASK
            if curr_level > levels[-1]:
                levels.append(curr_level)
            try:
                level_ind = levels.index(curr_level)
            except ValueError:
                break
            levels = levels[:level_ind + 1]
            if level_ind == 1 and self.SCIGetFoldExpanded(lineno):
                self.SCIToggleFold(lineno)
        win32ui.DoWaitCursor(-1)

    def FoldExpandEvent(self, event):
        if False:
            while True:
                i = 10
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        lineno = self.LineFromChar(self.GetSel()[0])
        if self.SCIGetFoldLevel(lineno) & scintillacon.SC_FOLDLEVELHEADERFLAG and (not self.SCIGetFoldExpanded(lineno)):
            self.SCIToggleFold(lineno)
        win32ui.DoWaitCursor(-1)

    def FoldExpandAllEvent(self, event):
        if False:
            return 10
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        for lineno in range(0, self.GetLineCount()):
            if self.SCIGetFoldLevel(lineno) & scintillacon.SC_FOLDLEVELHEADERFLAG and (not self.SCIGetFoldExpanded(lineno)):
                self.SCIToggleFold(lineno)
        win32ui.DoWaitCursor(-1)

    def FoldCollapseEvent(self, event):
        if False:
            return 10
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        lineno = self.LineFromChar(self.GetSel()[0])
        if self.SCIGetFoldLevel(lineno) & scintillacon.SC_FOLDLEVELHEADERFLAG and self.SCIGetFoldExpanded(lineno):
            self.SCIToggleFold(lineno)
        win32ui.DoWaitCursor(-1)

    def FoldCollapseAllEvent(self, event):
        if False:
            print('Hello World!')
        if not self.bFolding:
            return 1
        win32ui.DoWaitCursor(1)
        self.Colorize()
        for lineno in range(0, self.GetLineCount()):
            if self.SCIGetFoldLevel(lineno) & scintillacon.SC_FOLDLEVELHEADERFLAG and self.SCIGetFoldExpanded(lineno):
                self.SCIToggleFold(lineno)
        win32ui.DoWaitCursor(-1)
from pywin.framework.editor.frame import EditorFrame

class SplitterFrame(EditorFrame):

    def OnCreate(self, cs):
        if False:
            for i in range(10):
                print('nop')
        self.HookCommand(self.OnWindowSplit, win32ui.ID_WINDOW_SPLIT)
        return 1

    def OnWindowSplit(self, id, code):
        if False:
            i = 10
            return i + 15
        self.GetDlgItem(win32ui.AFX_IDW_PANE_FIRST).DoKeyboardSplit()
        return 1
from pywin.framework.editor.template import EditorTemplateBase

class SyntEditTemplate(EditorTemplateBase):

    def __init__(self, res=win32ui.IDR_TEXTTYPE, makeDoc=None, makeFrame=None, makeView=None):
        if False:
            return 10
        if makeDoc is None:
            makeDoc = SyntEditDocument
        if makeView is None:
            makeView = SyntEditView
        if makeFrame is None:
            makeFrame = SplitterFrame
        self.bSetMenus = 0
        EditorTemplateBase.__init__(self, res, makeDoc, makeFrame, makeView)

    def CheckIDLEMenus(self, idle):
        if False:
            while True:
                i = 10
        if self.bSetMenus:
            return
        self.bSetMenus = 1
        submenu = win32ui.CreatePopupMenu()
        newitems = idle.GetMenuItems('edit')
        flags = win32con.MF_STRING | win32con.MF_ENABLED
        for (text, event) in newitems:
            id = bindings.event_to_commands.get(event)
            if id is not None:
                keyname = pywin.scintilla.view.configManager.get_key_binding(event, ['editor'])
                if keyname is not None:
                    text = text + '\t' + keyname
                submenu.AppendMenu(flags, id, text)
        mainMenu = self.GetSharedMenu()
        editMenu = mainMenu.GetSubMenu(1)
        editMenu.AppendMenu(win32con.MF_SEPARATOR, 0, '')
        editMenu.AppendMenu(win32con.MF_STRING | win32con.MF_POPUP | win32con.MF_ENABLED, submenu.GetHandle(), '&Source Code')

    def _CreateDocTemplate(self, resourceId):
        if False:
            while True:
                i = 10
        return win32ui.CreateDocTemplate(resourceId)

    def CreateWin32uiDocument(self):
        if False:
            for i in range(10):
                print('nop')
        return self.DoCreateDoc()

    def GetPythonPropertyPages(self):
        if False:
            while True:
                i = 10
        'Returns a list of property pages'
        from pywin.scintilla import configui
        return EditorTemplateBase.GetPythonPropertyPages(self) + [configui.ScintillaFormatPropertyPage()]
try:
    win32ui.GetApp().RemoveDocTemplate(editorTemplate)
except NameError:
    pass
editorTemplate = SyntEditTemplate()
win32ui.GetApp().AddDocTemplate(editorTemplate)