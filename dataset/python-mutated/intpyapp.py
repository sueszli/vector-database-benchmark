import os
import sys
import traceback
import __main__
import commctrl
import win32api
import win32con
import win32ui
from pywin.mfc import afxres, dialog
from . import app, dbgcommands
lastLocateFileName = '.py'

def _SetupSharedMenu_(self):
    if False:
        for i in range(10):
            print('nop')
    sharedMenu = self.GetSharedMenu()
    from pywin.framework import toolmenu
    toolmenu.SetToolsMenu(sharedMenu)
    from pywin.framework import help
    help.SetHelpMenuOtherHelp(sharedMenu)
from pywin.mfc import docview
docview.DocTemplate._SetupSharedMenu_ = _SetupSharedMenu_

class MainFrame(app.MainFrame):

    def OnCreate(self, createStruct):
        if False:
            return 10
        self.closing = 0
        if app.MainFrame.OnCreate(self, createStruct) == -1:
            return -1
        style = win32con.WS_CHILD | afxres.CBRS_SIZE_DYNAMIC | afxres.CBRS_TOP | afxres.CBRS_TOOLTIPS | afxres.CBRS_FLYBY
        self.EnableDocking(afxres.CBRS_ALIGN_ANY)
        tb = win32ui.CreateToolBar(self, style | win32con.WS_VISIBLE)
        tb.ModifyStyle(0, commctrl.TBSTYLE_FLAT)
        tb.LoadToolBar(win32ui.IDR_MAINFRAME)
        tb.EnableDocking(afxres.CBRS_ALIGN_ANY)
        tb.SetWindowText('Standard')
        self.DockControlBar(tb)
        from pywin.debugger.debugger import PrepareControlBars
        PrepareControlBars(self)
        menu = self.GetMenu()
        from . import toolmenu
        toolmenu.SetToolsMenu(menu, 2)
        from pywin.framework import help
        help.SetHelpMenuOtherHelp(menu)

    def OnClose(self):
        if False:
            while True:
                i = 10
        try:
            import pywin.debugger
            if pywin.debugger.currentDebugger is not None and pywin.debugger.currentDebugger.pumping:
                try:
                    pywin.debugger.currentDebugger.close(1)
                except:
                    traceback.print_exc()
                return
        except win32ui.error:
            pass
        self.closing = 1
        self.SaveBarState('ToolbarDefault')
        self.SetActiveView(None)
        from pywin.framework import help
        help.FinalizeHelp()
        self.DestroyControlBar(afxres.AFX_IDW_TOOLBAR)
        self.DestroyControlBar(win32ui.ID_VIEW_TOOLBAR_DBG)
        return self._obj_.OnClose()

    def DestroyControlBar(self, id):
        if False:
            print('Hello World!')
        try:
            bar = self.GetControlBar(id)
        except win32ui.error:
            return
        bar.DestroyWindow()

    def OnCommand(self, wparam, lparam):
        if False:
            print('Hello World!')
        try:
            v = self.GetActiveView()
            if v.OnCommand(wparam, lparam):
                return 1
        except (win32ui.error, AttributeError):
            pass
        return self._obj_.OnCommand(wparam, lparam)

class InteractivePythonApp(app.CApp):

    def HookCommands(self):
        if False:
            i = 10
            return i + 15
        app.CApp.HookCommands(self)
        dbgcommands.DebuggerCommandHandler().HookCommands()
        self.HookCommand(self.OnViewBrowse, win32ui.ID_VIEW_BROWSE)
        self.HookCommand(self.OnFileImport, win32ui.ID_FILE_IMPORT)
        self.HookCommand(self.OnFileCheck, win32ui.ID_FILE_CHECK)
        self.HookCommandUpdate(self.OnUpdateFileCheck, win32ui.ID_FILE_CHECK)
        self.HookCommand(self.OnFileRun, win32ui.ID_FILE_RUN)
        self.HookCommand(self.OnFileLocate, win32ui.ID_FILE_LOCATE)
        self.HookCommand(self.OnInteractiveWindow, win32ui.ID_VIEW_INTERACTIVE)
        self.HookCommandUpdate(self.OnUpdateInteractiveWindow, win32ui.ID_VIEW_INTERACTIVE)
        self.HookCommand(self.OnViewOptions, win32ui.ID_VIEW_OPTIONS)
        self.HookCommand(self.OnHelpIndex, afxres.ID_HELP_INDEX)
        self.HookCommand(self.OnFileSaveAll, win32ui.ID_FILE_SAVE_ALL)
        self.HookCommand(self.OnViewToolbarDbg, win32ui.ID_VIEW_TOOLBAR_DBG)
        self.HookCommandUpdate(self.OnUpdateViewToolbarDbg, win32ui.ID_VIEW_TOOLBAR_DBG)

    def CreateMainFrame(self):
        if False:
            print('Hello World!')
        return MainFrame()

    def MakeExistingDDEConnection(self):
        if False:
            return 10
        try:
            from . import intpydde
        except ImportError:
            return None
        conv = intpydde.CreateConversation(self.ddeServer)
        try:
            conv.ConnectTo('Pythonwin', 'System')
            return conv
        except intpydde.error:
            return None

    def InitDDE(self):
        if False:
            return 10
        try:
            from . import intpydde
        except ImportError:
            self.ddeServer = None
            intpydde = None
        if intpydde is not None:
            self.ddeServer = intpydde.DDEServer(self)
            self.ddeServer.Create('Pythonwin', intpydde.CBF_FAIL_SELFCONNECTIONS)
            try:
                connection = self.MakeExistingDDEConnection()
                if connection is not None:
                    connection.Exec('self.Activate()')
                    if self.ProcessArgs(sys.argv, connection) is None:
                        return 1
            except:
                win32ui.DisplayTraceback(sys.exc_info(), ' - error in DDE conversation with Pythonwin')
                return 1

    def InitInstance(self):
        if False:
            return 10
        if '/nodde' not in sys.argv and '/new' not in sys.argv and ('-nodde' not in sys.argv) and ('-new' not in sys.argv):
            if self.InitDDE():
                return 1
        else:
            self.ddeServer = None
        win32ui.SetRegistryKey(f'Python {sys.winver}')
        app.CApp.InitInstance(self)
        win32ui.CreateDebuggerThread()
        win32ui.EnableControlContainer()
        from . import interact
        interact.CreateInteractiveWindowUserPreference()
        self.LoadSystemModules()
        self.LoadUserModules()
        try:
            self.frame.LoadBarState('ToolbarDefault')
        except win32ui.error:
            pass
        try:
            self.ProcessArgs(sys.argv)
        except:
            win32ui.DisplayTraceback(sys.exc_info(), ' - error processing command line args')

    def ExitInstance(self):
        if False:
            while True:
                i = 10
        win32ui.DestroyDebuggerThread()
        try:
            from . import interact
            interact.DestroyInteractiveWindow()
        except:
            pass
        if self.ddeServer is not None:
            self.ddeServer.Shutdown()
            self.ddeServer = None
        return app.CApp.ExitInstance(self)

    def Activate(self):
        if False:
            for i in range(10):
                print('nop')
        frame = win32ui.GetMainFrame()
        frame.SetForegroundWindow()
        if frame.GetWindowPlacement()[1] == win32con.SW_SHOWMINIMIZED:
            frame.ShowWindow(win32con.SW_RESTORE)

    def ProcessArgs(self, args, dde=None):
        if False:
            for i in range(10):
                print('nop')
        if len(args) < 1 or not args[0]:
            return
        i = 0
        while i < len(args):
            argType = args[i]
            i += 1
            if argType.startswith('-'):
                argType = '/' + argType[1:]
            if not argType.startswith('/'):
                argType = win32ui.GetProfileVal('Python', 'Default Arg Type', '/edit').lower()
                i -= 1
            par = i < len(args) and args[i] or 'MISSING'
            if argType in ('/nodde', '/new', '-nodde', '-new'):
                pass
            elif argType.startswith('/goto:'):
                gotoline = int(argType[len('/goto:'):])
                if dde:
                    dde.Exec('from pywin.framework import scriptutils\ned = scriptutils.GetActiveEditControl()\nif ed: ed.SetSel(ed.LineIndex(%s - 1))' % gotoline)
                else:
                    from . import scriptutils
                    ed = scriptutils.GetActiveEditControl()
                    if ed:
                        ed.SetSel(ed.LineIndex(gotoline - 1))
            elif argType == '/edit':
                i += 1
                fname = win32api.GetFullPathName(par)
                if not os.path.isfile(fname):
                    win32ui.MessageBox('No such file: {}\n\nCommand Line: {}'.format(fname, win32api.GetCommandLine()), 'Open file for edit', win32con.MB_ICONERROR)
                    continue
                if dde:
                    dde.Exec('win32ui.GetApp().OpenDocumentFile(%s)' % repr(fname))
                else:
                    win32ui.GetApp().OpenDocumentFile(par)
            elif argType == '/rundlg':
                if dde:
                    dde.Exec('from pywin.framework import scriptutils;scriptutils.RunScript({!r}, {!r}, 1)'.format(par, ' '.join(args[i + 1:])))
                else:
                    from . import scriptutils
                    scriptutils.RunScript(par, ' '.join(args[i + 1:]))
                return
            elif argType == '/run':
                if dde:
                    dde.Exec('from pywin.framework import scriptutils;scriptutils.RunScript({!r}, {!r}, 0)'.format(par, ' '.join(args[i + 1:])))
                else:
                    from . import scriptutils
                    scriptutils.RunScript(par, ' '.join(args[i + 1:]), 0)
                return
            elif argType == '/app':
                raise RuntimeError('/app only supported for new instances of Pythonwin.exe')
            elif argType == '/dde':
                if dde is not None:
                    dde.Exec(par)
                else:
                    win32ui.MessageBox('The /dde command can only be used\r\nwhen Pythonwin is already running')
                i += 1
            else:
                raise ValueError('Command line argument not recognised: %s' % argType)

    def LoadSystemModules(self):
        if False:
            while True:
                i = 10
        self.DoLoadModules('pywin.framework.editor,pywin.framework.stdin')

    def LoadUserModules(self, moduleNames=None):
        if False:
            print('Hello World!')
        if moduleNames is None:
            default = 'pywin.framework.sgrepmdi,pywin.framework.mdi_pychecker'
            moduleNames = win32ui.GetProfileVal('Python', 'Startup Modules', default)
        self.DoLoadModules(moduleNames)

    def DoLoadModules(self, moduleNames):
        if False:
            while True:
                i = 10
        if not moduleNames:
            return
        modules = moduleNames.split(',')
        for module in modules:
            try:
                __import__(module)
            except:
                traceback.print_exc()
                msg = 'Startup import of user module "%s" failed' % module
                print(msg)
                win32ui.MessageBox(msg)

    def OnDDECommand(self, command):
        if False:
            while True:
                i = 10
        try:
            exec(command + '\n')
        except:
            print('ERROR executing DDE command: ', command)
            traceback.print_exc()
            raise

    def OnViewBrowse(self, id, code):
        if False:
            while True:
                i = 10
        'Called when ViewBrowse message is received'
        from pywin.tools import browser
        obName = dialog.GetSimpleInput('Object', '__builtins__', 'Browse Python Object')
        if obName is None:
            return
        try:
            browser.Browse(eval(obName, __main__.__dict__, __main__.__dict__))
        except NameError:
            win32ui.MessageBox('This is no object with this name')
        except AttributeError:
            win32ui.MessageBox('The object has no attribute of that name')
        except:
            traceback.print_exc()
            win32ui.MessageBox('This object can not be browsed')

    def OnFileImport(self, id, code):
        if False:
            for i in range(10):
                print('nop')
        'Called when a FileImport message is received. Import the current or specified file'
        from . import scriptutils
        scriptutils.ImportFile()

    def OnFileCheck(self, id, code):
        if False:
            for i in range(10):
                print('nop')
        'Called when a FileCheck message is received. Check the current file.'
        from . import scriptutils
        scriptutils.CheckFile()

    def OnUpdateFileCheck(self, cmdui):
        if False:
            for i in range(10):
                print('nop')
        from . import scriptutils
        cmdui.Enable(scriptutils.GetActiveFileName(0) is not None)

    def OnFileRun(self, id, code):
        if False:
            for i in range(10):
                print('nop')
        'Called when a FileRun message is received.'
        from . import scriptutils
        showDlg = win32api.GetKeyState(win32con.VK_SHIFT) >= 0
        scriptutils.RunScript(None, None, showDlg)

    def OnFileLocate(self, id, code):
        if False:
            print('Hello World!')
        from . import scriptutils
        global lastLocateFileName
        name = dialog.GetSimpleInput('File name', lastLocateFileName, 'Locate Python File')
        if name is None:
            return
        lastLocateFileName = name
        if lastLocateFileName[-3:].lower() == '.py':
            lastLocateFileName = lastLocateFileName[:-3]
        lastLocateFileName = lastLocateFileName.replace('.', '\\')
        newName = scriptutils.LocatePythonFile(lastLocateFileName)
        if newName is None:
            win32ui.MessageBox("The file '%s' can not be located" % lastLocateFileName)
        else:
            win32ui.GetApp().OpenDocumentFile(newName)

    def OnViewOptions(self, id, code):
        if False:
            while True:
                i = 10
        win32ui.InitRichEdit()
        sheet = dialog.PropertySheet('Pythonwin Options')
        from pywin.dialogs import ideoptions
        sheet.AddPage(ideoptions.OptionsPropPage())
        from . import toolmenu
        sheet.AddPage(toolmenu.ToolMenuPropPage())
        pages = []
        for template in self.GetDocTemplateList():
            try:
                getter = template.GetPythonPropertyPages
            except AttributeError:
                continue
            pages = pages + getter()
        try:
            from pywin.debugger import configui
        except ImportError:
            configui = None
        if configui is not None:
            pages.append(configui.DebuggerOptionsPropPage())
        for page in pages:
            sheet.AddPage(page)
        if sheet.DoModal() == win32con.IDOK:
            win32ui.SetStatusText('Applying configuration changes...', 1)
            win32ui.DoWaitCursor(1)
            win32ui.GetMainFrame().SendMessageToDescendants(win32con.WM_WININICHANGE, 0, 0)
            win32ui.DoWaitCursor(0)

    def OnInteractiveWindow(self, id, code):
        if False:
            print('Hello World!')
        from . import interact
        interact.ToggleInteractiveWindow()

    def OnUpdateInteractiveWindow(self, cmdui):
        if False:
            while True:
                i = 10
        try:
            interact = sys.modules['pywin.framework.interact']
            state = interact.IsInteractiveWindowVisible()
        except KeyError:
            state = 0
        cmdui.Enable()
        cmdui.SetCheck(state)

    def OnFileSaveAll(self, id, code):
        if False:
            while True:
                i = 10
        from pywin.framework.editor import editorTemplate
        num = 0
        for doc in editorTemplate.GetDocumentList():
            if doc.IsModified() and doc.GetPathName():
                num = num = 1
                doc.OnSaveDocument(doc.GetPathName())
        win32ui.SetStatusText('%d documents saved' % num, 1)

    def OnViewToolbarDbg(self, id, code):
        if False:
            return 10
        if code == 0:
            return not win32ui.GetMainFrame().OnBarCheck(id)

    def OnUpdateViewToolbarDbg(self, cmdui):
        if False:
            while True:
                i = 10
        win32ui.GetMainFrame().OnUpdateControlBarMenu(cmdui)
        cmdui.Enable(1)

    def OnHelpIndex(self, id, code):
        if False:
            return 10
        from . import help
        help.SelectAndRunHelpFile()
thisApp = InteractivePythonApp()