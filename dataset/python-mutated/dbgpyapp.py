import sys
import win32con
import win32ui
from pywin.framework import intpyapp
version = '0.3.0'

class DebuggerPythonApp(intpyapp.InteractivePythonApp):

    def LoadMainFrame(self):
        if False:
            return 10
        'Create the main applications frame'
        self.frame = self.CreateMainFrame()
        self.SetMainFrame(self.frame)
        self.frame.LoadFrame(win32ui.IDR_DEBUGGER, win32con.WS_OVERLAPPEDWINDOW)
        self.frame.DragAcceptFiles()
        self.frame.ShowWindow(win32con.SW_HIDE)
        self.frame.UpdateWindow()
        self.HookCommands()

    def InitInstance(self):
        if False:
            return 10
        win32ui.SetAppName(win32ui.LoadString(win32ui.IDR_DEBUGGER))
        win32ui.SetRegistryKey(f'Python {sys.winver}')
        numMRU = win32ui.GetProfileVal('Settings', 'Recent File List Size', 10)
        win32ui.LoadStdProfileSettings(numMRU)
        self.LoadMainFrame()
        from pywin.framework import interact
        interact.CreateInteractiveWindowUserPreference()
        self.LoadSystemModules()
        self.LoadUserModules()
        win32ui.EnableControlContainer()