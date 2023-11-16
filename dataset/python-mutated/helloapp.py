import win32con
import win32ui
from pywin.mfc import window
from pywin.mfc.thread import WinApp

class HelloWindow(window.Wnd):

    def __init__(self):
        if False:
            return 10
        window.Wnd.__init__(self, win32ui.CreateWnd())
        self._obj_.CreateWindowEx(win32con.WS_EX_CLIENTEDGE, win32ui.RegisterWndClass(0, 0, win32con.COLOR_WINDOW + 1), 'Hello World!', win32con.WS_OVERLAPPEDWINDOW, (100, 100, 400, 300), None, 0, None)

class HelloApp(WinApp):

    def InitInstance(self):
        if False:
            print('Hello World!')
        self.frame = HelloWindow()
        self.frame.ShowWindow(win32con.SW_SHOWNORMAL)
        self.SetMainFrame(self.frame)
app = HelloApp()