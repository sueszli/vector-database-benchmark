import timer
import win32api
import win32con
import win32ui
from pywin.mfc import window
from pywin.mfc.thread import WinThread
WM_USER_PREPARE_TO_CLOSE = win32con.WM_USER + 32

class FontWindow(window.Wnd):

    def __init__(self, text='Python Rules!'):
        if False:
            while True:
                i = 10
        window.Wnd.__init__(self)
        self.text = text
        self.index = 0
        self.incr = 1
        self.width = self.height = 0
        self.ChangeAttributes()

    def Create(self, title, style, rect, parent):
        if False:
            for i in range(10):
                print('nop')
        classStyle = win32con.CS_HREDRAW | win32con.CS_VREDRAW
        className = win32ui.RegisterWndClass(classStyle, 0, win32con.COLOR_WINDOW + 1, 0)
        self._obj_ = win32ui.CreateWnd()
        self._obj_.AttachObject(self)
        self._obj_.CreateWindow(className, title, style, rect, parent, win32ui.AFX_IDW_PANE_FIRST)
        self.HookMessage(self.OnSize, win32con.WM_SIZE)
        self.HookMessage(self.OnPrepareToClose, WM_USER_PREPARE_TO_CLOSE)
        self.HookMessage(self.OnDestroy, win32con.WM_DESTROY)
        self.timerid = timer.set_timer(100, self.OnTimer)
        self.InvalidateRect()

    def OnDestroy(self, msg):
        if False:
            i = 10
            return i + 15
        timer.kill_timer(self.timerid)

    def OnTimer(self, id, timeVal):
        if False:
            return 10
        self.index = self.index + self.incr
        if self.index > len(self.text):
            self.incr = -1
            self.index = len(self.text)
        elif self.index < 0:
            self.incr = 1
            self.index = 0
        self.InvalidateRect()

    def OnPaint(self):
        if False:
            return 10
        (dc, paintStruct) = self.BeginPaint()
        self.OnPrepareDC(dc, None)
        if self.width == 0 and self.height == 0:
            (left, top, right, bottom) = self.GetClientRect()
            self.width = right - left
            self.height = bottom - top
        (x, y) = (self.width // 2, self.height // 2)
        dc.TextOut(x, y, self.text[:self.index])
        self.EndPaint(paintStruct)

    def ChangeAttributes(self):
        if False:
            return 10
        font_spec = {'name': 'Arial', 'height': 42}
        self.font = win32ui.CreateFont(font_spec)

    def OnPrepareToClose(self, params):
        if False:
            for i in range(10):
                print('nop')
        self.DestroyWindow()

    def OnSize(self, params):
        if False:
            i = 10
            return i + 15
        lParam = params[3]
        self.width = win32api.LOWORD(lParam)
        self.height = win32api.HIWORD(lParam)

    def OnPrepareDC(self, dc, printinfo):
        if False:
            i = 10
            return i + 15
        dc.SetTextColor(win32api.RGB(0, 0, 255))
        dc.SetBkColor(win32api.GetSysColor(win32con.COLOR_WINDOW))
        dc.SelectObject(self.font)
        dc.SetTextAlign(win32con.TA_CENTER | win32con.TA_BASELINE)

class FontFrame(window.MDIChildWnd):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def Create(self, title, rect=None, parent=None):
        if False:
            print('Hello World!')
        style = win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.WS_OVERLAPPEDWINDOW
        self._obj_ = win32ui.CreateMDIChild()
        self._obj_.AttachObject(self)
        self._obj_.CreateWindow(None, title, style, rect, parent)
        rect = self.GetClientRect()
        rect = (0, 0, rect[2] - rect[0], rect[3] - rect[1])
        self.child = FontWindow('Not threaded')
        self.child.Create('FontDemo', win32con.WS_CHILD | win32con.WS_VISIBLE, rect, self)

class TestThread(WinThread):

    def __init__(self, parentWindow):
        if False:
            for i in range(10):
                print('nop')
        self.parentWindow = parentWindow
        self.child = None
        WinThread.__init__(self)

    def InitInstance(self):
        if False:
            i = 10
            return i + 15
        rect = self.parentWindow.GetClientRect()
        rect = (0, 0, rect[2] - rect[0], rect[3] - rect[1])
        self.child = FontWindow()
        self.child.Create('FontDemo', win32con.WS_CHILD | win32con.WS_VISIBLE, rect, self.parentWindow)
        self.SetMainFrame(self.child)
        return WinThread.InitInstance(self)

    def ExitInstance(self):
        if False:
            i = 10
            return i + 15
        return 0

class ThreadedFontFrame(window.MDIChildWnd):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass
        self.thread = None

    def Create(self, title, rect=None, parent=None):
        if False:
            print('Hello World!')
        style = win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.WS_OVERLAPPEDWINDOW
        self._obj_ = win32ui.CreateMDIChild()
        self._obj_.CreateWindow(None, title, style, rect, parent)
        self._obj_.HookMessage(self.OnDestroy, win32con.WM_DESTROY)
        self._obj_.HookMessage(self.OnSize, win32con.WM_SIZE)
        self.thread = TestThread(self)
        self.thread.CreateThread()

    def OnSize(self, msg):
        if False:
            i = 10
            return i + 15
        pass

    def OnDestroy(self, msg):
        if False:
            while True:
                i = 10
        win32ui.OutputDebugString('OnDestroy\n')
        if self.thread and self.thread.child:
            child = self.thread.child
            child.SendMessage(WM_USER_PREPARE_TO_CLOSE, 0, 0)
            win32ui.OutputDebugString('Destroyed\n')

def Demo():
    if False:
        print('Hello World!')
    f = FontFrame()
    f.Create('Font Demo')

def ThreadedDemo():
    if False:
        i = 10
        return i + 15
    rect = win32ui.GetMainFrame().GetMDIClient().GetClientRect()
    rect = (rect[0], int(rect[3] * 3 / 4), int(rect[2] / 4), rect[3])
    incr = rect[2]
    for i in range(4):
        if i == 0:
            f = FontFrame()
            title = 'Not threaded'
        else:
            f = ThreadedFontFrame()
            title = 'Threaded GUI Demo'
        f.Create(title, rect)
        rect = (rect[0] + incr, rect[1], rect[2] + incr, rect[3])
    win32api.Sleep(100)
    win32ui.PumpWaitingMessages()
if __name__ == '__main__':
    import demoutils
    if demoutils.NeedGoodGUI():
        ThreadedDemo()