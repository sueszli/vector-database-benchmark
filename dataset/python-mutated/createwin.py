import win32api
import win32con
import win32ui
from pywin.mfc import dialog, window

class Control(window.Wnd):
    """Generic control class"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        window.Wnd.__init__(self, win32ui.CreateWnd())

    def OnPaint(self):
        if False:
            print('Hello World!')
        (dc, paintStruct) = self.BeginPaint()
        self.DoPaint(dc)
        self.EndPaint(paintStruct)

    def DoPaint(self, dc):
        if False:
            print('Hello World!')
        pass

class RedBox(Control):

    def DoPaint(self, dc):
        if False:
            return 10
        dc.FillSolidRect(self.GetClientRect(), win32api.RGB(255, 0, 0))

class RedBoxWithPie(RedBox):

    def DoPaint(self, dc):
        if False:
            while True:
                i = 10
        RedBox.DoPaint(self, dc)
        r = self.GetClientRect()
        dc.Pie(r[0], r[1], r[2], r[3], 0, 0, r[2], r[3] // 2)

def MakeDlgTemplate():
    if False:
        while True:
            i = 10
    style = win32con.DS_MODALFRAME | win32con.WS_POPUP | win32con.WS_VISIBLE | win32con.WS_CAPTION | win32con.WS_SYSMENU | win32con.DS_SETFONT
    cs = win32con.WS_CHILD | win32con.WS_VISIBLE
    w = 64
    h = 64
    dlg = [['Red box', (0, 0, w, h), style, None, (8, 'MS Sans Serif')]]
    s = win32con.WS_TABSTOP | cs
    dlg.append([128, 'Cancel', win32con.IDCANCEL, (7, h - 18, 50, 14), s | win32con.BS_PUSHBUTTON])
    return dlg

class TestDialog(dialog.Dialog):

    def OnInitDialog(self):
        if False:
            return 10
        rc = dialog.Dialog.OnInitDialog(self)
        self.redbox = RedBox()
        self.redbox.CreateWindow(None, 'RedBox', win32con.WS_CHILD | win32con.WS_VISIBLE, (5, 5, 90, 68), self, 1003)
        return rc

class TestPieDialog(dialog.Dialog):

    def OnInitDialog(self):
        if False:
            for i in range(10):
                print('nop')
        rc = dialog.Dialog.OnInitDialog(self)
        self.control = RedBoxWithPie()
        self.control.CreateWindow(None, 'RedBox with Pie', win32con.WS_CHILD | win32con.WS_VISIBLE, (5, 5, 90, 68), self, 1003)

def demo(modal=0):
    if False:
        print('Hello World!')
    d = TestPieDialog(MakeDlgTemplate())
    if modal:
        d.DoModal()
    else:
        d.CreateWindow()
if __name__ == '__main__':
    demo(1)