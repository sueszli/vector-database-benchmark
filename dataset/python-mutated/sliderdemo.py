import win32con
import win32ui
from pywin.mfc import dialog

class MyDialog(dialog.Dialog):
    """
    Example using simple controls
    """
    _dialogstyle = win32con.WS_MINIMIZEBOX | win32con.WS_DLGFRAME | win32con.DS_MODALFRAME | win32con.WS_POPUP | win32con.WS_VISIBLE | win32con.WS_CAPTION | win32con.WS_SYSMENU | win32con.DS_SETFONT
    _buttonstyle = win32con.BS_PUSHBUTTON | win32con.WS_TABSTOP | win32con.WS_CHILD | win32con.WS_VISIBLE
    DIALOGTEMPLATE = [['Example slider', (0, 0, 50, 43), _dialogstyle, None, (8, 'MS SansSerif')], [128, 'Close', win32con.IDCANCEL, (0, 30, 50, 13), _buttonstyle]]
    IDC_SLIDER = 9500

    def __init__(self):
        if False:
            while True:
                i = 10
        dialog.Dialog.__init__(self, self.DIALOGTEMPLATE)

    def OnInitDialog(self):
        if False:
            i = 10
            return i + 15
        rc = dialog.Dialog.OnInitDialog(self)
        win32ui.EnableControlContainer()
        self.slider = win32ui.CreateSliderCtrl()
        self.slider.CreateWindow(win32con.WS_TABSTOP | win32con.WS_VISIBLE, (0, 0, 100, 30), self._obj_, self.IDC_SLIDER)
        self.HookMessage(self.OnSliderMove, win32con.WM_HSCROLL)
        return rc

    def OnSliderMove(self, params):
        if False:
            return 10
        print('Slider moved')

    def OnCancel(self):
        if False:
            i = 10
            return i + 15
        print('The slider control is at position', self.slider.GetPos())
        self._obj_.OnCancel()

def demo():
    if False:
        print('Hello World!')
    dia = MyDialog()
    dia.DoModal()
if __name__ == '__main__':
    demo()