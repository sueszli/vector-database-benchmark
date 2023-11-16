import win32api
import win32con
import win32ui
from pywin.mfc import dialog
from . import app
error = 'Dialog Application Error'

class AppDialog(dialog.Dialog):
    """The dialog box for the application"""

    def __init__(self, id, dll=None):
        if False:
            i = 10
            return i + 15
        self.iconId = win32ui.IDR_MAINFRAME
        dialog.Dialog.__init__(self, id, dll)

    def OnInitDialog(self):
        if False:
            print('Hello World!')
        return dialog.Dialog.OnInitDialog(self)

    def OnPaint(self):
        if False:
            print('Hello World!')
        if not self.IsIconic():
            return self._obj_.OnPaint()
        self.DefWindowProc(win32con.WM_ICONERASEBKGND, dc.GetHandleOutput(), 0)
        (left, top, right, bottom) = self.GetClientRect()
        left = right - win32api.GetSystemMetrics(win32con.SM_CXICON) >> 1
        top = bottom - win32api.GetSystemMetrics(win32con.SM_CYICON) >> 1
        hIcon = win32ui.GetApp().LoadIcon(self.iconId)
        self.GetDC().DrawIcon((left, top), hIcon)

    def OnEraseBkgnd(self, dc):
        if False:
            i = 10
            return i + 15
        if self.IsIconic():
            return 1
        else:
            return self._obj_.OnEraseBkgnd(dc)

    def OnQueryDragIcon(self):
        if False:
            while True:
                i = 10
        return win32ui.GetApp().LoadIcon(self.iconId)

    def PreDoModal(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class DialogApp(app.CApp):
    """An application class, for an app with main dialog box"""

    def InitInstance(self):
        if False:
            i = 10
            return i + 15
        win32ui.LoadStdProfileSettings()
        win32ui.EnableControlContainer()
        win32ui.Enable3dControls()
        self.dlg = self.frame = self.CreateDialog()
        if self.frame is None:
            raise error('No dialog was created by CreateDialog()')
            return
        self._obj_.InitDlgInstance(self.dlg)
        self.PreDoModal()
        self.dlg.PreDoModal()
        self.dlg.DoModal()

    def CreateDialog(self):
        if False:
            return 10
        pass

    def PreDoModal(self):
        if False:
            i = 10
            return i + 15
        pass