import win32api
import win32con
import win32ui
from pywin.framework import app, dlgappcore

class DoJobAppDialog(dlgappcore.AppDialog):
    softspace = 1

    def __init__(self, appName=''):
        if False:
            for i in range(10):
                print('nop')
        self.appName = appName
        dlgappcore.AppDialog.__init__(self, win32ui.IDD_GENERAL_STATUS)

    def PreDoModal(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def ProcessArgs(self, args):
        if False:
            for i in range(10):
                print('nop')
        pass

    def OnInitDialog(self):
        if False:
            while True:
                i = 10
        self.SetWindowText(self.appName)
        butCancel = self.GetDlgItem(win32con.IDCANCEL)
        butCancel.ShowWindow(win32con.SW_HIDE)
        p1 = self.GetDlgItem(win32ui.IDC_PROMPT1)
        p2 = self.GetDlgItem(win32ui.IDC_PROMPT2)
        p1.SetWindowText('Hello there')
        p2.SetWindowText('from the demo')

    def OnDestroy(self, msg):
        if False:
            for i in range(10):
                print('nop')
        pass

class DoJobDialogApp(dlgappcore.DialogApp):

    def CreateDialog(self):
        if False:
            return 10
        return DoJobAppDialog('Do Something')

class CopyToDialogApp(DoJobDialogApp):

    def __init__(self):
        if False:
            return 10
        DoJobDialogApp.__init__(self)
app.AppBuilder = DoJobDialogApp

def t():
    if False:
        while True:
            i = 10
    t = DoJobAppDialog('Copy To')
    t.DoModal()
    return t
if __name__ == '__main__':
    import demoutils
    demoutils.NeedApp()