import sys
import win32ui
from pywin.framework import app, dlgappcore

class TestDialogApp(dlgappcore.DialogApp):

    def CreateDialog(self):
        if False:
            return 10
        return TestAppDialog()

class TestAppDialog(dlgappcore.AppDialog):

    def __init__(self):
        if False:
            return 10
        self.edit = None
        dlgappcore.AppDialog.__init__(self, win32ui.IDD_LARGE_EDIT)

    def OnInitDialog(self):
        if False:
            for i in range(10):
                print('nop')
        self.SetWindowText('Test dialog application')
        self.edit = self.GetDlgItem(win32ui.IDC_EDIT1)
        print('Hello from Python')
        print('args are:', end=' ')
        for arg in sys.argv:
            print(arg)
        return 1

    def PreDoModal(self):
        if False:
            print('Hello World!')
        sys.stdout = sys.stderr = self

    def write(self, str):
        if False:
            return 10
        if self.edit:
            self.edit.SetSel(-2)
            self.edit.ReplaceSel(str.replace('\n', '\r\n'))
        else:
            win32ui.OutputDebug('dlgapp - no edit control! >>\n%s\n<<\n' % str)
app.AppBuilder = TestDialogApp
if __name__ == '__main__':
    import demoutils
    demoutils.NeedApp()