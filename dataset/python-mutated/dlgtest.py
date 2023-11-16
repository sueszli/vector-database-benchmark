import win32con
import win32ui
from pywin.mfc import dialog
from win32con import IDCANCEL
from win32ui import IDC_EDIT_TABS, IDC_PROMPT_TABS, IDD_SET_TABSTOPS

class TestDialog(dialog.Dialog):

    def __init__(self, modal=1):
        if False:
            print('Hello World!')
        dialog.Dialog.__init__(self, IDD_SET_TABSTOPS)
        self.counter = 0
        if modal:
            self.DoModal()
        else:
            self.CreateWindow()

    def OnInitDialog(self):
        if False:
            while True:
                i = 10
        self.SetWindowText('Used to be Tab Stops!')
        self.edit = self.GetDlgItem(IDC_EDIT_TABS)
        self.edit.SetWindowText('Test')
        self.edit.HookMessage(self.KillFocus, win32con.WM_KILLFOCUS)
        prompt = self.GetDlgItem(IDC_PROMPT_TABS)
        prompt.SetWindowText('Prompt')
        cancel = self.GetDlgItem(IDCANCEL)
        cancel.SetWindowText('&Kill me')
        self.HookCommand(self.OnNotify, IDC_EDIT_TABS)

    def OnNotify(self, controlid, code):
        if False:
            print('Hello World!')
        if code == win32con.EN_CHANGE:
            print('Edit text changed!')
        return 1

    def KillFocus(self, msg):
        if False:
            print('Hello World!')
        self.counter = self.counter + 1
        if self.edit is not None:
            self.edit.SetWindowText(str(self.counter))

    def OnDestroy(self, msg):
        if False:
            print('Hello World!')
        del self.edit
        del self.counter

class TestSheet(dialog.PropertySheet):

    def __init__(self, title):
        if False:
            print('Hello World!')
        dialog.PropertySheet.__init__(self, title)
        self.HookMessage(self.OnActivate, win32con.WM_ACTIVATE)

    def OnActivate(self, msg):
        if False:
            i = 10
            return i + 15
        pass

class TestPage(dialog.PropertyPage):

    def OnInitDialog(self):
        if False:
            return 10
        self.HookNotify(self.OnNotify, win32con.BN_CLICKED)

    def OnNotify(self, std, extra):
        if False:
            return 10
        print('OnNotify', std, extra)

def demo(modal=0):
    if False:
        return 10
    TestDialog(modal)
    ps = win32ui.CreatePropertySheet('Property Sheet/Page Demo')
    page1 = win32ui.CreatePropertyPage(win32ui.IDD_PROPDEMO1)
    page2 = TestPage(win32ui.IDD_PROPDEMO2)
    ps.AddPage(page1)
    ps.AddPage(page2)
    if modal:
        ps.DoModal()
    else:
        style = win32con.WS_SYSMENU | win32con.WS_POPUP | win32con.WS_CAPTION | win32con.DS_MODALFRAME | win32con.WS_VISIBLE
        styleex = win32con.WS_EX_DLGMODALFRAME | win32con.WS_EX_PALETTEWINDOW
        ps.CreateWindow(win32ui.GetMainFrame(), style, styleex)

def test(modal=1):
    if False:
        return 10
    ps = TestSheet('Property Sheet/Page Demo')
    page1 = win32ui.CreatePropertyPage(win32ui.IDD_PROPDEMO1)
    page2 = win32ui.CreatePropertyPage(win32ui.IDD_PROPDEMO2)
    ps.AddPage(page1)
    ps.AddPage(page2)
    del page1
    del page2
    if modal:
        ps.DoModal()
    else:
        ps.CreateWindow(win32ui.GetMainFrame())
    return ps

def d():
    if False:
        for i in range(10):
            print('nop')
    dlg = win32ui.CreateDialog(win32ui.IDD_DEBUGGER)
    dlg.datalist.append((win32ui.IDC_DBG_RADIOSTACK, 'radio'))
    print('data list is ', dlg.datalist)
    dlg.data['radio'] = 1
    dlg.DoModal()
    print(dlg.data['radio'])
if __name__ == '__main__':
    demo(1)