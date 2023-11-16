import sys
import pythoncom
from win32com import universal
from win32com.client import Dispatch, DispatchWithEvents, constants, gencache
from win32com.server.exception import COMException
gencache.EnsureModule('{00020813-0000-0000-C000-000000000046}', 0, 1, 3, bForDemand=True)
gencache.EnsureModule('{2DF8D04C-5BFA-101B-BDE5-00AA0044DE52}', 0, 2, 1, bForDemand=True)
universal.RegisterInterfaces('{AC0714F2-3D04-11D1-AE7D-00A0C90F26F4}', 0, 1, 0, ['_IDTExtensibility2'])

class ButtonEvent:

    def OnClick(self, button, cancel):
        if False:
            return 10
        import win32con
        import win32ui
        win32ui.MessageBox('Hello from Python', 'Python Test', win32con.MB_OKCANCEL)
        return cancel

class ExcelAddin:
    _com_interfaces_ = ['_IDTExtensibility2']
    _public_methods_ = []
    _reg_clsctx_ = pythoncom.CLSCTX_INPROC_SERVER
    _reg_clsid_ = '{C5482ECA-F559-45A0-B078-B2036E6F011A}'
    _reg_progid_ = 'Python.Test.ExcelAddin'
    _reg_policy_spec_ = 'win32com.server.policy.EventHandlerPolicy'

    def __init__(self):
        if False:
            while True:
                i = 10
        self.appHostApp = None

    def OnConnection(self, application, connectMode, addin, custom):
        if False:
            return 10
        print('OnConnection', application, connectMode, addin, custom)
        try:
            self.appHostApp = application
            cbcMyBar = self.appHostApp.CommandBars.Add(Name='PythonBar', Position=constants.msoBarTop, MenuBar=constants.msoBarTypeNormal, Temporary=True)
            btnMyButton = cbcMyBar.Controls.Add(Type=constants.msoControlButton, Parameter='Greetings')
            btnMyButton = self.toolbarButton = DispatchWithEvents(btnMyButton, ButtonEvent)
            btnMyButton.Style = constants.msoButtonCaption
            btnMyButton.BeginGroup = True
            btnMyButton.Caption = '&Python'
            btnMyButton.TooltipText = 'Python rules the World'
            btnMyButton.Width = '34'
            cbcMyBar.Visible = True
        except pythoncom.com_error as xxx_todo_changeme:
            (hr, msg, exc, arg) = xxx_todo_changeme.args
            print('The Excel call failed with code %d: %s' % (hr, msg))
            if exc is None:
                print('There is no extended error information')
            else:
                (wcode, source, text, helpFile, helpId, scode) = exc
                print('The source of the error is', source)
                print('The error message is', text)
                print('More info can be found in %s (id=%d)' % (helpFile, helpId))

    def OnDisconnection(self, mode, custom):
        if False:
            for i in range(10):
                print('nop')
        print('OnDisconnection')
        self.appHostApp.CommandBars('PythonBar').Delete
        self.appHostApp = None

    def OnAddInsUpdate(self, custom):
        if False:
            i = 10
            return i + 15
        print('OnAddInsUpdate', custom)

    def OnStartupComplete(self, custom):
        if False:
            i = 10
            return i + 15
        print('OnStartupComplete', custom)

    def OnBeginShutdown(self, custom):
        if False:
            return 10
        print('OnBeginShutdown', custom)

def RegisterAddin(klass):
    if False:
        while True:
            i = 10
    import winreg
    key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, 'Software\\Microsoft\\Office\\Excel\\Addins')
    subkey = winreg.CreateKey(key, klass._reg_progid_)
    winreg.SetValueEx(subkey, 'CommandLineSafe', 0, winreg.REG_DWORD, 0)
    winreg.SetValueEx(subkey, 'LoadBehavior', 0, winreg.REG_DWORD, 3)
    winreg.SetValueEx(subkey, 'Description', 0, winreg.REG_SZ, 'Excel Addin')
    winreg.SetValueEx(subkey, 'FriendlyName', 0, winreg.REG_SZ, 'A Simple Excel Addin')

def UnregisterAddin(klass):
    if False:
        print('Hello World!')
    import winreg
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, 'Software\\Microsoft\\Office\\Excel\\Addins\\' + klass._reg_progid_)
    except OSError:
        pass
if __name__ == '__main__':
    import win32com.server.register
    win32com.server.register.UseCommandLine(ExcelAddin)
    if '--unregister' in sys.argv:
        UnregisterAddin(ExcelAddin)
    else:
        RegisterAddin(ExcelAddin)