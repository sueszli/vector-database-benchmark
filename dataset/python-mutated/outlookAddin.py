import sys
import pythoncom
from win32com import universal
from win32com.client import DispatchWithEvents, constants, gencache
from win32com.server.exception import COMException
gencache.EnsureModule('{00062FFF-0000-0000-C000-000000000046}', 0, 9, 0, bForDemand=True)
gencache.EnsureModule('{2DF8D04C-5BFA-101B-BDE5-00AA0044DE52}', 0, 2, 1, bForDemand=True)
universal.RegisterInterfaces('{AC0714F2-3D04-11D1-AE7D-00A0C90F26F4}', 0, 1, 0, ['_IDTExtensibility2'])

class ButtonEvent:

    def OnClick(self, button, cancel):
        if False:
            i = 10
            return i + 15
        import win32ui
        win32ui.MessageBox('Hello from Python')
        return cancel

class FolderEvent:

    def OnItemAdd(self, item):
        if False:
            i = 10
            return i + 15
        try:
            print('An item was added to the inbox with subject:', item.Subject)
        except AttributeError:
            print('An item was added to the inbox, but it has no subject! - ', repr(item))

class OutlookAddin:
    _com_interfaces_ = ['_IDTExtensibility2']
    _public_methods_ = []
    _reg_clsctx_ = pythoncom.CLSCTX_INPROC_SERVER
    _reg_clsid_ = '{0F47D9F3-598B-4d24-B7E3-92AC15ED27E2}'
    _reg_progid_ = 'Python.Test.OutlookAddin'
    _reg_policy_spec_ = 'win32com.server.policy.EventHandlerPolicy'

    def OnConnection(self, application, connectMode, addin, custom):
        if False:
            while True:
                i = 10
        print('OnConnection', application, connectMode, addin, custom)
        activeExplorer = application.ActiveExplorer()
        if activeExplorer is not None:
            bars = activeExplorer.CommandBars
            toolbar = bars.Item('Standard')
            item = toolbar.Controls.Add(Type=constants.msoControlButton, Temporary=True)
            item = self.toolbarButton = DispatchWithEvents(item, ButtonEvent)
            item.Caption = 'Python'
            item.TooltipText = 'Click for Python'
            item.Enabled = True
        inbox = application.Session.GetDefaultFolder(constants.olFolderInbox)
        self.inboxItems = DispatchWithEvents(inbox.Items, FolderEvent)

    def OnDisconnection(self, mode, custom):
        if False:
            print('Hello World!')
        print('OnDisconnection')

    def OnAddInsUpdate(self, custom):
        if False:
            for i in range(10):
                print('nop')
        print('OnAddInsUpdate', custom)

    def OnStartupComplete(self, custom):
        if False:
            print('Hello World!')
        print('OnStartupComplete', custom)

    def OnBeginShutdown(self, custom):
        if False:
            while True:
                i = 10
        print('OnBeginShutdown', custom)

def RegisterAddin(klass):
    if False:
        return 10
    import winreg
    key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, 'Software\\Microsoft\\Office\\Outlook\\Addins')
    subkey = winreg.CreateKey(key, klass._reg_progid_)
    winreg.SetValueEx(subkey, 'CommandLineSafe', 0, winreg.REG_DWORD, 0)
    winreg.SetValueEx(subkey, 'LoadBehavior', 0, winreg.REG_DWORD, 3)
    winreg.SetValueEx(subkey, 'Description', 0, winreg.REG_SZ, klass._reg_progid_)
    winreg.SetValueEx(subkey, 'FriendlyName', 0, winreg.REG_SZ, klass._reg_progid_)

def UnregisterAddin(klass):
    if False:
        return 10
    import winreg
    try:
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, 'Software\\Microsoft\\Office\\Outlook\\Addins\\' + klass._reg_progid_)
    except OSError:
        pass
if __name__ == '__main__':
    import win32com.server.register
    win32com.server.register.UseCommandLine(OutlookAddin)
    if '--unregister' in sys.argv:
        UnregisterAddin(OutlookAddin)
    else:
        RegisterAddin(OutlookAddin)