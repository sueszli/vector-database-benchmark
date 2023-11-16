import pythoncom
import win32con
import win32gui
from win32com.shell import shell, shellcon

class ShellExtension:
    _reg_progid_ = 'Python.ShellExtension.CopyHook'
    _reg_desc_ = 'Python Sample Shell Extension (copy hook)'
    _reg_clsid_ = '{1845b6ba-2bbd-4197-b930-46d8651497c1}'
    _com_interfaces_ = [shell.IID_ICopyHook]
    _public_methods_ = ['CopyCallBack']

    def CopyCallBack(self, hwnd, func, flags, srcName, srcAttr, destName, destAttr):
        if False:
            print('Hello World!')
        print('CopyCallBack', hwnd, func, flags, srcName, srcAttr, destName, destAttr)
        return win32gui.MessageBox(hwnd, 'Allow operation?', 'CopyHook', win32con.MB_YESNO)

def DllRegisterServer():
    if False:
        for i in range(10):
            print('nop')
    import winreg
    key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, 'directory\\shellex\\CopyHookHandlers\\' + ShellExtension._reg_desc_)
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ShellExtension._reg_clsid_)
    key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, '*\\shellex\\CopyHookHandlers\\' + ShellExtension._reg_desc_)
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ShellExtension._reg_clsid_)
    print(ShellExtension._reg_desc_, 'registration complete.')

def DllUnregisterServer():
    if False:
        return 10
    import winreg
    try:
        key = winreg.DeleteKey(winreg.HKEY_CLASSES_ROOT, 'directory\\shellex\\CopyHookHandlers\\' + ShellExtension._reg_desc_)
    except OSError as details:
        import errno
        if details.errno != errno.ENOENT:
            raise
    try:
        key = winreg.DeleteKey(winreg.HKEY_CLASSES_ROOT, '*\\shellex\\CopyHookHandlers\\' + ShellExtension._reg_desc_)
    except OSError as details:
        import errno
        if details.errno != errno.ENOENT:
            raise
    print(ShellExtension._reg_desc_, 'unregistration complete.')
if __name__ == '__main__':
    from win32com.server import register
    register.UseCommandLine(ShellExtension, finalize_register=DllRegisterServer, finalize_unregister=DllUnregisterServer)