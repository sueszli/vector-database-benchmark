import glob
import os
import random
import sys
import pythoncom
import win32gui
import winerror
from win32com.shell import shell, shellcon
ico_files = glob.glob(os.path.join(sys.prefix, '*.ico'))
if not ico_files:
    ico_files = glob.glob(os.path.join(sys.prefix, 'PC', '*.ico'))
if not ico_files:
    print("WARNING: Can't find any icon files")
IExtractIcon_Methods = 'Extract GetIconLocation'.split()
IPersistFile_Methods = 'IsDirty Load Save SaveCompleted GetCurFile'.split()

class ShellExtension:
    _reg_progid_ = 'Python.ShellExtension.IconHandler'
    _reg_desc_ = 'Python Sample Shell Extension (icon handler)'
    _reg_clsid_ = '{a97e32d7-3b78-448c-b341-418120ea9227}'
    _com_interfaces_ = [shell.IID_IExtractIcon, pythoncom.IID_IPersistFile]
    _public_methods_ = IExtractIcon_Methods + IPersistFile_Methods

    def Load(self, filename, mode):
        if False:
            print('Hello World!')
        self.filename = filename
        self.mode = mode

    def GetIconLocation(self, flags):
        if False:
            print('Hello World!')
        return (random.choice(ico_files), 0, 0)

    def Extract(self, fname, index, size):
        if False:
            while True:
                i = 10
        return winerror.S_FALSE

def DllRegisterServer():
    if False:
        return 10
    import winreg
    key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, 'Python.File\\shellex')
    subkey = winreg.CreateKey(key, 'IconHandler')
    winreg.SetValueEx(subkey, None, 0, winreg.REG_SZ, ShellExtension._reg_clsid_)
    print(ShellExtension._reg_desc_, 'registration complete.')

def DllUnregisterServer():
    if False:
        print('Hello World!')
    import winreg
    try:
        key = winreg.DeleteKey(winreg.HKEY_CLASSES_ROOT, 'Python.File\\shellex\\IconHandler')
    except OSError as details:
        import errno
        if details.errno != errno.ENOENT:
            raise
    print(ShellExtension._reg_desc_, 'unregistration complete.')
if __name__ == '__main__':
    from win32com.server import register
    register.UseCommandLine(ShellExtension, finalize_register=DllRegisterServer, finalize_unregister=DllUnregisterServer)