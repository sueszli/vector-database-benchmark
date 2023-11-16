import os
import stat
import commctrl
import pythoncom
from pywintypes import IID
from win32com.server.util import wrap
from win32com.shell import shell, shellcon
IPersist_Methods = ['GetClassID']
IColumnProvider_Methods = IPersist_Methods + ['Initialize', 'GetColumnInfo', 'GetItemData']

class ColumnProvider:
    _reg_progid_ = 'Python.ShellExtension.ColumnProvider'
    _reg_desc_ = 'Python Sample Shell Extension (Column Provider)'
    _reg_clsid_ = IID('{0F14101A-E05E-4070-BD54-83DFA58C3D68}')
    _com_interfaces_ = [pythoncom.IID_IPersist, shell.IID_IColumnProvider]
    _public_methods_ = IColumnProvider_Methods

    def GetClassID(self):
        if False:
            for i in range(10):
                print('nop')
        return self._reg_clsid_

    def Initialize(self, colInit):
        if False:
            print('Hello World!')
        (flags, reserved, name) = colInit
        print('ColumnProvider initializing for file', name)

    def GetColumnInfo(self, index):
        if False:
            for i in range(10):
                print('nop')
        if index in [0, 1]:
            if index == 0:
                ext = '.pyc'
            else:
                ext = '.pyo'
            title = ext + ' size'
            description = 'Size of compiled %s file' % ext
            col_id = (self._reg_clsid_, index)
            col_info = (col_id, pythoncom.VT_I4, commctrl.LVCFMT_RIGHT, 20, shellcon.SHCOLSTATE_TYPE_INT | shellcon.SHCOLSTATE_SECONDARYUI, title, description)
            return col_info
        return None

    def GetItemData(self, colid, colData):
        if False:
            for i in range(10):
                print('nop')
        (fmt_id, pid) = colid
        fmt_id == self._reg_clsid_
        (flags, attr, reserved, ext, name) = colData
        if ext.lower() not in ['.py', '.pyw']:
            return None
        if pid == 0:
            ext = '.pyc'
        else:
            ext = '.pyo'
        check_file = os.path.splitext(name)[0] + ext
        try:
            st = os.stat(check_file)
            return st[stat.ST_SIZE]
        except OSError:
            return None

def DllRegisterServer():
    if False:
        for i in range(10):
            print('nop')
    import winreg
    key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, 'Folder\\ShellEx\\ColumnHandlers\\' + str(ColumnProvider._reg_clsid_))
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, ColumnProvider._reg_desc_)
    print(ColumnProvider._reg_desc_, 'registration complete.')

def DllUnregisterServer():
    if False:
        print('Hello World!')
    import winreg
    try:
        key = winreg.DeleteKey(winreg.HKEY_CLASSES_ROOT, 'Folder\\ShellEx\\ColumnHandlers\\' + str(ColumnProvider._reg_clsid_))
    except OSError as details:
        import errno
        if details.errno != errno.ENOENT:
            raise
    print(ColumnProvider._reg_desc_, 'unregistration complete.')
if __name__ == '__main__':
    from win32com.server import register
    register.UseCommandLine(ColumnProvider, finalize_register=DllRegisterServer, finalize_unregister=DllUnregisterServer)