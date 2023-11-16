import os
import stat
import sys
import pythoncom
import win32gui
import winerror
from win32com.server.exception import COMException
from win32com.shell import shell, shellcon
IEmptyVolumeCache_Methods = 'Initialize GetSpaceUsed Purge ShowProperties Deactivate'.split()
IEmptyVolumeCache2_Methods = 'InitializeEx'.split()
ico = os.path.join(sys.prefix, 'py.ico')
if not os.path.isfile(ico):
    ico = os.path.join(sys.prefix, 'PC', 'py.ico')
if not os.path.isfile(ico):
    ico = None
    print("Can't find python.ico - no icon will be installed")

class EmptyVolumeCache:
    _reg_progid_ = 'Python.ShellExtension.EmptyVolumeCache'
    _reg_desc_ = 'Python Sample Shell Extension (disk cleanup)'
    _reg_clsid_ = '{EADD0777-2968-4c72-A999-2BF5F756259C}'
    _reg_icon_ = ico
    _com_interfaces_ = [shell.IID_IEmptyVolumeCache, shell.IID_IEmptyVolumeCache2]
    _public_methods_ = IEmptyVolumeCache_Methods + IEmptyVolumeCache2_Methods

    def Initialize(self, hkey, volume, flags):
        if False:
            while True:
                i = 10
        print('Unless we are on 98, Initialize call is unexpected!')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def InitializeEx(self, hkey, volume, key_name, flags):
        if False:
            for i in range(10):
                print('nop')
        print('InitializeEx called with', hkey, volume, key_name, flags)
        self.volume = volume
        if flags & shellcon.EVCF_SETTINGSMODE:
            print('We are being run on a schedule')
            self.volume = None
        elif flags & shellcon.EVCF_OUTOFDISKSPACE:
            print('We are being run as we are out of disk-space')
        else:
            print('We are being run because the user asked')
        flags = shellcon.EVCF_DONTSHOWIFZERO | shellcon.EVCF_ENABLEBYDEFAULT
        return ('pywin32 compiled files', 'Removes all .pyc and .pyo files in the pywin32 directories', 'click me!', flags)

    def _GetDirectories(self):
        if False:
            return 10
        root_dir = os.path.abspath(os.path.dirname(os.path.dirname(win32gui.__file__)))
        if self.volume is not None and (not root_dir.lower().startswith(self.volume.lower())):
            return []
        return [os.path.join(root_dir, p) for p in ('win32', 'win32com', 'win32comext', 'isapi')]

    def _WalkCallback(self, arg, directory, files):
        if False:
            while True:
                i = 10
        (callback, total_list) = arg
        for file in files:
            fqn = os.path.join(directory, file).lower()
            if file.endswith('.pyc') or file.endswith('.pyo'):
                if total_list is None:
                    print('Deleting file', fqn)
                    os.remove(fqn)
                else:
                    total_list[0] += os.stat(fqn)[stat.ST_SIZE]
                    if callback:
                        used = total_list[0]
                        callback.ScanProgress(used, 0, 'Looking at ' + fqn)

    def GetSpaceUsed(self, callback):
        if False:
            while True:
                i = 10
        total = [0]
        try:
            for d in self._GetDirectories():
                os.path.walk(d, self._WalkCallback, (callback, total))
                print('After looking in', d, 'we have', total[0], 'bytes')
        except pythoncom.error as exc:
            if exc.hresult != winerror.E_ABORT:
                raise
            print('User cancelled the operation')
        return total[0]

    def Purge(self, amt_to_free, callback):
        if False:
            print('Hello World!')
        print('Purging', amt_to_free, 'bytes...')
        try:
            for d in self._GetDirectories():
                os.path.walk(d, self._WalkCallback, (callback, None))
        except pythoncom.error as exc:
            if exc.hresult != winerror.E_ABORT:
                raise
            print('User cancelled the operation')

    def ShowProperties(self, hwnd):
        if False:
            print('Hello World!')
        raise COMException(hresult=winerror.E_NOTIMPL)

    def Deactivate(self):
        if False:
            return 10
        print('Deactivate called')
        return 0

def DllRegisterServer():
    if False:
        for i in range(10):
            print('nop')
    import winreg
    kn = 'Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\VolumeCaches\\{}'.format(EmptyVolumeCache._reg_desc_)
    key = winreg.CreateKey(winreg.HKEY_LOCAL_MACHINE, kn)
    winreg.SetValueEx(key, None, 0, winreg.REG_SZ, EmptyVolumeCache._reg_clsid_)

def DllUnregisterServer():
    if False:
        while True:
            i = 10
    import winreg
    kn = 'Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\VolumeCaches\\{}'.format(EmptyVolumeCache._reg_desc_)
    try:
        key = winreg.DeleteKey(winreg.HKEY_LOCAL_MACHINE, kn)
    except OSError as details:
        import errno
        if details.errno != errno.ENOENT:
            raise
    print(EmptyVolumeCache._reg_desc_, 'unregistration complete.')
if __name__ == '__main__':
    from win32com.server import register
    register.UseCommandLine(EmptyVolumeCache, finalize_register=DllRegisterServer, finalize_unregister=DllUnregisterServer)