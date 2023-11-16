from . import RemovableDrivePlugin
import string
import ctypes
from ctypes import wintypes
from UM.i18n import i18nCatalog
catalog = i18nCatalog('cura')
ctypes.windll.kernel32.SetErrorMode(1)
DRIVE_REMOVABLE = 2
GENERIC_READ = 2147483648
GENERIC_WRITE = 1073741824
FILE_SHARE_READ = 1
FILE_SHARE_WRITE = 2
IOCTL_STORAGE_EJECT_MEDIA = 2967560
OPEN_EXISTING = 3
ctypes.windll.kernel32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
ctypes.windll.kernel32.DeviceIoControl.restype = wintypes.BOOL

class WindowsRemovableDrivePlugin(RemovableDrivePlugin.RemovableDrivePlugin):
    """Removable drive support for windows"""

    def checkRemovableDrives(self):
        if False:
            i = 10
            return i + 15
        drives = {}
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        bitmask >>= 2
        for letter in string.ascii_uppercase[2:]:
            drive = '{0}:/'.format(letter)
            if bitmask & 1 and ctypes.windll.kernel32.GetDriveTypeA(drive.encode('ascii')) == DRIVE_REMOVABLE:
                volume_name = ''
                name_buffer = ctypes.create_unicode_buffer(1024)
                filesystem_buffer = ctypes.create_unicode_buffer(1024)
                error = ctypes.windll.kernel32.GetVolumeInformationW(ctypes.c_wchar_p(drive), name_buffer, ctypes.sizeof(name_buffer), None, None, None, filesystem_buffer, ctypes.sizeof(filesystem_buffer))
                if error != 0:
                    volume_name = name_buffer.value
                if not volume_name:
                    volume_name = catalog.i18nc('@item:intext', 'Removable Drive')
                if filesystem_buffer.value == '':
                    continue
                free_bytes = ctypes.c_longlong(0)
                if ctypes.windll.kernel32.GetDiskFreeSpaceExA(drive.encode('ascii'), ctypes.byref(free_bytes), None, None) == 0:
                    continue
                if free_bytes.value < 1:
                    continue
                drives[drive] = '{0} ({1}:)'.format(volume_name, letter)
            bitmask >>= 1
        return drives

    def performEjectDevice(self, device):
        if False:
            for i in range(10):
                print('nop')
        handle = ctypes.windll.kernel32.CreateFileA('\\\\.\\{0}'.format(device.getId()[:-1]).encode('ascii'), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, None, OPEN_EXISTING, 0, None)
        if handle == -1:
            raise ctypes.WinError()
        bytes_returned = wintypes.DWORD(0)
        error = None
        return_code = ctypes.windll.kernel32.DeviceIoControl(handle, IOCTL_STORAGE_EJECT_MEDIA, None, 0, None, 0, ctypes.pointer(bytes_returned), None)
        if return_code == 0:
            error = ctypes.WinError()
        ctypes.windll.kernel32.CloseHandle(handle)
        if error:
            raise error
        return True