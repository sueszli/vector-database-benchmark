"""
Get Version information from Windows
"""
import ctypes
HAS_WIN32 = True
try:
    from ctypes.wintypes import BYTE, DWORD, WCHAR, WORD
    import win32net
    import win32netcon
except (ImportError, ValueError):
    HAS_WIN32 = False
if HAS_WIN32:
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only load if Win32 Libraries are installed\n    '
    if not HAS_WIN32:
        return (False, 'This utility requires pywin32')
    return 'win_osinfo'

def os_version_info_ex():
    if False:
        print('Hello World!')
    '\n    Helper function to return the results of the GetVersionExW Windows API call.\n    It is a ctypes Structure that contains Windows OS Version information.\n\n    Returns:\n        class: An instance of a class containing version info\n    '
    if not HAS_WIN32:
        return

    class OSVersionInfo(ctypes.Structure):
        _fields_ = (('dwOSVersionInfoSize', DWORD), ('dwMajorVersion', DWORD), ('dwMinorVersion', DWORD), ('dwBuildNumber', DWORD), ('dwPlatformId', DWORD), ('szCSDVersion', WCHAR * 128))

        def __init__(self, *args, **kwds):
            if False:
                while True:
                    i = 10
            super().__init__(*args, **kwds)
            self.dwOSVersionInfoSize = ctypes.sizeof(self)
            kernel32.GetVersionExW(ctypes.byref(self))

    class OSVersionInfoEx(OSVersionInfo):
        _fields_ = (('wServicePackMajor', WORD), ('wServicePackMinor', WORD), ('wSuiteMask', WORD), ('wProductType', BYTE), ('wReserved', BYTE))
    return OSVersionInfoEx()

def get_os_version_info():
    if False:
        while True:
            i = 10
    info = os_version_info_ex()
    ret = {'MajorVersion': info.dwMajorVersion, 'MinorVersion': info.dwMinorVersion, 'BuildNumber': info.dwBuildNumber, 'PlatformID': info.dwPlatformId, 'ServicePackMajor': info.wServicePackMajor, 'ServicePackMinor': info.wServicePackMinor, 'SuiteMask': info.wSuiteMask, 'ProductType': info.wProductType}
    return ret

def get_join_info():
    if False:
        i = 10
        return i + 15
    '\n    Gets information about the domain/workgroup. This will tell you if the\n    system is joined to a domain or a workgroup\n\n    .. versionadded:: 2018.3.4\n\n    Returns:\n        dict: A dictionary containing the domain/workgroup and its status\n    '
    info = win32net.NetGetJoinInformation()
    status = {win32netcon.NetSetupUnknown: 'Unknown', win32netcon.NetSetupUnjoined: 'Unjoined', win32netcon.NetSetupWorkgroupName: 'Workgroup', win32netcon.NetSetupDomainName: 'Domain'}
    return {'Domain': info[0], 'DomainType': status[info[1]]}