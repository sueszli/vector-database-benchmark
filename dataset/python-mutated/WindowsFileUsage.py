""" This is work in progress and hangs unfortunately on some file handles.

"""
import ctypes.wintypes
import os
from nuitka.Tracing import my_print
STATUS_INFO_LENGTH_MISMATCH = 3221225476
STATUS_BUFFER_OVERFLOW = 2147483653
STATUS_INVALID_HANDLE = 3221225480
STATUS_BUFFER_TOO_SMALL = 3221225507

def getWindowsAllProcessandles():
    if False:
        i = 10
        return i + 15
    'Return all process system handles.'
    i = 2048

    class SYSTEM_HANDLE(ctypes.Structure):
        _fields_ = [('dwProcessId', ctypes.wintypes.DWORD), ('bObjectType', ctypes.wintypes.BYTE), ('bFlags', ctypes.wintypes.BYTE), ('wValue', ctypes.wintypes.WORD), ('pAddress', ctypes.wintypes.LPVOID), ('GrantedAccess', ctypes.wintypes.DWORD)]
    ctypes.windll.ntdll.ZwQuerySystemInformation.argtypes = (ctypes.c_ulong, ctypes.wintypes.LPVOID, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong))
    ctypes.windll.ntdll.ZwQuerySystemInformation.restype = ctypes.c_ulong
    return_length = ctypes.c_ulong()
    while True:

        class SYSTEM_HANDLE_INFORMATION(ctypes.Structure):
            _fields_ = [('HandleCount', ctypes.c_ulong), ('Handles', SYSTEM_HANDLE * i)]
        buf = SYSTEM_HANDLE_INFORMATION()
        rc = ctypes.windll.ntdll.ZwQuerySystemInformation(16, ctypes.byref(buf), ctypes.sizeof(buf), ctypes.byref(return_length))
        if rc == STATUS_INFO_LENGTH_MISMATCH:
            i *= 2
            continue
        if rc == 0:
            handles = {}
            for handle in buf.Handles[:buf.HandleCount]:
                if handle.dwProcessId in handles:
                    handles[handle.dwProcessId].append(handle.wValue)
                else:
                    handles[handle.dwProcessId] = [handle.wValue]
            return handles
        else:
            raise ctypes.WinError()

def getWindowsFileHandleFilename(handle):
    if False:
        return 10

    class FILE_NAME_INFORMATION(ctypes.Structure):
        _fields_ = [('FileNameLength', ctypes.wintypes.ULONG), ('FileName', ctypes.wintypes.WCHAR * 2048)]

    class IO_STATUS_BLOCK(ctypes.Structure):

        class _STATUS(ctypes.Union):
            _fields_ = (('Status', ctypes.wintypes.DWORD), ('Pointer', ctypes.wintypes.LPVOID))
        _anonymous_ = ('_Status',)
        _fields_ = (('_Status', _STATUS), ('Information', ctypes.wintypes.WPARAM))
    PIO_STATUS_BLOCK = ctypes.POINTER(IO_STATUS_BLOCK)
    ctypes.windll.ntdll.NtQueryInformationFile.argtypes = (ctypes.wintypes.HANDLE, PIO_STATUS_BLOCK, ctypes.wintypes.LPVOID, ctypes.wintypes.ULONG, ctypes.wintypes.ULONG)
    ctypes.windll.ntdll.NtQueryInformationFile.restype = ctypes.wintypes.DWORD
    file_name_information = FILE_NAME_INFORMATION()
    iosb = IO_STATUS_BLOCK()
    rv = ctypes.windll.ntdll.NtQueryInformationFile(handle, ctypes.byref(iosb), ctypes.byref(file_name_information), 2048, 9)
    if rv == 0:
        return file_name_information.FileName
    else:
        return None

def getWindowsAllProcessFileHandles():
    if False:
        i = 10
        return i + 15

    class LSA_UNICODE_STRING(ctypes.Structure):
        _fields_ = [('Length', ctypes.wintypes.USHORT), ('MaximumLength', ctypes.wintypes.USHORT), ('Buffer', ctypes.wintypes.LPWSTR)]

    class PUBLIC_OBJECT_TYPE_INFORMATION(ctypes.Structure):
        _fields_ = [('Name', LSA_UNICODE_STRING), ('Reserved', ctypes.wintypes.ULONG * 100)]
    ctypes.windll.ntdll.NtQueryObject.argtypes = (ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD, ctypes.POINTER(PUBLIC_OBJECT_TYPE_INFORMATION), ctypes.wintypes.ULONG, ctypes.POINTER(ctypes.wintypes.ULONG))
    ctypes.windll.ntdll.NtQueryObject.restype = ctypes.wintypes.DWORD
    ctypes.windll.kernel32.OpenProcess.argtypes = (ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD)
    ctypes.windll.kernel32.OpenProcess.restype = ctypes.wintypes.HANDLE
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
    ctypes.windll.kernel32.GetFileType.restype = ctypes.wintypes.DWORD
    ctypes.windll.kernel32.DuplicateHandle.argtypes = (ctypes.wintypes.HANDLE, ctypes.wintypes.HANDLE, ctypes.wintypes.HANDLE, ctypes.wintypes.LPHANDLE, ctypes.wintypes.DWORD, ctypes.wintypes.BOOL, ctypes.wintypes.DWORD)
    ctypes.windll.kernel32.DuplicateHandle.restype = ctypes.wintypes.BOOL
    DUPLICATE_SAME_ACCESS = 2
    PROCESS_DUP_HANDLE = 64
    PROCESS_QUERY_LIMITED_INFORMATION = 4096
    ObjectTypeInformation = 2
    psapi = ctypes.WinDLL('Psapi.dll')
    psapi.GetProcessImageFileNameW.restype = ctypes.wintypes.DWORD
    this_process = ctypes.windll.kernel32.GetCurrentProcess()
    this_process_id = ctypes.windll.kernel32.GetCurrentProcessId()
    for (process, handles) in getWindowsAllProcessandles().items():
        if this_process_id == process:
            continue
        process_handle = ctypes.windll.kernel32.OpenProcess(PROCESS_DUP_HANDLE | PROCESS_QUERY_LIMITED_INFORMATION, True, process)
        if not process_handle:
            continue
        ImageFileName = (ctypes.c_wchar * 2048)()
        rv = psapi.GetProcessImageFileNameW(process_handle, ImageFileName, 2048)
        if rv == 0:
            raise ctypes.WinError()
        process_name = os.path.basename(ImageFileName.value)
        if process_name in ('Code.exe', 'CodeHelper.exe', 'RadeonSoftware.exe', 'vsls-agent.exe', 'firefox.exe', 'EpicGamesLauncher.exe', 'Amazon Music Helper.exe', 'EpicWebHelper.exe', 'RuntimeBroker.exe', 'ShellExperienceHost.exe', 'StartMenuExperienceHost.exe', 'devenv.exe', 'ApplicationFrameHost.exe', 'cpptools-srv.exe', 'CompPkgSrv.exe', 'SettingSyncHost.exe', 'SecurityHealthSystray.exe', 'vcpkgsrv.exe', 'UserOOBEBroker.exe', 'TextInputHost.exe', 'sihost.exe', 'taskhostw.exe', 'RAVCpl64.exe', 'PerfWatson2.exe', 'conhost.exe', 'bash.exe', 'cpptools.exe', 'SystemSettings.exe', 'LockApp.exe', 'SearchApp.exe', 'dllhost.exe', 'vctip.exe'):
            continue
        if process_name.startswith('ServiceHub.'):
            continue
        my_print('Process: %s (%d files)' % (process_name, len(handles)))
        for h in handles:
            handle = ctypes.wintypes.HANDLE()
            rv = ctypes.windll.kernel32.DuplicateHandle(process_handle, h, this_process, ctypes.byref(handle), 0, 0, DUPLICATE_SAME_ACCESS)
            if rv == 0:
                continue
            if not handle:
                continue
            public_object_type_information = PUBLIC_OBJECT_TYPE_INFORMATION()
            size = ctypes.wintypes.DWORD(ctypes.sizeof(public_object_type_information))
            file_type = ctypes.windll.kernel32.GetFileType(handle)
            if file_type != 1:
                continue
            rv = ctypes.windll.ntdll.NtQueryObject(handle, ObjectTypeInformation, ctypes.byref(public_object_type_information), size, None)
            if rv == 0:
                kind = public_object_type_information.Name.Buffer
                if kind != 'File':
                    continue
            elif rv == STATUS_INVALID_HANDLE:
                continue
            else:
                assert rv != STATUS_BUFFER_TOO_SMALL
                assert rv != STATUS_BUFFER_OVERFLOW
                assert rv != STATUS_INFO_LENGTH_MISMATCH
                assert False, hex(rv)
            r = getWindowsFileHandleFilename(handle)
            ctypes.windll.kernel32.CloseHandle(handle)
            if r:
                my_print(r)
        ctypes.windll.kernel32.CloseHandle(process_handle)
if __name__ == '__main__':
    getWindowsAllProcessFileHandles()