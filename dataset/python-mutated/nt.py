import contextlib
import ctypes
import os
from ctypes.wintypes import BOOL, CHAR, DWORD, HANDLE, LONG, LPWSTR, MAX_PATH, PDWORD, ULONG
from pipenv.vendor.shellingham._core import SHELL_NAMES
INVALID_HANDLE_VALUE = HANDLE(-1).value
ERROR_NO_MORE_FILES = 18
ERROR_INSUFFICIENT_BUFFER = 122
TH32CS_SNAPPROCESS = 2
PROCESS_QUERY_LIMITED_INFORMATION = 4096
kernel32 = ctypes.windll.kernel32

def _check_handle(error_val=0):
    if False:
        print('Hello World!')

    def check(ret, func, args):
        if False:
            for i in range(10):
                print('nop')
        if ret == error_val:
            raise ctypes.WinError()
        return ret
    return check

def _check_expected(expected):
    if False:
        for i in range(10):
            print('nop')

    def check(ret, func, args):
        if False:
            i = 10
            return i + 15
        if ret:
            return True
        code = ctypes.GetLastError()
        if code == expected:
            return False
        raise ctypes.WinError(code)
    return check

class ProcessEntry32(ctypes.Structure):
    _fields_ = (('dwSize', DWORD), ('cntUsage', DWORD), ('th32ProcessID', DWORD), ('th32DefaultHeapID', ctypes.POINTER(ULONG)), ('th32ModuleID', DWORD), ('cntThreads', DWORD), ('th32ParentProcessID', DWORD), ('pcPriClassBase', LONG), ('dwFlags', DWORD), ('szExeFile', CHAR * MAX_PATH))
kernel32.CloseHandle.argtypes = [HANDLE]
kernel32.CloseHandle.restype = BOOL
kernel32.CreateToolhelp32Snapshot.argtypes = [DWORD, DWORD]
kernel32.CreateToolhelp32Snapshot.restype = HANDLE
kernel32.CreateToolhelp32Snapshot.errcheck = _check_handle(INVALID_HANDLE_VALUE)
kernel32.Process32First.argtypes = [HANDLE, ctypes.POINTER(ProcessEntry32)]
kernel32.Process32First.restype = BOOL
kernel32.Process32First.errcheck = _check_expected(ERROR_NO_MORE_FILES)
kernel32.Process32Next.argtypes = [HANDLE, ctypes.POINTER(ProcessEntry32)]
kernel32.Process32Next.restype = BOOL
kernel32.Process32Next.errcheck = _check_expected(ERROR_NO_MORE_FILES)
kernel32.GetCurrentProcessId.argtypes = []
kernel32.GetCurrentProcessId.restype = DWORD
kernel32.OpenProcess.argtypes = [DWORD, BOOL, DWORD]
kernel32.OpenProcess.restype = HANDLE
kernel32.OpenProcess.errcheck = _check_handle(INVALID_HANDLE_VALUE)
kernel32.QueryFullProcessImageNameW.argtypes = [HANDLE, DWORD, LPWSTR, PDWORD]
kernel32.QueryFullProcessImageNameW.restype = BOOL
kernel32.QueryFullProcessImageNameW.errcheck = _check_expected(ERROR_INSUFFICIENT_BUFFER)

@contextlib.contextmanager
def _handle(f, *args, **kwargs):
    if False:
        print('Hello World!')
    handle = f(*args, **kwargs)
    try:
        yield handle
    finally:
        kernel32.CloseHandle(handle)

def _iter_processes():
    if False:
        while True:
            i = 10
    f = kernel32.CreateToolhelp32Snapshot
    with _handle(f, TH32CS_SNAPPROCESS, 0) as snap:
        entry = ProcessEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        ret = kernel32.Process32First(snap, entry)
        while ret:
            yield entry
            ret = kernel32.Process32Next(snap, entry)

def _get_full_path(proch):
    if False:
        print('Hello World!')
    size = DWORD(MAX_PATH)
    while True:
        path_buff = ctypes.create_unicode_buffer('', size.value)
        if kernel32.QueryFullProcessImageNameW(proch, 0, path_buff, size):
            return path_buff.value
        size.value *= 2

def get_shell(pid=None, max_depth=10):
    if False:
        return 10
    proc_map = {proc.th32ProcessID: (proc.th32ParentProcessID, proc.szExeFile) for proc in _iter_processes()}
    pid = pid or os.getpid()
    for _ in range(0, max_depth + 1):
        try:
            (ppid, executable) = proc_map[pid]
        except KeyError:
            break
        if isinstance(executable, bytes):
            executable = executable.decode('mbcs', 'replace')
        name = executable.rpartition('.')[0].lower()
        if name not in SHELL_NAMES:
            pid = ppid
            continue
        key = PROCESS_QUERY_LIMITED_INFORMATION
        with _handle(kernel32.OpenProcess, key, 0, pid) as proch:
            return (name, _get_full_path(proch))
    return None