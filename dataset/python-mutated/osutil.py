"""
Some functions related to the os and os.path module
"""
from contextlib import contextmanager
import os
from os.path import join as opj
import shutil
import tempfile
import zipfile
if os.name == 'nt':
    import ctypes
    import win32service as ws
    import win32serviceutil as wsu

def listdir(dir, recursive=False):
    if False:
        return 10
    'Allow to recursively get the file listing'
    dir = os.path.normpath(dir)
    if not recursive:
        return os.listdir(dir)
    res = []
    for (root, dirs, files) in walksymlinks(dir):
        root = root[len(dir) + 1:]
        res.extend([opj(root, f) for f in files])
    return res

def walksymlinks(top, topdown=True, onerror=None):
    if False:
        while True:
            i = 10
    '\n    same as os.walk but follow symlinks\n    attention: all symlinks are walked before all normals directories\n    '
    for (dirpath, dirnames, filenames) in os.walk(top, topdown, onerror):
        if topdown:
            yield (dirpath, dirnames, filenames)
        symlinks = filter(lambda dirname: os.path.islink(os.path.join(dirpath, dirname)), dirnames)
        for s in symlinks:
            for x in walksymlinks(os.path.join(dirpath, s), topdown, onerror):
                yield x
        if not topdown:
            yield (dirpath, dirnames, filenames)

@contextmanager
def tempdir():
    if False:
        print('Hello World!')
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)

def zip_dir(path, stream, include_dir=True, fnct_sort=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    : param fnct_sort : Function to be passed to "key" parameter of built-in\n                        python sorted() to provide flexibility of sorting files\n                        inside ZIP archive according to specific requirements.\n    '
    path = os.path.normpath(path)
    len_prefix = len(os.path.dirname(path)) if include_dir else len(path)
    if len_prefix:
        len_prefix += 1
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for (dirpath, dirnames, filenames) in os.walk(path):
            filenames = sorted(filenames, key=fnct_sort)
            for fname in filenames:
                (bname, ext) = os.path.splitext(fname)
                ext = ext or bname
                if ext not in ['.pyc', '.pyo', '.swp', '.DS_Store']:
                    path = os.path.normpath(os.path.join(dirpath, fname))
                    if os.path.isfile(path):
                        zipf.write(path, path[len_prefix:])
if os.name != 'nt':
    getppid = os.getppid
    is_running_as_nt_service = lambda : False
else:
    _TH32CS_SNAPPROCESS = 2

    class _PROCESSENTRY32(ctypes.Structure):
        _fields_ = [('dwSize', ctypes.c_ulong), ('cntUsage', ctypes.c_ulong), ('th32ProcessID', ctypes.c_ulong), ('th32DefaultHeapID', ctypes.c_ulong), ('th32ModuleID', ctypes.c_ulong), ('cntThreads', ctypes.c_ulong), ('th32ParentProcessID', ctypes.c_ulong), ('pcPriClassBase', ctypes.c_ulong), ('dwFlags', ctypes.c_ulong), ('szExeFile', ctypes.c_char * 260)]

    def getppid():
        if False:
            print('Hello World!')
        CreateToolhelp32Snapshot = ctypes.windll.kernel32.CreateToolhelp32Snapshot
        Process32First = ctypes.windll.kernel32.Process32First
        Process32Next = ctypes.windll.kernel32.Process32Next
        CloseHandle = ctypes.windll.kernel32.CloseHandle
        hProcessSnap = CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        current_pid = os.getpid()
        try:
            pe32 = _PROCESSENTRY32()
            pe32.dwSize = ctypes.sizeof(_PROCESSENTRY32)
            if not Process32First(hProcessSnap, ctypes.byref(pe32)):
                raise OSError('Failed getting first process.')
            while True:
                if pe32.th32ProcessID == current_pid:
                    return pe32.th32ParentProcessID
                if not Process32Next(hProcessSnap, ctypes.byref(pe32)):
                    return None
        finally:
            CloseHandle(hProcessSnap)
    from contextlib import contextmanager
    from odoo.release import nt_service_name

    def is_running_as_nt_service():
        if False:
            return 10

        @contextmanager
        def close_srv(srv):
            if False:
                i = 10
                return i + 15
            try:
                yield srv
            finally:
                ws.CloseServiceHandle(srv)
        try:
            with close_srv(ws.OpenSCManager(None, None, ws.SC_MANAGER_ALL_ACCESS)) as hscm:
                with close_srv(wsu.SmartOpenService(hscm, nt_service_name, ws.SERVICE_ALL_ACCESS)) as hs:
                    info = ws.QueryServiceStatusEx(hs)
                    return info['ProcessId'] == getppid()
        except Exception:
            return False
if __name__ == '__main__':
    from pprint import pprint as pp
    pp(listdir('../report', True))