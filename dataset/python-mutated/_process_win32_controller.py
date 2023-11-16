"""Windows-specific implementation of process utilities with direct WinAPI.

This file is meant to be used by process.py
"""
import os, sys, threading
import ctypes, msvcrt
from ctypes import POINTER
from ctypes.wintypes import HANDLE, HLOCAL, LPVOID, WORD, DWORD, BOOL, ULONG, LPCWSTR
LPDWORD = POINTER(DWORD)
LPHANDLE = POINTER(HANDLE)
ULONG_PTR = POINTER(ULONG)

class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [('nLength', DWORD), ('lpSecurityDescriptor', LPVOID), ('bInheritHandle', BOOL)]
LPSECURITY_ATTRIBUTES = POINTER(SECURITY_ATTRIBUTES)

class STARTUPINFO(ctypes.Structure):
    _fields_ = [('cb', DWORD), ('lpReserved', LPCWSTR), ('lpDesktop', LPCWSTR), ('lpTitle', LPCWSTR), ('dwX', DWORD), ('dwY', DWORD), ('dwXSize', DWORD), ('dwYSize', DWORD), ('dwXCountChars', DWORD), ('dwYCountChars', DWORD), ('dwFillAttribute', DWORD), ('dwFlags', DWORD), ('wShowWindow', WORD), ('cbReserved2', WORD), ('lpReserved2', LPVOID), ('hStdInput', HANDLE), ('hStdOutput', HANDLE), ('hStdError', HANDLE)]
LPSTARTUPINFO = POINTER(STARTUPINFO)

class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [('hProcess', HANDLE), ('hThread', HANDLE), ('dwProcessId', DWORD), ('dwThreadId', DWORD)]
LPPROCESS_INFORMATION = POINTER(PROCESS_INFORMATION)
ERROR_HANDLE_EOF = 38
ERROR_BROKEN_PIPE = 109
ERROR_NO_DATA = 232
HANDLE_FLAG_INHERIT = 1
STARTF_USESTDHANDLES = 256
CREATE_SUSPENDED = 4
CREATE_NEW_CONSOLE = 16
CREATE_NO_WINDOW = 134217728
STILL_ACTIVE = 259
WAIT_TIMEOUT = 258
WAIT_FAILED = 4294967295
INFINITE = 4294967295
DUPLICATE_SAME_ACCESS = 2
ENABLE_ECHO_INPUT = 4
ENABLE_LINE_INPUT = 2
ENABLE_PROCESSED_INPUT = 1
GetLastError = ctypes.windll.kernel32.GetLastError
GetLastError.argtypes = []
GetLastError.restype = DWORD
CreateFile = ctypes.windll.kernel32.CreateFileW
CreateFile.argtypes = [LPCWSTR, DWORD, DWORD, LPVOID, DWORD, DWORD, HANDLE]
CreateFile.restype = HANDLE
CreatePipe = ctypes.windll.kernel32.CreatePipe
CreatePipe.argtypes = [POINTER(HANDLE), POINTER(HANDLE), LPSECURITY_ATTRIBUTES, DWORD]
CreatePipe.restype = BOOL
CreateProcess = ctypes.windll.kernel32.CreateProcessW
CreateProcess.argtypes = [LPCWSTR, LPCWSTR, LPSECURITY_ATTRIBUTES, LPSECURITY_ATTRIBUTES, BOOL, DWORD, LPVOID, LPCWSTR, LPSTARTUPINFO, LPPROCESS_INFORMATION]
CreateProcess.restype = BOOL
GetExitCodeProcess = ctypes.windll.kernel32.GetExitCodeProcess
GetExitCodeProcess.argtypes = [HANDLE, LPDWORD]
GetExitCodeProcess.restype = BOOL
GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
GetCurrentProcess.argtypes = []
GetCurrentProcess.restype = HANDLE
ResumeThread = ctypes.windll.kernel32.ResumeThread
ResumeThread.argtypes = [HANDLE]
ResumeThread.restype = DWORD
ReadFile = ctypes.windll.kernel32.ReadFile
ReadFile.argtypes = [HANDLE, LPVOID, DWORD, LPDWORD, LPVOID]
ReadFile.restype = BOOL
WriteFile = ctypes.windll.kernel32.WriteFile
WriteFile.argtypes = [HANDLE, LPVOID, DWORD, LPDWORD, LPVOID]
WriteFile.restype = BOOL
GetConsoleMode = ctypes.windll.kernel32.GetConsoleMode
GetConsoleMode.argtypes = [HANDLE, LPDWORD]
GetConsoleMode.restype = BOOL
SetConsoleMode = ctypes.windll.kernel32.SetConsoleMode
SetConsoleMode.argtypes = [HANDLE, DWORD]
SetConsoleMode.restype = BOOL
FlushConsoleInputBuffer = ctypes.windll.kernel32.FlushConsoleInputBuffer
FlushConsoleInputBuffer.argtypes = [HANDLE]
FlushConsoleInputBuffer.restype = BOOL
WaitForSingleObject = ctypes.windll.kernel32.WaitForSingleObject
WaitForSingleObject.argtypes = [HANDLE, DWORD]
WaitForSingleObject.restype = DWORD
DuplicateHandle = ctypes.windll.kernel32.DuplicateHandle
DuplicateHandle.argtypes = [HANDLE, HANDLE, HANDLE, LPHANDLE, DWORD, BOOL, DWORD]
DuplicateHandle.restype = BOOL
SetHandleInformation = ctypes.windll.kernel32.SetHandleInformation
SetHandleInformation.argtypes = [HANDLE, DWORD, DWORD]
SetHandleInformation.restype = BOOL
CloseHandle = ctypes.windll.kernel32.CloseHandle
CloseHandle.argtypes = [HANDLE]
CloseHandle.restype = BOOL
CommandLineToArgvW = ctypes.windll.shell32.CommandLineToArgvW
CommandLineToArgvW.argtypes = [LPCWSTR, POINTER(ctypes.c_int)]
CommandLineToArgvW.restype = POINTER(LPCWSTR)
LocalFree = ctypes.windll.kernel32.LocalFree
LocalFree.argtypes = [HLOCAL]
LocalFree.restype = HLOCAL

class AvoidUNCPath(object):
    """A context manager to protect command execution from UNC paths.

    In the Win32 API, commands can't be invoked with the cwd being a UNC path.
    This context manager temporarily changes directory to the 'C:' drive on
    entering, and restores the original working directory on exit.

    The context manager returns the starting working directory *if* it made a
    change and None otherwise, so that users can apply the necessary adjustment
    to their system calls in the event of a change.

    Examples
    --------
    ::
        cmd = 'dir'
        with AvoidUNCPath() as path:
            if path is not None:
                cmd = '"pushd %s &&"%s' % (path, cmd)
            os.system(cmd)
    """

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.path = os.getcwd()
        self.is_unc_path = self.path.startswith('\\\\')
        if self.is_unc_path:
            os.chdir('C:')
            return self.path
        else:
            return None

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            for i in range(10):
                print('nop')
        if self.is_unc_path:
            os.chdir(self.path)

class Win32ShellCommandController(object):
    """Runs a shell command in a 'with' context.

    This implementation is Win32-specific.

    Example:
        # Runs the command interactively with default console stdin/stdout
        with ShellCommandController('python -i') as scc:
            scc.run()

        # Runs the command using the provided functions for stdin/stdout
        def my_stdout_func(s):
            # print or save the string 's'
            write_to_stdout(s)
        def my_stdin_func():
            # If input is available, return it as a string.
            if input_available():
                return get_input()
            # If no input available, return None after a short delay to
            # keep from blocking.
            else:
                time.sleep(0.01)
                return None
      
        with ShellCommandController('python -i') as scc:
            scc.run(my_stdout_func, my_stdin_func)
    """

    def __init__(self, cmd, mergeout=True):
        if False:
            while True:
                i = 10
        'Initializes the shell command controller.\n\n        The cmd is the program to execute, and mergeout is\n        whether to blend stdout and stderr into one output\n        in stdout. Merging them together in this fashion more\n        reliably keeps stdout and stderr in the correct order\n        especially for interactive shell usage.\n        '
        self.cmd = cmd
        self.mergeout = mergeout

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        cmd = self.cmd
        mergeout = self.mergeout
        (self.hstdout, self.hstdin, self.hstderr) = (None, None, None)
        self.piProcInfo = None
        try:
            (p_hstdout, c_hstdout, p_hstderr, c_hstderr, p_hstdin, c_hstdin) = [None] * 6
            saAttr = SECURITY_ATTRIBUTES()
            saAttr.nLength = ctypes.sizeof(saAttr)
            saAttr.bInheritHandle = True
            saAttr.lpSecurityDescriptor = None

            def create_pipe(uninherit):
                if False:
                    return 10
                "Creates a Windows pipe, which consists of two handles.\n\n                The 'uninherit' parameter controls which handle is not\n                inherited by the child process.\n                "
                handles = (HANDLE(), HANDLE())
                if not CreatePipe(ctypes.byref(handles[0]), ctypes.byref(handles[1]), ctypes.byref(saAttr), 0):
                    raise ctypes.WinError()
                if not SetHandleInformation(handles[uninherit], HANDLE_FLAG_INHERIT, 0):
                    raise ctypes.WinError()
                return (handles[0].value, handles[1].value)
            (p_hstdout, c_hstdout) = create_pipe(uninherit=0)
            if mergeout:
                c_hstderr = HANDLE()
                if not DuplicateHandle(GetCurrentProcess(), c_hstdout, GetCurrentProcess(), ctypes.byref(c_hstderr), 0, True, DUPLICATE_SAME_ACCESS):
                    raise ctypes.WinError()
            else:
                (p_hstderr, c_hstderr) = create_pipe(uninherit=0)
            (c_hstdin, p_hstdin) = create_pipe(uninherit=1)
            piProcInfo = PROCESS_INFORMATION()
            siStartInfo = STARTUPINFO()
            siStartInfo.cb = ctypes.sizeof(siStartInfo)
            siStartInfo.hStdInput = c_hstdin
            siStartInfo.hStdOutput = c_hstdout
            siStartInfo.hStdError = c_hstderr
            siStartInfo.dwFlags = STARTF_USESTDHANDLES
            dwCreationFlags = CREATE_SUSPENDED | CREATE_NO_WINDOW
            if not CreateProcess(None, u'cmd.exe /c ' + cmd, None, None, True, dwCreationFlags, None, None, ctypes.byref(siStartInfo), ctypes.byref(piProcInfo)):
                raise ctypes.WinError()
            CloseHandle(c_hstdin)
            c_hstdin = None
            CloseHandle(c_hstdout)
            c_hstdout = None
            if c_hstderr is not None:
                CloseHandle(c_hstderr)
                c_hstderr = None
            self.hstdin = p_hstdin
            p_hstdin = None
            self.hstdout = p_hstdout
            p_hstdout = None
            if not mergeout:
                self.hstderr = p_hstderr
                p_hstderr = None
            self.piProcInfo = piProcInfo
        finally:
            if p_hstdin:
                CloseHandle(p_hstdin)
            if c_hstdin:
                CloseHandle(c_hstdin)
            if p_hstdout:
                CloseHandle(p_hstdout)
            if c_hstdout:
                CloseHandle(c_hstdout)
            if p_hstderr:
                CloseHandle(p_hstderr)
            if c_hstderr:
                CloseHandle(c_hstderr)
        return self

    def _stdin_thread(self, handle, hprocess, func, stdout_func):
        if False:
            return 10
        exitCode = DWORD()
        bytesWritten = DWORD(0)
        while True:
            data = func()
            if data is None:
                if not GetExitCodeProcess(hprocess, ctypes.byref(exitCode)):
                    raise ctypes.WinError()
                if exitCode.value != STILL_ACTIVE:
                    return
                if not WriteFile(handle, '', 0, ctypes.byref(bytesWritten), None):
                    raise ctypes.WinError()
                continue
            if isinstance(data, unicode):
                data = data.encode('utf_8')
            if not isinstance(data, str):
                raise RuntimeError('internal stdin function string error')
            if len(data) == 0:
                return
            stdout_func(data)
            while len(data) != 0:
                if not WriteFile(handle, data, len(data), ctypes.byref(bytesWritten), None):
                    if GetLastError() == ERROR_NO_DATA:
                        return
                    raise ctypes.WinError()
                data = data[bytesWritten.value:]

    def _stdout_thread(self, handle, func):
        if False:
            for i in range(10):
                print('nop')
        data = ctypes.create_string_buffer(4096)
        while True:
            bytesRead = DWORD(0)
            if not ReadFile(handle, data, 4096, ctypes.byref(bytesRead), None):
                le = GetLastError()
                if le == ERROR_BROKEN_PIPE:
                    return
                else:
                    raise ctypes.WinError()
            s = data.value[0:bytesRead.value]
            func(s.decode('utf_8', 'replace'))

    def run(self, stdout_func=None, stdin_func=None, stderr_func=None):
        if False:
            while True:
                i = 10
        'Runs the process, using the provided functions for I/O.\n\n        The function stdin_func should return strings whenever a\n        character or characters become available.\n        The functions stdout_func and stderr_func are called whenever\n        something is printed to stdout or stderr, respectively.\n        These functions are called from different threads (but not\n        concurrently, because of the GIL).\n        '
        if stdout_func is None and stdin_func is None and (stderr_func is None):
            return self._run_stdio()
        if stderr_func is not None and self.mergeout:
            raise RuntimeError('Shell command was initiated with merged stdin/stdout, but a separate stderr_func was provided to the run() method')
        stdin_thread = None
        threads = []
        if stdin_func:
            stdin_thread = threading.Thread(target=self._stdin_thread, args=(self.hstdin, self.piProcInfo.hProcess, stdin_func, stdout_func))
        threads.append(threading.Thread(target=self._stdout_thread, args=(self.hstdout, stdout_func)))
        if not self.mergeout:
            if stderr_func is None:
                stderr_func = stdout_func
            threads.append(threading.Thread(target=self._stdout_thread, args=(self.hstderr, stderr_func)))
        if ResumeThread(self.piProcInfo.hThread) == 4294967295:
            raise ctypes.WinError()
        if stdin_thread is not None:
            stdin_thread.start()
        for thread in threads:
            thread.start()
        if WaitForSingleObject(self.piProcInfo.hProcess, INFINITE) == WAIT_FAILED:
            raise ctypes.WinError()
        for thread in threads:
            thread.join()
        if stdin_thread is not None:
            stdin_thread.join()

    def _stdin_raw_nonblock(self):
        if False:
            while True:
                i = 10
        'Use the raw Win32 handle of sys.stdin to do non-blocking reads'
        handle = msvcrt.get_osfhandle(sys.stdin.fileno())
        result = WaitForSingleObject(handle, 100)
        if result == WAIT_FAILED:
            raise ctypes.WinError()
        elif result == WAIT_TIMEOUT:
            print('.', end='')
            return None
        else:
            data = ctypes.create_string_buffer(256)
            bytesRead = DWORD(0)
            print('?', end='')
            if not ReadFile(handle, data, 256, ctypes.byref(bytesRead), None):
                raise ctypes.WinError()
            FlushConsoleInputBuffer(handle)
            data = data.value
            data = data.replace('\r\n', '\n')
            data = data.replace('\r', '\n')
            print(repr(data) + ' ', end='')
            return data

    def _stdin_raw_block(self):
        if False:
            print('Hello World!')
        'Use a blocking stdin read'
        try:
            data = sys.stdin.read(1)
            data = data.replace('\r', '\n')
            return data
        except WindowsError as we:
            if we.winerror == ERROR_NO_DATA:
                return None
            else:
                raise we

    def _stdout_raw(self, s):
        if False:
            i = 10
            return i + 15
        'Writes the string to stdout'
        print(s, end='', file=sys.stdout)
        sys.stdout.flush()

    def _stderr_raw(self, s):
        if False:
            while True:
                i = 10
        'Writes the string to stdout'
        print(s, end='', file=sys.stderr)
        sys.stderr.flush()

    def _run_stdio(self):
        if False:
            while True:
                i = 10
        'Runs the process using the system standard I/O.\n\n        IMPORTANT: stdin needs to be asynchronous, so the Python\n                   sys.stdin object is not used. Instead,\n                   msvcrt.kbhit/getwch are used asynchronously.\n        '
        if self.mergeout:
            return self.run(stdout_func=self._stdout_raw, stdin_func=self._stdin_raw_block)
        else:
            return self.run(stdout_func=self._stdout_raw, stdin_func=self._stdin_raw_block, stderr_func=self._stderr_raw)

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            print('Hello World!')
        if self.hstdin:
            CloseHandle(self.hstdin)
            self.hstdin = None
        if self.hstdout:
            CloseHandle(self.hstdout)
            self.hstdout = None
        if self.hstderr:
            CloseHandle(self.hstderr)
            self.hstderr = None
        if self.piProcInfo is not None:
            CloseHandle(self.piProcInfo.hProcess)
            CloseHandle(self.piProcInfo.hThread)
            self.piProcInfo = None

def system(cmd):
    if False:
        print('Hello World!')
    'Win32 version of os.system() that works with network shares.\n\n    Note that this implementation returns None, as meant for use in IPython.\n\n    Parameters\n    ----------\n    cmd : str\n        A command to be executed in the system shell.\n\n    Returns\n    -------\n    None : we explicitly do NOT return the subprocess status code, as this\n    utility is meant to be used extensively in IPython, where any return value\n    would trigger : func:`sys.displayhook` calls.\n    '
    with AvoidUNCPath() as path:
        if path is not None:
            cmd = '"pushd %s &&"%s' % (path, cmd)
        with Win32ShellCommandController(cmd) as scc:
            scc.run()
if __name__ == '__main__':
    print('Test starting!')
    system('python -i')
    print('Test finished!')