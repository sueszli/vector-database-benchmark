"""runproc.py

start a process with three inherited pipes.
Try to write to and read from those.
"""
import msvcrt
import os
import win32api
import win32con
import win32file
import win32pipe
import win32process
import win32security

class Process:

    def run(self, cmdline):
        if False:
            while True:
                i = 10
        sAttrs = win32security.SECURITY_ATTRIBUTES()
        sAttrs.bInheritHandle = 1
        (hStdin_r, self.hStdin_w) = win32pipe.CreatePipe(sAttrs, 0)
        (self.hStdout_r, hStdout_w) = win32pipe.CreatePipe(sAttrs, 0)
        (self.hStderr_r, hStderr_w) = win32pipe.CreatePipe(sAttrs, 0)
        StartupInfo = win32process.STARTUPINFO()
        StartupInfo.hStdInput = hStdin_r
        StartupInfo.hStdOutput = hStdout_w
        StartupInfo.hStdError = hStderr_w
        StartupInfo.dwFlags = win32process.STARTF_USESTDHANDLES
        pid = win32api.GetCurrentProcess()
        tmp = win32api.DuplicateHandle(pid, self.hStdin_w, pid, 0, 0, win32con.DUPLICATE_SAME_ACCESS)
        win32file.CloseHandle(self.hStdin_w)
        self.hStdin_w = tmp
        tmp = win32api.DuplicateHandle(pid, self.hStdout_r, pid, 0, 0, win32con.DUPLICATE_SAME_ACCESS)
        win32file.CloseHandle(self.hStdout_r)
        self.hStdout_r = tmp
        (hProcess, hThread, dwPid, dwTid) = win32process.CreateProcess(None, cmdline, None, None, 1, 0, None, None, StartupInfo)
        win32file.CloseHandle(hStderr_w)
        win32file.CloseHandle(hStdout_w)
        win32file.CloseHandle(hStdin_r)
        self.stdin = os.fdopen(msvcrt.open_osfhandle(self.hStdin_w, 0), 'wb')
        self.stdin.write('hmmmmm\r\n')
        self.stdin.flush()
        self.stdin.close()
        self.stdout = os.fdopen(msvcrt.open_osfhandle(self.hStdout_r, 0), 'rb')
        print('Read on stdout: ', repr(self.stdout.read()))
        self.stderr = os.fdopen(msvcrt.open_osfhandle(self.hStderr_r, 0), 'rb')
        print('Read on stderr: ', repr(self.stderr.read()))
if __name__ == '__main__':
    p = Process()
    exe = win32api.GetModuleFileName(0)
    p.run(exe + ' cat.py')