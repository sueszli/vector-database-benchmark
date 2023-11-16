"""
Windows Process Control

winprocess.run launches a child process and returns the exit code.
Optionally, it can:
  redirect stdin, stdout & stderr to files
  run the command as another user
  limit the process's running time
  control the process window (location, size, window state, desktop)
Works on Windows NT, 2000 & XP. Requires Mark Hammond's win32
extensions.

This code is free for any purpose, with no warranty of any kind.
-- John B. Dell'Aquila <jbd@alum.mit.edu>
"""
import msvcrt
import os
import win32api
import win32con
import win32event
import win32gui
import win32process
import win32security

def logonUser(loginString):
    if False:
        print('Hello World!')
    "\n    Login as specified user and return handle.\n    loginString:  'Domain\nUser\nPassword'; for local\n        login use . or empty string as domain\n        e.g. '.\nadministrator\nsecret_password'\n    "
    (domain, user, passwd) = loginString.split('\n')
    return win32security.LogonUser(user, domain, passwd, win32con.LOGON32_LOGON_INTERACTIVE, win32con.LOGON32_PROVIDER_DEFAULT)

class Process:
    """
    A Windows process.
    """

    def __init__(self, cmd, login=None, hStdin=None, hStdout=None, hStderr=None, show=1, xy=None, xySize=None, desktop=None):
        if False:
            i = 10
            return i + 15
        "\n        Create a Windows process.\n        cmd:     command to run\n        login:   run as user 'Domain\nUser\nPassword'\n        hStdin, hStdout, hStderr:\n                 handles for process I/O; default is caller's stdin,\n                 stdout & stderr\n        show:    wShowWindow (0=SW_HIDE, 1=SW_NORMAL, ...)\n        xy:      window offset (x, y) of upper left corner in pixels\n        xySize:  window size (width, height) in pixels\n        desktop: lpDesktop - name of desktop e.g. 'winsta0\\default'\n                 None = inherit current desktop\n                 '' = create new desktop if necessary\n\n        User calling login requires additional privileges:\n          Act as part of the operating system [not needed on Windows XP]\n          Increase quotas\n          Replace a process level token\n        Login string must EITHER be an administrator's account\n        (ordinary user can't access current desktop - see Microsoft\n        Q165194) OR use desktop='' to run another desktop invisibly\n        (may be very slow to startup & finalize).\n        "
        si = win32process.STARTUPINFO()
        si.dwFlags = win32con.STARTF_USESTDHANDLES ^ win32con.STARTF_USESHOWWINDOW
        if hStdin is None:
            si.hStdInput = win32api.GetStdHandle(win32api.STD_INPUT_HANDLE)
        else:
            si.hStdInput = hStdin
        if hStdout is None:
            si.hStdOutput = win32api.GetStdHandle(win32api.STD_OUTPUT_HANDLE)
        else:
            si.hStdOutput = hStdout
        if hStderr is None:
            si.hStdError = win32api.GetStdHandle(win32api.STD_ERROR_HANDLE)
        else:
            si.hStdError = hStderr
        si.wShowWindow = show
        if xy is not None:
            (si.dwX, si.dwY) = xy
            si.dwFlags ^= win32con.STARTF_USEPOSITION
        if xySize is not None:
            (si.dwXSize, si.dwYSize) = xySize
            si.dwFlags ^= win32con.STARTF_USESIZE
        if desktop is not None:
            si.lpDesktop = desktop
        procArgs = (None, cmd, None, None, 1, win32process.CREATE_NEW_CONSOLE, None, None, si)
        if login is not None:
            hUser = logonUser(login)
            win32security.ImpersonateLoggedOnUser(hUser)
            procHandles = win32process.CreateProcessAsUser(hUser, *procArgs)
            win32security.RevertToSelf()
        else:
            procHandles = win32process.CreateProcess(*procArgs)
        (self.hProcess, self.hThread, self.PId, self.TId) = procHandles

    def wait(self, mSec=None):
        if False:
            while True:
                i = 10
        '\n        Wait for process to finish or for specified number of\n        milliseconds to elapse.\n        '
        if mSec is None:
            mSec = win32event.INFINITE
        return win32event.WaitForSingleObject(self.hProcess, mSec)

    def kill(self, gracePeriod=5000):
        if False:
            for i in range(10):
                print('nop')
        '\n        Kill process. Try for an orderly shutdown via WM_CLOSE.  If\n        still running after gracePeriod (5 sec. default), terminate.\n        '
        win32gui.EnumWindows(self.__close__, 0)
        if self.wait(gracePeriod) != win32event.WAIT_OBJECT_0:
            win32process.TerminateProcess(self.hProcess, 0)
            win32api.Sleep(100)

    def __close__(self, hwnd, dummy):
        if False:
            i = 10
            return i + 15
        '\n        EnumWindows callback - sends WM_CLOSE to any window\n        owned by this process.\n        '
        (TId, PId) = win32process.GetWindowThreadProcessId(hwnd)
        if PId == self.PId:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

    def exitCode(self):
        if False:
            i = 10
            return i + 15
        '\n        Return process exit code.\n        '
        return win32process.GetExitCodeProcess(self.hProcess)

def run(cmd, mSec=None, stdin=None, stdout=None, stderr=None, **kw):
    if False:
        print('Hello World!')
    "\n    Run cmd as a child process and return exit code.\n    mSec:  terminate cmd after specified number of milliseconds\n    stdin, stdout, stderr:\n           file objects for child I/O (use hStdin etc. to attach\n           handles instead of files); default is caller's stdin,\n           stdout & stderr;\n    kw:    see Process.__init__ for more keyword options\n    "
    if stdin is not None:
        kw['hStdin'] = msvcrt.get_osfhandle(stdin.fileno())
    if stdout is not None:
        kw['hStdout'] = msvcrt.get_osfhandle(stdout.fileno())
    if stderr is not None:
        kw['hStderr'] = msvcrt.get_osfhandle(stderr.fileno())
    child = Process(cmd, **kw)
    if child.wait(mSec) != win32event.WAIT_OBJECT_0:
        child.kill()
        raise OSError('process timeout exceeded')
    return child.exitCode()
if __name__ == '__main__':
    print('Testing winprocess.py...')
    import tempfile
    timeoutSeconds = 15
    cmdString = "REM      Test of winprocess.py piping commands to a shell.\r\nREM      This 'notepad' process will terminate in %d seconds.\r\nvol\r\nnet user\r\n_this_is_a_test_of_stderr_\r\n" % timeoutSeconds
    cmd_name = tempfile.mktemp()
    out_name = cmd_name + '.txt'
    try:
        cmd = open(cmd_name, 'w+b')
        out = open(out_name, 'w+b')
        cmd.write(cmdString.encode('mbcs'))
        cmd.seek(0)
        print('CMD.EXE exit code:', run('cmd.exe', show=0, stdin=cmd, stdout=out, stderr=out))
        cmd.close()
        print('NOTEPAD exit code:', run('notepad.exe %s' % out.name, show=win32con.SW_MAXIMIZE, mSec=timeoutSeconds * 1000))
        out.close()
    finally:
        for n in (cmd_name, out_name):
            try:
                os.unlink(cmd_name)
            except OSError:
                pass