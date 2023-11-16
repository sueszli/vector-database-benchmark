"""
Run processes as a different user in Windows
"""
import ctypes
import logging
import os
import time
from salt.exceptions import CommandExecutionError
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
try:
    import msvcrt
    import pywintypes
    import win32api
    import win32con
    import win32event
    import win32pipe
    import win32process
    import win32profile
    import win32security
    import salt.platform.win
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        for i in range(10):
            print('nop')
    '\n    Only load if Win32 Libraries are installed\n    '
    if not HAS_WIN32 or not HAS_PSUTIL:
        return (False, 'This utility requires pywin32 and psutil')
    return 'win_runas'

def split_username(username):
    if False:
        print('Hello World!')
    domain = '.'
    user_name = username
    if '@' in username:
        (user_name, domain) = username.split('@')
    if '\\' in username:
        (domain, user_name) = username.split('\\')
    return (user_name, domain)

def create_env(user_token, inherit, timeout=1):
    if False:
        print('Hello World!')
    '\n    CreateEnvironmentBlock might fail when we close a login session and then\n    try to re-open one very quickly. Run the method multiple times to work\n    around the async nature of logoffs.\n    '
    start = time.time()
    env = None
    exc = None
    while True:
        try:
            env = win32profile.CreateEnvironmentBlock(user_token, False)
        except pywintypes.error as exc:
            pass
        else:
            break
        if time.time() - start > timeout:
            break
    if env is not None:
        return env
    raise exc

def runas(cmdLine, username, password=None, cwd=None):
    if False:
        return 10
    '\n    Run a command as another user. If the process is running as an admin or\n    system account this method does not require a password. Other non\n    privileged accounts need to provide a password for the user to runas.\n    Commands are run in with the highest level privileges possible for the\n    account provided.\n    '
    try:
        (_, domain, _) = win32security.LookupAccountName(None, username)
        (username, _) = split_username(username)
    except pywintypes.error as exc:
        message = win32api.FormatMessage(exc.winerror).rstrip('\n')
        raise CommandExecutionError(message)
    access = win32security.TOKEN_QUERY | win32security.TOKEN_ADJUST_PRIVILEGES
    th = win32security.OpenProcessToken(win32api.GetCurrentProcess(), access)
    salt.platform.win.elevate_token(th)
    try:
        impersonation_token = salt.platform.win.impersonate_sid(salt.platform.win.SYSTEM_SID, session_id=0, privs=['SeTcbPrivilege'])
    except OSError:
        log.debug('Unable to impersonate SYSTEM user')
        impersonation_token = None
        win32api.CloseHandle(th)
    if not impersonation_token:
        log.debug('No impersonation token, using unprivileged runas')
        return runas_unpriv(cmdLine, username, password, cwd)
    if domain == 'NT AUTHORITY':
        user_token = win32security.LogonUser(username, domain, '', win32con.LOGON32_LOGON_SERVICE, win32con.LOGON32_PROVIDER_DEFAULT)
    elif password:
        user_token = win32security.LogonUser(username, domain, password, win32con.LOGON32_LOGON_INTERACTIVE, win32con.LOGON32_PROVIDER_DEFAULT)
    else:
        user_token = salt.platform.win.logon_msv1_s4u(username).Token
    elevation_type = win32security.GetTokenInformation(user_token, win32security.TokenElevationType)
    if elevation_type > 1:
        user_token = win32security.GetTokenInformation(user_token, win32security.TokenLinkedToken)
    salt.platform.win.elevate_token(user_token)
    salt.platform.win.grant_winsta_and_desktop(user_token)
    security_attributes = win32security.SECURITY_ATTRIBUTES()
    security_attributes.bInheritHandle = 1
    (stdin_read, stdin_write) = win32pipe.CreatePipe(security_attributes, 0)
    stdin_read = salt.platform.win.make_inheritable(stdin_read)
    (stdout_read, stdout_write) = win32pipe.CreatePipe(security_attributes, 0)
    stdout_write = salt.platform.win.make_inheritable(stdout_write)
    (stderr_read, stderr_write) = win32pipe.CreatePipe(security_attributes, 0)
    stderr_write = salt.platform.win.make_inheritable(stderr_write)
    creationflags = win32process.CREATE_NO_WINDOW | win32process.CREATE_NEW_CONSOLE | win32process.CREATE_SUSPENDED
    startup_info = salt.platform.win.STARTUPINFO(dwFlags=win32con.STARTF_USESTDHANDLES, hStdInput=stdin_read.handle, hStdOutput=stdout_write.handle, hStdError=stderr_write.handle)
    env = create_env(user_token, False)
    hProcess = None
    try:
        process_info = salt.platform.win.CreateProcessWithTokenW(int(user_token), logonflags=1, applicationname=None, commandline=cmdLine, currentdirectory=cwd, creationflags=creationflags, startupinfo=startup_info, environment=env)
        hProcess = process_info.hProcess
        hThread = process_info.hThread
        dwProcessId = process_info.dwProcessId
        dwThreadId = process_info.dwThreadId
        salt.platform.win.kernel32.CloseHandle(stdin_write.handle)
        salt.platform.win.kernel32.CloseHandle(stdout_write.handle)
        salt.platform.win.kernel32.CloseHandle(stderr_write.handle)
        ret = {'pid': dwProcessId}
        psutil.Process(dwProcessId).resume()
        if win32event.WaitForSingleObject(hProcess, win32event.INFINITE) == win32con.WAIT_OBJECT_0:
            exitcode = win32process.GetExitCodeProcess(hProcess)
            ret['retcode'] = exitcode
        fd_out = msvcrt.open_osfhandle(stdout_read.handle, os.O_RDONLY | os.O_TEXT)
        with os.fdopen(fd_out, 'r') as f_out:
            stdout = f_out.read()
            ret['stdout'] = stdout
        fd_err = msvcrt.open_osfhandle(stderr_read.handle, os.O_RDONLY | os.O_TEXT)
        with os.fdopen(fd_err, 'r') as f_err:
            stderr = f_err.read()
            ret['stderr'] = stderr
    finally:
        if hProcess is not None:
            salt.platform.win.kernel32.CloseHandle(hProcess)
        win32api.CloseHandle(th)
        win32api.CloseHandle(user_token)
        if impersonation_token:
            win32security.RevertToSelf()
        win32api.CloseHandle(impersonation_token)
    return ret

def runas_unpriv(cmd, username, password, cwd=None):
    if False:
        return 10
    '\n    Runas that works for non-privileged users\n    '
    try:
        (_, domain, _) = win32security.LookupAccountName(None, username)
        (username, _) = split_username(username)
    except pywintypes.error as exc:
        message = win32api.FormatMessage(exc.winerror).rstrip('\n')
        raise CommandExecutionError(message)
    (c2pread, c2pwrite) = salt.platform.win.CreatePipe(inherit_read=False, inherit_write=True)
    (errread, errwrite) = salt.platform.win.CreatePipe(inherit_read=False, inherit_write=True)
    stdin = salt.platform.win.kernel32.GetStdHandle(salt.platform.win.STD_INPUT_HANDLE)
    dupin = salt.platform.win.DuplicateHandle(srchandle=stdin, inherit=True)
    startup_info = salt.platform.win.STARTUPINFO(dwFlags=win32con.STARTF_USESTDHANDLES, hStdInput=dupin, hStdOutput=c2pwrite, hStdError=errwrite)
    try:
        process_info = salt.platform.win.CreateProcessWithLogonW(username=username, domain=domain, password=password, logonflags=salt.platform.win.LOGON_WITH_PROFILE, commandline=cmd, startupinfo=startup_info, currentdirectory=cwd)
        salt.platform.win.kernel32.CloseHandle(process_info.hThread)
    finally:
        salt.platform.win.kernel32.CloseHandle(dupin)
        salt.platform.win.kernel32.CloseHandle(c2pwrite)
        salt.platform.win.kernel32.CloseHandle(errwrite)
    ret = {'pid': process_info.dwProcessId}
    fd_out = msvcrt.open_osfhandle(c2pread, os.O_RDONLY | os.O_TEXT)
    with os.fdopen(fd_out, 'r') as f_out:
        ret['stdout'] = f_out.read()
    fd_err = msvcrt.open_osfhandle(errread, os.O_RDONLY | os.O_TEXT)
    with os.fdopen(fd_err, 'r') as f_err:
        ret['stderr'] = f_err.read()
    if salt.platform.win.kernel32.WaitForSingleObject(process_info.hProcess, win32event.INFINITE) == win32con.WAIT_OBJECT_0:
        exitcode = salt.platform.win.wintypes.DWORD()
        salt.platform.win.kernel32.GetExitCodeProcess(process_info.hProcess, ctypes.byref(exitcode))
        ret['retcode'] = exitcode.value
    salt.platform.win.kernel32.CloseHandle(process_info.hProcess)
    return ret