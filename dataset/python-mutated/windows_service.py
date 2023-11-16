import os
import sys
import threading
import pywintypes
import servicemanager
import win32api
import win32con
import win32event
import win32file
import win32pipe
import win32process
import win32security
import win32service
import win32serviceutil
import winerror
is_frozen = hasattr(sys, 'frozen')
CHILDCAPTURE_BLOCK_SIZE = 80
CHILDCAPTURE_MAX_BLOCKS = 200

class BBService(win32serviceutil.ServiceFramework):
    _svc_name_ = 'BuildBot'
    _svc_display_name_ = _svc_name_
    _svc_description_ = 'Manages local buildbot workers and masters - see http://buildbot.net'

    def __init__(self, args):
        if False:
            while True:
                i = 10
        super().__init__(args)
        sa = win32security.SECURITY_ATTRIBUTES()
        sa.bInheritHandle = True
        self.hWaitStop = win32event.CreateEvent(sa, True, False, None)
        self.args = args
        self.dirs = None
        self.runner_prefix = None
        if is_frozen and servicemanager.RunningAsService():
            msg_file = os.path.join(os.path.dirname(sys.executable), 'buildbot.msg')
            if os.path.isfile(msg_file):
                servicemanager.Initialize('BuildBot', msg_file)
            else:
                self.warning(f"Strange - '{msg_file}' does not exist")

    def _checkConfig(self):
        if False:
            while True:
                i = 10
        if not is_frozen:
            python_exe = os.path.join(sys.prefix, 'python.exe')
            if not os.path.isfile(python_exe):
                python_exe = os.path.join(sys.prefix, 'PCBuild', 'python.exe')
            if not os.path.isfile(python_exe):
                python_exe = os.path.join(sys.prefix, 'Scripts', 'python.exe')
            if not os.path.isfile(python_exe):
                self.error('Can not find python.exe to spawn subprocess')
                return False
            me = __file__
            if me.endswith('.pyc') or me.endswith('.pyo'):
                me = me[:-1]
            self.runner_prefix = f'"{python_exe}" "{me}"'
        else:
            self.runner_prefix = '"' + sys.executable + '"'
        self.dirs = []
        if len(self.args) > 1:
            dir_string = os.pathsep.join(self.args[1:])
            save_dirs = True
        else:
            dir_string = win32serviceutil.GetServiceCustomOption(self, 'directories')
            save_dirs = False
        if not dir_string:
            self.error('You must specify the buildbot directories as parameters to the service.\nStopping the service.')
            return False
        dirs = dir_string.split(os.pathsep)
        for d in dirs:
            d = os.path.abspath(d)
            sentinal = os.path.join(d, 'buildbot.tac')
            if os.path.isfile(sentinal):
                self.dirs.append(d)
            else:
                msg = f"Directory '{d}' is not a buildbot dir - ignoring"
                self.warning(msg)
        if not self.dirs:
            self.error('No valid buildbot directories were specified.\nStopping the service.')
            return False
        if save_dirs:
            dir_string = os.pathsep.join(self.dirs)
            win32serviceutil.SetServiceCustomOption(self, 'directories', dir_string)
        return True

    def SvcStop(self):
        if False:
            return 10
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
    SvcShutdown = SvcStop

    def SvcDoRun(self):
        if False:
            return 10
        if not self._checkConfig():
            return
        self.logmsg(servicemanager.PYS_SERVICE_STARTED)
        child_infos = []
        for bbdir in self.dirs:
            self.info(f"Starting BuildBot in directory '{bbdir}'")
            hstop = int(self.hWaitStop)
            cmd = f'{self.runner_prefix} --spawn {hstop} start --nodaemon {bbdir}'
            (h, t, output) = self.createProcess(cmd)
            child_infos.append((bbdir, h, t, output))
        while child_infos:
            handles = [self.hWaitStop] + [i[1] for i in child_infos]
            rc = win32event.WaitForMultipleObjects(handles, 0, win32event.INFINITE)
            if rc == win32event.WAIT_OBJECT_0:
                break
            index = rc - win32event.WAIT_OBJECT_0 - 1
            (bbdir, dead_handle, _, output_blocks) = child_infos[index]
            status = win32process.GetExitCodeProcess(dead_handle)
            output = ''.join(output_blocks)
            if not output:
                output = 'The child process generated no output. Please check the twistd.log file in the indicated directory.'
            self.warning(f'BuildBot for directory {repr(bbdir)} terminated with exit code {status}.\n{output}')
            del child_infos[index]
            if not child_infos:
                self.warning('All BuildBot child processes have terminated.  Service stopping.')
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        for (bbdir, h, t, output) in child_infos:
            for _ in range(10):
                self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
                rc = win32event.WaitForSingleObject(h, 3000)
                if rc == win32event.WAIT_OBJECT_0:
                    break
            if rc == win32event.WAIT_OBJECT_0:
                break
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            if win32process.GetExitCodeProcess(h) == win32con.STILL_ACTIVE:
                self.warning(f'BuildBot process at {repr(bbdir)} failed to terminate - killing it')
                win32api.TerminateProcess(h, 3)
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
            for _ in range(5):
                t.join(1)
                self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
                if not t.is_alive():
                    break
            else:
                self.warning('Redirect thread did not stop!')
        self.logmsg(servicemanager.PYS_SERVICE_STOPPED)

    def logmsg(self, event):
        if False:
            i = 10
            return i + 15
        try:
            servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE, event, (self._svc_name_, f' ({self._svc_display_name_})'))
        except win32api.error as details:
            try:
                print('FAILED to write INFO event', event, ':', details)
            except IOError:
                pass

    def _dolog(self, func, msg):
        if False:
            print('Hello World!')
        try:
            func(msg)
        except win32api.error as details:
            try:
                print('FAILED to write event log entry:', details)
                print(msg)
            except IOError:
                pass

    def info(self, s):
        if False:
            i = 10
            return i + 15
        self._dolog(servicemanager.LogInfoMsg, s)

    def warning(self, s):
        if False:
            return 10
        self._dolog(servicemanager.LogWarningMsg, s)

    def error(self, s):
        if False:
            i = 10
            return i + 15
        self._dolog(servicemanager.LogErrorMsg, s)

    def createProcess(self, cmd):
        if False:
            i = 10
            return i + 15
        (hInputRead, hInputWriteTemp) = self.newPipe()
        (hOutReadTemp, hOutWrite) = self.newPipe()
        pid = win32api.GetCurrentProcess()
        hErrWrite = win32api.DuplicateHandle(pid, hOutWrite, pid, 0, 1, win32con.DUPLICATE_SAME_ACCESS)
        hOutRead = self.dup(hOutReadTemp)
        hInputWrite = self.dup(hInputWriteTemp)
        si = win32process.STARTUPINFO()
        si.hStdInput = hInputRead
        si.hStdOutput = hOutWrite
        si.hStdError = hErrWrite
        si.dwFlags = win32process.STARTF_USESTDHANDLES | win32process.STARTF_USESHOWWINDOW
        si.wShowWindow = win32con.SW_HIDE
        create_flags = win32process.CREATE_NEW_CONSOLE
        info = win32process.CreateProcess(None, cmd, None, None, True, create_flags, None, None, si)
        hOutWrite.Close()
        hErrWrite.Close()
        hInputRead.Close()
        hInputWrite.Close()
        blocks = []
        t = threading.Thread(target=self.redirectCaptureThread, args=(hOutRead, blocks))
        t.start()
        return (info[0], t, blocks)

    def redirectCaptureThread(self, handle, captured_blocks):
        if False:
            while True:
                i = 10
        while True:
            try:
                (_, data) = win32file.ReadFile(handle, CHILDCAPTURE_BLOCK_SIZE)
            except pywintypes.error as err:
                if err[0] != winerror.ERROR_BROKEN_PIPE:
                    self.warning(f'Error reading output from process: {err}')
                break
            captured_blocks.append(data)
            del captured_blocks[CHILDCAPTURE_MAX_BLOCKS:]
        handle.Close()

    def newPipe(self):
        if False:
            return 10
        sa = win32security.SECURITY_ATTRIBUTES()
        sa.bInheritHandle = True
        return win32pipe.CreatePipe(sa, 0)

    def dup(self, pipe):
        if False:
            i = 10
            return i + 15
        pid = win32api.GetCurrentProcess()
        dup = win32api.DuplicateHandle(pid, pipe, pid, 0, 0, win32con.DUPLICATE_SAME_ACCESS)
        pipe.Close()
        return dup

def RegisterWithFirewall(exe_name, description):
    if False:
        i = 10
        return i + 15
    from win32com.client import Dispatch
    NET_FW_SCOPE_ALL = 0
    NET_FW_IP_VERSION_ANY = 2
    fwMgr = Dispatch('HNetCfg.FwMgr')
    profile = fwMgr.LocalPolicy.CurrentProfile
    app = Dispatch('HNetCfg.FwAuthorizedApplication')
    app.ProcessImageFileName = exe_name
    app.Name = description
    app.Scope = NET_FW_SCOPE_ALL
    app.IpVersion = NET_FW_IP_VERSION_ANY
    app.Enabled = True
    profile.AuthorizedApplications.Add(app)

def CustomInstall(opts):
    if False:
        return 10
    import pythoncom
    try:
        RegisterWithFirewall(sys.executable, 'BuildBot')
    except pythoncom.com_error as why:
        print('FAILED to register with the Windows firewall')
        print(why)

def _RunChild(runfn):
    if False:
        return 10
    del sys.argv[1]
    t = threading.Thread(target=_WaitForShutdown, args=(int(sys.argv[1]),))
    del sys.argv[1]

    def ConsoleHandler(what):
        if False:
            while True:
                i = 10
        return True
    win32api.SetConsoleCtrlHandler(ConsoleHandler, True)
    t.setDaemon(True)
    t.start()
    if hasattr(sys, 'frozen'):
        del os.environ['PYTHONPATH']
    runfn()
    print('Service child process terminating normally.')

def _WaitForShutdown(h):
    if False:
        for i in range(10):
            print('nop')
    win32event.WaitForSingleObject(h, win32event.INFINITE)
    print('Shutdown requested')
    from twisted.internet import reactor
    reactor.callLater(0, reactor.stop)

def DetermineRunner(bbdir):
    if False:
        return 10
    'Checks if the given directory is a worker or a master and returns the\n    appropriate run function.'
    tacfile = os.path.join(bbdir, 'buildbot.tac')
    if not os.path.exists(tacfile):
        import buildbot.scripts.runner
        return buildbot.scripts.runner.run
    with open(tacfile, 'r', encoding='utf-8') as f:
        contents = f.read()
    try:
        if 'import Worker' in contents:
            import buildbot_worker.scripts.runner
            return buildbot_worker.scripts.runner.run
    except ImportError:
        pass
    try:
        if 'import BuildSlave' in contents:
            import buildslave.scripts.runner
            return buildslave.scripts.runner.run
    except ImportError:
        pass
    import buildbot.scripts.runner
    return buildbot.scripts.runner.run

def HandleCommandLine():
    if False:
        while True:
            i = 10
    if len(sys.argv) > 1 and sys.argv[1] == '--spawn':
        _RunChild(DetermineRunner(sys.argv[5]))
    else:
        win32serviceutil.HandleCommandLine(BBService, customOptionHandler=CustomInstall)
if __name__ == '__main__':
    HandleCommandLine()