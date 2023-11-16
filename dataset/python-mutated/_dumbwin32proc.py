"""
Windows Process Management, used with reactor.spawnProcess
"""
import os
import sys
from zope.interface import implementer
import pywintypes
import win32api
import win32con
import win32event
import win32file
import win32pipe
import win32process
import win32security
from twisted.internet import _pollingfile, error
from twisted.internet._baseprocess import BaseProcess
from twisted.internet.interfaces import IConsumer, IProcessTransport, IProducer
from twisted.python.win32 import quoteArguments
PIPE_ATTRS_INHERITABLE = win32security.SECURITY_ATTRIBUTES()
PIPE_ATTRS_INHERITABLE.bInheritHandle = 1

def debug(msg):
    if False:
        i = 10
        return i + 15
    print(msg)
    sys.stdout.flush()

class _Reaper(_pollingfile._PollableResource):

    def __init__(self, proc):
        if False:
            i = 10
            return i + 15
        self.proc = proc

    def checkWork(self):
        if False:
            while True:
                i = 10
        if win32event.WaitForSingleObject(self.proc.hProcess, 0) != win32event.WAIT_OBJECT_0:
            return 0
        exitCode = win32process.GetExitCodeProcess(self.proc.hProcess)
        self.deactivate()
        self.proc.processEnded(exitCode)
        return 0

def _findShebang(filename):
    if False:
        print('Hello World!')
    "\n    Look for a #! line, and return the value following the #! if one exists, or\n    None if this file is not a script.\n\n    I don't know if there are any conventions for quoting in Windows shebang\n    lines, so this doesn't support any; therefore, you may not pass any\n    arguments to scripts invoked as filters.  That's probably wrong, so if\n    somebody knows more about the cultural expectations on Windows, please feel\n    free to fix.\n\n    This shebang line support was added in support of the CGI tests;\n    appropriately enough, I determined that shebang lines are culturally\n    accepted in the Windows world through this page::\n\n        http://www.cgi101.com/learn/connect/winxp.html\n\n    @param filename: str representing a filename\n\n    @return: a str representing another filename.\n    "
    with open(filename) as f:
        if f.read(2) == '#!':
            exe = f.readline(1024).strip('\n')
            return exe

def _invalidWin32App(pywinerr):
    if False:
        print('Hello World!')
    "\n    Determine if a pywintypes.error is telling us that the given process is\n    'not a valid win32 application', i.e. not a PE format executable.\n\n    @param pywinerr: a pywintypes.error instance raised by CreateProcess\n\n    @return: a boolean\n    "
    return pywinerr.args[0] == 193

@implementer(IProcessTransport, IConsumer, IProducer)
class Process(_pollingfile._PollingTimer, BaseProcess):
    """
    A process that integrates with the Twisted event loop.

    If your subprocess is a python program, you need to:

     - Run python.exe with the '-u' command line option - this turns on
       unbuffered I/O. Buffering stdout/err/in can cause problems, see e.g.
       http://support.microsoft.com/default.aspx?scid=kb;EN-US;q1903

     - If you don't want Windows messing with data passed over
       stdin/out/err, set the pipes to be in binary mode::

        import os, sys, mscvrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stderr.fileno(), os.O_BINARY)

    """
    closedNotifies = 0

    def __init__(self, reactor, protocol, command, args, environment, path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new child process.\n        '
        _pollingfile._PollingTimer.__init__(self, reactor)
        BaseProcess.__init__(self, protocol)
        sAttrs = win32security.SECURITY_ATTRIBUTES()
        sAttrs.bInheritHandle = 1
        (self.hStdoutR, hStdoutW) = win32pipe.CreatePipe(sAttrs, 0)
        (self.hStderrR, hStderrW) = win32pipe.CreatePipe(sAttrs, 0)
        (hStdinR, self.hStdinW) = win32pipe.CreatePipe(sAttrs, 0)
        win32pipe.SetNamedPipeHandleState(self.hStdinW, win32pipe.PIPE_NOWAIT, None, None)
        StartupInfo = win32process.STARTUPINFO()
        StartupInfo.hStdOutput = hStdoutW
        StartupInfo.hStdError = hStderrW
        StartupInfo.hStdInput = hStdinR
        StartupInfo.dwFlags = win32process.STARTF_USESTDHANDLES
        currentPid = win32api.GetCurrentProcess()
        tmp = win32api.DuplicateHandle(currentPid, self.hStdoutR, currentPid, 0, 0, win32con.DUPLICATE_SAME_ACCESS)
        win32file.CloseHandle(self.hStdoutR)
        self.hStdoutR = tmp
        tmp = win32api.DuplicateHandle(currentPid, self.hStderrR, currentPid, 0, 0, win32con.DUPLICATE_SAME_ACCESS)
        win32file.CloseHandle(self.hStderrR)
        self.hStderrR = tmp
        tmp = win32api.DuplicateHandle(currentPid, self.hStdinW, currentPid, 0, 0, win32con.DUPLICATE_SAME_ACCESS)
        win32file.CloseHandle(self.hStdinW)
        self.hStdinW = tmp
        env = os.environ.copy()
        env.update(environment or {})
        env = {os.fsdecode(key): os.fsdecode(value) for (key, value) in env.items()}
        args = [os.fsdecode(x) for x in args]
        cmdline = quoteArguments(args)
        command = os.fsdecode(command) if command else command
        path = os.fsdecode(path) if path else path

        def doCreate():
            if False:
                for i in range(10):
                    print('nop')
            flags = win32con.CREATE_NO_WINDOW
            (self.hProcess, self.hThread, self.pid, dwTid) = win32process.CreateProcess(command, cmdline, None, None, 1, flags, env, path, StartupInfo)
        try:
            doCreate()
        except pywintypes.error as pwte:
            if not _invalidWin32App(pwte):
                raise OSError(pwte)
            else:
                sheb = _findShebang(command)
                if sheb is None:
                    raise OSError('%r is neither a Windows executable, nor a script with a shebang line' % command)
                else:
                    args = list(args)
                    args.insert(0, command)
                    cmdline = quoteArguments(args)
                    origcmd = command
                    command = sheb
                    try:
                        doCreate()
                    except pywintypes.error as pwte2:
                        if _invalidWin32App(pwte2):
                            raise OSError('%r has an invalid shebang line: %r is not a valid executable' % (origcmd, sheb))
                        raise OSError(pwte2)
        win32file.CloseHandle(hStderrW)
        win32file.CloseHandle(hStdoutW)
        win32file.CloseHandle(hStdinR)
        self.stdout = _pollingfile._PollableReadPipe(self.hStdoutR, lambda data: self.proto.childDataReceived(1, data), self.outConnectionLost)
        self.stderr = _pollingfile._PollableReadPipe(self.hStderrR, lambda data: self.proto.childDataReceived(2, data), self.errConnectionLost)
        self.stdin = _pollingfile._PollableWritePipe(self.hStdinW, self.inConnectionLost)
        for pipewatcher in (self.stdout, self.stderr, self.stdin):
            self._addPollableResource(pipewatcher)
        self.proto.makeConnection(self)
        self._addPollableResource(_Reaper(self))

    def signalProcess(self, signalID):
        if False:
            i = 10
            return i + 15
        if self.pid is None:
            raise error.ProcessExitedAlready()
        if signalID in ('INT', 'TERM', 'KILL'):
            win32process.TerminateProcess(self.hProcess, 1)

    def _getReason(self, status):
        if False:
            print('Hello World!')
        if status == 0:
            return error.ProcessDone(status)
        return error.ProcessTerminated(status)

    def write(self, data):
        if False:
            return 10
        "\n        Write data to the process' stdin.\n\n        @type data: C{bytes}\n        "
        self.stdin.write(data)

    def writeSequence(self, seq):
        if False:
            return 10
        "\n        Write data to the process' stdin.\n\n        @type seq: C{list} of C{bytes}\n        "
        self.stdin.writeSequence(seq)

    def writeToChild(self, fd, data):
        if False:
            return 10
        "\n        Similar to L{ITransport.write} but also allows the file descriptor in\n        the child process which will receive the bytes to be specified.\n\n        This implementation is limited to writing to the child's standard input.\n\n        @param fd: The file descriptor to which to write.  Only stdin (C{0}) is\n            supported.\n        @type fd: C{int}\n\n        @param data: The bytes to write.\n        @type data: C{bytes}\n\n        @return: L{None}\n\n        @raise KeyError: If C{fd} is anything other than the stdin file\n            descriptor (C{0}).\n        "
        if fd == 0:
            self.stdin.write(data)
        else:
            raise KeyError(fd)

    def closeChildFD(self, fd):
        if False:
            for i in range(10):
                print('nop')
        if fd == 0:
            self.closeStdin()
        elif fd == 1:
            self.closeStdout()
        elif fd == 2:
            self.closeStderr()
        else:
            raise NotImplementedError('Only standard-IO file descriptors available on win32')

    def closeStdin(self):
        if False:
            for i in range(10):
                print('nop')
        "Close the process' stdin."
        self.stdin.close()

    def closeStderr(self):
        if False:
            print('Hello World!')
        self.stderr.close()

    def closeStdout(self):
        if False:
            print('Hello World!')
        self.stdout.close()

    def loseConnection(self):
        if False:
            return 10
        "\n        Close the process' stdout, in and err.\n        "
        self.closeStdin()
        self.closeStdout()
        self.closeStderr()

    def outConnectionLost(self):
        if False:
            for i in range(10):
                print('nop')
        self.proto.childConnectionLost(1)
        self.connectionLostNotify()

    def errConnectionLost(self):
        if False:
            print('Hello World!')
        self.proto.childConnectionLost(2)
        self.connectionLostNotify()

    def inConnectionLost(self):
        if False:
            i = 10
            return i + 15
        self.proto.childConnectionLost(0)
        self.connectionLostNotify()

    def connectionLostNotify(self):
        if False:
            while True:
                i = 10
        '\n        Will be called 3 times, by stdout/err threads and process handle.\n        '
        self.closedNotifies += 1
        self.maybeCallProcessEnded()

    def maybeCallProcessEnded(self):
        if False:
            for i in range(10):
                print('nop')
        if self.closedNotifies == 3 and self.lostProcess:
            win32file.CloseHandle(self.hProcess)
            win32file.CloseHandle(self.hThread)
            self.hProcess = None
            self.hThread = None
            BaseProcess.maybeCallProcessEnded(self)

    def registerProducer(self, producer, streaming):
        if False:
            while True:
                i = 10
        self.stdin.registerProducer(producer, streaming)

    def unregisterProducer(self):
        if False:
            while True:
                i = 10
        self.stdin.unregisterProducer()

    def pauseProducing(self):
        if False:
            return 10
        self._pause()

    def resumeProducing(self):
        if False:
            return 10
        self._unpause()

    def stopProducing(self):
        if False:
            for i in range(10):
                print('nop')
        self.loseConnection()

    def getHost(self):
        if False:
            return 10
        raise NotImplementedError('Unimplemented: Process.getHost')

    def getPeer(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('Unimplemented: Process.getPeer')

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        '\n        Return a string representation of the process.\n        '
        return f'<{self.__class__.__name__} pid={self.pid}>'