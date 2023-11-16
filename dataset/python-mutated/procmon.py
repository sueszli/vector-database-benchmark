"""
Support for starting, monitoring, and restarting child process.
"""
from typing import Dict, List, Optional
import attr
import incremental
from twisted.application import service
from twisted.internet import error, protocol, reactor as _reactor
from twisted.logger import Logger
from twisted.protocols import basic
from twisted.python import deprecate

@attr.s(frozen=True, auto_attribs=True)
class _Process:
    """
    The parameters of a process to be restarted.

    @ivar args: command-line arguments (including name of command as first one)
    @type args: C{list}

    @ivar uid: user-id to run process as, or None (which means inherit uid)
    @type uid: C{int}

    @ivar gid: group-id to run process as, or None (which means inherit gid)
    @type gid: C{int}

    @ivar env: environment for process
    @type env: C{dict}

    @ivar cwd: initial working directory for process or None
               (which means inherit cwd)
    @type cwd: C{str}
    """
    args: List[str]
    uid: Optional[int] = None
    gid: Optional[int] = None
    env: Dict[str, str] = attr.ib(default=attr.Factory(dict))
    cwd: Optional[str] = None

    @deprecate.deprecated(incremental.Version('Twisted', 18, 7, 0))
    def toTuple(self):
        if False:
            return 10
        '\n        Convert process to tuple.\n\n        Convert process to tuple that looks like the legacy structure\n        of processes, for potential users who inspected processes\n        directly.\n\n        This was only an accidental feature, and will be removed. If\n        you need to remember what processes were added to a process monitor,\n        keep track of that when they are added. The process list\n        inside the process monitor is no longer a public API.\n\n        This allows changing the internal structure of the process list,\n        when warranted by bug fixes or additional features.\n\n        @return: tuple representation of process\n        '
        return (self.args, self.uid, self.gid, self.env)

class DummyTransport:
    disconnecting = 0
transport = DummyTransport()

class LineLogger(basic.LineReceiver):
    tag = None
    stream = None
    delimiter = b'\n'
    service = None

    def lineReceived(self, line):
        if False:
            return 10
        try:
            line = line.decode('utf-8')
        except UnicodeDecodeError:
            line = repr(line)
        self.service.log.info('[{tag}] {line}', tag=self.tag, line=line, stream=self.stream)

class LoggingProtocol(protocol.ProcessProtocol):
    service = None
    name = None

    def connectionMade(self):
        if False:
            while True:
                i = 10
        self._output = LineLogger()
        self._output.tag = self.name
        self._output.stream = 'stdout'
        self._output.service = self.service
        self._outputEmpty = True
        self._error = LineLogger()
        self._error.tag = self.name
        self._error.stream = 'stderr'
        self._error.service = self.service
        self._errorEmpty = True
        self._output.makeConnection(transport)
        self._error.makeConnection(transport)

    def outReceived(self, data):
        if False:
            while True:
                i = 10
        self._output.dataReceived(data)
        self._outputEmpty = data[-1] == b'\n'

    def errReceived(self, data):
        if False:
            for i in range(10):
                print('nop')
        self._error.dataReceived(data)
        self._errorEmpty = data[-1] == b'\n'

    def processEnded(self, reason):
        if False:
            for i in range(10):
                print('nop')
        if not self._outputEmpty:
            self._output.dataReceived(b'\n')
        if not self._errorEmpty:
            self._error.dataReceived(b'\n')
        self.service.connectionLost(self.name)

    @property
    def output(self):
        if False:
            while True:
                i = 10
        return self._output

    @property
    def empty(self):
        if False:
            for i in range(10):
                print('nop')
        return self._outputEmpty

class ProcessMonitor(service.Service):
    """
    ProcessMonitor runs processes, monitors their progress, and restarts
    them when they die.

    The ProcessMonitor will not attempt to restart a process that appears to
    die instantly -- with each "instant" death (less than 1 second, by
    default), it will delay approximately twice as long before restarting
    it.  A successful run will reset the counter.

    The primary interface is L{addProcess} and L{removeProcess}. When the
    service is running (that is, when the application it is attached to is
    running), adding a process automatically starts it.

    Each process has a name. This name string must uniquely identify the
    process.  In particular, attempting to add two processes with the same
    name will result in a C{KeyError}.

    @type threshold: C{float}
    @ivar threshold: How long a process has to live before the death is
        considered instant, in seconds.  The default value is 1 second.

    @type killTime: C{float}
    @ivar killTime: How long a process being killed has to get its affairs
        in order before it gets killed with an unmaskable signal.  The
        default value is 5 seconds.

    @type minRestartDelay: C{float}
    @ivar minRestartDelay: The minimum time (in seconds) to wait before
        attempting to restart a process.  Default 1s.

    @type maxRestartDelay: C{float}
    @ivar maxRestartDelay: The maximum time (in seconds) to wait before
        attempting to restart a process.  Default 3600s (1h).

    @type _reactor: L{IReactorProcess} provider
    @ivar _reactor: A provider of L{IReactorProcess} and L{IReactorTime}
        which will be used to spawn processes and register delayed calls.

    @type log: L{Logger}
    @ivar log: The logger used to propagate log messages from spawned
        processes.

    """
    threshold = 1
    killTime = 5
    minRestartDelay = 1
    maxRestartDelay = 3600
    log = Logger()

    def __init__(self, reactor=_reactor):
        if False:
            i = 10
            return i + 15
        self._reactor = reactor
        self._processes = {}
        self.protocols = {}
        self.delay = {}
        self.timeStarted = {}
        self.murder = {}
        self.restart = {}

    @deprecate.deprecatedProperty(incremental.Version('Twisted', 18, 7, 0))
    def processes(self):
        if False:
            print('Hello World!')
        '\n        Processes as dict of tuples\n\n        @return: Dict of process name to monitored processes as tuples\n        '
        return {name: process.toTuple() for (name, process) in self._processes.items()}

    @deprecate.deprecated(incremental.Version('Twisted', 18, 7, 0))
    def __getstate__(self):
        if False:
            while True:
                i = 10
        dct = service.Service.__getstate__(self)
        del dct['_reactor']
        dct['protocols'] = {}
        dct['delay'] = {}
        dct['timeStarted'] = {}
        dct['murder'] = {}
        dct['restart'] = {}
        del dct['_processes']
        dct['processes'] = self.processes
        return dct

    def addProcess(self, name, args, uid=None, gid=None, env={}, cwd=None):
        if False:
            return 10
        '\n        Add a new monitored process and start it immediately if the\n        L{ProcessMonitor} service is running.\n\n        Note that args are passed to the system call, not to the shell. If\n        running the shell is desired, the common idiom is to use\n        C{ProcessMonitor.addProcess("name", [\'/bin/sh\', \'-c\', shell_script])}\n\n        @param name: A name for this process.  This value must be\n            unique across all processes added to this monitor.\n        @type name: C{str}\n        @param args: The argv sequence for the process to launch.\n        @param uid: The user ID to use to run the process.  If L{None},\n            the current UID is used.\n        @type uid: C{int}\n        @param gid: The group ID to use to run the process.  If L{None},\n            the current GID is used.\n        @type uid: C{int}\n        @param env: The environment to give to the launched process. See\n            L{IReactorProcess.spawnProcess}\'s C{env} parameter.\n        @type env: C{dict}\n        @param cwd: The initial working directory of the launched process.\n            The default of C{None} means inheriting the laucnhing process\'s\n            working directory.\n        @type env: C{dict}\n        @raise KeyError: If a process with the given name already exists.\n        '
        if name in self._processes:
            raise KeyError(f'remove {name} first')
        self._processes[name] = _Process(args, uid, gid, env, cwd)
        self.delay[name] = self.minRestartDelay
        if self.running:
            self.startProcess(name)

    def removeProcess(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Stop the named process and remove it from the list of monitored\n        processes.\n\n        @type name: C{str}\n        @param name: A string that uniquely identifies the process.\n        '
        self.stopProcess(name)
        del self._processes[name]

    def startService(self):
        if False:
            i = 10
            return i + 15
        '\n        Start all monitored processes.\n        '
        service.Service.startService(self)
        for name in list(self._processes):
            self.startProcess(name)

    def stopService(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Stop all monitored processes and cancel all scheduled process restarts.\n        '
        service.Service.stopService(self)
        for (name, delayedCall) in list(self.restart.items()):
            if delayedCall.active():
                delayedCall.cancel()
        for name in list(self._processes):
            self.stopProcess(name)

    def connectionLost(self, name):
        if False:
            return 10
        '\n        Called when a monitored processes exits. If\n        L{service.IService.running} is L{True} (ie the service is started), the\n        process will be restarted.\n        If the process had been running for more than\n        L{ProcessMonitor.threshold} seconds it will be restarted immediately.\n        If the process had been running for less than\n        L{ProcessMonitor.threshold} seconds, the restart will be delayed and\n        each time the process dies before the configured threshold, the restart\n        delay will be doubled - up to a maximum delay of maxRestartDelay sec.\n\n        @type name: C{str}\n        @param name: A string that uniquely identifies the process\n            which exited.\n        '
        if name in self.murder:
            if self.murder[name].active():
                self.murder[name].cancel()
            del self.murder[name]
        del self.protocols[name]
        if self._reactor.seconds() - self.timeStarted[name] < self.threshold:
            nextDelay = self.delay[name]
            self.delay[name] = min(self.delay[name] * 2, self.maxRestartDelay)
        else:
            nextDelay = 0
            self.delay[name] = self.minRestartDelay
        if self.running and name in self._processes:
            self.restart[name] = self._reactor.callLater(nextDelay, self.startProcess, name)

    def startProcess(self, name):
        if False:
            i = 10
            return i + 15
        '\n        @param name: The name of the process to be started\n        '
        if name in self.protocols:
            return
        process = self._processes[name]
        proto = LoggingProtocol()
        proto.service = self
        proto.name = name
        self.protocols[name] = proto
        self.timeStarted[name] = self._reactor.seconds()
        self._reactor.spawnProcess(proto, process.args[0], process.args, uid=process.uid, gid=process.gid, env=process.env, path=process.cwd)

    def _forceStopProcess(self, proc):
        if False:
            for i in range(10):
                print('nop')
        '\n        @param proc: An L{IProcessTransport} provider\n        '
        try:
            proc.signalProcess('KILL')
        except error.ProcessExitedAlready:
            pass

    def stopProcess(self, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        @param name: The name of the process to be stopped\n        '
        if name not in self._processes:
            raise KeyError(f'Unrecognized process name: {name}')
        proto = self.protocols.get(name, None)
        if proto is not None:
            proc = proto.transport
            try:
                proc.signalProcess('TERM')
            except error.ProcessExitedAlready:
                pass
            else:
                self.murder[name] = self._reactor.callLater(self.killTime, self._forceStopProcess, proc)

    def restartAll(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Restart all processes. This is useful for third party management\n        services to allow a user to restart servers because of an outside change\n        in circumstances -- for example, a new version of a library is\n        installed.\n        '
        for name in self._processes:
            self.stopProcess(name)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        lst = []
        for (name, proc) in self._processes.items():
            uidgid = ''
            if proc.uid is not None:
                uidgid = str(proc.uid)
            if proc.gid is not None:
                uidgid += ':' + str(proc.gid)
            if uidgid:
                uidgid = '(' + uidgid + ')'
            lst.append(f'{name!r}{uidgid}: {proc.args!r}')
        return '<' + self.__class__.__name__ + ' ' + ' '.join(lst) + '>'