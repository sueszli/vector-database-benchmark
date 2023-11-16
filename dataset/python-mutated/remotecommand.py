from twisted.internet import defer
from twisted.internet import error
from twisted.python import log
from twisted.python.failure import Failure
from twisted.spread import pb
from buildbot import util
from buildbot.pbutil import decode
from buildbot.process import metrics
from buildbot.process.results import CANCELLED
from buildbot.process.results import FAILURE
from buildbot.process.results import SUCCESS
from buildbot.util import lineboundaries
from buildbot.util.eventual import eventually
from buildbot.worker.protocols import base

class RemoteException(Exception):
    pass

class RemoteCommand(base.RemoteCommandImpl):
    _commandCounter = 0
    active = False
    rc = None
    debug = False

    def __init__(self, remote_command, args, ignore_updates=False, collectStdout=False, collectStderr=False, decodeRC=None, stdioLogName='stdio'):
        if False:
            for i in range(10):
                print('nop')
        if decodeRC is None:
            decodeRC = {0: SUCCESS}
        self.logs = {}
        self.delayedLogs = {}
        self._closeWhenFinished = {}
        self.collectStdout = collectStdout
        self.collectStderr = collectStderr
        self.stdout = ''
        self.stderr = ''
        self.updates = {}
        self.stdioLogName = stdioLogName
        self._startTime = None
        self._remoteElapsed = None
        self.remote_failure_reason = None
        self.remote_command = remote_command
        self.args = args
        self.ignore_updates = ignore_updates
        self.decodeRC = decodeRC
        self.conn = None
        self._is_conn_test_fake = False
        self.worker = None
        self.step = None
        self.builder_name = None
        self.commandID = None
        self.deferred = None
        self.interrupted = False
        self.loglock = defer.DeferredLock()
        self._line_boundary_finders = {}

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f"<RemoteCommand '{self.remote_command}' at {id(self)}>"

    @classmethod
    def generate_new_command_id(cls):
        if False:
            while True:
                i = 10
        cmd_id = cls._commandCounter
        cls._commandCounter += 1
        return f'{cmd_id}'

    @classmethod
    def get_last_generated_command_id(cls):
        if False:
            i = 10
            return i + 15
        cmd_id = cls._commandCounter - 1
        return f'{cmd_id}'

    def run(self, step, conn, builder_name):
        if False:
            print('Hello World!')
        self.active = True
        self.step = step
        self.conn = conn
        self.builder_name = builder_name
        self._is_conn_test_fake = hasattr(self.conn, 'is_fake_test_connection')
        self.commandID = RemoteCommand.generate_new_command_id()
        log.msg(f'{self}: RemoteCommand.run [{self.commandID}]')
        self.deferred = defer.Deferred()
        d = defer.maybeDeferred(self._start)
        d.addErrback(self._finished)
        return self.deferred

    def useLog(self, log_, closeWhenFinished=False, logfileName=None):
        if False:
            return 10
        if not logfileName:
            logfileName = log_.getName()
        assert logfileName not in self.logs
        assert logfileName not in self.delayedLogs
        self.logs[logfileName] = log_
        self._closeWhenFinished[logfileName] = closeWhenFinished

    def useLogDelayed(self, logfileName, activateCallBack, closeWhenFinished=False):
        if False:
            while True:
                i = 10
        assert logfileName not in self.logs
        assert logfileName not in self.delayedLogs
        self.delayedLogs[logfileName] = (activateCallBack, closeWhenFinished)

    def _start(self):
        if False:
            for i in range(10):
                print('nop')
        self._startTime = util.now()
        d = self.conn.remoteStartCommand(self, self.builder_name, self.commandID, self.remote_command, self.args)
        return d

    @defer.inlineCallbacks
    def _finished(self, failure=None):
        if False:
            for i in range(10):
                print('nop')
        if not self.active:
            return
        self.active = False
        if not self._is_conn_test_fake:
            timeout = 10
            while self.rc is None and timeout > 0:
                yield util.asyncSleep(0.1)
                timeout -= 1
        try:
            yield self.remoteComplete(failure)
            self.deferred.callback(self)
        except Exception as e:
            self.deferred.errback(e)

    @defer.inlineCallbacks
    def interrupt(self, why):
        if False:
            while True:
                i = 10
        log.msg('RemoteCommand.interrupt', self, why)
        if self.conn and isinstance(why, Failure) and why.check(error.ConnectionLost):
            log.msg('RemoteCommand.interrupt: lost worker')
            self.conn = None
            self._finished(why)
            return
        if not self.active or self.interrupted:
            log.msg(' but this RemoteCommand is already inactive')
            return
        if not self.conn:
            log.msg(' but our .conn went away')
            return
        self.interrupted = True
        try:
            yield self.conn.remoteInterruptCommand(self.builder_name, self.commandID, str(why))
        except Exception as e:
            log.msg('RemoteCommand.interrupt failed', self, e)

    def remote_update_msgpack(self, updates):
        if False:
            return 10
        self.worker.messageReceivedFromWorker()
        try:
            for (key, value) in updates:
                if self.active and (not self.ignore_updates):
                    if key in ['stdout', 'stderr', 'header']:
                        self.remoteUpdate(key, value[0], False)
                    elif key == 'log':
                        (logname, data) = value
                        self.remoteUpdate(key, (logname, data[0]), False)
                    else:
                        self.remoteUpdate(key, value, False)
        except Exception:
            self._finished(Failure())

    def split_line(self, stream, text):
        if False:
            while True:
                i = 10
        try:
            return self._line_boundary_finders[stream].append(text)
        except KeyError:
            lbf = self._line_boundary_finders[stream] = lineboundaries.LineBoundaryFinder()
            return lbf.append(text)

    def remote_update(self, updates):
        if False:
            while True:
                i = 10
        "\n        I am called by the worker's\n        L{buildbot_worker.base.WorkerForBuilderBase.sendUpdate} so\n        I can receive updates from the running remote command.\n\n        @type  updates: list of [object, int]\n        @param updates: list of updates from the remote command\n        "
        updates = decode(updates)
        self.worker.messageReceivedFromWorker()
        max_updatenum = 0
        for (update, num) in updates:
            try:
                if self.active and (not self.ignore_updates):
                    for (key, value) in update.items():
                        if key in ['stdout', 'stderr', 'header']:
                            whole_line = self.split_line(key, value)
                            if whole_line is not None:
                                self.remoteUpdate(key, whole_line, False)
                        elif key == 'log':
                            (logname, data) = value
                            whole_line = self.split_line(logname, data)
                            value = (logname, whole_line)
                            if whole_line is not None:
                                self.remoteUpdate(key, value, False)
                        else:
                            self.remoteUpdate(key, value, False)
            except Exception:
                self._finished(Failure())
            if num > max_updatenum:
                max_updatenum = num
        return max_updatenum

    def remote_complete(self, failure=None):
        if False:
            return 10
        "\n        Called by the worker's\n        L{buildbot_worker.base.WorkerForBuilderBase.commandComplete} to\n        notify me the remote command has finished.\n\n        @type  failure: L{twisted.python.failure.Failure} or None\n\n        @rtype: None\n        "
        self.worker.messageReceivedFromWorker()
        if self.active:
            eventually(self._finished, failure)
        return None

    @util.deferredLocked('loglock')
    def addStdout(self, data):
        if False:
            for i in range(10):
                print('nop')
        if self.collectStdout:
            self.stdout += data
        if self.stdioLogName is not None and self.stdioLogName in self.logs:
            self.logs[self.stdioLogName].addStdout(data)
        return defer.succeed(None)

    @util.deferredLocked('loglock')
    def add_stdout_lines(self, data, is_flushed):
        if False:
            return 10
        if self.collectStdout:
            if is_flushed:
                data = data[:-1]
            self.stdout += data
        if self.stdioLogName is not None and self.stdioLogName in self.logs:
            self.logs[self.stdioLogName].add_stdout_lines(data)
        return defer.succeed(None)

    @util.deferredLocked('loglock')
    def addStderr(self, data):
        if False:
            print('Hello World!')
        if self.collectStderr:
            self.stderr += data
        if self.stdioLogName is not None and self.stdioLogName in self.logs:
            self.logs[self.stdioLogName].addStderr(data)
        return defer.succeed(None)

    @util.deferredLocked('loglock')
    def add_stderr_lines(self, data, is_flushed):
        if False:
            print('Hello World!')
        if self.collectStderr:
            if is_flushed:
                data = data[:-1]
            self.stderr += data
        if self.stdioLogName is not None and self.stdioLogName in self.logs:
            self.logs[self.stdioLogName].add_stderr_lines(data)
        return defer.succeed(None)

    @util.deferredLocked('loglock')
    def addHeader(self, data):
        if False:
            print('Hello World!')
        if self.stdioLogName is not None and self.stdioLogName in self.logs:
            self.logs[self.stdioLogName].addHeader(data)
        return defer.succeed(None)

    @util.deferredLocked('loglock')
    def add_header_lines(self, data):
        if False:
            return 10
        if self.stdioLogName is not None and self.stdioLogName in self.logs:
            self.logs[self.stdioLogName].add_header_lines(data)
        return defer.succeed(None)

    @util.deferredLocked('loglock')
    @defer.inlineCallbacks
    def addToLog(self, logname, data):
        if False:
            for i in range(10):
                print('nop')
        if logname in self.delayedLogs:
            (activateCallBack, closeWhenFinished) = self.delayedLogs[logname]
            del self.delayedLogs[logname]
            loog = (yield activateCallBack(self))
            self.logs[logname] = loog
            self._closeWhenFinished[logname] = closeWhenFinished
        if logname in self.logs:
            yield self.logs[logname].add_stdout_lines(data)
        else:
            log.msg(f'{self}.addToLog: no such log {logname}')

    @metrics.countMethod('RemoteCommand.remoteUpdate()')
    @defer.inlineCallbacks
    def remoteUpdate(self, key, value, is_flushed):
        if False:
            print('Hello World!')

        def cleanup(data):
            if False:
                i = 10
                return i + 15
            if self.step is None:
                return data
            return self.step.build.properties.cleanupTextFromSecrets(data)
        if self.debug:
            log.msg(f'Update[{key}]: {value}')
        if key == 'stdout':
            yield self.add_stdout_lines(cleanup(value), is_flushed)
        if key == 'stderr':
            yield self.add_stderr_lines(cleanup(value), is_flushed)
        if key == 'header':
            yield self.add_header_lines(cleanup(value))
        if key == 'log':
            (logname, data) = value
            yield self.addToLog(logname, cleanup(data))
        if key == 'rc':
            rc = self.rc = value
            log.msg(f'{self} rc={rc}')
            yield self.add_header_lines(f'program finished with exit code {rc}\n')
        if key == 'elapsed':
            self._remoteElapsed = value
        if key == 'failure_reason':
            self.remote_failure_reason = value
        if key not in ('stdout', 'stderr', 'header', 'rc', 'failure_reason'):
            if key not in self.updates:
                self.updates[key] = []
            self.updates[key].append(value)

    @defer.inlineCallbacks
    def remoteComplete(self, maybeFailure):
        if False:
            while True:
                i = 10
        if self._startTime and self._remoteElapsed:
            delta = util.now() - self._startTime - self._remoteElapsed
            metrics.MetricTimeEvent.log('RemoteCommand.overhead', delta)
        for (key, lbf) in self._line_boundary_finders.items():
            if key in ['stdout', 'stderr', 'header']:
                whole_line = lbf.flush()
                if whole_line is not None:
                    yield self.remoteUpdate(key, whole_line, True)
            else:
                logname = key
                whole_line = lbf.flush()
                value = (logname, whole_line)
                if whole_line is not None:
                    yield self.remoteUpdate('log', value, True)
        try:
            yield self.loglock.acquire()
            for (name, loog) in self.logs.items():
                if self._closeWhenFinished[name]:
                    if maybeFailure:
                        yield loog.addHeader(f'\nremoteFailed: {maybeFailure}')
                    else:
                        log.msg(f'closing log {loog}')
                    yield loog.finish()
        finally:
            yield self.loglock.release()
        if maybeFailure:
            if isinstance(maybeFailure, str):
                raise RemoteException(maybeFailure)
            if isinstance(maybeFailure, pb.CopiedFailure):
                maybeFailure.value = RemoteException(f'{maybeFailure.type}: {maybeFailure.value}\n{maybeFailure.traceback}')
                maybeFailure.type = RemoteException
            maybeFailure.raiseException()

    def results(self):
        if False:
            i = 10
            return i + 15
        if self.interrupted:
            return CANCELLED
        if self.rc in self.decodeRC:
            return self.decodeRC[self.rc]
        return FAILURE

    def didFail(self):
        if False:
            while True:
                i = 10
        return self.results() == FAILURE
LoggedRemoteCommand = RemoteCommand

class RemoteShellCommand(RemoteCommand):

    def __init__(self, workdir, command, env=None, want_stdout=1, want_stderr=1, timeout=20 * 60, maxTime=None, sigtermTime=None, logfiles=None, usePTY=None, logEnviron=True, collectStdout=False, collectStderr=False, interruptSignal=None, initialStdin=None, decodeRC=None, stdioLogName='stdio'):
        if False:
            for i in range(10):
                print('nop')
        if logfiles is None:
            logfiles = {}
        if decodeRC is None:
            decodeRC = {0: SUCCESS}
        self.command = command
        if isinstance(self.command, (str, bytes)):
            self.fake_command = command
        else:

            def obfuscate(arg):
                if False:
                    for i in range(10):
                        print('nop')
                if isinstance(arg, tuple) and len(arg) == 3 and (arg[0] == 'obfuscated'):
                    return arg[2]
                return arg
            self.fake_command = [obfuscate(c) for c in self.command]
        if env is not None:
            env = env.copy()
        args = {'workdir': workdir, 'env': env, 'want_stdout': want_stdout, 'want_stderr': want_stderr, 'logfiles': logfiles, 'timeout': timeout, 'maxTime': maxTime, 'sigtermTime': sigtermTime, 'usePTY': usePTY, 'logEnviron': logEnviron, 'initial_stdin': initialStdin}
        if interruptSignal is not None:
            args['interruptSignal'] = interruptSignal
        super().__init__('shell', args, collectStdout=collectStdout, collectStderr=collectStderr, decodeRC=decodeRC, stdioLogName=stdioLogName)

    def _start(self):
        if False:
            i = 10
            return i + 15
        if self.args['usePTY'] is None:
            if self.step.workerVersionIsOlderThan('shell', '3.0'):
                self.args['usePTY'] = 'slave-config'
            else:
                self.args['usePTY'] = False
        self.args['command'] = self.command
        if self.remote_command == 'shell':
            if self.step.workerVersion('shell', 'old') == 'old':
                self.args['dir'] = self.args['workdir']
            if self.step.workerVersionIsOlderThan('shell', '2.16'):
                self.args.pop('sigtermTime', None)
        what = f"command '{self.fake_command}' in dir '{self.args['workdir']}'"
        log.msg(what)
        return super()._start()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f"<RemoteShellCommand '{repr(self.fake_command)}'>"