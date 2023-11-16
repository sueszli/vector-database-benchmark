"""
Source step code for Monotone
"""
from twisted.internet import defer
from twisted.internet import reactor
from twisted.python import log
from buildbot.config import ConfigErrors
from buildbot.interfaces import WorkerSetupError
from buildbot.process import buildstep
from buildbot.process import remotecommand
from buildbot.process.results import SUCCESS
from buildbot.steps.source.base import Source

class Monotone(Source):
    """ Class for Monotone with all smarts """
    name = 'monotone'
    renderables = ['repourl']
    possible_methods = ('clobber', 'copy', 'fresh', 'clean')

    def __init__(self, repourl=None, branch=None, progress=False, mode='incremental', method=None, **kwargs):
        if False:
            i = 10
            return i + 15
        self.repourl = repourl
        self.method = method
        self.mode = mode
        self.branch = branch
        self.sourcedata = f'{self.repourl}?{self.branch}'
        self.database = 'db.mtn'
        self.progress = progress
        super().__init__(**kwargs)
        errors = []
        if not self._hasAttrGroupMember('mode', self.mode):
            errors.append(f"mode {self.mode} is not one of {self._listAttrGroupMembers('mode')}")
        if self.mode == 'incremental' and self.method:
            errors.append('Incremental mode does not require method')
        if self.mode == 'full':
            if self.method is None:
                self.method = 'copy'
            elif self.method not in self.possible_methods:
                errors.append(f'Invalid method for mode == {self.mode}')
        if repourl is None:
            errors.append('you must provide repourl')
        if branch is None:
            errors.append('you must provide branch')
        if errors:
            raise ConfigErrors(errors)

    @defer.inlineCallbacks
    def run_vc(self, branch, revision, patch):
        if False:
            print('Hello World!')
        self.revision = revision
        self.stdio_log = (yield self.addLogForRemoteCommands('stdio'))
        try:
            monotoneInstalled = (yield self.checkMonotone())
            if not monotoneInstalled:
                raise WorkerSetupError('Monotone is not installed on worker')
            yield self._checkDb()
            yield self._retryPull()
            if self.mode != 'full' or self.method not in ('clobber', 'copy'):
                patched = (yield self.sourcedirIsPatched())
                if patched:
                    yield self.clean()
            fn = self._getAttrGroupMember('mode', self.mode)
            yield fn()
            if patch:
                yield self.patch(patch)
            yield self.parseGotRevision()
            return SUCCESS
        finally:
            pass

    @defer.inlineCallbacks
    def mode_full(self):
        if False:
            return 10
        if self.method == 'clobber':
            yield self.clobber()
            return
        elif self.method == 'copy':
            yield self.copy()
            return
        updatable = (yield self._sourcedirIsUpdatable())
        if not updatable:
            yield self.clobber()
        elif self.method == 'clean':
            yield self.clean()
            yield self._update()
        elif self.method == 'fresh':
            yield self.clean(False)
            yield self._update()
        else:
            raise ValueError('Unknown method, check your configuration')

    @defer.inlineCallbacks
    def mode_incremental(self):
        if False:
            while True:
                i = 10
        updatable = (yield self._sourcedirIsUpdatable())
        if not updatable:
            yield self.clobber()
        else:
            yield self._update()

    @defer.inlineCallbacks
    def clobber(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.runRmdir(self.workdir)
        yield self._checkout()

    @defer.inlineCallbacks
    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        cmd = remotecommand.RemoteCommand('rmdir', {'dir': self.workdir, 'logEnviron': self.logEnviron, 'timeout': self.timeout})
        cmd.useLog(self.stdio_log, False)
        yield self.runCommand(cmd)
        self.workdir = 'source'
        yield self.mode_incremental()
        cmd = remotecommand.RemoteCommand('cpdir', {'fromdir': 'source', 'todir': 'build', 'logEnviron': self.logEnviron, 'timeout': self.timeout})
        cmd.useLog(self.stdio_log, False)
        yield self.runCommand(cmd)
        self.workdir = 'build'
        return 0

    @defer.inlineCallbacks
    def checkMonotone(self):
        if False:
            while True:
                i = 10
        cmd = remotecommand.RemoteShellCommand(self.workdir, ['mtn', '--version'], env=self.env, logEnviron=self.logEnviron, timeout=self.timeout)
        cmd.useLog(self.stdio_log, False)
        yield self.runCommand(cmd)
        return cmd.rc == 0

    @defer.inlineCallbacks
    def clean(self, ignore_ignored=True):
        if False:
            for i in range(10):
                print('nop')
        files = []
        commands = [['mtn', 'ls', 'unknown']]
        if not ignore_ignored:
            commands.append(['mtn', 'ls', 'ignored'])
        for cmd in commands:
            stdout = (yield self._dovccmd(cmd, workdir=self.workdir, collectStdout=True))
            if not stdout:
                continue
            for filename in stdout.strip().split('\n'):
                filename = self.workdir + '/' + str(filename)
                files.append(filename)
        if not files:
            rc = 0
        elif self.workerVersionIsOlderThan('rmdir', '2.14'):
            rc = (yield self.removeFiles(files))
        else:
            rc = (yield self.runRmdir(files, abandonOnFailure=False))
        if rc != 0:
            log.msg('Failed removing files')
            raise buildstep.BuildStepFailed()

    @defer.inlineCallbacks
    def removeFiles(self, files):
        if False:
            for i in range(10):
                print('nop')
        for filename in files:
            res = (yield self.runRmdir(filename, abandonOnFailure=False))
            if res:
                return res
        return 0

    def _checkout(self, abandonOnFailure=False):
        if False:
            i = 10
            return i + 15
        command = ['mtn', 'checkout', self.workdir, '--db', self.database]
        if self.revision:
            command.extend(['--revision', self.revision])
        command.extend(['--branch', self.branch])
        return self._dovccmd(command, workdir='.', abandonOnFailure=abandonOnFailure)

    def _update(self, abandonOnFailure=False):
        if False:
            while True:
                i = 10
        command = ['mtn', 'update']
        if self.revision:
            command.extend(['--revision', self.revision])
        else:
            command.extend(['--revision', 'h:' + self.branch])
        command.extend(['--branch', self.branch])
        return self._dovccmd(command, workdir=self.workdir, abandonOnFailure=abandonOnFailure)

    def _pull(self, abandonOnFailure=False):
        if False:
            while True:
                i = 10
        command = ['mtn', 'pull', self.sourcedata, '--db', self.database]
        if self.progress:
            command.extend(['--ticker=dot'])
        else:
            command.extend(['--ticker=none'])
        d = self._dovccmd(command, workdir='.', abandonOnFailure=abandonOnFailure)
        return d

    @defer.inlineCallbacks
    def _retryPull(self):
        if False:
            while True:
                i = 10
        if self.retry:
            abandonOnFailure = self.retry[1] <= 0
        else:
            abandonOnFailure = True
        res = (yield self._pull(abandonOnFailure))
        if self.retry:
            (delay, repeats) = self.retry
            if self.stopped or res == 0 or repeats <= 0:
                return res
            else:
                log.msg(f'Checkout failed, trying {repeats} more times after {delay} seconds')
                self.retry = (delay, repeats - 1)
                df = defer.Deferred()
                df.addCallback(lambda _: self._retryPull())
                reactor.callLater(delay, df.callback, None)
                yield df
        return None

    @defer.inlineCallbacks
    def parseGotRevision(self):
        if False:
            i = 10
            return i + 15
        stdout = (yield self._dovccmd(['mtn', 'automate', 'select', 'w:'], workdir=self.workdir, collectStdout=True))
        revision = stdout.strip()
        if len(revision) != 40:
            raise buildstep.BuildStepFailed()
        log.msg(f'Got Monotone revision {revision}')
        self.updateSourceProperty('got_revision', revision)
        return 0

    @defer.inlineCallbacks
    def _dovccmd(self, command, workdir, collectStdout=False, initialStdin=None, decodeRC=None, abandonOnFailure=True):
        if False:
            for i in range(10):
                print('nop')
        if not command:
            raise ValueError('No command specified')
        if decodeRC is None:
            decodeRC = {0: SUCCESS}
        cmd = remotecommand.RemoteShellCommand(workdir, command, env=self.env, logEnviron=self.logEnviron, timeout=self.timeout, collectStdout=collectStdout, initialStdin=initialStdin, decodeRC=decodeRC)
        cmd.useLog(self.stdio_log, False)
        yield self.runCommand(cmd)
        if abandonOnFailure and cmd.didFail():
            log.msg(f'Source step failed while running command {cmd}')
            raise buildstep.BuildStepFailed()
        if collectStdout:
            return cmd.stdout
        else:
            return cmd.rc

    @defer.inlineCallbacks
    def _checkDb(self):
        if False:
            while True:
                i = 10
        db_exists = (yield self.pathExists(self.database))
        db_needs_init = False
        if db_exists:
            stdout = (yield self._dovccmd(['mtn', 'db', 'info', '--db', self.database], workdir='.', collectStdout=True))
            if stdout.find('migration needed') >= 0:
                log.msg('Older format database found, migrating it')
                yield self._dovccmd(['mtn', 'db', 'migrate', '--db', self.database], workdir='.')
            elif stdout.find('too new, cannot use') >= 0 or stdout.find('database has no tables') >= 0:
                yield self.runRmdir(self.database)
                db_needs_init = True
            elif stdout.find('not a monotone database') >= 0:
                raise buildstep.BuildStepFailed()
            else:
                log.msg('Database exists and compatible')
        else:
            db_needs_init = True
            log.msg('Database does not exist')
        if db_needs_init:
            command = ['mtn', 'db', 'init', '--db', self.database]
            yield self._dovccmd(command, workdir='.')

    @defer.inlineCallbacks
    def _sourcedirIsUpdatable(self):
        if False:
            return 10
        workdir_path = self.build.path_module.join(self.workdir, '_MTN')
        workdir_exists = (yield self.pathExists(workdir_path))
        if not workdir_exists:
            log.msg('Workdir does not exist, falling back to a fresh clone')
        return workdir_exists