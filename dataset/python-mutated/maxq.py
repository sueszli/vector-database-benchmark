from twisted.internet import defer
from buildbot import config
from buildbot.process import buildstep
from buildbot.process import logobserver
from buildbot.process.results import FAILURE
from buildbot.process.results import SUCCESS

class MaxQObserver(logobserver.LogLineObserver):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.failures = 0

    def outLineReceived(self, line):
        if False:
            print('Hello World!')
        if line.startswith('TEST FAILURE:'):
            self.failures += 1

class MaxQ(buildstep.ShellMixin, buildstep.BuildStep):
    flunkOnFailure = True
    name = 'maxq'
    binary = 'run_maxq.py'
    failures = 0

    def __init__(self, testdir=None, **kwargs):
        if False:
            while True:
                i = 10
        if not testdir:
            config.error('please pass testdir')
        self.testdir = testdir
        kwargs = self.setupShellMixin(kwargs)
        super().__init__(**kwargs)
        self.observer = MaxQObserver()
        self.addLogObserver('stdio', self.observer)

    @defer.inlineCallbacks
    def run(self):
        if False:
            while True:
                i = 10
        command = [self.binary]
        command.append(self.testdir)
        cmd = (yield self.makeRemoteShellCommand(command=command))
        yield self.runCommand(cmd)
        stdio_log = (yield self.getLog('stdio'))
        yield stdio_log.finish()
        self.failures = self.observer.failures
        if not self.failures and cmd.didFail():
            self.failures = 1
        if self.failures:
            return FAILURE
        return SUCCESS

    def getResultSummary(self):
        if False:
            return 10
        if self.failures:
            return {'step': f'{self.failures} maxq failures'}
        return {'step': 'success'}