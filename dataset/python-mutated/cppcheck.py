import re
from twisted.internet import defer
from buildbot.process import logobserver
from buildbot.process.buildstep import BuildStep
from buildbot.process.buildstep import ShellMixin
from buildbot.process.results import FAILURE
from buildbot.process.results import SUCCESS
from buildbot.process.results import WARNINGS

class Cppcheck(ShellMixin, BuildStep):
    name = 'cppcheck'
    description = ['running', 'cppcheck']
    descriptionDone = ['cppcheck']
    flunkingIssues = ('error',)
    MESSAGES = ('error', 'warning', 'style', 'performance', 'portability', 'information')
    renderables = ('binary', 'source', 'extra_args')

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        for (name, default) in [('binary', 'cppcheck'), ('source', ['.']), ('enable', []), ('inconclusive', False), ('extra_args', [])]:
            setattr(self, name, kwargs.pop(name, default))
        kwargs = self.setupShellMixin(kwargs, prohibitArgs=['command'])
        super().__init__(*args, **kwargs)
        self.addLogObserver('stdio', logobserver.LineConsumerLogObserver(self._log_consumer))
        self.counts = {}
        summaries = self.summaries = {}
        for m in self.MESSAGES:
            self.counts[m] = 0
            summaries[m] = []

    def _log_consumer(self):
        if False:
            while True:
                i = 10
        line_re = re.compile(f"(?:\\[.+\\]: )?\\((?P<severity>{'|'.join(self.MESSAGES)})\\) .+")
        while True:
            (_, line) = (yield)
            m = line_re.match(line)
            if m is not None:
                msgsev = m.group('severity')
                self.summaries[msgsev].append(line)
                self.counts[msgsev] += 1

    @defer.inlineCallbacks
    def run(self):
        if False:
            for i in range(10):
                print('nop')
        command = [self.binary]
        command.extend(self.source)
        if self.enable:
            command.append(f"--enable={','.join(self.enable)}")
        if self.inconclusive:
            command.append('--inconclusive')
        command.extend(self.extra_args)
        cmd = (yield self.makeRemoteShellCommand(command=command))
        yield self.runCommand(cmd)
        stdio_log = (yield self.getLog('stdio'))
        yield stdio_log.finish()
        self.descriptionDone = self.descriptionDone[:]
        for msg in self.MESSAGES:
            self.setProperty(f'cppcheck-{msg}', self.counts[msg], 'Cppcheck')
            if not self.counts[msg]:
                continue
            self.descriptionDone.append(f'{msg}={self.counts[msg]}')
            yield self.addCompleteLog(msg, '\n'.join(self.summaries[msg]))
        self.setProperty('cppcheck-total', sum(self.counts.values()), 'Cppcheck')
        yield self.updateSummary()
        if cmd.results() != SUCCESS:
            return cmd.results()
        for msg in self.flunkingIssues:
            if self.counts[msg] != 0:
                return FAILURE
        if sum(self.counts.values()) > 0:
            return WARNINGS
        return SUCCESS