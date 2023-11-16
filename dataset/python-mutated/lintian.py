"""
Steps and objects related to lintian
"""
from twisted.internet import defer
from buildbot import config
from buildbot.process import buildstep
from buildbot.process import logobserver
from buildbot.process.results import FAILURE
from buildbot.process.results import SUCCESS
from buildbot.process.results import WARNINGS
from buildbot.steps.package import util as pkgutil

class MaxQObserver(logobserver.LogLineObserver):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.failures = 0

    def outLineReceived(self, line):
        if False:
            i = 10
            return i + 15
        if line.startswith('TEST FAILURE:'):
            self.failures += 1

class DebLintian(buildstep.ShellMixin, buildstep.BuildStep):
    name = 'lintian'
    description = 'Lintian running'
    descriptionDone = 'Lintian'
    fileloc = None
    suppressTags = []
    flunkOnFailure = False
    warnOnFailure = True

    def __init__(self, fileloc=None, suppressTags=None, **kwargs):
        if False:
            print('Hello World!')
        kwargs = self.setupShellMixin(kwargs)
        super().__init__(**kwargs)
        if fileloc:
            self.fileloc = fileloc
        if suppressTags:
            self.suppressTags = suppressTags
        if not self.fileloc:
            config.error('You must specify a fileloc')
        self.command = ['lintian', '-v', self.fileloc]
        if self.suppressTags:
            for tag in self.suppressTags:
                self.command += ['--suppress-tags', tag]
        self.obs = pkgutil.WEObserver()
        self.addLogObserver('stdio', self.obs)

    @defer.inlineCallbacks
    def run(self):
        if False:
            i = 10
            return i + 15
        cmd = (yield self.makeRemoteShellCommand())
        yield self.runCommand(cmd)
        stdio_log = (yield self.getLog('stdio'))
        yield stdio_log.finish()
        warnings = self.obs.warnings
        errors = self.obs.errors
        if warnings:
            yield self.addCompleteLog(f'{len(warnings)} Warnings', '\n'.join(warnings))
        if errors:
            yield self.addCompleteLog(f'{len(errors)} Errors', '\n'.join(errors))
        if cmd.rc != 0 or errors:
            return FAILURE
        if warnings:
            return WARNINGS
        return SUCCESS