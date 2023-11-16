"""
Steps and objects related to rpmlint.
"""
from twisted.internet import defer
from buildbot.steps.package import util as pkgutil
from buildbot.steps.shell import Test

class RpmLint(Test):
    """
    Rpmlint build step.
    """
    name = 'rpmlint'
    description = ['Checking for RPM/SPEC issues']
    descriptionDone = ['Finished checking RPM/SPEC issues']
    fileloc = '.'
    config = None

    def __init__(self, fileloc=None, config=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Create the Rpmlint object.\n\n        @type fileloc: str\n        @param fileloc: Location glob of the specs or rpms.\n        @type config: str\n        @param config: path to the rpmlint user config.\n        @type kwargs: dict\n        @param fileloc: all other keyword arguments.\n        '
        super().__init__(**kwargs)
        if fileloc:
            self.fileloc = fileloc
        if config:
            self.config = config
        self.command = ['rpmlint', '-i']
        if self.config:
            self.command += ['-f', self.config]
        self.command.append(self.fileloc)
        self.obs = pkgutil.WEObserver()
        self.addLogObserver('stdio', self.obs)

    @defer.inlineCallbacks
    def createSummary(self):
        if False:
            i = 10
            return i + 15
        '\n        Create nice summary logs.\n\n        @param log: log to create summary off of.\n        '
        warnings = self.obs.warnings
        errors = []
        if warnings:
            yield self.addCompleteLog(f'{len(warnings)} Warnings', '\n'.join(warnings))
        if errors:
            yield self.addCompleteLog(f'{len(errors)} Errors', '\n'.join(errors))