import os
from twisted.internet import defer
from buildbot import config
from buildbot.process import buildstep
from buildbot.process import logobserver

class RpmBuild(buildstep.ShellMixin, buildstep.BuildStep):
    """
    RpmBuild build step.
    """
    renderables = ['dist']
    name = 'rpmbuilder'
    haltOnFailure = 1
    flunkOnFailure = 1
    description = ['RPMBUILD']
    descriptionDone = ['RPMBUILD']

    def __init__(self, specfile=None, topdir='`pwd`', builddir='`pwd`', rpmdir='`pwd`', sourcedir='`pwd`', specdir='`pwd`', srcrpmdir='`pwd`', dist='.el6', define=None, autoRelease=False, vcsRevision=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        kwargs = self.setupShellMixin(kwargs, prohibitArgs=['command'])
        super().__init__(**kwargs)
        self.dist = dist
        self.base_rpmbuild = f'rpmbuild --define "_topdir {topdir}" --define "_builddir {builddir}" --define "_rpmdir {rpmdir}" --define "_sourcedir {sourcedir}" --define "_specdir {specdir}" --define "_srcrpmdir {srcrpmdir}"'
        if define is None:
            define = {}
        for (k, v) in define.items():
            self.base_rpmbuild += f' --define "{k} {v}"'
        self.specfile = specfile
        self.autoRelease = autoRelease
        self.vcsRevision = vcsRevision
        if not self.specfile:
            config.error('You must specify a specfile')
        self.addLogObserver('stdio', logobserver.LineConsumerLogObserver(self.logConsumer))

    @defer.inlineCallbacks
    def run(self):
        if False:
            return 10
        rpm_extras_dict = {}
        rpm_extras_dict['dist'] = self.dist
        if self.autoRelease:
            relfile = f"{os.path.basename(self.specfile).split('.')[0]}.release"
            try:
                with open(relfile, 'r', encoding='utf-8') as rfile:
                    rel = int(rfile.readline().strip())
            except (IOError, TypeError, ValueError):
                rel = 0
            rpm_extras_dict['_release'] = rel
            with open(relfile, 'w', encoding='utf-8') as rfile:
                rfile.write(str(rel + 1))
        if self.vcsRevision:
            revision = self.getProperty('got_revision')
            if revision and (not isinstance(revision, dict)):
                rpm_extras_dict['_revision'] = revision
        self.rpmbuild = self.base_rpmbuild
        for (k, v) in sorted(rpm_extras_dict.items()):
            self.rpmbuild = f'{self.rpmbuild} --define "{k} {v}"'
        command = f'{self.rpmbuild} -ba {self.specfile}'
        cmd = (yield self.makeRemoteShellCommand(command=command))
        yield self.runCommand(cmd)
        stdio_log = (yield self.getLog('stdio'))
        yield stdio_log.finish()
        yield self.addCompleteLog('RPM Command Log', '\n'.join(self.rpmcmdlog))
        if self.rpmerrors:
            yield self.addCompleteLog('RPM Errors', '\n'.join(self.rpmerrors))
        return cmd.results()

    def logConsumer(self):
        if False:
            i = 10
            return i + 15
        rpm_prefixes = ['Provides:', 'Requires(', 'Requires:', 'Checking for unpackaged', 'Wrote:', 'Executing(%', '+ ', 'Processing files:']
        rpm_err_pfx = ['   ', 'RPM build errors:', 'error: ']
        self.rpmcmdlog = []
        self.rpmerrors = []
        while True:
            (_, line) = (yield)
            for pfx in rpm_prefixes:
                if line.startswith(pfx):
                    self.rpmcmdlog.append(line)
                    break
            for err in rpm_err_pfx:
                if line.startswith(err):
                    self.rpmerrors.append(line)
                    break