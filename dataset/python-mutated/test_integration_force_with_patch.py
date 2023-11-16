from twisted.internet import defer
from buildbot.process.results import FAILURE
from buildbot.process.results import SUCCESS
from buildbot.steps.source.base import Source
from buildbot.test.util.decorators import skipUnlessPlatformIs
from buildbot.test.util.integration import RunMasterBase
PATCH = b'diff --git a/Makefile b/Makefile\nnew file mode 100644\nindex 0000000..8a5cf80\n--- /dev/null\n+++ b/Makefile\n@@ -0,0 +1,2 @@\n+all:\n+\techo OK\n'

class MySource(Source):
    """A source class which only applies the patch"""

    @defer.inlineCallbacks
    def run_vc(self, branch, revision, patch):
        if False:
            for i in range(10):
                print('nop')
        self.stdio_log = (yield self.addLogForRemoteCommands('stdio'))
        if patch:
            yield self.patch(patch)
        return SUCCESS

class ShellMaster(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.plugins import steps
        from buildbot.plugins import util
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.ForceScheduler(name='force', codebases=[util.CodebaseParameter('foo', patch=util.PatchParameter())], builderNames=['testy'])]
        f = BuildFactory()
        f.addStep(MySource(codebase='foo'))
        f.addStep(steps.ShellCommand(command=['make']))
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    @skipUnlessPlatformIs('posix')
    @defer.inlineCallbacks
    def test_shell(self):
        if False:
            while True:
                i = 10
        yield self.setup_config()
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True, forceParams={'foo_patch_body': PATCH}))
        self.assertEqual(build['buildid'], 1)
        self.assertEqual(build['results'], SUCCESS)

    @defer.inlineCallbacks
    def test_shell_no_patch(self):
        if False:
            return 10
        yield self.setup_config()
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        self.assertEqual(build['steps'][1]['results'], SUCCESS)
        self.assertEqual(build['steps'][2]['results'], FAILURE)
        self.assertEqual(build['results'], FAILURE)