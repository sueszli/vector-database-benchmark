import sys
from twisted.internet import defer
from buildbot.config import BuilderConfig
from buildbot.plugins import schedulers
from buildbot.plugins import steps
from buildbot.process.factory import BuildFactory
from buildbot.test.util.integration import RunMasterBase
from buildbot.util import asyncSleep

class ShellMaster(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config_for_master_command(self, **kwargs):
        if False:
            return 10
        c = {}
        c['schedulers'] = [schedulers.AnyBranchScheduler(name='sched', builderNames=['testy'])]
        f = BuildFactory()
        f.addStep(steps.MasterShellCommand(**kwargs))
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    def get_change(self):
        if False:
            print('Hello World!')
        return {'branch': 'master', 'files': ['foo.c'], 'author': 'me@foo.com', 'committer': 'me@foo.com', 'comments': 'good stuff', 'revision': 'HEAD', 'project': 'none'}

    @defer.inlineCallbacks
    def test_shell(self):
        if False:
            i = 10
            return i + 15
        yield self.setup_config_for_master_command(command='echo hello')
        build = (yield self.doForceBuild(wantSteps=True, useChange=self.get_change(), wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        self.assertEqual(build['steps'][1]['state_string'], 'Ran')

    @defer.inlineCallbacks
    def test_logs(self):
        if False:
            print('Hello World!')
        yield self.setup_config_for_master_command(command=[sys.executable, '-c', 'print("hello")'])
        build = (yield self.doForceBuild(wantSteps=True, useChange=self.get_change(), wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        res = (yield self.checkBuildStepLogExist(build, 'hello'))
        self.assertTrue(res)
        self.assertEqual(build['steps'][1]['state_string'], 'Ran')

    @defer.inlineCallbacks
    def test_fails(self):
        if False:
            print('Hello World!')
        yield self.setup_config_for_master_command(command=[sys.executable, '-c', 'exit(1)'])
        build = (yield self.doForceBuild(wantSteps=True, useChange=self.get_change(), wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        self.assertEqual(build['steps'][1]['state_string'], 'failed (1) (failure)')

    @defer.inlineCallbacks
    def test_interrupt(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.setup_config_for_master_command(name='sleep', command=[sys.executable, '-c', 'while True: pass'])
        d = self.doForceBuild(wantSteps=True, useChange=self.get_change(), wantLogs=True)

        @defer.inlineCallbacks
        def on_new_step(_, data):
            if False:
                print('Hello World!')
            if data['name'] == 'sleep':
                yield asyncSleep(1)
                brs = (yield self.master.data.get(('buildrequests',)))
                brid = brs[-1]['buildrequestid']
                self.master.data.control('cancel', {'reason': 'cancelled by test'}, ('buildrequests', brid))
        yield self.master.mq.startConsuming(on_new_step, ('steps', None, 'new'))
        build = (yield d)
        self.assertEqual(build['buildid'], 1)
        if sys.platform == 'win32':
            self.assertEqual(build['steps'][1]['state_string'], 'failed (1) (exception)')
        else:
            self.assertEqual(build['steps'][1]['state_string'], 'killed (9) (exception)')