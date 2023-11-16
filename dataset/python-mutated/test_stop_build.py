from twisted.internet import defer
from buildbot.test.util.integration import RunMasterBase

class ShellMaster(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config(self):
        if False:
            print('Hello World!')
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.plugins import steps
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.AnyBranchScheduler(name='sched', builderNames=['testy']), schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        f = BuildFactory()
        f.addStep(steps.ShellCommand(command='sleep 100', name='sleep'))
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    @defer.inlineCallbacks
    def test_shell(self):
        if False:
            for i in range(10):
                print('nop')
        yield self.setup_config()

        @defer.inlineCallbacks
        def newStepCallback(_, data):
            if False:
                i = 10
                return i + 15
            if data['name'] == 'sleep':
                brs = (yield self.master.data.get(('buildrequests',)))
                brid = brs[-1]['buildrequestid']
                self.master.data.control('cancel', {'reason': 'cancelled by test'}, ('buildrequests', brid))
        yield self.master.mq.startConsuming(newStepCallback, ('steps', None, 'new'))
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True, wantProperties=True))
        self.assertEqual(build['buildid'], 1)
        cancel_logs = [log for log in build['steps'][1]['logs'] if log['name'] == 'cancelled']
        self.assertEqual(len(cancel_logs), 1)
        self.assertIn('cancelled by test', cancel_logs[0]['contents']['content'])