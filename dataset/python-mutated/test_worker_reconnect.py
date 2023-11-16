from twisted.internet import defer
from buildbot.process.buildstep import BuildStep
from buildbot.process.results import SUCCESS
from buildbot.test.util.integration import RunMasterBase

class DisconnectingStep(BuildStep):
    disconnection_list = []

    def run(self):
        if False:
            return 10
        self.disconnection_list.append(self)
        assert self.worker.conn.get_peer().startswith('127.0.0.1:')
        if len(self.disconnection_list) < 2:
            self.worker.disconnect()
        return SUCCESS

class WorkerReconnectPb(RunMasterBase):
    """integration test for testing worker disconnection and reconnection"""
    proto = 'pb'

    @defer.inlineCallbacks
    def setup_config(self):
        if False:
            i = 10
            return i + 15
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.AnyBranchScheduler(name='sched', builderNames=['testy']), schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        f = BuildFactory()
        f.addStep(DisconnectingStep())
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    @defer.inlineCallbacks
    def test_eventually_reconnect(self):
        if False:
            print('Hello World!')
        DisconnectingStep.disconnection_list = []
        yield self.setup_config()
        build = (yield self.doForceBuild())
        self.assertEqual(build['buildid'], 2)
        self.assertEqual(len(DisconnectingStep.disconnection_list), 2)

class WorkerReconnectMsgPack(WorkerReconnectPb):
    proto = 'msgpack'