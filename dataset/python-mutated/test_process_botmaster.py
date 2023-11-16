from twisted.internet import defer
from buildbot.config import BuilderConfig
from buildbot.process.factory import BuildFactory
from buildbot.process.workerforbuilder import PingException
from buildbot.test.fake.worker import WorkerController
from buildbot.test.util.integration import RunFakeMasterTestCase

class Tests(RunFakeMasterTestCase):

    @defer.inlineCallbacks
    def do_terminates_ping_on_shutdown(self, quick_mode):
        if False:
            print('Hello World!')
        '\n        During shutdown we want to terminate any outstanding pings.\n        '
        controller = WorkerController(self, 'local')
        config_dict = {'builders': [BuilderConfig(name='testy', workernames=['local'], factory=BuildFactory())], 'workers': [controller.worker], 'protocols': {'null': {}}, 'multiMaster': True}
        yield self.setup_master(config_dict)
        builder_id = (yield self.master.data.updates.findBuilderId('testy'))
        yield controller.connect_worker()
        controller.sever_connection()
        yield self.create_build_request([builder_id])
        self.reactor.advance(1)
        yield self.master.botmaster.cleanShutdown(quickMode=quick_mode, stopReactor=False)
        self.flushLoggedErrors(PingException)

    def test_terminates_ping_on_shutdown_quick_mode(self):
        if False:
            print('Hello World!')
        return self.do_terminates_ping_on_shutdown(quick_mode=True)

    def test_terminates_ping_on_shutdown_slow_mode(self):
        if False:
            i = 10
            return i + 15
        return self.do_terminates_ping_on_shutdown(quick_mode=False)