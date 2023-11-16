from twisted.internet import defer
from buildbot.test.util.integration import RunFakeMasterTestCase

class CustomServiceMaster(RunFakeMasterTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.num_reconfig = 0

    def create_master_config(self):
        if False:
            i = 10
            return i + 15
        self.num_reconfig += 1
        from buildbot.config import BuilderConfig
        from buildbot.process.factory import BuildFactory
        from buildbot.steps.shell import ShellCommand
        from buildbot.util.service import BuildbotService

        class MyShellCommand(ShellCommand):

            def getResultSummary(self):
                if False:
                    for i in range(10):
                        print('nop')
                service = self.master.service_manager.namedServices['myService']
                return {'step': f'num reconfig: {service.num_reconfig}'}

        class MyService(BuildbotService):
            name = 'myService'

            def reconfigService(self, num_reconfig):
                if False:
                    i = 10
                    return i + 15
                self.num_reconfig = num_reconfig
                return defer.succeed(None)
        config_dict = {'builders': [BuilderConfig(name='builder', workernames=['worker1'], factory=BuildFactory([MyShellCommand(command='echo hei')]))], 'workers': [self.createLocalWorker('worker1')], 'protocols': {'null': {}}, 'multiMaster': True, 'db_url': 'sqlite://', 'services': [MyService(num_reconfig=self.num_reconfig)]}
        if self.num_reconfig == 3:
            config_dict['services'].append(MyService(name='myService2', num_reconfig=self.num_reconfig))
        return config_dict

    @defer.inlineCallbacks
    def test_custom_service(self):
        if False:
            while True:
                i = 10
        yield self.setup_master(self.create_master_config())
        yield self.do_test_build_by_name('builder')
        self.assertStepStateString(1, 'worker worker1 ready')
        self.assertStepStateString(2, 'num reconfig: 1')
        myService = self.master.service_manager.namedServices['myService']
        self.assertEqual(myService.num_reconfig, 1)
        self.assertTrue(myService.running)
        yield self.reconfig_master(self.create_master_config())
        yield self.do_test_build_by_name('builder')
        self.assertEqual(myService.num_reconfig, 2)
        self.assertStepStateString(1, 'worker worker1 ready')
        self.assertStepStateString(2, 'num reconfig: 1')
        yield self.reconfig_master(self.create_master_config())
        myService2 = self.master.service_manager.namedServices['myService2']
        self.assertTrue(myService2.running)
        self.assertEqual(myService2.num_reconfig, 3)
        self.assertEqual(myService.num_reconfig, 3)
        yield self.reconfig_master(self.create_master_config())
        self.assertNotIn('myService2', self.master.service_manager.namedServices)
        self.assertFalse(myService2.running)
        self.assertEqual(myService2.num_reconfig, 3)
        self.assertEqual(myService.num_reconfig, 4)