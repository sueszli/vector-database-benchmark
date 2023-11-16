from parameterized import parameterized
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.spread import pb
from buildbot.config import BuilderConfig
from buildbot.config.master import MasterConfig
from buildbot.interfaces import LatentWorkerCannotSubstantiate
from buildbot.interfaces import LatentWorkerFailedToSubstantiate
from buildbot.interfaces import LatentWorkerSubstantiatiationCancelled
from buildbot.machine.latent import States as MachineStates
from buildbot.process.factory import BuildFactory
from buildbot.process.properties import Interpolate
from buildbot.process.properties import Properties
from buildbot.process.results import CANCELLED
from buildbot.process.results import EXCEPTION
from buildbot.process.results import FAILURE
from buildbot.process.results import RETRY
from buildbot.process.results import SUCCESS
from buildbot.test.fake.latent import ControllableLatentWorker
from buildbot.test.fake.latent import LatentController
from buildbot.test.fake.machine import LatentMachineController
from buildbot.test.fake.step import BuildStepController
from buildbot.test.util.integration import RunFakeMasterTestCase
from buildbot.test.util.misc import TimeoutableTestCase
from buildbot.test.util.patch_delay import patchForDelay
from buildbot.worker import manager
from buildbot.worker.latent import States

class TestException(Exception):
    """
    An exception thrown in tests.
    """

class Latent(TimeoutableTestCase, RunFakeMasterTestCase):

    def tearDown(self):
        if False:
            print('Hello World!')
        self.flushLoggedErrors(LatentWorkerSubstantiatiationCancelled)
        super().tearDown()

    @defer.inlineCallbacks
    def create_single_worker_config(self, controller_kwargs=None):
        if False:
            return 10
        if not controller_kwargs:
            controller_kwargs = {}
        controller = LatentController(self, 'local', **controller_kwargs)
        config_dict = {'builders': [BuilderConfig(name='testy', workernames=['local'], factory=BuildFactory())], 'workers': [controller.worker], 'protocols': {'null': {}}, 'multiMaster': True}
        yield self.setup_master(config_dict)
        builder_id = (yield self.master.data.updates.findBuilderId('testy'))
        return (controller, builder_id)

    @defer.inlineCallbacks
    def create_single_worker_config_with_step(self, controller_kwargs=None):
        if False:
            return 10
        if not controller_kwargs:
            controller_kwargs = {}
        controller = LatentController(self, 'local', **controller_kwargs)
        stepcontroller = BuildStepController()
        config_dict = {'builders': [BuilderConfig(name='testy', workernames=['local'], factory=BuildFactory([stepcontroller.step]))], 'workers': [controller.worker], 'protocols': {'null': {}}, 'multiMaster': True}
        yield self.setup_master(config_dict)
        builder_id = (yield self.master.data.updates.findBuilderId('testy'))
        return (controller, stepcontroller, builder_id)

    @defer.inlineCallbacks
    def create_single_worker_two_builder_config(self, controller_kwargs=None):
        if False:
            print('Hello World!')
        if not controller_kwargs:
            controller_kwargs = {}
        controller = LatentController(self, 'local', **controller_kwargs)
        config_dict = {'builders': [BuilderConfig(name='testy-1', workernames=['local'], factory=BuildFactory()), BuilderConfig(name='testy-2', workernames=['local'], factory=BuildFactory())], 'workers': [controller.worker], 'protocols': {'null': {}}, 'multiMaster': True}
        yield self.setup_master(config_dict)
        builder_ids = [(yield self.master.data.updates.findBuilderId('testy-1')), (yield self.master.data.updates.findBuilderId('testy-2'))]
        return (controller, builder_ids)

    @defer.inlineCallbacks
    def reconfig_workers_remove_all(self):
        if False:
            for i in range(10):
                print('nop')
        config_dict = {'workers': [], 'multiMaster': True}
        config = MasterConfig.loadFromDict(config_dict, '<dict>')
        yield self.master.workers.reconfigServiceWithBuildbotConfig(config)

    def stop_first_build(self, results):
        if False:
            while True:
                i = 10
        stopped_d = defer.Deferred()

        def new_callback(_, data):
            if False:
                while True:
                    i = 10
            if stopped_d.called:
                return
            buildid = data['buildid']
            self.master.mq.produce(('control', 'builds', str(buildid), 'stop'), {'reason': 'no reason', 'results': results})
            stopped_d.callback(None)
        consumed_d = self.master.mq.startConsuming(new_callback, ('builds', None, 'new'))
        return (consumed_d, stopped_d)

    @defer.inlineCallbacks
    def test_latent_workers_start_in_parallel(self):
        if False:
            return 10
        '\n        If there are two latent workers configured, and two build\n        requests for them, both workers will start substantiating\n        concurrently.\n        '
        controllers = [LatentController(self, 'local1'), LatentController(self, 'local2')]
        config_dict = {'builders': [BuilderConfig(name='testy', workernames=['local1', 'local2'], factory=BuildFactory())], 'workers': [controller.worker for controller in controllers], 'protocols': {'null': {}}, 'multiMaster': True}
        yield self.setup_master(config_dict)
        builder_id = (yield self.master.data.updates.findBuilderId('testy'))
        for _ in range(2):
            yield self.create_build_request([builder_id])
        self.assertEqual(controllers[0].starting, True)
        self.assertEqual(controllers[1].starting, True)
        for controller in controllers:
            yield controller.start_instance(True)
            yield controller.auto_stop(True)

    @defer.inlineCallbacks
    def test_refused_substantiations_get_requeued(self):
        if False:
            i = 10
            return i + 15
        '\n        If a latent worker refuses to substantiate, the build request becomes\n        unclaimed.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config())
        (_, brids) = (yield self.create_build_request([builder_id]))
        unclaimed_build_requests = []
        yield self.master.mq.startConsuming(lambda key, request: unclaimed_build_requests.append(request), ('buildrequests', None, 'unclaimed'))
        yield controller.start_instance(False)
        self.assertEqual(set(brids), {req['buildrequestid'] for req in unclaimed_build_requests})
        yield self.assertBuildResults(1, RETRY)
        yield controller.auto_stop(True)
        self.flushLoggedErrors(LatentWorkerFailedToSubstantiate)

    @defer.inlineCallbacks
    def test_failed_substantiations_get_requeued(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If a latent worker fails to substantiate, the build request becomes\n        unclaimed.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config())
        (_, brids) = (yield self.create_build_request([builder_id]))
        unclaimed_build_requests = []
        yield self.master.mq.startConsuming(lambda key, request: unclaimed_build_requests.append(request), ('buildrequests', None, 'unclaimed'))
        yield controller.start_instance(Failure(TestException('substantiation failed')))
        self.flushLoggedErrors(TestException)
        self.assertEqual(set(brids), {req['buildrequestid'] for req in unclaimed_build_requests})
        yield self.assertBuildResults(1, RETRY)
        yield controller.auto_stop(True)

    @defer.inlineCallbacks
    def test_failed_substantiations_get_exception(self):
        if False:
            while True:
                i = 10
        '\n        If a latent worker fails to substantiate, the result is an exception.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config())
        yield self.create_build_request([builder_id])
        yield controller.start_instance(Failure(LatentWorkerCannotSubstantiate('substantiation failed')))
        self.flushLoggedErrors(LatentWorkerCannotSubstantiate)
        yield self.assertBuildResults(1, EXCEPTION)
        yield controller.auto_stop(True)

    @defer.inlineCallbacks
    def test_worker_accepts_builds_after_failure(self):
        if False:
            print('Hello World!')
        '\n        If a latent worker fails to substantiate, the worker is still able to\n        accept jobs.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config())
        yield controller.auto_stop(True)
        yield self.create_build_request([builder_id])
        unclaimed_build_requests = []
        yield self.master.mq.startConsuming(lambda key, request: unclaimed_build_requests.append(request), ('buildrequests', None, 'unclaimed'))
        yield controller.start_instance(Failure(TestException('substantiation failed')))
        self.flushLoggedErrors(TestException)
        self.assertEqual(controller.starting, False)
        self.reactor.advance(controller.worker.quarantine_initial_timeout)
        self.assertEqual(controller.starting, True)
        yield controller.start_instance(Failure(TestException('substantiation failed')))
        self.flushLoggedErrors(TestException)
        yield self.assertBuildResults(1, RETRY)
        self.reactor.advance(controller.worker.quarantine_initial_timeout)
        self.assertEqual(controller.starting, False)
        self.reactor.advance(controller.worker.quarantine_initial_timeout)
        self.assertEqual(controller.starting, True)
        controller.auto_start(True)
        controller.auto_stop(True)

    @defer.inlineCallbacks
    def test_worker_multiple_substantiations_succeed(self):
        if False:
            return 10
        '\n        If multiple builders trigger try to substantiate a worker at\n        the same time, if the substantiation succeeds then all of\n        the builds proceed.\n        '
        (controller, builder_ids) = (yield self.create_single_worker_two_builder_config())
        yield self.create_build_request(builder_ids)
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, SUCCESS)
        yield self.assertBuildResults(2, SUCCESS)
        yield controller.auto_stop(True)

    @defer.inlineCallbacks
    def test_very_late_detached_after_substantiation(self):
        if False:
            i = 10
            return i + 15
        "\n        A latent worker may detach at any time after stop_instance() call.\n        Make sure it works at the most late detachment point, i.e. when we're\n        substantiating again.\n        "
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': 1}))
        yield self.create_build_request([builder_id])
        self.assertTrue(controller.starting)
        controller.auto_disconnect_worker = False
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, SUCCESS)
        self.reactor.advance(1)
        self.assertTrue(controller.stopping)
        yield controller.stop_instance(True)
        self.assertTrue(controller.stopped)
        yield self.create_build_request([builder_id])
        self.assertTrue(controller.starting)
        yield controller.disconnect_worker()
        yield controller.start_instance(True)
        yield self.assertBuildResults(2, SUCCESS)
        self.reactor.advance(1)
        yield controller.stop_instance(True)
        yield controller.disconnect_worker()

    @defer.inlineCallbacks
    def test_substantiation_during_stop_instance(self):
        if False:
            while True:
                i = 10
        '\n        If a latent worker detaches before stop_instance() completes and we\n        start a build then it should start successfully without causing an\n        erroneous cancellation of the substantiation request.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': 1}))
        yield self.create_build_request([builder_id])
        self.assertEqual(True, controller.starting)
        controller.auto_disconnect_worker = False
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, SUCCESS)
        self.reactor.advance(1)
        self.assertTrue(controller.stopping)
        yield controller.disconnect_worker()
        yield self.create_build_request([builder_id])
        yield controller.stop_instance(True)
        yield controller.start_instance(True)
        yield self.assertBuildResults(2, SUCCESS)
        self.reactor.advance(1)
        yield controller.stop_instance(True)
        yield controller.disconnect_worker()

    @defer.inlineCallbacks
    def test_substantiation_during_stop_instance_canStartBuild_race(self):
        if False:
            i = 10
            return i + 15
        '\n        If build attempts substantiation after the latent worker detaches,\n        but stop_instance() is not completed yet, then we should successfully\n        complete substantiation without causing an erroneous cancellation.\n        The above sequence of events was possible even if canStartBuild\n        checked for a in-progress insubstantiation, as if the build is scheduled\n        before insubstantiation, its start could be delayed until when\n        stop_instance() is in progress.\n        '
        (controller, builder_ids) = (yield self.create_single_worker_two_builder_config(controller_kwargs={'build_wait_timeout': 1}))
        yield self.create_build_request([builder_ids[0]])
        self.assertEqual(True, controller.starting)
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, SUCCESS)
        with patchForDelay('buildbot.process.builder.Builder.maybeStartBuild') as delay:
            yield self.create_build_request([builder_ids[1]])
            self.assertEqual(len(delay), 1)
            self.reactor.advance(1)
            self.assertTrue(controller.stopping)
            delay.fire()
            yield controller.stop_instance(True)
        self.assertTrue(controller.starting)
        yield controller.start_instance(True)
        yield self.assertBuildResults(2, SUCCESS)
        self.reactor.advance(1)
        yield controller.stop_instance(True)

    @defer.inlineCallbacks
    def test_insubstantiation_during_substantiation_refuses_substantiation(self):
        if False:
            i = 10
            return i + 15
        '\n        If a latent worker gets insubstantiation() during substantiation, then it should refuse\n        to substantiate.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': 1}))
        yield self.create_build_request([builder_id])
        d = controller.worker.insubstantiate()
        yield controller.start_instance(False)
        yield controller.stop_instance(True)
        yield d
        yield self.assertBuildResults(1, RETRY)

    @defer.inlineCallbacks
    def test_stopservice_during_insubstantiation_completes(self):
        if False:
            i = 10
            return i + 15
        '\n        When stopService is called and a worker is insubstantiating, we should wait for this\n        process to complete.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': 1}))
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, SUCCESS)
        self.assertTrue(controller.started)
        self.reactor.advance(1)
        self.assertTrue(controller.stopping)
        d = self.reconfig_workers_remove_all()
        self.assertFalse(d.called)
        yield controller.stop_instance(True)
        yield d

    @parameterized.expand([('with_substantiation_failure', False, False), ('without_worker_connecting', True, False), ('with_worker_connecting', True, True)])
    @defer.inlineCallbacks
    def test_stopservice_during_substantiation_completes(self, name, subst_success, worker_connects):
        if False:
            print('Hello World!')
        '\n        When stopService is called and a worker is substantiating, we should wait for this\n        process to complete.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': 1}))
        controller.auto_connect_worker = worker_connects
        yield self.create_build_request([builder_id])
        self.assertTrue(controller.starting)
        d = self.reconfig_workers_remove_all()
        self.assertFalse(d.called)
        yield controller.start_instance(subst_success)
        self.assertTrue(controller.stopping)
        yield controller.stop_instance(True)
        yield d

    @defer.inlineCallbacks
    def test_substantiation_is_cancelled_by_build_stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Stopping a build during substantiation should cancel the substantiation itself.\n        Otherwise we will be left with a substantiating worker without a corresponding build\n        which means that master shutdown may not work correctly.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config())
        controller.auto_connect_worker = False
        controller.auto_stop(True)
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        self.master.mq.produce(('control', 'builds', '1', 'stop'), {'reason': 'no reason'})
        self.reactor.advance(1)
        self.assertTrue(controller.stopped)

    @parameterized.expand([('after_start_instance_no_worker', False, False), ('after_start_instance_with_worker', True, False), ('before_start_instance_no_worker', False, True), ('before_start_instance_with_worker', True, True)])
    @defer.inlineCallbacks
    def test_reconfigservice_during_substantiation_clean_shutdown_after(self, name, worker_connects, before_start_service):
        if False:
            while True:
                i = 10
        '\n        When stopService is called and a worker is substantiating, we should wait for this\n        process to complete.\n        '
        registered_workers = []

        def registration_updates(reg, worker_config, global_config):
            if False:
                return 10
            registered_workers.append((worker_config.workername, worker_config.password))
        self.patch(manager.WorkerRegistration, 'update', registration_updates)
        (controller, builder_id) = (yield self.create_single_worker_config())
        controller.auto_connect_worker = worker_connects
        controller.auto_stop(True)
        yield self.create_build_request([builder_id])
        self.assertTrue(controller.starting)
        self.master.config_loader.config_dict['workers'] = [ControllableLatentWorker('local', controller, max_builds=3)]
        if before_start_service:
            yield self.reconfig_master()
            yield controller.start_instance(True)
        else:
            yield controller.start_instance(True)
            yield self.reconfig_master()
        yield self.clean_master_shutdown(quick=True)
        self.assertEqual(registered_workers, [('local', 'password_1'), ('local', 'password_1')])

    @defer.inlineCallbacks
    def test_substantiation_cancelled_by_insubstantiation_when_waiting_for_insubstantiation(self):
        if False:
            return 10
        '\n        We should cancel substantiation if we insubstantiate when that substantiation is waiting\n        on current insubstantiation to finish\n        '
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': 1}))
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, SUCCESS)
        self.reactor.advance(1)
        self.assertTrue(controller.stopping)
        yield self.create_build_request([builder_id])
        self.assertEqual(controller.worker.state, States.INSUBSTANTIATING_SUBSTANTIATING)
        d = controller.worker.insubstantiate()
        yield controller.stop_instance(True)
        yield d
        yield self.assertBuildResults(2, RETRY)

    @defer.inlineCallbacks
    def test_stalled_substantiation_then_timeout_get_requeued(self):
        if False:
            print('Hello World!')
        '\n        If a latent worker substantiate, but not connect, and then be\n        unsubstantiated, the build request becomes unclaimed.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config())
        (_, brids) = (yield self.create_build_request([builder_id]))
        unclaimed_build_requests = []
        yield self.master.mq.startConsuming(lambda key, request: unclaimed_build_requests.append(request), ('buildrequests', None, 'unclaimed'))
        self.reactor.advance(controller.worker.missing_timeout)
        self.flushLoggedErrors(defer.TimeoutError)
        self.assertEqual(set(brids), {req['buildrequestid'] for req in unclaimed_build_requests})
        yield controller.start_instance(False)
        yield controller.auto_stop(True)

    @defer.inlineCallbacks
    def test_stalled_substantiation_then_check_instance_fails_get_requeued(self):
        if False:
            print('Hello World!')
        '\n        If a latent worker substantiate, but not connect and check_instance() indicates a crash,\n        the build request should become unclaimed as soon as check_instance_interval passes\n        '
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'check_instance_interval': 10}))
        controller.auto_connect_worker = False
        (_, brids) = (yield self.create_build_request([builder_id]))
        unclaimed_build_requests = []
        yield self.master.mq.startConsuming(lambda key, request: unclaimed_build_requests.append(request), ('buildrequests', None, 'unclaimed'))
        yield controller.start_instance(True)
        self.reactor.advance(10)
        controller.has_crashed = True
        self.reactor.advance(10)
        self.flushLoggedErrors(LatentWorkerFailedToSubstantiate)
        self.assertEqual(set(brids), {req['buildrequestid'] for req in unclaimed_build_requests})
        yield controller.auto_stop(True)

    @defer.inlineCallbacks
    def test_sever_connection_before_ping_then_timeout_get_requeued(self):
        if False:
            return 10
        '\n        If a latent worker connects, but its connection is severed without\n        notification in the TCP layer, we successfully wait until TCP times\n        out and requeue the build.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': 1}))
        yield self.create_build_request([builder_id])
        with patchForDelay('buildbot.process.workerforbuilder.AbstractWorkerForBuilder.ping') as delay:
            yield controller.start_instance(True)
            controller.sever_connection()
            delay.fire()
        self.reactor.advance(100)
        yield controller.disconnect_worker()
        yield self.assertBuildResults(1, RETRY)
        self.reactor.advance(controller.worker.quarantine_initial_timeout)
        yield controller.stop_instance(True)
        yield controller.start_instance(True)
        yield self.assertBuildResults(2, SUCCESS)
        self.reactor.advance(1)
        yield controller.stop_instance(True)
        self.flushLoggedErrors(pb.PBConnectionLost)

    @defer.inlineCallbacks
    def test_failed_sendBuilderList_get_requeued(self):
        if False:
            print('Hello World!')
        '\n        sendBuilderList can fail due to missing permissions on the workdir,\n        the build request becomes unclaimed\n        '
        (controller, builder_id) = (yield self.create_single_worker_config())
        (_, brids) = (yield self.create_build_request([builder_id]))
        unclaimed_build_requests = []
        yield self.master.mq.startConsuming(lambda key, request: unclaimed_build_requests.append(request), ('buildrequests', None, 'unclaimed'))
        logs = []
        yield self.master.mq.startConsuming(lambda key, log: logs.append(log), ('logs', None, 'new'))

        def remote_setBuilderList(self, dirs):
            if False:
                print('Hello World!')
            raise TestException("can't create dir")
        controller.patchBot(self, 'remote_setBuilderList', remote_setBuilderList)
        yield controller.start_instance(True)
        self.flushLoggedErrors(TestException)
        self.assertEqual(set(brids), {req['buildrequestid'] for req in unclaimed_build_requests})
        self.assertEqual(len(logs), 2)
        logs_by_name = {}
        for _log in logs:
            fulllog = (yield self.master.data.get(('logs', str(_log['logid']), 'raw')))
            logs_by_name[fulllog['filename']] = fulllog['raw']
        for i in ['err_text', 'err_html']:
            self.assertIn("can't create dir", logs_by_name[i])
            self.assertIn('buildbot.test.integration.test_worker_latent.TestException', logs_by_name[i])
        yield controller.auto_stop(True)

    @defer.inlineCallbacks
    def test_failed_ping_get_requeued(self):
        if False:
            return 10
        '\n        sendBuilderList can fail due to missing permissions on the workdir,\n        the build request becomes unclaimed\n        '
        (controller, builder_id) = (yield self.create_single_worker_config())
        (_, brids) = (yield self.create_build_request([builder_id]))
        unclaimed_build_requests = []
        yield self.master.mq.startConsuming(lambda key, request: unclaimed_build_requests.append(request), ('buildrequests', None, 'unclaimed'))
        logs = []
        yield self.master.mq.startConsuming(lambda key, log: logs.append(log), ('logs', None, 'new'))

        def remote_print(self, msg):
            if False:
                i = 10
                return i + 15
            if msg == 'ping':
                raise TestException("can't ping")
        controller.patchBot(self, 'remote_print', remote_print)
        yield controller.start_instance(True)
        self.flushLoggedErrors(TestException)
        self.assertEqual(set(brids), {req['buildrequestid'] for req in unclaimed_build_requests})
        self.assertEqual(len(logs), 2)
        logs_by_name = {}
        for _log in logs:
            fulllog = (yield self.master.data.get(('logs', str(_log['logid']), 'raw')))
            logs_by_name[fulllog['filename']] = fulllog['raw']
        for i in ['err_text', 'err_html']:
            self.assertIn("can't ping", logs_by_name[i])
            self.assertIn('buildbot.test.integration.test_worker_latent.TestException', logs_by_name[i])
        yield controller.auto_stop(True)

    @defer.inlineCallbacks
    def test_worker_close_connection_while_building(self):
        if False:
            return 10
        '\n        If the worker close connection in the middle of the build, the next\n        build can start correctly\n        '
        (controller, stepcontroller, builder_id) = (yield self.create_single_worker_config_with_step(controller_kwargs={'build_wait_timeout': 0}))
        controller.auto_disconnect_worker = False
        yield self.create_build_request([builder_id])
        yield controller.auto_stop(True)
        self.assertTrue(controller.starting)
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, None)
        yield controller.disconnect_worker()
        yield self.assertBuildResults(1, RETRY)
        yield controller.start_instance(True)
        yield self.assertBuildResults(2, None)
        stepcontroller.finish_step(SUCCESS)
        yield self.assertBuildResults(2, SUCCESS)
        yield controller.disconnect_worker()

    @defer.inlineCallbacks
    def test_negative_build_timeout_reattach_substantiated(self):
        if False:
            print('Hello World!')
        "\n        When build_wait_timeout is negative, we don't disconnect the worker from\n        our side. We should still support accidental disconnections from\n        worker side due to, e.g. network problems.\n        "
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': -1}))
        controller.auto_disconnect_worker = False
        controller.auto_connect_worker = False
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        yield controller.connect_worker()
        yield self.assertBuildResults(1, SUCCESS)
        self.assertTrue(controller.started)
        yield controller.disconnect_worker()
        self.assertTrue(controller.started)
        yield controller.connect_worker()
        self.assertTrue(controller.started)
        yield self.create_build_request([builder_id])
        yield self.assertBuildResults(1, SUCCESS)
        yield controller.auto_stop(True)
        yield controller.worker.insubstantiate()
        yield controller.disconnect_worker()

    @defer.inlineCallbacks
    def test_sever_connection_while_building(self):
        if False:
            while True:
                i = 10
        '\n        If the connection to worker is severed without TCP notification in the\n        middle of the build, the build is re-queued and successfully restarted.\n        '
        (controller, stepcontroller, builder_id) = (yield self.create_single_worker_config_with_step(controller_kwargs={'build_wait_timeout': 0}))
        yield self.create_build_request([builder_id])
        yield controller.auto_stop(True)
        self.assertTrue(controller.starting)
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, None)
        controller.sever_connection()
        self.reactor.advance(100)
        yield controller.disconnect_worker()
        yield self.assertBuildResults(1, RETRY)
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        yield self.assertBuildResults(2, None)
        stepcontroller.finish_step(SUCCESS)
        yield self.assertBuildResults(2, SUCCESS)

    @defer.inlineCallbacks
    def test_sever_connection_during_insubstantiation(self):
        if False:
            while True:
                i = 10
        '\n        If latent worker connection is severed without notification in the TCP\n        layer, we successfully wait until TCP times out, insubstantiate and\n        can substantiate after that.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': 1}))
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, SUCCESS)
        with patchForDelay('buildbot.worker.base.AbstractWorker.disconnect') as delay:
            self.reactor.advance(1)
            self.assertTrue(controller.stopping)
            controller.sever_connection()
            delay.fire()
        yield controller.stop_instance(True)
        self.reactor.advance(100)
        yield controller.disconnect_worker()
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, SUCCESS)
        self.reactor.advance(1)
        yield controller.stop_instance(True)
        self.flushLoggedErrors(pb.PBConnectionLost)

    @defer.inlineCallbacks
    def test_sever_connection_during_insubstantiation_and_buildrequest(self):
        if False:
            return 10
        '\n        If latent worker connection is severed without notification in the TCP\n        layer, we successfully wait until TCP times out, insubstantiate and\n        can substantiate after that. In this the subsequent build request is\n        created during insubstantiation\n        '
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': 1}))
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, SUCCESS)
        with patchForDelay('buildbot.worker.base.AbstractWorker.disconnect') as delay:
            self.reactor.advance(1)
            self.assertTrue(controller.stopping)
            yield self.create_build_request([builder_id])
            controller.sever_connection()
            delay.fire()
        yield controller.stop_instance(True)
        self.reactor.advance(100)
        yield controller.disconnect_worker()
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, SUCCESS)
        self.reactor.advance(1)
        yield controller.stop_instance(True)
        self.flushLoggedErrors(pb.PBConnectionLost)

    @defer.inlineCallbacks
    def test_negative_build_timeout_reattach_insubstantiating(self):
        if False:
            print('Hello World!')
        "\n        When build_wait_timeout is negative, we don't disconnect the worker from\n        our side, but it can disconnect and reattach from worker side due to,\n        e.g. network problems.\n        "
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': -1}))
        controller.auto_disconnect_worker = False
        controller.auto_connect_worker = False
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        yield controller.connect_worker()
        yield self.assertBuildResults(1, SUCCESS)
        self.assertTrue(controller.started)
        d = controller.worker.insubstantiate()
        self.assertTrue(controller.stopping)
        yield controller.disconnect_worker()
        self.assertTrue(controller.stopping)
        yield controller.connect_worker()
        self.assertTrue(controller.stopping)
        yield controller.stop_instance(True)
        yield d
        self.assertTrue(controller.stopped)
        yield controller.disconnect_worker()
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        yield controller.connect_worker()
        yield self.assertBuildResults(1, SUCCESS)
        controller.auto_disconnect_worker = True
        yield controller.auto_stop(True)

    @defer.inlineCallbacks
    def test_negative_build_timeout_no_disconnect_insubstantiating(self):
        if False:
            i = 10
            return i + 15
        "\n        When build_wait_timeout is negative, we don't disconnect the worker from\n        our side, so it should be possible to insubstantiate and substantiate\n        it without problems if the worker does not disconnect either.\n        "
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': -1}))
        controller.auto_disconnect_worker = False
        controller.auto_connect_worker = False
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        yield controller.connect_worker()
        yield self.assertBuildResults(1, SUCCESS)
        self.assertTrue(controller.started)
        d = controller.worker.insubstantiate()
        self.assertTrue(controller.stopping)
        yield controller.stop_instance(True)
        yield d
        self.assertTrue(controller.stopped)
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, SUCCESS)
        controller.auto_disconnect_worker = True
        yield controller.auto_stop(True)

    @defer.inlineCallbacks
    def test_negative_build_timeout_insubstantiates_on_master_shutdown(self):
        if False:
            print('Hello World!')
        '\n        When build_wait_timeout is negative, we should still insubstantiate when master shuts down.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': -1}))
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        yield self.assertBuildResults(1, SUCCESS)
        self.assertTrue(controller.started)
        d = self.master.stopService()
        yield controller.stop_instance(True)
        yield d

    @defer.inlineCallbacks
    def test_stop_instance_synchronous_exception(self):
        if False:
            while True:
                i = 10
        '\n        Throwing a synchronous exception from stop_instance should allow subsequent build to start.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config(controller_kwargs={'build_wait_timeout': 1}))
        controller.auto_stop(True)

        def raise_stop_instance(fast):
            if False:
                for i in range(10):
                    print('nop')
            raise TestException()
        real_stop_instance = controller.worker.stop_instance
        controller.worker.stop_instance = raise_stop_instance
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        self.reactor.advance(1)
        yield self.assertBuildResults(1, SUCCESS)
        self.flushLoggedErrors(TestException)
        controller.worker.stop_instance = real_stop_instance
        yield controller.worker.stop_instance(False)
        self.reactor.advance(1)
        yield self.create_build_request([builder_id])
        yield controller.start_instance(True)
        self.reactor.advance(1)
        yield self.assertBuildResults(2, SUCCESS)

    @defer.inlineCallbacks
    def test_build_stop_with_cancelled_during_substantiation(self):
        if False:
            return 10
        '\n        If a build is stopping during latent worker substantiating, the build\n        becomes cancelled\n        '
        (controller, builder_id) = (yield self.create_single_worker_config())
        (consumed_d, stopped_d) = self.stop_first_build(CANCELLED)
        yield consumed_d
        yield self.create_build_request([builder_id])
        yield stopped_d
        yield controller.start_instance(False)
        yield self.assertBuildResults(1, CANCELLED)
        yield controller.auto_stop(True)
        self.flushLoggedErrors(LatentWorkerFailedToSubstantiate)

    @defer.inlineCallbacks
    def test_build_stop_with_retry_during_substantiation(self):
        if False:
            i = 10
            return i + 15
        '\n        If master is shutting down during latent worker substantiating, the build becomes retry.\n        '
        (controller, builder_id) = (yield self.create_single_worker_config())
        (consumed_d, stopped_d) = self.stop_first_build(RETRY)
        yield consumed_d
        (_, brids) = (yield self.create_build_request([builder_id]))
        unclaimed_build_requests = []
        yield self.master.mq.startConsuming(lambda key, request: unclaimed_build_requests.append(request), ('buildrequests', None, 'unclaimed'))
        yield stopped_d
        yield controller.start_instance(False)
        yield self.assertBuildResults(1, RETRY)
        self.assertEqual(set(brids), {req['buildrequestid'] for req in unclaimed_build_requests})
        yield controller.auto_stop(True)
        self.flushLoggedErrors(LatentWorkerFailedToSubstantiate)

    @defer.inlineCallbacks
    def test_rejects_build_on_instance_with_different_type_timeout_zero(self):
        if False:
            i = 10
            return i + 15
        '\n        If latent worker supports getting its instance type from properties that\n        are rendered from build then the buildrequestdistributor must not\n        schedule any builds on workers that are running different instance type\n        than what these builds will require.\n        '
        (controller, stepcontroller, builder_id) = (yield self.create_single_worker_config_with_step(controller_kwargs={'kind': Interpolate('%(prop:worker_kind)s'), 'build_wait_timeout': 0}))
        yield self.create_build_request([builder_id], properties=Properties(worker_kind='a'))
        self.assertEqual(True, controller.starting)
        controller.auto_start(True)
        yield controller.auto_stop(True)
        self.assertEqual((yield controller.get_started_kind()), 'a')
        yield self.create_build_request([builder_id], properties=Properties(worker_kind='b'))
        stepcontroller.finish_step(SUCCESS)
        self.reactor.advance(0.1)
        self.assertEqual((yield controller.get_started_kind()), 'b')
        stepcontroller.finish_step(SUCCESS)
        yield self.assertBuildResults(1, SUCCESS)
        yield self.assertBuildResults(2, SUCCESS)

    @defer.inlineCallbacks
    def test_rejects_build_on_instance_with_different_type_timeout_nonzero(self):
        if False:
            return 10
        '\n        If latent worker supports getting its instance type from properties that\n        are rendered from build then the buildrequestdistributor must not\n        schedule any builds on workers that are running different instance type\n        than what these builds will require.\n        '
        (controller, stepcontroller, builder_id) = (yield self.create_single_worker_config_with_step(controller_kwargs={'kind': Interpolate('%(prop:worker_kind)s'), 'build_wait_timeout': 5}))
        yield self.create_build_request([builder_id], properties=Properties(worker_kind='a'))
        self.assertEqual(True, controller.starting)
        controller.auto_start(True)
        yield controller.auto_stop(True)
        self.assertEqual((yield controller.get_started_kind()), 'a')
        yield self.create_build_request([builder_id], properties=Properties(worker_kind='b'))
        stepcontroller.finish_step(SUCCESS)
        self.reactor.advance(0.1)
        self.assertIsNone((yield self.master.db.builds.getBuild(2)))
        self.assertTrue(controller.started)
        self.reactor.advance(6)
        self.assertIsNotNone((yield self.master.db.builds.getBuild(2)))
        self.assertEqual((yield controller.get_started_kind()), 'b')
        stepcontroller.finish_step(SUCCESS)
        yield self.assertBuildResults(1, SUCCESS)
        yield self.assertBuildResults(2, SUCCESS)

    @defer.inlineCallbacks
    def test_supports_no_build_for_substantiation(self):
        if False:
            print('Hello World!')
        '\n        Abstract latent worker should support being substantiated without a\n        build and then insubstantiated.\n        '
        (controller, _) = (yield self.create_single_worker_config())
        controller.worker.substantiate(None, None)
        yield controller.start_instance(True)
        self.assertTrue(controller.started)
        d = controller.worker.insubstantiate()
        yield controller.stop_instance(True)
        yield d

    @defer.inlineCallbacks
    def test_supports_no_build_for_substantiation_accepts_build_later(self):
        if False:
            i = 10
            return i + 15
        '\n        Abstract latent worker should support being substantiated without a\n        build and then accept a build request.\n        '
        (controller, stepcontroller, builder_id) = (yield self.create_single_worker_config_with_step(controller_kwargs={'build_wait_timeout': 1}))
        controller.worker.substantiate(None, None)
        yield controller.start_instance(True)
        self.assertTrue(controller.started)
        self.create_build_request([builder_id])
        stepcontroller.finish_step(SUCCESS)
        self.reactor.advance(1)
        yield controller.stop_instance(True)

class LatentWithLatentMachine(TimeoutableTestCase, RunFakeMasterTestCase):

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.flushLoggedErrors(LatentWorkerSubstantiatiationCancelled)
        super().tearDown()

    @defer.inlineCallbacks
    def create_single_worker_config(self, build_wait_timeout=0):
        if False:
            return 10
        machine_controller = LatentMachineController(name='machine1', build_wait_timeout=build_wait_timeout)
        worker_controller = LatentController(self, 'worker1', machine_name='machine1')
        step_controller = BuildStepController()
        config_dict = {'machines': [machine_controller.machine], 'builders': [BuilderConfig(name='builder1', workernames=['worker1'], factory=BuildFactory([step_controller.step]))], 'workers': [worker_controller.worker], 'protocols': {'null': {}}, 'multiMaster': True}
        yield self.setup_master(config_dict)
        builder_id = (yield self.master.data.updates.findBuilderId('builder1'))
        return (machine_controller, worker_controller, step_controller, builder_id)

    @defer.inlineCallbacks
    def create_two_worker_config(self, build_wait_timeout=0, controller_kwargs=None):
        if False:
            i = 10
            return i + 15
        if not controller_kwargs:
            controller_kwargs = {}
        machine_controller = LatentMachineController(name='machine1', build_wait_timeout=build_wait_timeout)
        worker1_controller = LatentController(self, 'worker1', machine_name='machine1', **controller_kwargs)
        worker2_controller = LatentController(self, 'worker2', machine_name='machine1', **controller_kwargs)
        step1_controller = BuildStepController()
        step2_controller = BuildStepController()
        config_dict = {'machines': [machine_controller.machine], 'builders': [BuilderConfig(name='builder1', workernames=['worker1'], factory=BuildFactory([step1_controller.step])), BuilderConfig(name='builder2', workernames=['worker2'], factory=BuildFactory([step2_controller.step]))], 'workers': [worker1_controller.worker, worker2_controller.worker], 'protocols': {'null': {}}, 'multiMaster': True}
        yield self.setup_master(config_dict)
        builder1_id = (yield self.master.data.updates.findBuilderId('builder1'))
        builder2_id = (yield self.master.data.updates.findBuilderId('builder2'))
        return (machine_controller, [worker1_controller, worker2_controller], [step1_controller, step2_controller], [builder1_id, builder2_id])

    @defer.inlineCallbacks
    def test_1worker_starts_and_stops_after_single_build_success(self):
        if False:
            for i in range(10):
                print('nop')
        (machine_controller, worker_controller, step_controller, builder_id) = (yield self.create_single_worker_config())
        worker_controller.auto_start(True)
        worker_controller.auto_stop(True)
        yield self.create_build_request([builder_id])
        machine_controller.start_machine(True)
        self.assertTrue(worker_controller.started)
        step_controller.finish_step(SUCCESS)
        self.reactor.advance(0)
        machine_controller.stop_machine()
        self.assertEqual(machine_controller.machine.state, MachineStates.STOPPED)

    @defer.inlineCallbacks
    def test_1worker_starts_and_stops_after_single_build_failure(self):
        if False:
            for i in range(10):
                print('nop')
        (machine_controller, worker_controller, step_controller, builder_id) = (yield self.create_single_worker_config())
        worker_controller.auto_start(True)
        worker_controller.auto_stop(True)
        yield self.create_build_request([builder_id])
        machine_controller.start_machine(True)
        self.assertTrue(worker_controller.started)
        step_controller.finish_step(FAILURE)
        self.reactor.advance(0)
        machine_controller.stop_machine()
        self.assertEqual(machine_controller.machine.state, MachineStates.STOPPED)

    @defer.inlineCallbacks
    def test_1worker_stops_machine_after_timeout(self):
        if False:
            i = 10
            return i + 15
        (machine_controller, worker_controller, step_controller, builder_id) = (yield self.create_single_worker_config(build_wait_timeout=5))
        worker_controller.auto_start(True)
        worker_controller.auto_stop(True)
        yield self.create_build_request([builder_id])
        machine_controller.start_machine(True)
        self.reactor.advance(10.0)
        step_controller.finish_step(SUCCESS)
        self.assertEqual(machine_controller.machine.state, MachineStates.STARTED)
        self.reactor.advance(4.9)
        self.assertEqual(machine_controller.machine.state, MachineStates.STARTED)
        self.reactor.advance(0.1)
        self.assertEqual(machine_controller.machine.state, MachineStates.STOPPING)
        machine_controller.stop_machine()
        self.assertEqual(machine_controller.machine.state, MachineStates.STOPPED)

    @defer.inlineCallbacks
    def test_1worker_does_not_stop_machine_machine_after_timeout_during_build(self):
        if False:
            while True:
                i = 10
        (machine_controller, worker_controller, step_controller, builder_id) = (yield self.create_single_worker_config(build_wait_timeout=5))
        worker_controller.auto_start(True)
        worker_controller.auto_stop(True)
        yield self.create_build_request([builder_id])
        machine_controller.start_machine(True)
        self.reactor.advance(10.0)
        step_controller.finish_step(SUCCESS)
        self.assertEqual(machine_controller.machine.state, MachineStates.STARTED)
        self.reactor.advance(4.9)
        self.assertEqual(machine_controller.machine.state, MachineStates.STARTED)
        yield self.create_build_request([builder_id])
        self.reactor.advance(5.1)
        self.assertEqual(machine_controller.machine.state, MachineStates.STARTED)
        step_controller.finish_step(SUCCESS)
        self.reactor.advance(4.9)
        self.assertEqual(machine_controller.machine.state, MachineStates.STARTED)
        self.reactor.advance(0.1)
        self.assertEqual(machine_controller.machine.state, MachineStates.STOPPING)
        machine_controller.stop_machine()
        self.assertEqual(machine_controller.machine.state, MachineStates.STOPPED)

    @defer.inlineCallbacks
    def test_1worker_insubstantiated_after_start_failure(self):
        if False:
            for i in range(10):
                print('nop')
        (machine_controller, worker_controller, _, builder_id) = (yield self.create_single_worker_config())
        worker_controller.auto_connect_worker = False
        worker_controller.auto_start(True)
        worker_controller.auto_stop(True)
        yield self.create_build_request([builder_id])
        machine_controller.start_machine(False)
        self.assertEqual(machine_controller.machine.state, MachineStates.STOPPED)
        self.assertEqual(worker_controller.started, False)

    @defer.inlineCallbacks
    def test_1worker_eats_exception_from_start_machine(self):
        if False:
            return 10
        (machine_controller, worker_controller, _, builder_id) = (yield self.create_single_worker_config())
        worker_controller.auto_connect_worker = False
        worker_controller.auto_start(True)
        worker_controller.auto_stop(True)
        yield self.create_build_request([builder_id])

        class FakeError(Exception):
            pass
        machine_controller.start_machine(FakeError('start error'))
        self.assertEqual(machine_controller.machine.state, MachineStates.STOPPED)
        self.assertEqual(worker_controller.started, False)
        self.flushLoggedErrors(FakeError)

    @defer.inlineCallbacks
    def test_1worker_eats_exception_from_stop_machine(self):
        if False:
            return 10
        (machine_controller, worker_controller, step_controller, builder_id) = (yield self.create_single_worker_config())
        worker_controller.auto_start(True)
        worker_controller.auto_stop(True)
        yield self.create_build_request([builder_id])
        machine_controller.start_machine(True)
        step_controller.finish_step(SUCCESS)
        self.reactor.advance(0)

        class FakeError(Exception):
            pass
        machine_controller.stop_machine(FakeError('stop error'))
        self.assertEqual(machine_controller.machine.state, MachineStates.STOPPED)
        self.flushLoggedErrors(FakeError)

    @defer.inlineCallbacks
    def test_2workers_build_substantiates_insubstantiates_both_workers(self):
        if False:
            for i in range(10):
                print('nop')
        (machine_controller, worker_controllers, step_controllers, builder_ids) = (yield self.create_two_worker_config(controller_kwargs={'starts_without_substantiate': True}))
        for wc in worker_controllers:
            wc.auto_start(True)
            wc.auto_stop(True)
        yield self.create_build_request([builder_ids[0]])
        machine_controller.start_machine(True)
        for wc in worker_controllers:
            self.assertTrue(wc.started)
        step_controllers[0].finish_step(SUCCESS)
        self.reactor.advance(0)
        machine_controller.stop_machine()
        for wc in worker_controllers:
            self.assertFalse(wc.started)
        self.assertEqual(machine_controller.machine.state, MachineStates.STOPPED)

    @defer.inlineCallbacks
    def test_2workers_two_builds_start_machine_concurrently(self):
        if False:
            return 10
        (machine_controller, worker_controllers, step_controllers, builder_ids) = (yield self.create_two_worker_config())
        for wc in worker_controllers:
            wc.auto_start(True)
            wc.auto_stop(True)
        yield self.create_build_request([builder_ids[0]])
        self.assertEqual(machine_controller.machine.state, MachineStates.STARTING)
        yield self.create_build_request([builder_ids[1]])
        machine_controller.start_machine(True)
        for wc in worker_controllers:
            self.assertTrue(wc.started)
        step_controllers[0].finish_step(SUCCESS)
        step_controllers[1].finish_step(SUCCESS)
        self.reactor.advance(0)
        machine_controller.stop_machine()
        for wc in worker_controllers:
            self.assertFalse(wc.started)
        self.assertEqual(machine_controller.machine.state, MachineStates.STOPPED)

    @defer.inlineCallbacks
    def test_2workers_insubstantiated_after_one_start_failure(self):
        if False:
            return 10
        (machine_controller, worker_controllers, _, builder_ids) = (yield self.create_two_worker_config())
        for wc in worker_controllers:
            wc.auto_connect_worker = False
            wc.auto_start(True)
            wc.auto_stop(True)
        yield self.create_build_request([builder_ids[0]])
        machine_controller.start_machine(False)
        self.assertEqual(machine_controller.machine.state, MachineStates.STOPPED)
        for wc in worker_controllers:
            self.assertEqual(wc.started, False)