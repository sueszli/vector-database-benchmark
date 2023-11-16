from twisted.internet import defer
from twisted.internet import error
from buildbot.config import BuilderConfig
from buildbot.process import buildstep
from buildbot.process import logobserver
from buildbot.process import results
from buildbot.process.factory import BuildFactory
from buildbot.test.util.integration import RunFakeMasterTestCase

class TestLogObserver(logobserver.LogObserver):

    def __init__(self):
        if False:
            print('Hello World!')
        self.observed = []

    def outReceived(self, data):
        if False:
            while True:
                i = 10
        self.observed.append(data)

class Latin1ProducingCustomBuildStep(buildstep.BuildStep):

    @defer.inlineCallbacks
    def run(self):
        if False:
            i = 10
            return i + 15
        _log = (yield self.addLog('xx'))
        output_str = '¢'
        yield _log.addStdout(output_str)
        yield _log.finish()
        return results.SUCCESS

class BuildStepWithFailingLogObserver(buildstep.BuildStep):

    @defer.inlineCallbacks
    def run(self):
        if False:
            i = 10
            return i + 15
        self.addLogObserver('xx', logobserver.LineConsumerLogObserver(self.log_consumer))
        _log = (yield self.addLog('xx'))
        yield _log.addStdout('line1\nline2\n')
        yield _log.finish()
        return results.SUCCESS

    def log_consumer(self):
        if False:
            while True:
                i = 10
        (_, _) = (yield)
        raise RuntimeError('fail')

class FailingCustomStep(buildstep.BuildStep):
    flunkOnFailure = True

    def __init__(self, exception=buildstep.BuildStepFailed, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.exception = exception

    @defer.inlineCallbacks
    def run(self):
        if False:
            while True:
                i = 10
        yield defer.succeed(None)
        raise self.exception()

class RunSteps(RunFakeMasterTestCase):

    @defer.inlineCallbacks
    def create_config_for_step(self, step):
        if False:
            return 10
        config_dict = {'builders': [BuilderConfig(name='builder', workernames=['worker1'], factory=BuildFactory([step]))], 'workers': [self.createLocalWorker('worker1')], 'protocols': {'null': {}}, 'multiMaster': True}
        yield self.setup_master(config_dict)
        builder_id = (yield self.master.data.updates.findBuilderId('builder'))
        return builder_id

    @defer.inlineCallbacks
    def test_step_raising_buildstepfailed_in_start(self):
        if False:
            return 10
        builder_id = (yield self.create_config_for_step(FailingCustomStep()))
        yield self.do_test_build(builder_id)
        yield self.assertBuildResults(1, results.FAILURE)

    @defer.inlineCallbacks
    def test_step_raising_exception_in_start(self):
        if False:
            return 10
        builder_id = (yield self.create_config_for_step(FailingCustomStep(exception=ValueError)))
        yield self.do_test_build(builder_id)
        yield self.assertBuildResults(1, results.EXCEPTION)
        self.assertEqual(len(self.flushLoggedErrors(ValueError)), 1)

    @defer.inlineCallbacks
    def test_step_raising_connectionlost_in_start(self):
        if False:
            while True:
                i = 10
        ' Check whether we can recover from raising ConnectionLost from a step if the worker\n            did not actually disconnect\n        '
        step = FailingCustomStep(exception=error.ConnectionLost)
        builder_id = (yield self.create_config_for_step(step))
        yield self.do_test_build(builder_id)
        yield self.assertBuildResults(1, results.EXCEPTION)
    test_step_raising_connectionlost_in_start.skip = 'Results in infinite loop'

    @defer.inlineCallbacks
    def test_step_raising_in_log_observer(self):
        if False:
            print('Hello World!')
        step = BuildStepWithFailingLogObserver()
        builder_id = (yield self.create_config_for_step(step))
        yield self.do_test_build(builder_id)
        yield self.assertBuildResults(1, results.EXCEPTION)
        yield self.assertStepStateString(2, 'finished (exception)')
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)

    @defer.inlineCallbacks
    def test_Latin1ProducingCustomBuildStep(self):
        if False:
            i = 10
            return i + 15
        step = Latin1ProducingCustomBuildStep(logEncoding='latin-1')
        builder_id = (yield self.create_config_for_step(step))
        yield self.do_test_build(builder_id)
        yield self.assertLogs(1, {'xx': 'o¢\n'})