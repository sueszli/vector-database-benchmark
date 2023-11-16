from unittest import mock
from parameterized import parameterized
from twisted.internet import defer
from twisted.internet import error
from twisted.internet import reactor
from twisted.internet.task import deferLater
from twisted.python import failure
from twisted.python import log
from twisted.trial import unittest
from buildbot import locks
from buildbot.interfaces import WorkerSetupError
from buildbot.plugins import util
from buildbot.process import buildstep
from buildbot.process import properties
from buildbot.process import remotecommand
from buildbot.process.properties import renderer
from buildbot.process.results import ALL_RESULTS
from buildbot.process.results import CANCELLED
from buildbot.process.results import EXCEPTION
from buildbot.process.results import FAILURE
from buildbot.process.results import RETRY
from buildbot.process.results import SKIPPED
from buildbot.process.results import SUCCESS
from buildbot.process.results import WARNINGS
from buildbot.test.fake import fakebuild
from buildbot.test.fake import fakemaster
from buildbot.test.fake import worker
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.steps import ExpectGlob
from buildbot.test.steps import ExpectMkdir
from buildbot.test.steps import ExpectRmdir
from buildbot.test.steps import ExpectShell
from buildbot.test.steps import ExpectStat
from buildbot.test.steps import TestBuildStepMixin
from buildbot.test.util import config
from buildbot.test.util import interfaces
from buildbot.util.eventual import eventually

class NewStyleStep(buildstep.BuildStep):

    def run(self):
        if False:
            while True:
                i = 10
        pass

class CustomActionBuildStep(buildstep.BuildStep):

    def run(self):
        if False:
            while True:
                i = 10
        return self.action()

class TestBuildStep(TestBuildStepMixin, config.ConfigErrorsMixin, TestReactorMixin, unittest.TestCase):

    class FakeBuildStep(buildstep.BuildStep):

        def run(self):
            if False:
                print('Hello World!')
            d = defer.Deferred()
            eventually(d.callback, 0)
            return d

    class SkippingBuildStep(buildstep.BuildStep):

        def run(self):
            if False:
                for i in range(10):
                    print('nop')
            return SKIPPED

    class LockBuildStep(buildstep.BuildStep):

        def __init__(self, testcase=None, lock_accesses=None, **kwargs):
            if False:
                while True:
                    i = 10
            super().__init__(**kwargs)
            self.testcase = testcase
            self.lock_accesses = lock_accesses

        @defer.inlineCallbacks
        def run(self):
            if False:
                for i in range(10):
                    print('nop')
            botmaster = self.build.builder.botmaster
            real_master_lock = (yield botmaster.getLockFromLockAccess(self.lock_accesses[0], self.build.config_version))
            real_worker_lock = (yield botmaster.getLockFromLockAccess(self.lock_accesses[1], self.build.config_version))
            self.testcase.assertFalse(real_master_lock.isAvailable(self.testcase, self.lock_accesses[0]))
            self.testcase.assertIn('workername', real_worker_lock.locks)
            self.testcase.assertFalse(real_worker_lock.locks['workername'].isAvailable(self.testcase, self.lock_accesses[1]))
            return SUCCESS

    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        return self.setup_test_build_step()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        return self.tear_down_test_build_step()

    def _setupWaterfallTest(self, hideStepIf, expect, expectedResult=SUCCESS):
        if False:
            for i in range(10):
                print('nop')
        self.setup_step(TestBuildStep.FakeBuildStep(hideStepIf=hideStepIf))
        self.expect_outcome(result=expectedResult)
        self.expect_hidden(expect)

    def test_nameIsntString(self):
        if False:
            while True:
                i = 10
        "\n        When BuildStep is passed a name that isn't a string, it reports\n        a config error.\n        "
        with self.assertRaisesConfigError('BuildStep name must be a string'):
            buildstep.BuildStep(name=5)

    def test_name_too_long(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesConfigError('exceeds maximum length of'):
            buildstep.BuildStep(name='b' * 100)

    def test_unexpectedKeywordArgument(self):
        if False:
            print('Hello World!')
        '\n        When BuildStep is passed an unknown keyword argument, it reports\n        a config error.\n        '
        with self.assertRaisesConfigError("__init__ got unexpected keyword argument(s) ['oogaBooga']"):
            buildstep.BuildStep(oogaBooga=5)

    def test_updateBuildSummaryPolicyDefaults(self):
        if False:
            print('Hello World!')
        '\n        updateBuildSummaryPolicy builds default value according to resultsMixin\n        parameters (flunkOnFailure..)\n        '
        step = buildstep.BuildStep()
        self.assertEqual(sorted(step.updateBuildSummaryPolicy), sorted([EXCEPTION, RETRY, CANCELLED, FAILURE]))
        step = buildstep.BuildStep(warnOnWarnings=True)
        self.assertEqual(sorted(step.updateBuildSummaryPolicy), sorted([EXCEPTION, RETRY, CANCELLED, FAILURE, WARNINGS]))
        step = buildstep.BuildStep(flunkOnFailure=False)
        self.assertEqual(sorted(step.updateBuildSummaryPolicy), sorted([EXCEPTION, RETRY, CANCELLED]))
        step = buildstep.BuildStep(updateBuildSummaryPolicy=False)
        self.assertEqual(sorted(step.updateBuildSummaryPolicy), [])
        step = buildstep.BuildStep(updateBuildSummaryPolicy=True)
        self.assertEqual(sorted(step.updateBuildSummaryPolicy), sorted(ALL_RESULTS))

    def test_updateBuildSummaryPolicyBadType(self):
        if False:
            return 10
        '\n        updateBuildSummaryPolicy raise ConfigError in case of bad type\n        '
        with self.assertRaisesConfigError('BuildStep updateBuildSummaryPolicy must be a list of result ids or boolean but it is 2'):
            buildstep.BuildStep(updateBuildSummaryPolicy=FAILURE)

    def test_getProperty(self):
        if False:
            i = 10
            return i + 15
        bs = buildstep.BuildStep()
        bs.build = fakebuild.FakeBuild()
        props = bs.build.properties = mock.Mock()
        bs.getProperty('xyz', 'b')
        props.getProperty.assert_called_with('xyz', 'b')
        bs.getProperty('xyz')
        props.getProperty.assert_called_with('xyz', None)

    def test_setProperty(self):
        if False:
            for i in range(10):
                print('nop')
        bs = buildstep.BuildStep()
        bs.build = fakebuild.FakeBuild()
        props = bs.build.properties = mock.Mock()
        bs.setProperty('x', 'y', 't')
        props.setProperty.assert_called_with('x', 'y', 't', runtime=True)
        bs.setProperty('x', 'abc', 'test', runtime=True)
        props.setProperty.assert_called_with('x', 'abc', 'test', runtime=True)

    @defer.inlineCallbacks
    def test_renderableLocks(self):
        if False:
            i = 10
            return i + 15
        master_lock = locks.MasterLock('masterlock')
        worker_lock = locks.WorkerLock('workerlock')
        lock_accesses = []

        @renderer
        def rendered_locks(props):
            if False:
                while True:
                    i = 10
            master_access = locks.LockAccess(master_lock, 'counting')
            worker_access = locks.LockAccess(worker_lock, 'exclusive')
            lock_accesses.append(master_access)
            lock_accesses.append(worker_access)
            return [master_access, worker_access]
        self.setup_step(self.LockBuildStep(testcase=self, lock_accesses=lock_accesses, locks=rendered_locks))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertEqual(len(lock_accesses), 2)
        botmaster = self.step.build.builder.botmaster
        real_master_lock = (yield botmaster.getLockFromLockAccess(lock_accesses[0], self.build.config_version))
        real_worker_lock = (yield botmaster.getLockFromLockAccess(lock_accesses[1], self.build.config_version))
        self.assertTrue(real_master_lock.isAvailable(self, lock_accesses[0]))
        self.assertIn('workername', real_worker_lock.locks)
        self.assertTrue(real_worker_lock.locks['workername'].isAvailable(self, lock_accesses[1]))

    def test_compare(self):
        if False:
            return 10
        lbs1 = buildstep.BuildStep(name='me')
        lbs2 = buildstep.BuildStep(name='me')
        lbs3 = buildstep.BuildStep(name='me2')
        self.assertEqual(lbs1, lbs2)
        self.assertNotEqual(lbs1, lbs3)

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(repr(buildstep.BuildStep(name='me')), 'BuildStep(name=' + repr('me') + ')')
        self.assertEqual(repr(NewStyleStep(name='me')), 'NewStyleStep(name=' + repr('me') + ')')

    @defer.inlineCallbacks
    def test_regularLocks(self):
        if False:
            for i in range(10):
                print('nop')
        master_lock = locks.MasterLock('masterlock')
        worker_lock = locks.WorkerLock('workerlock')
        lock_accesses = [locks.LockAccess(master_lock, 'counting'), locks.LockAccess(worker_lock, 'exclusive')]
        self.setup_step(self.LockBuildStep(testcase=self, lock_accesses=lock_accesses, locks=lock_accesses))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        botmaster = self.step.build.builder.botmaster
        real_master_lock = (yield botmaster.getLockFromLockAccess(lock_accesses[0], self.build.config_version))
        real_worker_lock = (yield botmaster.getLockFromLockAccess(lock_accesses[1], self.build.config_version))
        self.assertTrue(real_master_lock.isAvailable(self, lock_accesses[0]))
        self.assertIn('workername', real_worker_lock.locks)
        self.assertTrue(real_worker_lock.locks['workername'].isAvailable(self, lock_accesses[1]))

    @defer.inlineCallbacks
    def test_cancelWhileLocksAvailable(self):
        if False:
            i = 10
            return i + 15

        def _owns_lock(step, lock):
            if False:
                i = 10
                return i + 15
            access = [step_access for (step_lock, step_access) in step.locks if step_lock == lock][0]
            return lock.isOwner(step, access)

        def _lock_available(step, lock):
            if False:
                while True:
                    i = 10
            access = [step_access for (step_lock, step_access) in step.locks if step_lock == lock][0]
            return lock.isAvailable(step, access)
        lock1 = locks.MasterLock('masterlock1')
        real_lock1 = locks.RealMasterLock(lock1)
        lock2 = locks.MasterLock('masterlock2')
        real_lock2 = locks.RealMasterLock(lock2)
        stepa = self.setup_step(self.FakeBuildStep(locks=[(real_lock1, locks.LockAccess(lock1, 'exclusive'))]))
        stepb = self.setup_step(self.FakeBuildStep(locks=[(real_lock2, locks.LockAccess(lock2, 'exclusive'))]))
        stepc = self.setup_step(self.FakeBuildStep(locks=[(real_lock1, locks.LockAccess(lock1, 'exclusive')), (real_lock2, locks.LockAccess(lock2, 'exclusive'))]))
        stepd = self.setup_step(self.FakeBuildStep(locks=[(real_lock1, locks.LockAccess(lock1, 'exclusive')), (real_lock2, locks.LockAccess(lock2, 'exclusive'))]))
        yield stepa.acquireLocks()
        yield stepb.acquireLocks()
        c_d = stepc.acquireLocks()
        d_d = stepd.acquireLocks()
        self.assertTrue(_owns_lock(stepa, real_lock1))
        self.assertTrue(_owns_lock(stepb, real_lock2))
        self.assertFalse(_owns_lock(stepc, real_lock1))
        self.assertFalse(_owns_lock(stepc, real_lock2))
        self.assertFalse(_owns_lock(stepd, real_lock1))
        self.assertFalse(_owns_lock(stepd, real_lock2))
        stepa.releaseLocks()
        yield deferLater(reactor, 0, lambda : None)
        self.assertTrue(_lock_available(stepc, real_lock1))
        self.assertFalse(_lock_available(stepc, real_lock2))
        self.assertFalse(_lock_available(stepd, real_lock1))
        self.assertFalse(_lock_available(stepd, real_lock2))
        stepc.interrupt('cancelling')
        yield c_d
        self.assertFalse(_owns_lock(stepc, real_lock1))
        self.assertFalse(_owns_lock(stepc, real_lock2))
        self.assertFalse(_lock_available(stepc, real_lock1))
        self.assertFalse(_lock_available(stepc, real_lock2))
        self.assertTrue(_lock_available(stepd, real_lock1))
        self.assertFalse(_lock_available(stepd, real_lock2))
        stepb.releaseLocks()
        self.assertTrue(_lock_available(stepd, real_lock1))
        self.assertTrue(_lock_available(stepd, real_lock2))
        yield d_d
        self.assertTrue(_owns_lock(stepd, real_lock1))
        self.assertTrue(_owns_lock(stepd, real_lock2))

    @defer.inlineCallbacks
    def test_multiple_cancel(self):
        if False:
            return 10
        step = self.setup_step(CustomActionBuildStep())

        def double_interrupt():
            if False:
                print('Hello World!')
            step.interrupt('reason1')
            step.interrupt('reason2')
            return CANCELLED
        step.action = double_interrupt
        self.expect_outcome(result=CANCELLED)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_runCommand(self):
        if False:
            return 10
        bs = buildstep.BuildStep()
        bs.worker = worker.FakeWorker(master=None)
        bs.remote = 'dummy'
        bs.build = fakebuild.FakeBuild()
        bs.build.builder.name = 'fake'
        cmd = remotecommand.RemoteShellCommand('build', ['echo', 'hello'])

        def run(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            self.assertIdentical(bs.cmd, cmd)
            return SUCCESS
        cmd.run = run
        yield bs.runCommand(cmd)
        self.assertEqual(bs.cmd, None)

    @defer.inlineCallbacks
    def test_run_command_after_interrupt(self):
        if False:
            while True:
                i = 10
        step = self.setup_step(CustomActionBuildStep())
        cmd = remotecommand.RemoteShellCommand('build', ['echo', 'hello'])

        def run(*args, **kwargs):
            if False:
                while True:
                    i = 10
            raise RuntimeError('Command must not be run when step is interrupted')
        cmd.run = run

        @defer.inlineCallbacks
        def interrupt_and_run_command():
            if False:
                i = 10
                return i + 15
            step.interrupt('reason1')
            res = (yield step.runCommand(cmd))
            return res
        step.action = interrupt_and_run_command
        self.expect_outcome(result=CANCELLED)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_lost_remote_during_interrupt(self):
        if False:
            i = 10
            return i + 15
        step = self.setup_step(CustomActionBuildStep())
        cmd = remotecommand.RemoteShellCommand('build', ['echo', 'hello'])

        @defer.inlineCallbacks
        def on_command(cmd):
            if False:
                for i in range(10):
                    print('nop')
            cmd.conn.set_expect_interrupt()
            cmd.conn.set_block_on_interrupt()
            d1 = step.interrupt('interrupt reason')
            d2 = step.interrupt(failure.Failure(error.ConnectionLost()))
            cmd.conn.unblock_waiters()
            yield d1
            yield d2
        self.expect_commands(ExpectShell(workdir='build', command=['echo', 'hello']).behavior(on_command).break_connection())

        @defer.inlineCallbacks
        def run_command():
            if False:
                while True:
                    i = 10
            res = (yield step.runCommand(cmd))
            return res.results()
        step.action = run_command
        self.expect_outcome(result=RETRY)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_start_returns_SKIPPED(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(self.SkippingBuildStep())
        self.step.finished = mock.Mock()
        self.expect_outcome(result=SKIPPED, state_string='finished (skipped)')
        yield self.run_step()
        self.step.finished.assert_not_called()

    @defer.inlineCallbacks
    def test_doStepIf_false(self):
        if False:
            while True:
                i = 10
        self.setup_step(self.FakeBuildStep(doStepIf=False))
        self.step.finished = mock.Mock()
        self.expect_outcome(result=SKIPPED, state_string='finished (skipped)')
        yield self.run_step()
        self.step.finished.assert_not_called()

    @defer.inlineCallbacks
    def test_doStepIf_renderable_false(self):
        if False:
            for i in range(10):
                print('nop')

        @util.renderer
        def dostepif(props):
            if False:
                while True:
                    i = 10
            return False
        self.setup_step(self.FakeBuildStep(doStepIf=dostepif))
        self.step.finished = mock.Mock()
        self.expect_outcome(result=SKIPPED, state_string='finished (skipped)')
        yield self.run_step()
        self.step.finished.assert_not_called()

    @defer.inlineCallbacks
    def test_doStepIf_returns_false(self):
        if False:
            while True:
                i = 10
        self.setup_step(self.FakeBuildStep(doStepIf=lambda step: False))
        self.step.finished = mock.Mock()
        self.expect_outcome(result=SKIPPED, state_string='finished (skipped)')
        yield self.run_step()
        self.step.finished.assert_not_called()

    @defer.inlineCallbacks
    def test_doStepIf_returns_deferred_false(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(self.FakeBuildStep(doStepIf=lambda step: defer.succeed(False)))
        self.step.finished = mock.Mock()
        self.expect_outcome(result=SKIPPED, state_string='finished (skipped)')
        yield self.run_step()
        self.step.finished.assert_not_called()

    def test_hideStepIf_False(self):
        if False:
            print('Hello World!')
        self._setupWaterfallTest(False, False)
        return self.run_step()

    def test_hideStepIf_True(self):
        if False:
            while True:
                i = 10
        self._setupWaterfallTest(True, True)
        return self.run_step()

    @defer.inlineCallbacks
    def test_hideStepIf_Callable_False(self):
        if False:
            return 10
        called = [False]

        def shouldHide(result, step):
            if False:
                while True:
                    i = 10
            called[0] = True
            self.assertTrue(step is self.step)
            self.assertEqual(result, SUCCESS)
            return False
        self._setupWaterfallTest(shouldHide, False)
        yield self.run_step()
        self.assertTrue(called[0])

    @defer.inlineCallbacks
    def test_hideStepIf_Callable_True(self):
        if False:
            return 10
        called = [False]

        def shouldHide(result, step):
            if False:
                print('Hello World!')
            called[0] = True
            self.assertTrue(step is self.step)
            self.assertEqual(result, SUCCESS)
            return True
        self._setupWaterfallTest(shouldHide, True)
        yield self.run_step()
        self.assertTrue(called[0])

    @defer.inlineCallbacks
    def test_hideStepIf_fails(self):
        if False:
            print('Hello World!')
        self._setupWaterfallTest(lambda x, y: 0 / 0, False, expectedResult=EXCEPTION)
        self.step.addLogWithFailure = mock.Mock()
        yield self.run_step()
        self.assertEqual(len(self.flushLoggedErrors(ZeroDivisionError)), 1)

    @defer.inlineCallbacks
    def test_hideStepIf_Callable_Exception(self):
        if False:
            i = 10
            return i + 15
        called = [False]

        def shouldHide(result, step):
            if False:
                i = 10
                return i + 15
            called[0] = True
            self.assertTrue(step is self.step)
            self.assertEqual(result, EXCEPTION)
            return True

        def createException(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError()
        self.setup_step(self.FakeBuildStep(hideStepIf=shouldHide, doStepIf=createException))
        self.expect_outcome(result=EXCEPTION, state_string='finished (exception)')
        self.expect_hidden(True)
        try:
            yield self.run_step()
        except Exception as e:
            log.err(e)
        self.assertEqual(len(self.flushLoggedErrors(RuntimeError)), 1)
        self.assertTrue(called[0])

    @defer.inlineCallbacks
    def test_step_getLog(self):
        if False:
            for i in range(10):
                print('nop')
        testcase = self

        class TestGetLogStep(buildstep.BuildStep):

            @defer.inlineCallbacks
            def run(self):
                if False:
                    print('Hello World!')
                testcase.assertRaises(KeyError, lambda : self.getLog('testy'))
                log1 = (yield self.addLog('testy'))
                log2 = self.getLog('testy')
                testcase.assertIdentical(log1, log2)
                return SUCCESS
        self.setup_step(TestGetLogStep())
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_step_renders_flunkOnFailure(self):
        if False:
            print('Hello World!')
        self.setup_step(TestBuildStep.FakeBuildStep(flunkOnFailure=properties.Property('fOF')))
        self.properties.setProperty('fOF', 'yes', 'test')
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertEqual(self.step.flunkOnFailure, 'yes')

    def test_hasStatistic(self):
        if False:
            for i in range(10):
                print('nop')
        step = buildstep.BuildStep()
        self.assertFalse(step.hasStatistic('rbi'))
        step.setStatistic('rbi', 13)
        self.assertTrue(step.hasStatistic('rbi'))

    def test_setStatistic(self):
        if False:
            i = 10
            return i + 15
        step = buildstep.BuildStep()
        step.setStatistic('rbi', 13)
        self.assertEqual(step.getStatistic('rbi'), 13)

    def test_getStatistic(self):
        if False:
            print('Hello World!')
        step = buildstep.BuildStep()
        self.assertEqual(step.getStatistic('rbi', 99), 99)
        self.assertEqual(step.getStatistic('rbi'), None)
        step.setStatistic('rbi', 13)
        self.assertEqual(step.getStatistic('rbi'), 13)

    def test_getStatistics(self):
        if False:
            print('Hello World!')
        step = buildstep.BuildStep()
        step.setStatistic('rbi', 13)
        step.setStatistic('ba', 0.298)
        self.assertEqual(step.getStatistics(), {'rbi': 13, 'ba': 0.298})

    def setup_summary_test(self):
        if False:
            while True:
                i = 10
        self.patch(NewStyleStep, 'getCurrentSummary', lambda self: defer.succeed({'step': 'C'}))
        self.patch(NewStyleStep, 'getResultSummary', lambda self: defer.succeed({'step': 'CS', 'build': 'CB'}))
        step = NewStyleStep()
        step.master = fakemaster.make_master(self, wantData=True, wantDb=True)
        step.stepid = 13
        step.build = fakebuild.FakeBuild()
        return step

    def test_updateSummary_running(self):
        if False:
            while True:
                i = 10
        step = self.setup_summary_test()
        step._running = True
        step.updateSummary()
        self.reactor.advance(1)
        self.assertEqual(step.master.data.updates.stepStateString[13], 'C')

    def test_updateSummary_running_empty_dict(self):
        if False:
            for i in range(10):
                print('nop')
        step = self.setup_summary_test()
        step.getCurrentSummary = lambda : {}
        step._running = True
        step.updateSummary()
        self.reactor.advance(1)
        self.assertEqual(step.master.data.updates.stepStateString[13], 'finished')

    def test_updateSummary_running_not_unicode(self):
        if False:
            while True:
                i = 10
        step = self.setup_summary_test()
        step.getCurrentSummary = lambda : {'step': b'bytestring'}
        step._running = True
        step.updateSummary()
        self.reactor.advance(1)
        self.assertEqual(len(self.flushLoggedErrors(TypeError)), 1)

    def test_updateSummary_running_not_dict(self):
        if False:
            while True:
                i = 10
        step = self.setup_summary_test()
        step.getCurrentSummary = lambda : 'foo!'
        step._running = True
        step.updateSummary()
        self.reactor.advance(1)
        self.assertEqual(len(self.flushLoggedErrors(TypeError)), 1)

    def test_updateSummary_finished(self):
        if False:
            print('Hello World!')
        step = self.setup_summary_test()
        step._running = False
        step.updateSummary()
        self.reactor.advance(1)
        self.assertEqual(step.master.data.updates.stepStateString[13], 'CS')

    def test_updateSummary_finished_empty_dict(self):
        if False:
            return 10
        step = self.setup_summary_test()
        step.getResultSummary = lambda : {}
        step._running = False
        step.updateSummary()
        self.reactor.advance(1)
        self.assertEqual(step.master.data.updates.stepStateString[13], 'finished')

    def test_updateSummary_finished_not_dict(self):
        if False:
            return 10
        step = self.setup_summary_test()
        step.getResultSummary = lambda : 'foo!'
        step._running = False
        step.updateSummary()
        self.reactor.advance(1)
        self.assertEqual(len(self.flushLoggedErrors(TypeError)), 1)

    def checkSummary(self, got, step, build=None):
        if False:
            i = 10
            return i + 15
        self.assertTrue(all((isinstance(k, str) for k in got)))
        self.assertTrue(all((isinstance(k, str) for k in got.values())))
        exp = {'step': step}
        if build:
            exp['build'] = build
        self.assertEqual(got, exp)

    def test_getCurrentSummary(self):
        if False:
            for i in range(10):
                print('nop')
        st = buildstep.BuildStep()
        st.description = None
        self.checkSummary(st.getCurrentSummary(), 'running')

    def test_getCurrentSummary_description(self):
        if False:
            while True:
                i = 10
        st = buildstep.BuildStep()
        st.description = 'fooing'
        self.checkSummary(st.getCurrentSummary(), 'fooing')

    def test_getCurrentSummary_descriptionSuffix(self):
        if False:
            while True:
                i = 10
        st = buildstep.BuildStep()
        st.description = 'fooing'
        st.descriptionSuffix = 'bar'
        self.checkSummary(st.getCurrentSummary(), 'fooing bar')

    def test_getCurrentSummary_description_list(self):
        if False:
            for i in range(10):
                print('nop')
        st = buildstep.BuildStep()
        st.description = ['foo', 'ing']
        self.checkSummary(st.getCurrentSummary(), 'foo ing')

    def test_getCurrentSummary_descriptionSuffix_list(self):
        if False:
            return 10
        st = buildstep.BuildStep()
        st.results = SUCCESS
        st.description = ['foo', 'ing']
        st.descriptionSuffix = ['bar', 'bar2']
        self.checkSummary(st.getCurrentSummary(), 'foo ing bar bar2')

    def test_getResultSummary(self):
        if False:
            while True:
                i = 10
        st = buildstep.BuildStep()
        st.results = SUCCESS
        st.description = None
        self.checkSummary(st.getResultSummary(), 'finished')

    def test_getResultSummary_description(self):
        if False:
            i = 10
            return i + 15
        st = buildstep.BuildStep()
        st.results = SUCCESS
        st.description = 'fooing'
        self.checkSummary(st.getResultSummary(), 'fooing')

    def test_getResultSummary_descriptionDone(self):
        if False:
            print('Hello World!')
        st = buildstep.BuildStep()
        st.results = SUCCESS
        st.description = 'fooing'
        st.descriptionDone = 'fooed'
        self.checkSummary(st.getResultSummary(), 'fooed')

    def test_getResultSummary_descriptionSuffix(self):
        if False:
            i = 10
            return i + 15
        st = buildstep.BuildStep()
        st.results = SUCCESS
        st.description = 'fooing'
        st.descriptionSuffix = 'bar'
        self.checkSummary(st.getResultSummary(), 'fooing bar')

    def test_getResultSummary_descriptionDone_and_Suffix(self):
        if False:
            i = 10
            return i + 15
        st = buildstep.BuildStep()
        st.results = SUCCESS
        st.descriptionDone = 'fooed'
        st.descriptionSuffix = 'bar'
        self.checkSummary(st.getResultSummary(), 'fooed bar')

    def test_getResultSummary_description_list(self):
        if False:
            for i in range(10):
                print('nop')
        st = buildstep.BuildStep()
        st.results = SUCCESS
        st.description = ['foo', 'ing']
        self.checkSummary(st.getResultSummary(), 'foo ing')

    def test_getResultSummary_descriptionSuffix_list(self):
        if False:
            while True:
                i = 10
        st = buildstep.BuildStep()
        st.results = SUCCESS
        st.description = ['foo', 'ing']
        st.descriptionSuffix = ['bar', 'bar2']
        self.checkSummary(st.getResultSummary(), 'foo ing bar bar2')

    @defer.inlineCallbacks
    def test_getResultSummary_descriptionSuffix_failure(self):
        if False:
            i = 10
            return i + 15
        st = buildstep.BuildStep()
        st.results = FAILURE
        st.description = 'fooing'
        self.checkSummary((yield st.getBuildResultSummary()), 'fooing (failure)', 'fooing (failure)')
        self.checkSummary(st.getResultSummary(), 'fooing (failure)')

    @defer.inlineCallbacks
    def test_getResultSummary_descriptionSuffix_skipped(self):
        if False:
            i = 10
            return i + 15
        st = buildstep.BuildStep()
        st.results = SKIPPED
        st.description = 'fooing'
        self.checkSummary((yield st.getBuildResultSummary()), 'fooing (skipped)')
        self.checkSummary(st.getResultSummary(), 'fooing (skipped)')

    @defer.inlineCallbacks
    def test_getResultSummary_description_failure_timed_out(self):
        if False:
            while True:
                i = 10
        st = buildstep.BuildStep()
        st.results = FAILURE
        st.description = 'fooing'
        st.timed_out = True
        self.checkSummary((yield st.getBuildResultSummary()), 'fooing (failure) (timed out)', 'fooing (failure) (timed out)')
        self.checkSummary(st.getResultSummary(), 'fooing (failure) (timed out)')

    def testcheckWorkerHasCommandGood(self):
        if False:
            while True:
                i = 10
        mockedWorkerVersion = mock.Mock()
        self.patch(buildstep.BuildStep, 'workerVersion', mockedWorkerVersion)
        buildstep.BuildStep().checkWorkerHasCommand('foo')
        mockedWorkerVersion.assert_called_once_with('foo')

    def testcheckWorkerHasCommandTooOld(self):
        if False:
            for i in range(10):
                print('nop')
        self.patch(buildstep.BuildStep, 'workerVersion', mock.Mock(return_value=None))
        step = buildstep.BuildStep()
        with self.assertRaisesRegex(WorkerSetupError, 'worker is too old, does not know about foo'):
            step.checkWorkerHasCommand('foo')

    @defer.inlineCallbacks
    def testRunRaisesException(self):
        if False:
            print('Hello World!')
        step = NewStyleStep()
        step.master = mock.Mock()
        step.build = mock.Mock()
        step.build.builder.botmaster.getLockFromLockAccesses = mock.Mock(return_value=[])
        step.locks = []
        step.renderables = []
        step.build.render = defer.succeed
        step.master.data.updates.addStep = lambda **kwargs: defer.succeed((0, 0, 0))
        step.addLogWithFailure = lambda x: defer.succeed(None)
        step.run = lambda : defer.fail(RuntimeError('got exception'))
        res = (yield step.startStep(mock.Mock()))
        self.assertFalse(step._running)
        errors = self.flushLoggedErrors()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].getErrorMessage(), 'got exception')
        self.assertEqual(res, EXCEPTION)

class InterfaceTests(interfaces.InterfaceTests):

    def test_signature_attributes(self):
        if False:
            for i in range(10):
                print('nop')
        for attr in ['name', 'description', 'descriptionDone', 'descriptionSuffix', 'locks', 'progressMetrics', 'useProgress', 'doStepIf', 'hideStepIf', 'haltOnFailure', 'flunkOnWarnings', 'flunkOnFailure', 'warnOnWarnings', 'warnOnFailure', 'alwaysRun', 'build', 'worker', 'progress', 'stopped']:
            self.assertTrue(hasattr(self.step, attr))

    def test_signature_setBuild(self):
        if False:
            i = 10
            return i + 15

        @self.assertArgSpecMatches(self.step.setBuild)
        def setBuild(self, build):
            if False:
                return 10
            pass

    def test_signature_setWorker(self):
        if False:
            print('Hello World!')

        @self.assertArgSpecMatches(self.step.setWorker)
        def setWorker(self, worker):
            if False:
                print('Hello World!')
            pass

    def test_signature_setupProgress(self):
        if False:
            for i in range(10):
                print('nop')

        @self.assertArgSpecMatches(self.step.setupProgress)
        def setupProgress(self):
            if False:
                print('Hello World!')
            pass

    def test_signature_startStep(self):
        if False:
            i = 10
            return i + 15

        @self.assertArgSpecMatches(self.step.startStep)
        def startStep(self, remote):
            if False:
                print('Hello World!')
            pass

    def test_signature_run(self):
        if False:
            for i in range(10):
                print('nop')

        @self.assertArgSpecMatches(self.step.run)
        def run(self):
            if False:
                i = 10
                return i + 15
            pass

    def test_signature_interrupt(self):
        if False:
            while True:
                i = 10

        @self.assertArgSpecMatches(self.step.interrupt)
        def interrupt(self, reason):
            if False:
                print('Hello World!')
            pass

    def test_signature_setProgress(self):
        if False:
            i = 10
            return i + 15

        @self.assertArgSpecMatches(self.step.setProgress)
        def setProgress(self, metric, value):
            if False:
                while True:
                    i = 10
            pass

    def test_signature_workerVersion(self):
        if False:
            i = 10
            return i + 15

        @self.assertArgSpecMatches(self.step.workerVersion)
        def workerVersion(self, command, oldversion=None):
            if False:
                i = 10
                return i + 15
            pass

    def test_signature_workerVersionIsOlderThan(self):
        if False:
            i = 10
            return i + 15

        @self.assertArgSpecMatches(self.step.workerVersionIsOlderThan)
        def workerVersionIsOlderThan(self, command, minversion):
            if False:
                print('Hello World!')
            pass

    def test_signature_getWorkerName(self):
        if False:
            print('Hello World!')

        @self.assertArgSpecMatches(self.step.getWorkerName)
        def getWorkerName(self):
            if False:
                i = 10
                return i + 15
            pass

    def test_signature_runCommand(self):
        if False:
            i = 10
            return i + 15

        @self.assertArgSpecMatches(self.step.runCommand)
        def runCommand(self, command):
            if False:
                print('Hello World!')
            pass

    def test_signature_addURL(self):
        if False:
            while True:
                i = 10

        @self.assertArgSpecMatches(self.step.addURL)
        def addURL(self, name, url):
            if False:
                for i in range(10):
                    print('nop')
            pass

    def test_signature_addLog(self):
        if False:
            return 10

        @self.assertArgSpecMatches(self.step.addLog)
        def addLog(self, name, type='s', logEncoding=None):
            if False:
                i = 10
                return i + 15
            pass

    def test_signature_getLog(self):
        if False:
            return 10

        @self.assertArgSpecMatches(self.step.getLog)
        def getLog(self, name):
            if False:
                i = 10
                return i + 15
            pass

    def test_signature_addCompleteLog(self):
        if False:
            print('Hello World!')

        @self.assertArgSpecMatches(self.step.addCompleteLog)
        def addCompleteLog(self, name, text):
            if False:
                i = 10
                return i + 15
            pass

    def test_signature_addHTMLLog(self):
        if False:
            i = 10
            return i + 15

        @self.assertArgSpecMatches(self.step.addHTMLLog)
        def addHTMLLog(self, name, html):
            if False:
                for i in range(10):
                    print('nop')
            pass

    def test_signature_addLogObserver(self):
        if False:
            for i in range(10):
                print('nop')

        @self.assertArgSpecMatches(self.step.addLogObserver)
        def addLogObserver(self, logname, observer):
            if False:
                print('Hello World!')
            pass

class TestFakeItfc(unittest.TestCase, TestBuildStepMixin, TestReactorMixin, InterfaceTests):

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.setup_test_build_step()
        self.setup_step(buildstep.BuildStep())

class TestRealItfc(unittest.TestCase, InterfaceTests):

    def setUp(self):
        if False:
            print('Hello World!')
        self.step = buildstep.BuildStep()

class CommandMixinExample(buildstep.CommandMixin, buildstep.BuildStep):

    @defer.inlineCallbacks
    def run(self):
        if False:
            print('Hello World!')
        rv = (yield self.testMethod())
        self.method_return_value = rv
        return SUCCESS

class TestCommandMixin(TestBuildStepMixin, TestReactorMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setup_test_reactor()
        yield self.setup_test_build_step()
        self.step = CommandMixinExample()
        self.setup_step(self.step)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        return self.tear_down_test_build_step()

    @defer.inlineCallbacks
    def test_runRmdir(self):
        if False:
            return 10
        self.step.testMethod = lambda : self.step.runRmdir('/some/path')
        self.expect_commands(ExpectRmdir(dir='/some/path', log_environ=False).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertTrue(self.step.method_return_value)

    @defer.inlineCallbacks
    def test_runMkdir(self):
        if False:
            for i in range(10):
                print('nop')
        self.step.testMethod = lambda : self.step.runMkdir('/some/path')
        self.expect_commands(ExpectMkdir(dir='/some/path', log_environ=False).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertTrue(self.step.method_return_value)

    @defer.inlineCallbacks
    def test_runMkdir_fails(self):
        if False:
            print('Hello World!')
        self.step.testMethod = lambda : self.step.runMkdir('/some/path')
        self.expect_commands(ExpectMkdir(dir='/some/path', log_environ=False).exit(1))
        self.expect_outcome(result=FAILURE)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_runMkdir_fails_no_abandon(self):
        if False:
            while True:
                i = 10
        self.step.testMethod = lambda : self.step.runMkdir('/some/path', abandonOnFailure=False)
        self.expect_commands(ExpectMkdir(dir='/some/path', log_environ=False).exit(1))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertFalse(self.step.method_return_value)

    @defer.inlineCallbacks
    def test_pathExists(self):
        if False:
            return 10
        self.step.testMethod = lambda : self.step.pathExists('/some/path')
        self.expect_commands(ExpectStat(file='/some/path', log_environ=False).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertTrue(self.step.method_return_value)

    @defer.inlineCallbacks
    def test_pathExists_doesnt(self):
        if False:
            i = 10
            return i + 15
        self.step.testMethod = lambda : self.step.pathExists('/some/path')
        self.expect_commands(ExpectStat(file='/some/path', log_environ=False).exit(1))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertFalse(self.step.method_return_value)

    @defer.inlineCallbacks
    def test_pathExists_logging(self):
        if False:
            print('Hello World!')
        self.step.testMethod = lambda : self.step.pathExists('/some/path')
        self.expect_commands(ExpectStat(file='/some/path', log_environ=False).log('stdio', header='NOTE: never mind\n').exit(1))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertFalse(self.step.method_return_value)
        self.assertEqual(self.step.getLog('stdio').header, 'NOTE: never mind\nprogram finished with exit code 1\n')

    def test_glob(self):
        if False:
            return 10

        @defer.inlineCallbacks
        def testFunc():
            if False:
                return 10
            res = (yield self.step.runGlob('*.pyc'))
            self.assertEqual(res, ['one.pyc', 'two.pyc'])
        self.step.testMethod = testFunc
        self.expect_commands(ExpectGlob(path='*.pyc', log_environ=False).files(['one.pyc', 'two.pyc']).exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_glob_empty(self):
        if False:
            while True:
                i = 10
        self.step.testMethod = lambda : self.step.runGlob('*.pyc')
        self.expect_commands(ExpectGlob(path='*.pyc', log_environ=False).files().exit(0))
        self.expect_outcome(result=SUCCESS)
        return self.run_step()

    def test_glob_fail(self):
        if False:
            i = 10
            return i + 15
        self.step.testMethod = lambda : self.step.runGlob('*.pyc')
        self.expect_commands(ExpectGlob(path='*.pyc', log_environ=False).exit(1))
        self.expect_outcome(result=FAILURE)
        return self.run_step()

class SimpleShellCommand(buildstep.ShellMixin, buildstep.BuildStep):

    def __init__(self, make_cmd_kwargs=None, prohibit_args=None, **kwargs):
        if False:
            i = 10
            return i + 15
        self.make_cmd_kwargs = make_cmd_kwargs or {}
        kwargs = self.setupShellMixin(kwargs, prohibitArgs=prohibit_args)
        super().__init__(**kwargs)

    @defer.inlineCallbacks
    def run(self):
        if False:
            while True:
                i = 10
        cmd = (yield self.makeRemoteShellCommand(**self.make_cmd_kwargs))
        yield self.runCommand(cmd)
        return cmd.results()

class TestShellMixin(TestBuildStepMixin, config.ConfigErrorsMixin, TestReactorMixin, unittest.TestCase):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            print('Hello World!')
        self.setup_test_reactor()
        yield self.setup_test_build_step()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tear_down_test_build_step()

    def test_setupShellMixin_bad_arg(self):
        if False:
            return 10
        mixin = SimpleShellCommand()
        with self.assertRaisesConfigError('invalid SimpleShellCommand argument invarg'):
            mixin.setupShellMixin({'invarg': 13})

    def test_setupShellMixin_prohibited_arg(self):
        if False:
            return 10
        mixin = SimpleShellCommand()
        with self.assertRaisesConfigError('invalid SimpleShellCommand argument logfiles'):
            mixin.setupShellMixin({'logfiles': None}, prohibitArgs=['logfiles'])

    def test_constructor_defaults(self):
        if False:
            return 10

        class MySubclass(SimpleShellCommand):
            timeout = 9999
        self.assertEqual(MySubclass().timeout, 9999)
        self.assertEqual(MySubclass(timeout=88).timeout, 88)
        self.assertEqual(MySubclass().logEncoding, None)
        self.assertEqual(MySubclass(logEncoding='latin-1').logEncoding, 'latin-1')
        self.assertEqual(MySubclass().description, None)
        self.assertEqual(MySubclass(description='charming').description, ['charming'])

    @defer.inlineCallbacks
    def test_prohibit_args(self):
        if False:
            print('Hello World!')
        self.setup_step(SimpleShellCommand(prohibit_args=['command'], make_cmd_kwargs={'command': ['cmd', 'arg']}))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['cmd', 'arg']).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_no_default_workdir(self):
        if False:
            return 10
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg']), want_default_work_dir=False)
        self.expect_commands(ExpectShell(workdir='build', command=['cmd', 'arg']).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_build_workdir(self):
        if False:
            return 10
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg']), want_default_work_dir=False)
        self.build.workdir = '/alternate'
        self.expect_commands(ExpectShell(workdir='/alternate', command=['cmd', 'arg']).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_build_workdir_callable(self):
        if False:
            return 10
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg']), want_default_work_dir=False)
        self.build.workdir = lambda x: '/alternate'
        self.expect_commands(ExpectShell(workdir='/alternate', command=['cmd', 'arg']).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_build_workdir_callable_error(self):
        if False:
            while True:
                i = 10
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg']), want_default_work_dir=False)
        self.build.workdir = lambda x: x.nosuchattribute
        self.expect_exception(buildstep.CallableAttributeError)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_build_workdir_renderable(self):
        if False:
            print('Hello World!')
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg']), want_default_work_dir=False)
        self.build.workdir = properties.Property('myproperty')
        self.properties.setProperty('myproperty', '/myproperty', 'test')
        self.expect_commands(ExpectShell(workdir='/myproperty', command=['cmd', 'arg']).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_step_workdir(self):
        if False:
            return 10
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], workdir='/stepdir'))
        self.build.workdir = '/builddir'
        self.expect_commands(ExpectShell(workdir='/stepdir', command=['cmd', 'arg']).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_step_renderable_workdir(self):
        if False:
            i = 10
            return i + 15

        @renderer
        def rendered_workdir(_):
            if False:
                while True:
                    i = 10
            return '/stepdir'
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], workdir=rendered_workdir))
        self.build.workdir = '/builddir'
        self.expect_commands(ExpectShell(workdir='/stepdir', command=['cmd', 'arg']).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_step_workdir_overridden(self):
        if False:
            return 10
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], workdir='/stepdir', make_cmd_kwargs={'workdir': '/overridden'}))
        self.build.workdir = '/builddir'
        self.expect_commands(ExpectShell(workdir='/overridden', command=['cmd', 'arg']).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_step_env_default(self):
        if False:
            for i in range(10):
                print('nop')
        env = {'ENV': 'TRUE'}
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], env=env))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['cmd', 'arg'], env=env).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_step_env_overridden(self):
        if False:
            print('Hello World!')
        env = {'ENV': 'TRUE'}
        env_override = {'OVERRIDE': 'TRUE'}
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], env=env, make_cmd_kwargs={'env': env_override}))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['cmd', 'arg'], env=env_override).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_extra_logfile(self):
        if False:
            while True:
                i = 10
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], logfiles={'logname': 'logpath.log'}))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['cmd', 'arg'], logfiles={'logname': 'logpath.log'}).log('logname', stdout='logline\nlogline2\n').stdout('some log\n').exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertEqual(self.step.getLog('logname').stdout, 'logline\nlogline2\n')

    @defer.inlineCallbacks
    def test_lazy_logfiles_stdout_has_stdout(self):
        if False:
            print('Hello World!')
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], lazylogfiles=True))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['cmd', 'arg']).stdout('some log\n').exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertEqual(self.step.getLog('stdio').stdout, 'some log\n')

    @defer.inlineCallbacks
    def test_lazy_logfiles_stdout_no_stdout(self):
        if False:
            print('Hello World!')
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], lazylogfiles=True))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['cmd', 'arg']).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertEqual(self.step.getLog('stdio').stdout, '')

    @defer.inlineCallbacks
    def test_lazy_logfiles_logfile(self):
        if False:
            return 10
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], lazylogfiles=True, logfiles={'logname': 'logpath.log'}))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['cmd', 'arg'], logfiles={'logname': 'logpath.log'}).log('logname', stdout='logline\nlogline2\n').exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertEqual(self.step.getLog('logname').stdout, 'logline\nlogline2\n')

    @defer.inlineCallbacks
    def test_lazy_logfiles_no_logfile(self):
        if False:
            while True:
                i = 10
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], lazylogfiles=True, logfiles={'logname': 'logpath.log'}))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['cmd', 'arg'], logfiles={'logname': 'logpath.log'}).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        with self.assertRaises(KeyError):
            self.step.getLog('logname')

    @defer.inlineCallbacks
    def test_env(self):
        if False:
            return 10
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], env={'BAR': 'BAR'}))
        self.build.builder.config.env = {'FOO': 'FOO'}
        self.expect_commands(ExpectShell(workdir='wkdir', command=['cmd', 'arg'], env={'FOO': 'FOO', 'BAR': 'BAR'}).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()

    @defer.inlineCallbacks
    def test_old_worker_args(self):
        if False:
            print('Hello World!')
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], usePTY=False, interruptSignal='DIE'), worker_version={'*': '1.1'})
        self.expect_commands(ExpectShell(workdir='wkdir', command=['cmd', 'arg']).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertEqual(self.step.getLog('stdio').header, 'NOTE: worker does not allow master to override usePTY\nNOTE: worker does not allow master to specify interruptSignal\nprogram finished with exit code 0\n')

    @defer.inlineCallbacks
    def test_new_worker_args(self):
        if False:
            print('Hello World!')
        self.setup_step(SimpleShellCommand(command=['cmd', 'arg'], usePTY=False, interruptSignal='DIE'), worker_version={'*': '3.0'})
        self.expect_commands(ExpectShell(workdir='wkdir', use_pty=False, interrupt_signal='DIE', command=['cmd', 'arg']).exit(0))
        self.expect_outcome(result=SUCCESS)
        yield self.run_step()
        self.assertEqual(self.step.getLog('stdio').header, 'program finished with exit code 0\n')

    @defer.inlineCallbacks
    def test_description(self):
        if False:
            i = 10
            return i + 15
        self.setup_step(SimpleShellCommand(command=['foo', properties.Property('bar', 'BAR')]))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['foo', 'BAR']).exit(0))
        self.expect_outcome(result=SUCCESS, state_string="'foo BAR'")
        yield self.run_step()

    @parameterized.expand(['timeout', 'timeout_without_output'])
    @defer.inlineCallbacks
    def test_description_timed_out(self, failure_reason):
        if False:
            return 10
        self.setup_step(SimpleShellCommand(command=['foo']))
        self.expect_commands(ExpectShell(workdir='wkdir', command=['foo']).update('failure_reason', failure_reason).exit(1))
        self.expect_outcome(result=FAILURE, state_string="'foo' (failure) (timed out)")
        yield self.run_step()

    def test_getResultSummary(self):
        if False:
            while True:
                i = 10
        self.setup_step(SimpleShellCommand(command=['a', ['b', 'c']]))
        self.step.results = SUCCESS
        self.assertEqual(self.step.getResultSummary(), {'step': "'a b ...'"})