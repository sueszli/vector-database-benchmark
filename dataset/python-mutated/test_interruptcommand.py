from twisted.internet import defer
from buildbot.process.results import CANCELLED
from buildbot.test.util.decorators import flaky
from buildbot.test.util.integration import RunMasterBase
from buildbot.util import asyncSleep

class InterruptCommand(RunMasterBase):
    """Make sure we can interrupt a command"""

    @defer.inlineCallbacks
    def setup_config(self):
        if False:
            print('Hello World!')
        c = {}
        from buildbot.plugins import schedulers
        from buildbot.plugins import steps
        from buildbot.plugins import util

        class SleepAndInterrupt(steps.ShellSequence):

            @defer.inlineCallbacks
            def run(self):
                if False:
                    print('Hello World!')
                if self.worker.worker_system == 'nt':
                    sleep = 'waitfor SomethingThatIsNeverHappening /t 100 >nul 2>&1'
                else:
                    sleep = ['sleep', '100']
                d = self.runShellSequence([util.ShellArg(sleep)])
                yield asyncSleep(1)
                self.interrupt('just testing')
                res = (yield d)
                return res
        c['schedulers'] = [schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        f = util.BuildFactory()
        f.addStep(SleepAndInterrupt())
        c['builders'] = [util.BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    @flaky(bugNumber=4404, onPlatform='win32')
    @defer.inlineCallbacks
    def test_interrupt(self):
        if False:
            return 10
        yield self.setup_config()
        build = (yield self.doForceBuild(wantSteps=True))
        self.assertEqual(build['steps'][-1]['results'], CANCELLED)

class InterruptCommandPb(InterruptCommand):
    proto = 'pb'

class InterruptCommandMsgPack(InterruptCommand):
    proto = 'msgpack'