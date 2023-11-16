import os
from twisted.internet import defer
from twisted.internet import reactor
from buildbot.test.util.decorators import flaky
from buildbot.test.util.integration import RunMasterBase

class TryClientE2E(RunMasterBase):
    timeout = 15

    @defer.inlineCallbacks
    def setup_config(self):
        if False:
            print('Hello World!')
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.plugins import steps
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.Try_Userpass(name='try', builderNames=['testy'], port='tcp:0', userpass=[('alice', 'pw1')])]
        f = BuildFactory()
        f.addStep(steps.ShellCommand(command='echo hello'))
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    @flaky(bugNumber=7084)
    @defer.inlineCallbacks
    def test_shell(self):
        if False:
            print('Hello World!')
        yield self.setup_config()

        def trigger_callback():
            if False:
                for i in range(10):
                    print('nop')
            port = self.master.pbmanager.dispatchers['tcp:0'].port.getHost().port

            def thd():
                if False:
                    return 10
                os.system(f'buildbot try --connect=pb --master=127.0.0.1:{port} -b testy --property=foo:bar --username=alice --passwd=pw1 --vc=none')
            reactor.callInThread(thd)
        build = (yield self.doForceBuild(wantSteps=True, triggerCallback=trigger_callback, wantLogs=True, wantProperties=True))
        self.assertEqual(build['buildid'], 1)