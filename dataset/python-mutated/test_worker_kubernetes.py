import os
from unittest.case import SkipTest
from twisted.internet import defer
from buildbot.config import BuilderConfig
from buildbot.plugins import schedulers
from buildbot.plugins import steps
from buildbot.process.factory import BuildFactory
from buildbot.process.results import SUCCESS
from buildbot.test.util.integration import RunMasterBase
from buildbot.util import kubeclientservice
from buildbot.worker import kubernetes
NUM_CONCURRENT = int(os.environ.get('KUBE_TEST_NUM_CONCURRENT_BUILD', 1))

class KubernetesMaster(RunMasterBase):
    timeout = 200

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if 'TEST_KUBERNETES' not in os.environ:
            raise SkipTest('kubernetes integration tests only run when environment variable TEST_KUBERNETES is set')
        if 'masterFQDN' not in os.environ:
            raise SkipTest("you need to export masterFQDN. You have example in the test file. Make sure that you're spawned worker can callback this IP")

    @defer.inlineCallbacks
    def setup_config(self, num_concurrent, extra_steps=None):
        if False:
            while True:
                i = 10
        if extra_steps is None:
            extra_steps = []
        c = {}
        c['schedulers'] = [schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        triggereables = []
        for i in range(num_concurrent):
            c['schedulers'].append(schedulers.Triggerable(name='trigsched' + str(i), builderNames=['build']))
            triggereables.append('trigsched' + str(i))
        f = BuildFactory()
        f.addStep(steps.ShellCommand(command='echo hello'))
        f.addStep(steps.Trigger(schedulerNames=triggereables, waitForFinish=True, updateSourceStamp=True))
        f.addStep(steps.ShellCommand(command='echo world'))
        f2 = BuildFactory()
        f2.addStep(steps.ShellCommand(command='echo ola'))
        for step in extra_steps:
            f2.addStep(step)
        c['builders'] = [BuilderConfig(name='testy', workernames=['kubernetes0'], factory=f), BuilderConfig(name='build', workernames=['kubernetes' + str(i) for i in range(num_concurrent)], factory=f2)]
        masterFQDN = os.environ.get('masterFQDN')
        c['workers'] = [kubernetes.KubeLatentWorker('kubernetes' + str(i), 'buildbot/buildbot-worker', kube_config=kubeclientservice.KubeCtlProxyConfigLoader(namespace=os.getenv('KUBE_NAMESPACE', 'default')), masterFQDN=masterFQDN) for i in range(num_concurrent)]
        c['protocols'] = {'pb': {'port': 'tcp:9989'}}
        yield self.setup_master(c, startWorker=False)

    @defer.inlineCallbacks
    def test_trigger(self):
        if False:
            print('Hello World!')
        yield self.setup_config(num_concurrent=NUM_CONCURRENT)
        yield self.doForceBuild()
        builds = (yield self.master.data.get(('builds',)))
        self.assertEqual(len(builds), 1 + NUM_CONCURRENT)
        for b in builds:
            self.assertEqual(b['results'], SUCCESS)

class KubernetesMasterTReq(KubernetesMaster):

    def setup(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.patch(kubernetes.KubeClientService, 'PREFER_TREQ', True)