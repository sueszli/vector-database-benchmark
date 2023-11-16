import os
from unittest.case import SkipTest
from twisted.internet import defer
from buildbot.config import BuilderConfig
from buildbot.plugins import schedulers
from buildbot.plugins import steps
from buildbot.process.factory import BuildFactory
from buildbot.process.results import SUCCESS
from buildbot.test.util.integration import RunMasterBase
from buildbot.worker.marathon import MarathonLatentWorker
NUM_CONCURRENT = int(os.environ.get('MARATHON_TEST_NUM_CONCURRENT_BUILD', 1))

class MarathonMaster(RunMasterBase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if 'BBTEST_MARATHON_URL' not in os.environ:
            raise SkipTest('marathon integration tests only run when environment variable BBTEST_MARATHON_URL is with url to Marathon api ')

    @defer.inlineCallbacks
    def setup_config(self, num_concurrent, extra_steps=None):
        if False:
            print('Hello World!')
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
        c['builders'] = [BuilderConfig(name='testy', workernames=['marathon0'], factory=f), BuilderConfig(name='build', workernames=['marathon' + str(i) for i in range(num_concurrent)], factory=f2)]
        url = os.environ.get('BBTEST_MARATHON_URL')
        creds = os.environ.get('BBTEST_MARATHON_CREDS')
        if creds is not None:
            (user, password) = creds.split(':')
        else:
            user = password = None
        masterFQDN = os.environ.get('masterFQDN')
        marathon_extra_config = {}
        c['workers'] = [MarathonLatentWorker('marathon' + str(i), url, user, password, 'buildbot/buildbot-worker:master', marathon_extra_config=marathon_extra_config, masterFQDN=masterFQDN) for i in range(num_concurrent)]
        if masterFQDN is not None:
            c['protocols'] = {'pb': {'port': 'tcp:9989'}}
        else:
            c['protocols'] = {'pb': {'port': 'tcp:0'}}
        yield self.setup_master(c, startWorker=False)

    @defer.inlineCallbacks
    def test_trigger(self):
        if False:
            while True:
                i = 10
        yield self.setup_master(num_concurrent=NUM_CONCURRENT)
        yield self.doForceBuild()
        builds = (yield self.master.data.get(('builds',)))
        self.assertEqual(len(builds), 1 + NUM_CONCURRENT)
        for b in builds:
            self.assertEqual(b['results'], SUCCESS)