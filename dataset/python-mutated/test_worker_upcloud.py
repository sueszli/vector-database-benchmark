import os
from unittest.case import SkipTest
from twisted.internet import defer
from buildbot.config import BuilderConfig
from buildbot.plugins import schedulers
from buildbot.plugins import steps
from buildbot.process.factory import BuildFactory
from buildbot.process.results import SUCCESS
from buildbot.test.util.integration import RunMasterBase
from buildbot.worker.upcloud import UpcloudLatentWorker
NUM_CONCURRENT = int(os.environ.get('BUILDBOT_TEST_NUM_CONCURRENT_BUILD', 1))

class UpcloudMaster(RunMasterBase):
    timeout = 300

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if 'BBTEST_UPCLOUD_CREDS' not in os.environ:
            raise SkipTest('upcloud integration tests only run when environment variable BBTEST_UPCLOUD_CREDS is set to valid upcloud credentials ')

    @defer.inlineCallbacks
    def test_trigger(self):
        if False:
            while True:
                i = 10
        yield self.setup_master(masterConfig(num_concurrent=1), startWorker=False)
        yield self.doForceBuild()
        builds = (yield self.master.data.get(('builds',)))
        self.assertEqual(len(builds), 1 + NUM_CONCURRENT)
        for b in builds:
            self.assertEqual(b['results'], SUCCESS)

def masterConfig(num_concurrent, extra_steps=None):
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
    c['builders'] = [BuilderConfig(name='testy', workernames=['upcloud0'], factory=f), BuilderConfig(name='build', workernames=['upcloud' + str(i) for i in range(num_concurrent)], factory=f2)]
    creds = os.environ.get('BBTEST_UPCLOUD_CREDS')
    if creds is not None:
        (user, password) = creds.split(':')
    else:
        raise RuntimeError('Cannot run this test without credentials')
    masterFQDN = os.environ.get('masterFQDN', 'localhost')
    c['workers'] = []
    for i in range(num_concurrent):
        upcloud_host_config = {'user_data': f'\n#!/usr/bin/env bash\ngroupadd -g 999 buildbot\nuseradd -u 999 -g buildbot -s /bin/bash -d /buildworker -m buildbot\npasswd -l buildbot\napt update\napt install -y git python3 python3-dev python3-pip sudo gnupg curl\npip3 install buildbot-worker service_identity\nchown -R buildbot:buildbot /buildworker\ncat <<EOF >> /etc/hosts\n127.0.1.1    upcloud{i}\nEOF\ncat <<EOF >/etc/sudoers.d/buildbot\nbuidbot ALL=(ALL) NOPASSWD:ALL\nEOF\nsudo -H -u buildbot bash -c "buildbot-worker create-worker /buildworker {masterFQDN} upcloud{i} pass"\nsudo -H -u buildbot bash -c "buildbot-worker start /buildworker"\n'}
        c['workers'].append(UpcloudLatentWorker('upcloud' + str(i), api_username=user, api_password=password, image='Debian GNU/Linux 9 (Stretch)', hostconfig=upcloud_host_config, masterFQDN=masterFQDN))
    if masterFQDN is not None:
        c['protocols'] = {'pb': {'port': 'tcp:9989'}}
    else:
        c['protocols'] = {'pb': {'port': 'tcp:0'}}
    return c