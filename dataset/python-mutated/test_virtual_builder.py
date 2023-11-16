from twisted.internet import defer
from buildbot.test.util.integration import RunMasterBase

class ShellMaster(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config(self):
        if False:
            for i in range(10):
                print('nop')
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.plugins import schedulers
        from buildbot.plugins import steps
        from buildbot.process.factory import BuildFactory
        c['schedulers'] = [schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        f = BuildFactory()
        f.addStep(steps.ShellCommand(command='echo hello'))
        c['builders'] = [BuilderConfig(name='testy', workernames=['local1'], properties={'virtual_builder_name': 'virtual_testy', 'virtual_builder_description': 'I am a virtual builder', 'virtual_builder_project': 'virtual_project', 'virtual_builder_tags': ['virtual']}, factory=f)]
        yield self.setup_master(c)

    @defer.inlineCallbacks
    def test_shell(self):
        if False:
            print('Hello World!')
        yield self.setup_config()
        build = (yield self.doForceBuild(wantSteps=True, wantLogs=True))
        self.assertEqual(build['buildid'], 1)
        builders = (yield self.master.data.get(('builders',)))
        self.assertEqual(len(builders), 2)
        self.assertEqual(builders[1], {'masterids': [], 'tags': ['virtual', '_virtual_'], 'projectid': 1, 'description': 'I am a virtual builder', 'description_format': None, 'description_html': None, 'name': 'virtual_testy', 'builderid': 2})
        self.assertEqual(build['builderid'], builders[1]['builderid'])