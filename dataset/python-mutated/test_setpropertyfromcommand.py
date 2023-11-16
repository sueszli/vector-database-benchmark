from twisted.internet import defer
from twisted.internet import reactor
from twisted.internet import task
from buildbot.test.util.integration import RunMasterBase

class SetPropertyFromCommand(RunMasterBase):

    @defer.inlineCallbacks
    def setup_config(self):
        if False:
            i = 10
            return i + 15
        c = {}
        from buildbot.plugins import schedulers
        from buildbot.plugins import steps
        from buildbot.plugins import util
        c['schedulers'] = [schedulers.ForceScheduler(name='force', builderNames=['testy'])]
        f = util.BuildFactory()
        f.addStep(steps.SetPropertyFromCommand(property='test', command=['echo', 'foo']))
        c['builders'] = [util.BuilderConfig(name='testy', workernames=['local1'], factory=f)]
        yield self.setup_master(c)

    @defer.inlineCallbacks
    def test_setProp(self):
        if False:
            print('Hello World!')
        yield self.setup_config()
        oldNewLog = self.master.data.updates.addLog

        @defer.inlineCallbacks
        def newLog(*arg, **kw):
            if False:
                print('Hello World!')
            yield task.deferLater(reactor, 0.1, lambda : None)
            res = (yield oldNewLog(*arg, **kw))
            return res
        self.master.data.updates.addLog = newLog
        build = (yield self.doForceBuild(wantProperties=True))
        self.assertEqual(build['properties']['test'], ('foo', 'SetPropertyFromCommand Step'))

class SetPropertyFromCommandPB(SetPropertyFromCommand):
    proto = 'pb'

class SetPropertyFromCommandMsgPack(SetPropertyFromCommand):
    proto = 'msgpack'