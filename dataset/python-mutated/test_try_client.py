import os
from unittest import mock
from twisted.internet import defer
from twisted.internet import reactor
from twisted.python import log
from twisted.python.filepath import FilePath
from buildbot import util
from buildbot.clients import tryclient
from buildbot.schedulers import trysched
from buildbot.test.util import www
from buildbot.test.util.integration import RunMasterBase

@defer.inlineCallbacks
def waitFor(fn):
    if False:
        for i in range(10):
            print('nop')
    while True:
        res = (yield fn())
        if res:
            return res
        yield util.asyncSleep(0.01)

class Schedulers(RunMasterBase, www.RequiresWwwMixin):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.master = None
        self.sch = None

        def spawnProcess(pp, executable, args, environ):
            if False:
                return 10
            tmpfile = os.path.join(self.jobdir, 'tmp', 'testy')
            newfile = os.path.join(self.jobdir, 'new', 'testy')
            with open(tmpfile, 'w', encoding='utf-8') as f:
                f.write(pp.job)
            os.rename(tmpfile, newfile)
            log.msg(f'wrote jobfile {newfile}')
            d = self.sch.watcher.poll()
            d.addErrback(log.err, 'while polling')

            @d.addCallback
            def finished(_):
                if False:
                    for i in range(10):
                        print('nop')
                st = mock.Mock()
                st.value.signal = None
                st.value.exitCode = 0
                pp.processEnded(st)
        self.patch(reactor, 'spawnProcess', spawnProcess)
        self.sourcestamp = tryclient.SourceStamp(branch='br', revision='rr', patch=(0, '++--'))

        def getSourceStamp(vctype, treetop, branch=None, repository=None):
            if False:
                return 10
            return defer.succeed(self.sourcestamp)
        self.patch(tryclient, 'getSourceStamp', getSourceStamp)
        self.output = []
        self.patch(tryclient.Try, 'printStatus', lambda _: None)

        def output(*msg):
            if False:
                i = 10
                return i + 15
            msg = ' '.join(map(str, msg))
            log.msg(f'output: {msg}')
            self.output.append(msg)
        self.patch(tryclient, 'output', output)

    def setupJobdir(self):
        if False:
            i = 10
            return i + 15
        jobdir = FilePath(self.mktemp())
        jobdir.createDirectory()
        self.jobdir = jobdir.path
        for sub in ('new', 'tmp', 'cur'):
            jobdir.child(sub).createDirectory()
        return self.jobdir

    @defer.inlineCallbacks
    def setup_config(self, extra_config):
        if False:
            for i in range(10):
                print('nop')
        c = {}
        from buildbot.config import BuilderConfig
        from buildbot.process import results
        from buildbot.process.buildstep import BuildStep
        from buildbot.process.factory import BuildFactory

        class MyBuildStep(BuildStep):

            def run(self):
                if False:
                    i = 10
                    return i + 15
                return results.SUCCESS
        c['change_source'] = []
        c['schedulers'] = []
        f1 = BuildFactory()
        f1.addStep(MyBuildStep(name='one'))
        f1.addStep(MyBuildStep(name='two'))
        c['builders'] = [BuilderConfig(name='a', workernames=['local1'], factory=f1)]
        c['title'] = 'test'
        c['titleURL'] = 'test'
        c['buildbotURL'] = 'http://localhost:8010/'
        c['mq'] = {'debug': True}
        c.update(extra_config)
        yield self.setup_master(c)

    @defer.inlineCallbacks
    def startMaster(self, sch):
        if False:
            while True:
                i = 10
        extra_config = {'schedulers': [sch]}
        self.sch = sch
        yield self.setup_config(extra_config)
        yield waitFor(lambda : self.sch.active)
        if isinstance(self.sch, trysched.Try_Userpass):

            def getSchedulerPort():
                if False:
                    for i in range(10):
                        print('nop')
                if not self.sch.registrations:
                    return None
                self.serverPort = self.sch.registrations[0].getPort()
                log.msg(f'Scheduler registered at port {self.serverPort}')
                return True
            yield waitFor(getSchedulerPort)

    def runClient(self, config):
        if False:
            while True:
                i = 10
        self.clt = tryclient.Try(config)
        return self.clt.run_impl()

    @defer.inlineCallbacks
    def test_userpass_no_wait(self):
        if False:
            print('Hello World!')
        yield self.startMaster(trysched.Try_Userpass('try', ['a'], 0, [('u', b'p')]))
        yield self.runClient({'connect': 'pb', 'master': f'127.0.0.1:{self.serverPort}', 'username': 'u', 'passwd': b'p'})
        self.assertEqual(self.output, ["using 'pb' connect method", 'job created', 'Delivering job; comment= None', 'job has been delivered', 'not waiting for builds to finish'])
        buildsets = (yield self.master.db.buildsets.getBuildsets())
        self.assertEqual(len(buildsets), 1)

    @defer.inlineCallbacks
    def test_userpass_wait(self):
        if False:
            return 10
        yield self.startMaster(trysched.Try_Userpass('try', ['a'], 0, [('u', b'p')]))
        yield self.runClient({'connect': 'pb', 'master': f'127.0.0.1:{self.serverPort}', 'username': 'u', 'passwd': b'p', 'wait': True})
        self.assertEqual(self.output, ["using 'pb' connect method", 'job created', 'Delivering job; comment= None', 'job has been delivered', 'All Builds Complete', 'a: success (build successful)'])
        buildsets = (yield self.master.db.buildsets.getBuildsets())
        self.assertEqual(len(buildsets), 1)

    @defer.inlineCallbacks
    def test_userpass_wait_bytes(self):
        if False:
            i = 10
            return i + 15
        self.sourcestamp = tryclient.SourceStamp(branch=b'br', revision=b'rr', patch=(0, b'++--'))
        yield self.startMaster(trysched.Try_Userpass('try', ['a'], 0, [('u', b'p')]))
        yield self.runClient({'connect': 'pb', 'master': f'127.0.0.1:{self.serverPort}', 'username': 'u', 'passwd': b'p', 'wait': True})
        self.assertEqual(self.output, ["using 'pb' connect method", 'job created', 'Delivering job; comment= None', 'job has been delivered', 'All Builds Complete', 'a: success (build successful)'])
        buildsets = (yield self.master.db.buildsets.getBuildsets())
        self.assertEqual(len(buildsets), 1)

    @defer.inlineCallbacks
    def test_userpass_wait_dryrun(self):
        if False:
            return 10
        yield self.startMaster(trysched.Try_Userpass('try', ['a'], 0, [('u', b'p')]))
        yield self.runClient({'connect': 'pb', 'master': f'127.0.0.1:{self.serverPort}', 'username': 'u', 'passwd': b'p', 'wait': True, 'dryrun': True})
        self.assertEqual(self.output, ["using 'pb' connect method", 'job created', 'Job:\n\tRepository: \n\tProject: \n\tBranch: br\n\tRevision: rr\n\tBuilders: None\n++--', 'job has been delivered', 'All Builds Complete'])
        buildsets = (yield self.master.db.buildsets.getBuildsets())
        self.assertEqual(len(buildsets), 0)

    @defer.inlineCallbacks
    def test_userpass_list_builders(self):
        if False:
            return 10
        yield self.startMaster(trysched.Try_Userpass('try', ['a'], 0, [('u', b'p')]))
        yield self.runClient({'connect': 'pb', 'get-builder-names': True, 'master': f'127.0.0.1:{self.serverPort}', 'username': 'u', 'passwd': b'p'})
        self.assertEqual(self.output, ["using 'pb' connect method", 'The following builders are available for the try scheduler: ', 'a'])
        buildsets = (yield self.master.db.buildsets.getBuildsets())
        self.assertEqual(len(buildsets), 0)

    @defer.inlineCallbacks
    def test_jobdir_no_wait(self):
        if False:
            for i in range(10):
                print('nop')
        jobdir = self.setupJobdir()
        yield self.startMaster(trysched.Try_Jobdir('try', ['a'], jobdir))
        yield self.runClient({'connect': 'ssh', 'master': '127.0.0.1', 'username': 'u', 'passwd': b'p', 'builders': 'a'})
        self.assertEqual(self.output, ["using 'ssh' connect method", 'job created', 'job has been delivered', 'not waiting for builds to finish'])
        buildsets = (yield self.master.db.buildsets.getBuildsets())
        self.assertEqual(len(buildsets), 1)

    @defer.inlineCallbacks
    def test_jobdir_wait(self):
        if False:
            i = 10
            return i + 15
        jobdir = self.setupJobdir()
        yield self.startMaster(trysched.Try_Jobdir('try', ['a'], jobdir))
        yield self.runClient({'connect': 'ssh', 'wait': True, 'host': '127.0.0.1', 'username': 'u', 'passwd': b'p', 'builders': 'a'})
        self.assertEqual(self.output, ["using 'ssh' connect method", 'job created', 'job has been delivered', 'waiting for builds with ssh is not supported'])
        buildsets = (yield self.master.db.buildsets.getBuildsets())
        self.assertEqual(len(buildsets), 1)