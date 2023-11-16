import posixpath
from unittest import mock
from buildbot import config
from buildbot.process import factory
from buildbot.process import properties
from buildbot.process import workerforbuilder
from buildbot.test.fake import fakemaster
from buildbot.worker import base

class FakeWorkerStatus(properties.PropertiesMixin):

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.info = properties.Properties()
        self.info.setProperty('test', 'test', 'Worker')

class FakeBuild(properties.PropertiesMixin):

    def __init__(self, props=None, master=None):
        if False:
            for i in range(10):
                print('nop')
        self.builder = fakemaster.FakeBuilder(master)
        self.workerforbuilder = mock.Mock(spec=workerforbuilder.WorkerForBuilder)
        self.workerforbuilder.worker = mock.Mock(spec=base.Worker)
        self.workerforbuilder.worker.info = properties.Properties()
        self.workerforbuilder.worker.workername = 'workername'
        self.builder.config = config.BuilderConfig(name='bldr', workernames=['a'], factory=factory.BuildFactory())
        self.path_module = posixpath
        self.buildid = 92
        self.number = 13
        self.workdir = 'build'
        self.locks = []
        self.sources = {}
        if props is None:
            props = properties.Properties()
        props.build = self
        self.properties = props
        self.master = None
        self.config_version = 0

    def getProperties(self):
        if False:
            i = 10
            return i + 15
        return self.properties

    def getSourceStamp(self, codebase):
        if False:
            i = 10
            return i + 15
        if codebase in self.sources:
            return self.sources[codebase]
        return None

    def getAllSourceStamps(self):
        if False:
            i = 10
            return i + 15
        return list(self.sources.values())

    def allChanges(self):
        if False:
            for i in range(10):
                print('nop')
        for s in self.sources.values():
            for c in s.changes:
                yield c

    def allFiles(self):
        if False:
            print('Hello World!')
        files = []
        for c in self.allChanges():
            for f in c.files:
                files.append(f)
        return files

    def getBuilder(self):
        if False:
            return 10
        return self.builder

    def getWorkerInfo(self):
        if False:
            print('Hello World!')
        return self.workerforbuilder.worker.info

    def setUniqueStepName(self, step):
        if False:
            return 10
        pass

class FakeBuildForRendering:

    def render(self, r):
        if False:
            i = 10
            return i + 15
        if isinstance(r, str):
            return 'rendered:' + r
        if isinstance(r, list):
            return list((self.render(i) for i in r))
        if isinstance(r, tuple):
            return tuple((self.render(i) for i in r))
        return r