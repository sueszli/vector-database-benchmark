from twisted.internet import defer
from twisted.trial import unittest
from buildbot.test.fake import fakemaster
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util import scheduler
from buildbot.util import codebase
from buildbot.util import state

class FakeObject(codebase.AbsoluteSourceStampsMixin, state.StateMixin):
    name = 'fake-name'

    def __init__(self, master, codebases):
        if False:
            return 10
        self.master = master
        self.codebases = codebases

class TestAbsoluteSourceStampsMixin(unittest.TestCase, scheduler.SchedulerMixin, TestReactorMixin):
    codebases = {'a': {'repository': '', 'branch': 'master'}, 'b': {'repository': '', 'branch': 'master'}}

    def setUp(self):
        if False:
            return 10
        self.setup_test_reactor()
        self.master = fakemaster.make_master(self, wantDb=True, wantData=True)
        self.db = self.master.db
        self.object = FakeObject(self.master, self.codebases)

    def mkch(self, **kwargs):
        if False:
            while True:
                i = 10
        ch = self.makeFakeChange(**kwargs)
        self.master.db.changes.fakeAddChangeInstance(ch)
        return ch

    @defer.inlineCallbacks
    def test_getCodebaseDict(self):
        if False:
            i = 10
            return i + 15
        cbd = (yield self.object.getCodebaseDict('a'))
        self.assertEqual(cbd, {'repository': '', 'branch': 'master'})

    @defer.inlineCallbacks
    def test_getCodebaseDict_not_found(self):
        if False:
            print('Hello World!')
        d = self.object.getCodebaseDict('c')
        yield self.assertFailure(d, KeyError)

    @defer.inlineCallbacks
    def test_getCodebaseDict_existing(self):
        if False:
            return 10
        self.db.state.set_fake_state(self.object, 'lastCodebases', {'a': {'repository': 'A', 'revision': '1234:abc', 'branch': 'master', 'lastChange': 10}})
        cbd = (yield self.object.getCodebaseDict('a'))
        self.assertEqual(cbd, {'repository': 'A', 'revision': '1234:abc', 'branch': 'master', 'lastChange': 10})
        cbd = (yield self.object.getCodebaseDict('b'))
        self.assertEqual(cbd, {'repository': '', 'branch': 'master'})

    @defer.inlineCallbacks
    def test_recordChange(self):
        if False:
            return 10
        yield self.object.recordChange(self.mkch(codebase='a', repository='A', revision='1234:abc', branch='master', number=10))
        self.db.state.assertStateByClass('fake-name', 'FakeObject', lastCodebases={'a': {'repository': 'A', 'revision': '1234:abc', 'branch': 'master', 'lastChange': 10}})

    @defer.inlineCallbacks
    def test_recordChange_older(self):
        if False:
            i = 10
            return i + 15
        self.db.state.set_fake_state(self.object, 'lastCodebases', {'a': {'repository': 'A', 'revision': '2345:bcd', 'branch': 'master', 'lastChange': 20}})
        yield self.object.getCodebaseDict('a')
        yield self.object.recordChange(self.mkch(codebase='a', repository='A', revision='1234:abc', branch='master', number=10))
        self.db.state.assertStateByClass('fake-name', 'FakeObject', lastCodebases={'a': {'repository': 'A', 'revision': '2345:bcd', 'branch': 'master', 'lastChange': 20}})

    @defer.inlineCallbacks
    def test_recordChange_newer(self):
        if False:
            while True:
                i = 10
        self.db.state.set_fake_state(self.object, 'lastCodebases', {'a': {'repository': 'A', 'revision': '1234:abc', 'branch': 'master', 'lastChange': 10}})
        yield self.object.getCodebaseDict('a')
        yield self.object.recordChange(self.mkch(codebase='a', repository='A', revision='2345:bcd', branch='master', number=20))
        self.db.state.assertStateByClass('fake-name', 'FakeObject', lastCodebases={'a': {'repository': 'A', 'revision': '2345:bcd', 'branch': 'master', 'lastChange': 20}})