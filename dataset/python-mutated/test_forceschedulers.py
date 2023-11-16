from twisted.internet import defer
from twisted.trial import unittest
from buildbot.data import forceschedulers
from buildbot.schedulers.forcesched import ForceScheduler
from buildbot.test.util import endpoint
expected_default = {'all_fields': [{'columns': 1, 'autopopulate': None, 'default': '', 'fields': [{'default': '', 'autopopulate': None, 'fullName': 'username', 'hide': False, 'label': 'Your name:', 'maxsize': None, 'multiple': False, 'name': 'username', 'need_email': True, 'regex': None, 'required': False, 'size': 30, 'tablabel': 'Your name:', 'type': 'username'}, {'default': 'force build', 'autopopulate': None, 'fullName': 'reason', 'hide': False, 'label': 'reason', 'maxsize': None, 'multiple': False, 'name': 'reason', 'regex': None, 'required': False, 'size': 20, 'tablabel': 'reason', 'type': 'text'}, {'default': 0, 'autopopulate': None, 'fullName': 'priority', 'hide': False, 'label': 'priority', 'maxsize': None, 'multiple': False, 'name': 'priority', 'regex': None, 'required': False, 'size': 10, 'tablabel': 'priority', 'type': 'int'}], 'fullName': None, 'hide': False, 'label': '', 'layout': 'vertical', 'maxsize': None, 'multiple': False, 'name': '', 'regex': None, 'required': False, 'tablabel': '', 'type': 'nested'}, {'columns': 2, 'default': '', 'fields': [{'default': '', 'autopopulate': None, 'fullName': 'branch', 'hide': False, 'label': 'Branch:', 'multiple': False, 'maxsize': None, 'name': 'branch', 'regex': None, 'required': False, 'size': 10, 'tablabel': 'Branch:', 'type': 'text'}, {'default': '', 'autopopulate': None, 'fullName': 'project', 'hide': False, 'label': 'Project:', 'maxsize': None, 'multiple': False, 'name': 'project', 'regex': None, 'required': False, 'size': 10, 'tablabel': 'Project:', 'type': 'text'}, {'default': '', 'autopopulate': None, 'fullName': 'repository', 'hide': False, 'label': 'Repository:', 'maxsize': None, 'multiple': False, 'name': 'repository', 'regex': None, 'required': False, 'size': 10, 'tablabel': 'Repository:', 'type': 'text'}, {'default': '', 'autopopulate': None, 'fullName': 'revision', 'hide': False, 'label': 'Revision:', 'maxsize': None, 'multiple': False, 'name': 'revision', 'regex': None, 'required': False, 'size': 10, 'tablabel': 'Revision:', 'type': 'text'}], 'autopopulate': None, 'fullName': None, 'hide': False, 'label': '', 'layout': 'vertical', 'maxsize': None, 'multiple': False, 'name': '', 'regex': None, 'required': False, 'tablabel': '', 'type': 'nested'}], 'builder_names': ['builder'], 'button_name': 'defaultforce', 'label': 'defaultforce', 'name': 'defaultforce', 'enabled': True}

class ForceschedulerEndpoint(endpoint.EndpointMixin, unittest.TestCase):
    endpointClass = forceschedulers.ForceSchedulerEndpoint
    resourceTypeClass = forceschedulers.ForceScheduler
    maxDiff = None

    def setUp(self):
        if False:
            while True:
                i = 10
        self.setUpEndpoint()
        scheds = [ForceScheduler(name='defaultforce', builderNames=['builder'])]
        self.master.allSchedulers = lambda : scheds

    def tearDown(self):
        if False:
            return 10
        self.tearDownEndpoint()

    @defer.inlineCallbacks
    def test_get_existing(self):
        if False:
            i = 10
            return i + 15
        res = (yield self.callGet(('forceschedulers', 'defaultforce')))
        self.validateData(res)
        self.assertEqual(res, expected_default)

    @defer.inlineCallbacks
    def test_get_missing(self):
        if False:
            for i in range(10):
                print('nop')
        res = (yield self.callGet(('forceschedulers', 'foo')))
        self.assertEqual(res, None)

class ForceSchedulersEndpoint(endpoint.EndpointMixin, unittest.TestCase):
    endpointClass = forceschedulers.ForceSchedulersEndpoint
    resourceTypeClass = forceschedulers.ForceScheduler
    maxDiff = None

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.setUpEndpoint()
        scheds = [ForceScheduler(name='defaultforce', builderNames=['builder'])]
        self.master.allSchedulers = lambda : scheds

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.tearDownEndpoint()

    @defer.inlineCallbacks
    def test_get_existing(self):
        if False:
            for i in range(10):
                print('nop')
        res = (yield self.callGet(('forceschedulers',)))
        self.assertEqual(res, [expected_default])