import pickle
from unittest import mock
from unittest import TestCase
from apps.core.task.coretaskstate import Options, TaskDefinition, TaskDesc
from golem.environments.environment import Environment
from golem.testutils import PEP8MixIn

class TestTaskDesc(TestCase):

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        td = TaskDesc()
        self.assertIsInstance(td, TaskDesc)

class TestOptions(TestCase):

    def test_option(self):
        if False:
            for i in range(10):
                print('nop')
        opt = Options()
        assert isinstance(opt.environment, Environment)
        assert opt.name == ''

class TestCoreTaskStateStyle(TestCase, PEP8MixIn):
    PEP8_FILES = ['apps/core/task/coretaskstate.py']

class TestPicklesFrom_0_17_1(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.task_definition = TaskDefinition()
        self.assertTrue(hasattr(self.task_definition, 'compute_on'))
        self.assertTrue(hasattr(self.task_definition, 'concent_enabled'))
        super().setUp()

    def ser_deser(self):
        if False:
            for i in range(10):
                print('nop')
        pickled = pickle.dumps(self.task_definition)
        self.task_definition = pickle.loads(pickled)

    def test_missing_compute_on(self, *_):
        if False:
            i = 10
            return i + 15
        del self.task_definition.compute_on
        with mock.patch('apps.core.task.coretaskstate.TaskDefinition.__getstate__', side_effect=lambda : self.task_definition.__dict__):
            self.ser_deser()
        self.assertEqual(self.task_definition.compute_on, 'cpu')

    def test_missing_concent_enabled(self, *_):
        if False:
            i = 10
            return i + 15
        del self.task_definition.concent_enabled
        with mock.patch('apps.core.task.coretaskstate.TaskDefinition.__getstate__', side_effect=lambda : self.task_definition.__dict__):
            self.ser_deser()
        self.assertEqual(self.task_definition.concent_enabled, False)

class TestPicklesFrom_0_18_0(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.task_definition = TaskDefinition()
        self.assertTrue(hasattr(self.task_definition, 'name'))
        self.assertTrue(hasattr(self.task_definition, 'timeout'))
        self.assertTrue(hasattr(self.task_definition, 'subtasks_count'))
        super().setUp()

    def ser_deser(self):
        if False:
            return 10
        pickled = pickle.dumps(self.task_definition)
        self.task_definition = pickle.loads(pickled)

    def test_old_attributes_version_0(self, *_):
        if False:
            for i in range(10):
                print('nop')
        del self.task_definition.name
        del self.task_definition.timeout
        del self.task_definition.subtasks_count
        self.task_definition.task_name = 'some_name'
        self.task_definition.full_task_timeout = '00:01:00'
        self.task_definition.total_subtasks = 1
        with mock.patch('apps.core.task.coretaskstate.TaskDefinition.__getstate__', side_effect=lambda : self.task_definition.__dict__):
            self.ser_deser()
        self.assertEqual(self.task_definition.name, 'some_name')
        self.assertEqual(self.task_definition.timeout, '00:01:00')
        self.assertEqual(self.task_definition.subtasks_count, 1)

    def test_old_attributes_version_1(self, *_):
        if False:
            while True:
                i = 10
        del self.task_definition.name
        del self.task_definition.timeout
        del self.task_definition.subtasks_count
        self.task_definition.task_name = 'some_name'
        self.task_definition.full_task_timeout = '00:01:00'
        self.task_definition.total_subtasks = 1
        with mock.patch('apps.core.task.coretaskstate.TaskDefinition.__getstate__', side_effect=lambda : ('0.18.0', self.task_definition.__dict__)):
            self.ser_deser()
        self.assertEqual(self.task_definition.name, 'some_name')
        self.assertEqual(self.task_definition.timeout, '00:01:00')
        self.assertEqual(self.task_definition.subtasks_count, 1)