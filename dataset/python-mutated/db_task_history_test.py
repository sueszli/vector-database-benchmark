from helpers import unittest
from helpers import with_config
import luigi
from luigi.db_task_history import DbTaskHistory
from luigi.task_status import DONE, PENDING, RUNNING
import luigi.scheduler
from luigi.parameter import ParameterVisibility

class DummyTask(luigi.Task):
    foo = luigi.Parameter(default='foo')

class ParamTask(luigi.Task):
    param1 = luigi.Parameter()
    param2 = luigi.IntParameter(visibility=ParameterVisibility.HIDDEN)
    param3 = luigi.Parameter(default='empty', visibility=ParameterVisibility.PRIVATE)

class DbTaskHistoryTest(unittest.TestCase):

    @with_config(dict(task_history=dict(db_connection='sqlite:///:memory:')))
    def setUp(self):
        if False:
            print('Hello World!')
        self.history = DbTaskHistory()

    def test_task_list(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_task(DummyTask())
        self.run_task(DummyTask(foo='bar'))
        with self.history._session() as session:
            tasks = list(self.history.find_all_by_name('DummyTask', session))
            self.assertEqual(len(tasks), 2)
            for task in tasks:
                self.assertEqual(task.name, 'DummyTask')
                self.assertEqual(task.host, 'hostname')

    def test_task_events(self):
        if False:
            print('Hello World!')
        self.run_task(DummyTask())
        with self.history._session() as session:
            tasks = list(self.history.find_all_by_name('DummyTask', session))
            self.assertEqual(len(tasks), 1)
            [task] = tasks
            self.assertEqual(task.name, 'DummyTask')
            self.assertEqual(len(task.events), 3)
            for (event, name) in zip(task.events, [DONE, RUNNING, PENDING]):
                self.assertEqual(event.event_name, name)

    def test_task_by_params(self):
        if False:
            print('Hello World!')
        task1 = ParamTask('foo', 'bar')
        task2 = ParamTask('bar', 'foo')
        with self.history._session() as session:
            self.run_task(task1)
            self.run_task(task2)
            task1_record = self.history.find_all_by_parameters(task_name='ParamTask', session=session, param1='foo', param2='bar')
            task2_record = self.history.find_all_by_parameters(task_name='ParamTask', session=session, param1='bar', param2='foo')
            for (task, records) in zip((task1, task2), (task1_record, task2_record)):
                records = list(records)
                self.assertEqual(len(records), 1)
                [record] = records
                self.assertEqual(task.task_family, record.name)
                for (param_name, param_value) in task.param_kwargs.items():
                    self.assertTrue(param_name in record.parameters)
                    self.assertEqual(str(param_value), record.parameters[param_name].value)

    def test_task_blank_param(self):
        if False:
            while True:
                i = 10
        self.run_task(DummyTask(foo=''))
        with self.history._session() as session:
            tasks = list(self.history.find_all_by_name('DummyTask', session))
            self.assertEqual(len(tasks), 1)
            task_record = tasks[0]
            self.assertEqual(task_record.name, 'DummyTask')
            self.assertEqual(task_record.host, 'hostname')
            self.assertIn('foo', task_record.parameters)
            self.assertEqual(task_record.parameters['foo'].value, '')

    def run_task(self, task):
        if False:
            for i in range(10):
                print('nop')
        task2 = luigi.scheduler.Task(task.task_id, PENDING, [], family=task.task_family, params=task.param_kwargs, retry_policy=luigi.scheduler._get_empty_retry_policy())
        self.history.task_scheduled(task2)
        self.history.task_started(task2, 'hostname')
        self.history.task_finished(task2, successful=True)

class MySQLDbTaskHistoryTest(unittest.TestCase):

    @with_config(dict(task_history=dict(db_connection='mysql+mysqlconnector://travis@localhost/luigi_test')))
    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.history = DbTaskHistory()
        except Exception:
            raise unittest.SkipTest('DBTaskHistory cannot be created: probably no MySQL available')

    def test_subsecond_timestamp(self):
        if False:
            while True:
                i = 10
        with self.history._session() as session:
            task = DummyTask()
            self.run_task(task)
            task_record = next(self.history.find_all_by_name('DummyTask', session))
            print(task_record.events)
            self.assertEqual(task_record.events[0].event_name, DONE)

    def test_utc_conversion(self):
        if False:
            i = 10
            return i + 15
        from luigi.server import from_utc
        with self.history._session() as session:
            task = DummyTask()
            self.run_task(task)
            task_record = next(self.history.find_all_by_name('DummyTask', session))
            last_event = task_record.events[0]
            try:
                print(from_utc(str(last_event.ts)))
            except ValueError:
                self.fail('Failed to convert timestamp {} to UTC'.format(last_event.ts))

    def run_task(self, task):
        if False:
            return 10
        task2 = luigi.scheduler.Task(task.task_id, PENDING, [], family=task.task_family, params=task.param_kwargs, retry_policy=luigi.scheduler._get_empty_retry_policy())
        self.history.task_scheduled(task2)
        self.history.task_started(task2, 'hostname')
        self.history.task_finished(task2, successful=True)