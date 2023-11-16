from __future__ import absolute_import
from datetime import timedelta
from st2common import log as logging
from st2common.garbage_collection.workflows import purge_task_executions
from st2common.models.db.workflow import TaskExecutionDB
from st2common.persistence.workflow import TaskExecution
from st2common.util import date as date_utils
from st2tests.base import CleanDbTestCase
LOG = logging.getLogger(__name__)

class TestPurgeTaskExecutionInstances(CleanDbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        CleanDbTestCase.setUpClass()
        super(TestPurgeTaskExecutionInstances, cls).setUpClass()

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestPurgeTaskExecutionInstances, self).setUp()

    def test_no_timestamp_doesnt_delete(self):
        if False:
            print('Hello World!')
        now = date_utils.get_datetime_utc_now()
        instance_db = TaskExecutionDB(start_timestamp=now - timedelta(days=20), end_timestamp=now - timedelta(days=20), status='succeeded')
        TaskExecution.add_or_update(instance_db)
        self.assertEqual(len(TaskExecution.get_all()), 1)
        expected_msg = 'Specify a valid timestamp'
        self.assertRaisesRegexp(ValueError, expected_msg, purge_task_executions, logger=LOG, timestamp=None)
        self.assertEqual(len(TaskExecution.get_all()), 1)

    def test_purge(self):
        if False:
            return 10
        now = date_utils.get_datetime_utc_now()
        instance_db = TaskExecutionDB(start_timestamp=now - timedelta(days=20), end_timestamp=now - timedelta(days=20), status='failed')
        TaskExecution.add_or_update(instance_db)
        instance_db = TaskExecutionDB(start_timestamp=now - timedelta(days=20), status='running')
        TaskExecution.add_or_update(instance_db)
        instance_db = TaskExecutionDB(start_timestamp=now - timedelta(days=5), end_timestamp=now - timedelta(days=5), status='canceled')
        TaskExecution.add_or_update(instance_db)
        self.assertEqual(len(TaskExecution.get_all()), 3)
        purge_task_executions(logger=LOG, timestamp=now - timedelta(days=10))
        self.assertEqual(len(TaskExecution.get_all()), 2)

    def test_purge_incomplete(self):
        if False:
            return 10
        now = date_utils.get_datetime_utc_now()
        instance_db = TaskExecutionDB(start_timestamp=now - timedelta(days=20), end_timestamp=now - timedelta(days=20), status='failed')
        TaskExecution.add_or_update(instance_db)
        instance_db = TaskExecutionDB(start_timestamp=now - timedelta(days=20), status='running')
        TaskExecution.add_or_update(instance_db)
        instance_db = TaskExecutionDB(start_timestamp=now - timedelta(days=5), end_timestamp=now - timedelta(days=5), status='canceled')
        TaskExecution.add_or_update(instance_db)
        self.assertEqual(len(TaskExecution.get_all()), 3)
        purge_task_executions(logger=LOG, timestamp=now - timedelta(days=10), purge_incomplete=True)
        self.assertEqual(len(TaskExecution.get_all()), 1)