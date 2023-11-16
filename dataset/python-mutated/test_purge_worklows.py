from __future__ import absolute_import
from datetime import timedelta
from st2common import log as logging
from st2common.garbage_collection.workflows import purge_workflow_executions
from st2common.models.db.workflow import WorkflowExecutionDB
from st2common.persistence.workflow import WorkflowExecution
from st2common.util import date as date_utils
from st2tests.base import CleanDbTestCase
LOG = logging.getLogger(__name__)

class TestPurgeWorkflowExecutionInstances(CleanDbTestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        CleanDbTestCase.setUpClass()
        super(TestPurgeWorkflowExecutionInstances, cls).setUpClass()

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestPurgeWorkflowExecutionInstances, self).setUp()

    def test_no_timestamp_doesnt_delete(self):
        if False:
            while True:
                i = 10
        now = date_utils.get_datetime_utc_now()
        instance_db = WorkflowExecutionDB(start_timestamp=now - timedelta(days=20), end_timestamp=now - timedelta(days=20), status='running')
        WorkflowExecution.add_or_update(instance_db)
        self.assertEqual(len(WorkflowExecution.get_all()), 1)
        expected_msg = 'Specify a valid timestamp'
        self.assertRaisesRegexp(ValueError, expected_msg, purge_workflow_executions, logger=LOG, timestamp=None)
        self.assertEqual(len(WorkflowExecution.get_all()), 1)

    def test_purge(self):
        if False:
            return 10
        now = date_utils.get_datetime_utc_now()
        instance_db = WorkflowExecutionDB(start_timestamp=now - timedelta(days=20), end_timestamp=now - timedelta(days=20), status='failed')
        WorkflowExecution.add_or_update(instance_db)
        instance_db = WorkflowExecutionDB(start_timestamp=now - timedelta(days=20), status='running')
        WorkflowExecution.add_or_update(instance_db)
        instance_db = WorkflowExecutionDB(start_timestamp=now - timedelta(days=5), end_timestamp=now - timedelta(days=5), status='succeeded')
        WorkflowExecution.add_or_update(instance_db)
        self.assertEqual(len(WorkflowExecution.get_all()), 3)
        purge_workflow_executions(logger=LOG, timestamp=now - timedelta(days=10))
        self.assertEqual(len(WorkflowExecution.get_all()), 2)

    def test_purge_incomplete(self):
        if False:
            while True:
                i = 10
        now = date_utils.get_datetime_utc_now()
        instance_db = WorkflowExecutionDB(start_timestamp=now - timedelta(days=20), end_timestamp=now - timedelta(days=20), status='cancelled')
        WorkflowExecution.add_or_update(instance_db)
        instance_db = WorkflowExecutionDB(start_timestamp=now - timedelta(days=20), status='running')
        WorkflowExecution.add_or_update(instance_db)
        instance_db = WorkflowExecutionDB(start_timestamp=now - timedelta(days=5), end_timestamp=now - timedelta(days=5), status='succeeded')
        WorkflowExecution.add_or_update(instance_db)
        self.assertEqual(len(WorkflowExecution.get_all()), 3)
        purge_workflow_executions(logger=LOG, timestamp=now - timedelta(days=10), purge_incomplete=True)
        self.assertEqual(len(WorkflowExecution.get_all()), 1)