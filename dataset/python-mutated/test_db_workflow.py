from __future__ import absolute_import
import mock
import uuid
import st2tests
from st2common.models.db import workflow as wf_db_models
from st2common.persistence import workflow as wf_db_access
from st2common.transport import publishers
from st2common.exceptions import db as db_exc

@mock.patch.object(publishers.PoolPublisher, 'publish', mock.MagicMock())
class WorkflowExecutionModelTest(st2tests.DbTestCase):

    def test_workflow_execution_crud(self):
        if False:
            for i in range(10):
                print('nop')
        initial = wf_db_models.WorkflowExecutionDB()
        initial.action_execution = uuid.uuid4().hex
        initial.graph = {'var1': 'foobar'}
        initial.status = 'requested'
        created = wf_db_access.WorkflowExecution.add_or_update(initial)
        self.assertEqual(initial.rev, 1)
        doc_id = created.id
        retrieved = wf_db_access.WorkflowExecution.get_by_id(doc_id)
        self.assertEqual(created.action_execution, retrieved.action_execution)
        self.assertDictEqual(created.graph, retrieved.graph)
        self.assertEqual(created.status, retrieved.status)
        graph = {'var1': 'fubar'}
        status = 'running'
        retrieved = wf_db_access.WorkflowExecution.update(retrieved, graph=graph, status=status)
        updated = wf_db_access.WorkflowExecution.get_by_id(doc_id)
        self.assertNotEqual(created.rev, updated.rev)
        self.assertEqual(retrieved.rev, updated.rev)
        self.assertEqual(retrieved.action_execution, updated.action_execution)
        self.assertDictEqual(retrieved.graph, updated.graph)
        self.assertEqual(retrieved.status, updated.status)
        retrieved.graph = {'var2': 'fubar'}
        retrieved = wf_db_access.WorkflowExecution.add_or_update(retrieved)
        updated = wf_db_access.WorkflowExecution.get_by_id(doc_id)
        self.assertNotEqual(created.rev, updated.rev)
        self.assertEqual(retrieved.rev, updated.rev)
        self.assertEqual(retrieved.action_execution, updated.action_execution)
        self.assertDictEqual(retrieved.graph, updated.graph)
        self.assertEqual(retrieved.status, updated.status)
        created.delete()
        self.assertRaises(db_exc.StackStormDBObjectNotFoundError, wf_db_access.WorkflowExecution.get_by_id, doc_id)

    def test_workflow_execution_write_conflict(self):
        if False:
            print('Hello World!')
        initial = wf_db_models.WorkflowExecutionDB()
        initial.action_execution = uuid.uuid4().hex
        initial.graph = {'var1': 'foobar'}
        initial.status = 'requested'
        created = wf_db_access.WorkflowExecution.add_or_update(initial)
        self.assertEqual(initial.rev, 1)
        doc_id = created.id
        retrieved1 = wf_db_access.WorkflowExecution.get_by_id(doc_id)
        retrieved2 = wf_db_access.WorkflowExecution.get_by_id(doc_id)
        graph = {'var1': 'fubar'}
        status = 'running'
        retrieved1 = wf_db_access.WorkflowExecution.update(retrieved1, graph=graph, status=status)
        updated = wf_db_access.WorkflowExecution.get_by_id(doc_id)
        self.assertNotEqual(created.rev, updated.rev)
        self.assertEqual(retrieved1.rev, updated.rev)
        self.assertEqual(retrieved1.action_execution, updated.action_execution)
        self.assertDictEqual(retrieved1.graph, updated.graph)
        self.assertEqual(retrieved1.status, updated.status)
        self.assertRaises(db_exc.StackStormDBObjectWriteConflictError, wf_db_access.WorkflowExecution.update, retrieved2, graph={'var2': 'fubar'})
        created.delete()
        self.assertRaises(db_exc.StackStormDBObjectNotFoundError, wf_db_access.WorkflowExecution.get_by_id, doc_id)