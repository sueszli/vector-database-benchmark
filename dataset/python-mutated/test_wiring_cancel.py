from __future__ import absolute_import
import os
from integration.orquesta import base
from st2common.constants import action as ac_const

class CancellationWiringTest(base.TestWorkflowExecution, base.WorkflowControlTestCaseMixin):
    temp_file_path = None

    def setUp(self):
        if False:
            while True:
                i = 10
        super(CancellationWiringTest, self).setUp()
        self.temp_file_path = self._create_temp_file()

    def tearDown(self):
        if False:
            print('Hello World!')
        self._delete_temp_file(self.temp_file_path)
        super(CancellationWiringTest, self).tearDown()

    def test_cancellation(self):
        if False:
            while True:
                i = 10
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        params = {'tempfile': path, 'message': 'foobar'}
        ex = self._execute_workflow('examples.orquesta-test-cancel', params)
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        self.st2client.executions.delete(ex)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELING)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELED)
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_SUCCEEDED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELED)

    def test_task_cancellation(self):
        if False:
            while True:
                i = 10
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        params = {'tempfile': path, 'message': 'foobar'}
        ex = self._execute_workflow('examples.orquesta-test-cancel', params)
        task_exs = self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        self.st2client.executions.delete(task_exs[0])
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_CANCELED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELED)

    def test_cancellation_cascade_down_to_subworkflow(self):
        if False:
            i = 10
            return i + 15
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        params = {'tempfile': path, 'message': 'foobar'}
        action_ref = 'examples.orquesta-test-cancel-subworkflow'
        ex = self._execute_workflow(action_ref, params)
        task_exs = self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        subwf_ex = task_exs[0]
        self.st2client.executions.delete(ex)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELING)
        subwf_ex = self._wait_for_state(subwf_ex, ac_const.LIVEACTION_STATUS_CANCELING)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        subwf_ex = self._wait_for_state(subwf_ex, ac_const.LIVEACTION_STATUS_CANCELED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELED)

    def test_cancellation_cascade_up_from_subworkflow(self):
        if False:
            for i in range(10):
                print('nop')
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        params = {'tempfile': path, 'message': 'foobar'}
        action_ref = 'examples.orquesta-test-cancel-subworkflow'
        ex = self._execute_workflow(action_ref, params)
        task_exs = self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        subwf_ex = task_exs[0]
        self.st2client.executions.delete(subwf_ex)
        subwf_ex = self._wait_for_state(subwf_ex, ac_const.LIVEACTION_STATUS_CANCELING)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELING)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        subwf_ex = self._wait_for_state(subwf_ex, ac_const.LIVEACTION_STATUS_CANCELED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELED)

    def test_cancellation_cascade_up_to_workflow_with_other_subworkflow(self):
        if False:
            for i in range(10):
                print('nop')
        path = self.temp_file_path
        self.assertTrue(os.path.exists(path))
        params = {'file1': path, 'file2': path}
        action_ref = 'examples.orquesta-test-cancel-subworkflows'
        ex = self._execute_workflow(action_ref, params)
        task_exs = self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        subwf_ex_1 = task_exs[0]
        task_exs = self._wait_for_task(ex, 'task2', ac_const.LIVEACTION_STATUS_RUNNING)
        subwf_ex_2 = task_exs[0]
        self.st2client.executions.delete(subwf_ex_1)
        subwf_ex_1 = self._wait_for_state(subwf_ex_1, ac_const.LIVEACTION_STATUS_CANCELING)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELING)
        subwf_ex_2 = self._wait_for_state(subwf_ex_2, ac_const.LIVEACTION_STATUS_CANCELING)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        subwf_ex_1 = self._wait_for_state(subwf_ex_1, ac_const.LIVEACTION_STATUS_CANCELED)
        subwf_ex_2 = self._wait_for_state(subwf_ex_2, ac_const.LIVEACTION_STATUS_CANCELED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELED)