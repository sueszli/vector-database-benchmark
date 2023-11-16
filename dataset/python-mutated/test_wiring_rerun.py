from __future__ import absolute_import
import os
import shutil
import tempfile
from integration.orquesta import base
from st2common.constants import action as action_constants

class RerunWiringTest(base.TestWorkflowExecution):
    temp_dir_path = None

    def setUp(self):
        if False:
            return 10
        super(RerunWiringTest, self).setUp()
        (_, self.temp_dir_path) = tempfile.mkstemp()
        os.chmod(self.temp_dir_path, 493)

    def tearDown(self):
        if False:
            return 10
        if self.temp_dir_path and os.path.exists(self.temp_dir_path):
            if os.path.isdir(self.temp_dir_path):
                shutil.rmtree(self.temp_dir_path)
            else:
                os.remove(self.temp_dir_path)

    def test_rerun_workflow(self):
        if False:
            while True:
                i = 10
        path = self.temp_dir_path
        with open(path, 'w') as f:
            f.write('1')
        params = {'tempfile': path}
        ex = self._execute_workflow('examples.orquesta-test-rerun', params)
        ex = self._wait_for_state(ex, action_constants.LIVEACTION_STATUS_FAILED)
        orig_st2_ex_id = ex.id
        orig_wf_ex_id = ex.context['workflow_execution']
        with open(path, 'w') as f:
            f.write('0')
        ex = self.st2client.executions.re_run(orig_st2_ex_id)
        self.assertNotEqual(ex.id, orig_st2_ex_id)
        ex = self._wait_for_state(ex, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertNotEqual(ex.context['workflow_execution'], orig_wf_ex_id)

    def test_rerun_task(self):
        if False:
            i = 10
            return i + 15
        path = self.temp_dir_path
        with open(path, 'w') as f:
            f.write('1')
        params = {'tempfile': path}
        ex = self._execute_workflow('examples.orquesta-test-rerun', params)
        ex = self._wait_for_state(ex, action_constants.LIVEACTION_STATUS_FAILED)
        orig_st2_ex_id = ex.id
        orig_wf_ex_id = ex.context['workflow_execution']
        with open(path, 'w') as f:
            f.write('0')
        ex = self.st2client.executions.re_run(orig_st2_ex_id, tasks=['task2'])
        self.assertNotEqual(ex.id, orig_st2_ex_id)
        ex = self._wait_for_state(ex, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(ex.context['workflow_execution'], orig_wf_ex_id)

    def test_rerun_task_of_workflow_already_succeeded(self):
        if False:
            print('Hello World!')
        path = self.temp_dir_path
        with open(path, 'w') as f:
            f.write('0')
        params = {'tempfile': path}
        ex = self._execute_workflow('examples.orquesta-test-rerun', params)
        ex = self._wait_for_state(ex, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        orig_st2_ex_id = ex.id
        orig_wf_ex_id = ex.context['workflow_execution']
        ex = self.st2client.executions.re_run(orig_st2_ex_id, tasks=['task2'])
        self.assertNotEqual(ex.id, orig_st2_ex_id)
        ex = self._wait_for_state(ex, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(ex.context['workflow_execution'], orig_wf_ex_id)

    def test_rerun_and_reset_with_items_task(self):
        if False:
            print('Hello World!')
        path = self.temp_dir_path
        with open(path, 'w') as f:
            f.write('1')
        params = {'tempfile': path}
        ex = self._execute_workflow('examples.orquesta-test-rerun-with-items', params)
        ex = self._wait_for_state(ex, action_constants.LIVEACTION_STATUS_FAILED)
        orig_st2_ex_id = ex.id
        orig_wf_ex_id = ex.context['workflow_execution']
        with open(path, 'w') as f:
            f.write('0')
        ex = self.st2client.executions.re_run(orig_st2_ex_id, tasks=['task1'])
        self.assertNotEqual(ex.id, orig_st2_ex_id)
        ex = self._wait_for_state(ex, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(ex.context['workflow_execution'], orig_wf_ex_id)
        children = self.st2client.executions.get_property(ex.id, 'children')
        self.assertEqual(len(children), 4)

    def test_rerun_and_resume_with_items_task(self):
        if False:
            for i in range(10):
                print('nop')
        path = self.temp_dir_path
        with open(path, 'w') as f:
            f.write('1')
        params = {'tempfile': path}
        ex = self._execute_workflow('examples.orquesta-test-rerun-with-items', params)
        ex = self._wait_for_state(ex, action_constants.LIVEACTION_STATUS_FAILED)
        orig_st2_ex_id = ex.id
        orig_wf_ex_id = ex.context['workflow_execution']
        with open(path, 'w') as f:
            f.write('0')
        ex = self.st2client.executions.re_run(orig_st2_ex_id, tasks=['task1'], no_reset=['task1'])
        self.assertNotEqual(ex.id, orig_st2_ex_id)
        ex = self._wait_for_state(ex, action_constants.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(ex.context['workflow_execution'], orig_wf_ex_id)
        children = self.st2client.executions.get_property(ex.id, 'children')
        self.assertEqual(len(children), 2)