from __future__ import absolute_import
import os
import unittest
from integration.orquesta import base
from st2common.constants import action as ac_const

@unittest.skipIf(os.environ.get('ST2_CI_RUN_ORQUESTA_PAUSE_RESUME_TESTS', 'false').lower() not in ['1', 'true'], 'Skipping race prone tests')
class PauseResumeWiringTest(base.TestWorkflowExecution, base.WorkflowControlTestCaseMixin):
    temp_file_path_x = None
    temp_file_path_y = None

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(PauseResumeWiringTest, self).setUp()
        self.temp_file_path_x = self._create_temp_file()
        self.temp_file_path_y = self._create_temp_file()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self._delete_temp_file(self.temp_file_path_x)
        self._delete_temp_file(self.temp_file_path_y)
        super(PauseResumeWiringTest, self).tearDown()

    def test_pause_and_resume(self):
        if False:
            while True:
                i = 10
        path = self.temp_file_path_x
        self.assertTrue(os.path.exists(path))
        params = {'tempfile': path}
        ex = self._execute_workflow('examples.orquesta-test-pause', params)
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        self.st2client.executions.pause(ex.id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSING)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ex = self.st2client.executions.resume(ex.id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_pause_and_resume_cascade_to_subworkflow(self):
        if False:
            for i in range(10):
                print('nop')
        path = self.temp_file_path_x
        self.assertTrue(os.path.exists(path))
        params = {'tempfile': path}
        ex = self._execute_workflow('examples.orquesta-test-pause-subworkflow', params)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk_exs = self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        ex = self.st2client.executions.pause(ex.id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSING)
        tk_ac_ex = self._wait_for_state(tk_exs[0], ac_const.LIVEACTION_STATUS_PAUSING)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        tk_ac_ex = self._wait_for_state(tk_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ex = self.st2client.executions.resume(ex.id)
        tk_ac_ex = self._wait_for_state(tk_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_pause_and_resume_cascade_to_subworkflows(self):
        if False:
            print('Hello World!')
        path1 = self.temp_file_path_x
        self.assertTrue(os.path.exists(path1))
        path2 = self.temp_file_path_y
        self.assertTrue(os.path.exists(path2))
        params = {'file1': path1, 'file2': path2}
        ex = self._execute_workflow('examples.orquesta-test-pause-subworkflows', params)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_exs = self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        tk2_exs = self._wait_for_task(ex, 'task2', ac_const.LIVEACTION_STATUS_RUNNING)
        ex = self.st2client.executions.pause(ex.id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSING)
        tk1_ac_ex = self._wait_for_state(tk1_exs[0], ac_const.LIVEACTION_STATUS_PAUSING)
        tk2_ac_ex = self._wait_for_state(tk2_exs[0], ac_const.LIVEACTION_STATUS_PAUSING)
        os.remove(path1)
        self.assertFalse(os.path.exists(path1))
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        tk1_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_PAUSING)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSING)
        os.remove(path2)
        self.assertFalse(os.path.exists(path2))
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        tk1_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ex = self.st2client.executions.resume(ex.id)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_pause_and_resume_cascade_from_subworkflow(self):
        if False:
            print('Hello World!')
        path = self.temp_file_path_x
        self.assertTrue(os.path.exists(path))
        params = {'tempfile': path}
        ex = self._execute_workflow('examples.orquesta-test-pause-subworkflow', params)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk_exs = self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        tk_ac_ex = self.st2client.executions.pause(tk_exs[0].id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk_ac_ex = self._wait_for_state(tk_ac_ex, ac_const.LIVEACTION_STATUS_PAUSING)
        os.remove(path)
        self.assertFalse(os.path.exists(path))
        tk_ac_ex = self._wait_for_state(tk_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        tk_ac_ex = self.st2client.executions.resume(tk_ac_ex.id)
        tk_ac_ex = self._wait_for_state(tk_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_pause_from_1_of_2_subworkflows_and_resume_subworkflow_when_workflow_paused(self):
        if False:
            while True:
                i = 10
        path1 = self.temp_file_path_x
        self.assertTrue(os.path.exists(path1))
        path2 = self.temp_file_path_y
        self.assertTrue(os.path.exists(path2))
        params = {'file1': path1, 'file2': path2}
        ex = self._execute_workflow('examples.orquesta-test-pause-subworkflows', params)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_exs = self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        tk2_exs = self._wait_for_task(ex, 'task2', ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self.st2client.executions.pause(tk1_exs[0].id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSING)
        tk2_ac_ex = self._wait_for_state(tk2_exs[0], ac_const.LIVEACTION_STATUS_RUNNING)
        os.remove(path1)
        self.assertFalse(os.path.exists(path1))
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_RUNNING)
        os.remove(path2)
        self.assertFalse(os.path.exists(path2))
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        tk1_ac_ex = self.st2client.executions.resume(tk1_ac_ex.id)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_pause_from_1_of_2_subworkflows_and_resume_subworkflow_while_workflow_running(self):
        if False:
            print('Hello World!')
        path1 = self.temp_file_path_x
        self.assertTrue(os.path.exists(path1))
        path2 = self.temp_file_path_y
        self.assertTrue(os.path.exists(path2))
        params = {'file1': path1, 'file2': path2}
        ex = self._execute_workflow('examples.orquesta-test-pause-subworkflows', params)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_exs = self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        tk2_exs = self._wait_for_task(ex, 'task2', ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self.st2client.executions.pause(tk1_exs[0].id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSING)
        tk2_ac_ex = self._wait_for_state(tk2_exs[0], ac_const.LIVEACTION_STATUS_RUNNING)
        os.remove(path1)
        self.assertFalse(os.path.exists(path1))
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self.st2client.executions.resume(tk1_ac_ex.id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_RUNNING)
        os.remove(path2)
        self.assertFalse(os.path.exists(path2))
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_pause_from_all_subworkflows_and_resume_from_subworkflows(self):
        if False:
            for i in range(10):
                print('nop')
        path1 = self.temp_file_path_x
        self.assertTrue(os.path.exists(path1))
        path2 = self.temp_file_path_y
        self.assertTrue(os.path.exists(path2))
        params = {'file1': path1, 'file2': path2}
        ex = self._execute_workflow('examples.orquesta-test-pause-subworkflows', params)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_exs = self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        tk2_exs = self._wait_for_task(ex, 'task2', ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self.st2client.executions.pause(tk1_exs[0].id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSING)
        tk2_ac_ex = self._wait_for_state(tk2_exs[0], ac_const.LIVEACTION_STATUS_RUNNING)
        tk2_ac_ex = self.st2client.executions.pause(tk2_exs[0].id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSING)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_PAUSING)
        os.remove(path1)
        self.assertFalse(os.path.exists(path1))
        os.remove(path2)
        self.assertFalse(os.path.exists(path2))
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        tk1_ac_ex = self.st2client.executions.resume(tk1_ac_ex.id)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        tk2_ac_ex = self.st2client.executions.resume(tk2_ac_ex.id)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_pause_from_all_subworkflows_and_resume_from_parent_workflow(self):
        if False:
            while True:
                i = 10
        path1 = self.temp_file_path_x
        self.assertTrue(os.path.exists(path1))
        path2 = self.temp_file_path_y
        self.assertTrue(os.path.exists(path2))
        params = {'file1': path1, 'file2': path2}
        ex = self._execute_workflow('examples.orquesta-test-pause-subworkflows', params)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_exs = self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_RUNNING)
        tk2_exs = self._wait_for_task(ex, 'task2', ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self.st2client.executions.pause(tk1_exs[0].id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSING)
        tk2_ac_ex = self._wait_for_state(tk2_exs[0], ac_const.LIVEACTION_STATUS_RUNNING)
        tk2_ac_ex = self.st2client.executions.pause(tk2_exs[0].id)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_RUNNING)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSING)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_PAUSING)
        os.remove(path1)
        self.assertFalse(os.path.exists(path1))
        os.remove(path2)
        self.assertFalse(os.path.exists(path2))
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_PAUSED)
        ex = self.st2client.executions.resume(ex.id)
        tk1_ac_ex = self._wait_for_state(tk1_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        tk2_ac_ex = self._wait_for_state(tk2_ac_ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_SUCCEEDED)