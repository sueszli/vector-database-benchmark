from __future__ import absolute_import
from integration.orquesta import base
from st2common.constants import action as ac_const

class TaskRetryWiringTest(base.TestWorkflowExecution):

    def test_task_retry(self):
        if False:
            print('Hello World!')
        wf_name = 'examples.orquesta-task-retry'
        ex = self._execute_workflow(wf_name)
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        task_exs = [task_ex for task_ex in self._get_children(ex) if task_ex.context.get('orquesta', {}).get('task_name', '') == 'check']
        self.assertGreater(len(task_exs), 1)

    def test_task_retry_exhausted(self):
        if False:
            for i in range(10):
                print('nop')
        wf_name = 'examples.orquesta-task-retry-exhausted'
        ex = self._execute_workflow(wf_name)
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        task_exs = [task_ex for task_ex in self._get_children(ex) if task_ex.context.get('orquesta', {}).get('task_name', '') == 'check']
        self.assertListEqual(['failed'] * 3, [task_ex.status for task_ex in task_exs])
        task_exs = [task_ex for task_ex in self._get_children(ex) if task_ex.context.get('orquesta', {}).get('task_name', '') == 'delete']
        self.assertEqual(len(task_exs), 0)