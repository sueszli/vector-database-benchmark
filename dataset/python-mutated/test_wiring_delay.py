from __future__ import absolute_import
from integration.orquesta import base
from st2common.constants import action as ac_const

class TaskDelayWiringTest(base.TestWorkflowExecution):

    def test_task_delay(self):
        if False:
            print('Hello World!')
        wf_name = 'examples.orquesta-delay'
        wf_input = {'name': 'Thanos', 'delay': 1}
        expected_output = {'greeting': 'Thanos, All your base are belong to us!'}
        expected_result = {'output': expected_output}
        ex = self._execute_workflow(wf_name, wf_input)
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertDictEqual(ex.result, expected_result)

    def test_task_delay_workflow_cancellation(self):
        if False:
            while True:
                i = 10
        wf_name = 'examples.orquesta-delay'
        wf_input = {'name': 'Thanos', 'delay': 300}
        ex = self._execute_workflow(wf_name, wf_input)
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_DELAYED)
        self.st2client.executions.delete(ex)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELED)
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_CANCELED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELED)

    def test_task_delay_task_cancellation(self):
        if False:
            i = 10
            return i + 15
        wf_name = 'examples.orquesta-delay'
        wf_input = {'name': 'Thanos', 'delay': 300}
        ex = self._execute_workflow(wf_name, wf_input)
        task_exs = self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_DELAYED)
        self.st2client.executions.delete(task_exs[0])
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_CANCELED)
        ex = self._wait_for_state(ex, ac_const.LIVEACTION_STATUS_CANCELED)