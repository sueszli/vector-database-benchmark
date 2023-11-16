from __future__ import absolute_import
from integration.orquesta import base
from st2common.constants import action as ac_const

class WiringTest(base.TestWorkflowExecution):

    def test_sequential(self):
        if False:
            for i in range(10):
                print('nop')
        wf_name = 'examples.orquesta-sequential'
        wf_input = {'name': 'Thanos'}
        expected_output = {'greeting': 'Thanos, All your base are belong to us!'}
        expected_result = {'output': expected_output}
        ex = self._execute_workflow(wf_name, wf_input)
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertDictEqual(ex.result, expected_result)

    def test_join(self):
        if False:
            print('Hello World!')
        wf_name = 'examples.orquesta-join'
        expected_output = {'messages': ['Fee fi fo fum', 'I smell the blood of an English man', 'Be alive, or be he dead', "I'll grind his bones to make my bread"]}
        expected_result = {'output': expected_output}
        ex = self._execute_workflow(wf_name)
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertDictEqual(ex.result, expected_result)

    def test_cycle(self):
        if False:
            for i in range(10):
                print('nop')
        wf_name = 'examples.orquesta-rollback-retry'
        expected_output = None
        expected_result = {'output': expected_output}
        ex = self._execute_workflow(wf_name)
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertDictEqual(ex.result, expected_result)

    def test_action_less(self):
        if False:
            for i in range(10):
                print('nop')
        wf_name = 'examples.orquesta-test-action-less-tasks'
        wf_input = {'name': 'Thanos'}
        message = 'Thanos, All your base are belong to us!'
        expected_output = {'greeting': message.upper()}
        expected_result = {'output': expected_output}
        ex = self._execute_workflow(wf_name, wf_input)
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertDictEqual(ex.result, expected_result)

    def test_st2_runtime_context(self):
        if False:
            for i in range(10):
                print('nop')
        wf_name = 'examples.orquesta-st2-ctx'
        ex = self._execute_workflow(wf_name)
        ex = self._wait_for_completion(ex)
        expected_output = {'callback': 'http://127.0.0.1:9101/v1/executions/%s' % str(ex.id)}
        expected_result = {'output': expected_output}
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertDictEqual(ex.result, expected_result)

    def test_subworkflow(self):
        if False:
            while True:
                i = 10
        wf_name = 'examples.orquesta-subworkflow'
        ex = self._execute_workflow(wf_name)
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self._wait_for_task(ex, 'start', ac_const.LIVEACTION_STATUS_SUCCEEDED)
        t2_ex = self._wait_for_task(ex, 'subworkflow', ac_const.LIVEACTION_STATUS_SUCCEEDED)[0]
        self._wait_for_task(t2_ex, 'task1', ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self._wait_for_task(t2_ex, 'task2', ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self._wait_for_task(t2_ex, 'task3', ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self._wait_for_task(ex, 'finish', ac_const.LIVEACTION_STATUS_SUCCEEDED)

    def test_output_on_error(self):
        if False:
            while True:
                i = 10
        wf_name = 'examples.orquesta-output-on-error'
        ex = self._execute_workflow(wf_name)
        ex = self._wait_for_completion(ex)
        expected_output = {'progress': 25}
        expected_errors = [{'type': 'error', 'task_id': 'task2', 'message': 'Execution failed. See result for details.', 'result': {'failed': True, 'return_code': 1, 'stderr': '', 'stdout': '', 'succeeded': False}}]
        expected_result = {'errors': expected_errors, 'output': expected_output}
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        self.assertDictEqual(ex.result, expected_result)

    def test_config_context_renders(self):
        if False:
            while True:
                i = 10
        config_value = 'Testing'
        wf_name = 'examples.render_config_context'
        expected_output = {'context_value': config_value}
        expected_result = {'output': expected_output}
        ex = self._execute_workflow(wf_name)
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertDictEqual(ex.result, expected_result)

    def test_field_escaping(self):
        if False:
            while True:
                i = 10
        wf_name = 'examples.orquesta-test-field-escaping'
        ex = self._execute_workflow(wf_name)
        ex = self._wait_for_completion(ex)
        expected_output = {'wf.hostname.with.periods': {'hostname.domain.tld': 'vars.value.with.periods', 'hostname2.domain.tld': {'stdout': 'vars.nested.value.with.periods'}}, 'wf.output.with.periods': 'vars.nested.value.with.periods'}
        expected_result = {'output': expected_output}
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertDictEqual(ex.result, expected_result)