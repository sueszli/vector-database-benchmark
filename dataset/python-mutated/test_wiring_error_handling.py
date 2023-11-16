from __future__ import absolute_import
import eventlet
from integration.orquesta import base
from st2common.constants import action as ac_const

class ErrorHandlingTest(base.TestWorkflowExecution):

    def test_inspection_error(self):
        if False:
            while True:
                i = 10
        expected_errors = [{'type': 'content', 'message': 'The action "std.noop" is not registered in the database.', 'schema_path': 'properties.tasks.patternProperties.^\\w+$.properties.action', 'spec_path': 'tasks.task3.action'}, {'type': 'context', 'language': 'yaql', 'expression': '<% ctx().foobar %>', 'message': 'Variable "foobar" is referenced before assignment.', 'schema_path': 'properties.tasks.patternProperties.^\\w+$.properties.input', 'spec_path': 'tasks.task1.input'}, {'type': 'expression', 'language': 'yaql', 'expression': '<% <% succeeded() %>', 'message': "Parse error: unexpected '<' at position 0 of expression '<% succeeded()'", 'schema_path': 'properties.tasks.patternProperties.^\\w+$.properties.next.items.properties.when', 'spec_path': 'tasks.task2.next[0].when'}, {'type': 'syntax', 'message': "[{'cmd': 'echo <% ctx().macro %>'}] is not valid under any of the given schemas", 'schema_path': 'properties.tasks.patternProperties.^\\w+$.properties.input.oneOf', 'spec_path': 'tasks.task2.input'}]
        ex = self._execute_workflow('examples.orquesta-fail-inspection')
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        self.assertDictEqual(ex.result, {'errors': expected_errors, 'output': None})

    def test_input_error(self):
        if False:
            while True:
                i = 10
        expected_errors = [{'type': 'error', 'message': 'YaqlEvaluationException: Unable to evaluate expression \'<% abs(8).value %>\'. NoFunctionRegisteredException: Unknown function "#property#value"'}]
        ex = self._execute_workflow('examples.orquesta-fail-input-rendering')
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        self.assertDictEqual(ex.result, {'errors': expected_errors, 'output': None})

    def test_vars_error(self):
        if False:
            while True:
                i = 10
        expected_errors = [{'type': 'error', 'message': 'YaqlEvaluationException: Unable to evaluate expression \'<% abs(8).value %>\'. NoFunctionRegisteredException: Unknown function "#property#value"'}]
        ex = self._execute_workflow('examples.orquesta-fail-vars-rendering')
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        self.assertDictEqual(ex.result, {'errors': expected_errors, 'output': None})

    def test_start_task_error(self):
        if False:
            return 10
        self.maxDiff = None
        expected_errors = [{'type': 'error', 'message': 'YaqlEvaluationException: Unable to evaluate expression \'<% ctx().name.value %>\'. NoFunctionRegisteredException: Unknown function "#property#value"', 'task_id': 'task1', 'route': 0}, {'type': 'error', 'message': "YaqlEvaluationException: Unable to resolve key 'greeting' in expression '<% ctx().greeting %>' from context."}]
        ex = self._execute_workflow('examples.orquesta-fail-start-task')
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        self.assertDictEqual(ex.result, {'errors': expected_errors, 'output': None})

    def test_task_transition_error(self):
        if False:
            while True:
                i = 10
        expected_errors = [{'type': 'error', 'message': "YaqlEvaluationException: Unable to resolve key 'value' in expression '<% succeeded() and result().value %>' from context.", 'task_transition_id': 'task2__t0', 'task_id': 'task1', 'route': 0}]
        expected_output = {'greeting': None}
        ex = self._execute_workflow('examples.orquesta-fail-task-transition')
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        self.assertDictEqual(ex.result, {'errors': expected_errors, 'output': expected_output})

    def test_task_publish_error(self):
        if False:
            while True:
                i = 10
        expected_errors = [{'type': 'error', 'message': "YaqlEvaluationException: Unable to resolve key 'value' in expression '<% result().value %>' from context.", 'task_transition_id': 'task2__t0', 'task_id': 'task1', 'route': 0}]
        expected_output = {'greeting': None}
        ex = self._execute_workflow('examples.orquesta-fail-task-publish')
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        self.assertDictEqual(ex.result, {'errors': expected_errors, 'output': expected_output})

    def test_output_error(self):
        if False:
            print('Hello World!')
        expected_errors = [{'type': 'error', 'message': 'YaqlEvaluationException: Unable to evaluate expression \'<% abs(8).value %>\'. NoFunctionRegisteredException: Unknown function "#property#value"'}]
        ex = self._execute_workflow('examples.orquesta-fail-output-rendering')
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        self.assertDictEqual(ex.result, {'errors': expected_errors, 'output': None})

    def test_task_content_errors(self):
        if False:
            i = 10
            return i + 15
        expected_errors = [{'type': 'content', 'message': 'The action reference "echo" is not formatted correctly.', 'schema_path': 'properties.tasks.patternProperties.^\\w+$.properties.action', 'spec_path': 'tasks.task1.action'}, {'type': 'content', 'message': 'The action "core.echoz" is not registered in the database.', 'schema_path': 'properties.tasks.patternProperties.^\\w+$.properties.action', 'spec_path': 'tasks.task2.action'}, {'type': 'content', 'message': 'Action "core.echo" is missing required input "message".', 'schema_path': 'properties.tasks.patternProperties.^\\w+$.properties.input', 'spec_path': 'tasks.task3.input'}, {'type': 'content', 'message': 'Action "core.echo" has unexpected input "messages".', 'schema_path': 'properties.tasks.patternProperties.^\\w+$.properties.input.patternProperties.^\\w+$', 'spec_path': 'tasks.task3.input.messages'}]
        ex = self._execute_workflow('examples.orquesta-fail-inspection-task-contents')
        ex = self._wait_for_completion(ex)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        self.assertDictEqual(ex.result, {'errors': expected_errors, 'output': None})

    def test_remediate_then_fail(self):
        if False:
            print('Hello World!')
        expected_errors = [{'task_id': 'task1', 'type': 'error', 'message': 'Execution failed. See result for details.', 'result': {'failed': True, 'return_code': 1, 'stderr': '', 'stdout': '', 'succeeded': False}}, {'task_id': 'fail', 'type': 'error', 'message': 'Execution failed. See result for details.'}]
        ex = self._execute_workflow('examples.orquesta-remediate-then-fail')
        ex = self._wait_for_completion(ex)
        eventlet.sleep(2)
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_FAILED)
        self._wait_for_task(ex, 'log', ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        self.assertDictEqual(ex.result, {'errors': expected_errors, 'output': None})

    def test_fail_manually(self):
        if False:
            return 10
        expected_errors = [{'task_id': 'task1', 'type': 'error', 'message': 'Execution failed. See result for details.', 'result': {'failed': True, 'return_code': 1, 'stderr': '', 'stdout': '', 'succeeded': False}}, {'task_id': 'fail', 'type': 'error', 'message': 'Execution failed. See result for details.'}]
        expected_output = {'message': '$%#&@#$!!!'}
        wf_input = {'cmd': 'exit 1'}
        ex = self._execute_workflow('examples.orquesta-error-handling-fail-manually', wf_input)
        ex = self._wait_for_completion(ex)
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_FAILED)
        self._wait_for_task(ex, 'task3', ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        self.assertDictEqual(ex.result, {'errors': expected_errors, 'output': expected_output})

    def test_fail_continue(self):
        if False:
            return 10
        expected_errors = [{'task_id': 'task1', 'type': 'error', 'message': 'Execution failed. See result for details.', 'result': {'failed': True, 'return_code': 1, 'stderr': '', 'stdout': '', 'succeeded': False}}]
        expected_output = {'message': '$%#&@#$!!!'}
        wf_input = {'cmd': 'exit 1'}
        ex = self._execute_workflow('examples.orquesta-error-handling-continue', wf_input)
        ex = self._wait_for_completion(ex)
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_FAILED)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_FAILED)
        self.assertDictEqual(ex.result, {'errors': expected_errors, 'output': expected_output})

    def test_fail_noop(self):
        if False:
            print('Hello World!')
        expected_output = {'message': '$%#&@#$!!!'}
        wf_input = {'cmd': 'exit 1'}
        ex = self._execute_workflow('examples.orquesta-error-handling-noop', wf_input)
        ex = self._wait_for_completion(ex)
        self._wait_for_task(ex, 'task1', ac_const.LIVEACTION_STATUS_FAILED)
        self.assertEqual(ex.status, ac_const.LIVEACTION_STATUS_SUCCEEDED)
        self.assertDictEqual(ex.result, {'output': expected_output})