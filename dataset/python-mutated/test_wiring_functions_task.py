from __future__ import absolute_import
from integration.orquesta import base
from st2common.constants import action as action_constants

class FunctionsWiringTest(base.TestWorkflowExecution):

    def test_task_functions_in_yaql(self):
        if False:
            print('Hello World!')
        wf_name = 'examples.orquesta-test-yaql-task-functions'
        expected_output = {'last_task4_result': 'False', 'task9__1__parent': 'task8__1', 'task9__2__parent': 'task8__2', 'that_task_by_name': 'task1', 'this_task_by_name': 'task1', 'this_task_no_arg': 'task1'}
        expected_result = {'output': expected_output}
        self._execute_workflow(wf_name, execute_async=False, expected_result=expected_result)

    def test_task_functions_in_jinja(self):
        if False:
            print('Hello World!')
        wf_name = 'examples.orquesta-test-jinja-task-functions'
        expected_output = {'last_task4_result': 'False', 'task9__1__parent': 'task8__1', 'task9__2__parent': 'task8__2', 'that_task_by_name': 'task1', 'this_task_by_name': 'task1', 'this_task_no_arg': 'task1'}
        expected_result = {'output': expected_output}
        self._execute_workflow(wf_name, execute_async=False, expected_result=expected_result)

    def test_task_nonexistent_in_yaql(self):
        if False:
            for i in range(10):
                print('nop')
        wf_name = 'examples.orquesta-test-yaql-task-nonexistent'
        expected_output = None
        expected_errors = [{'type': 'error', 'message': 'YaqlEvaluationException: Unable to evaluate expression \'<% task("task0") %>\'. ExpressionEvaluationException: Unable to find task execution for "task0".', 'task_transition_id': 'continue__t0', 'task_id': 'task1', 'route': 0}]
        expected_result = {'output': expected_output, 'errors': expected_errors}
        self._execute_workflow(wf_name, execute_async=False, expected_status=action_constants.LIVEACTION_STATUS_FAILED, expected_result=expected_result)

    def test_task_nonexistent_in_jinja(self):
        if False:
            i = 10
            return i + 15
        wf_name = 'examples.orquesta-test-jinja-task-nonexistent'
        expected_output = None
        expected_errors = [{'type': 'error', 'message': 'JinjaEvaluationException: Unable to evaluate expression \'{{ task("task0") }}\'. ExpressionEvaluationException: Unable to find task execution for "task0".', 'task_transition_id': 'continue__t0', 'task_id': 'task1', 'route': 0}]
        expected_result = {'output': expected_output, 'errors': expected_errors}
        self._execute_workflow(wf_name, execute_async=False, expected_status=action_constants.LIVEACTION_STATUS_FAILED, expected_result=expected_result)