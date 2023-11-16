from __future__ import absolute_import
from integration.orquesta import base

class FunctionsWiringTest(base.TestWorkflowExecution):

    def test_data_functions_in_yaql(self):
        if False:
            return 10
        wf_name = 'examples.orquesta-test-yaql-data-functions'
        expected_output = {'data_json_str_1': '{"foo": {"bar": "foobar"}}', 'data_json_str_2': '{"foo": {"bar": "foobar"}}', 'data_json_str_3': '{"foo": {"bar": "foobar"}}', 'data_json_obj_1': {'foo': {'bar': 'foobar'}}, 'data_json_obj_2': {'foo': {'bar': 'foobar'}}, 'data_json_obj_3': {'foo': {'bar': 'foobar'}}, 'data_json_obj_4': {'foo': {'bar': 'foobar'}}, 'data_yaml_str_1': 'foo:\n  bar: foobar\n', 'data_yaml_str_2': 'foo:\n  bar: foobar\n', 'data_query_1': ['foobar'], 'data_none_str': '%*****__%NONE%__*****%', 'data_str': 'foobar'}
        expected_result = {'output': expected_output}
        self._execute_workflow(wf_name, execute_async=False, expected_result=expected_result)

    def test_data_functions_in_jinja(self):
        if False:
            for i in range(10):
                print('nop')
        wf_name = 'examples.orquesta-test-jinja-data-functions'
        expected_output = {'data_json_str_1': '{"foo": {"bar": "foobar"}}', 'data_json_str_2': '{"foo": {"bar": "foobar"}}', 'data_json_str_3': '{"foo": {"bar": "foobar"}}', 'data_json_obj_1': {'foo': {'bar': 'foobar'}}, 'data_json_obj_2': {'foo': {'bar': 'foobar'}}, 'data_json_obj_3': {'foo': {'bar': 'foobar'}}, 'data_json_obj_4': {'foo': {'bar': 'foobar'}}, 'data_yaml_str_1': 'foo:\n  bar: foobar\n', 'data_yaml_str_2': 'foo:\n  bar: foobar\n', 'data_query_1': ['foobar'], 'data_pipe_str_1': '{"foo": {"bar": "foobar"}}', 'data_none_str': '%*****__%NONE%__*****%', 'data_str': 'foobar', 'data_list_str': '- a: 1\n  b: 2\n- x: 3\n  y: 4\n'}
        expected_result = {'output': expected_output}
        self._execute_workflow(wf_name, execute_async=False, expected_result=expected_result)

    def test_path_functions_in_yaql(self):
        if False:
            while True:
                i = 10
        wf_name = 'examples.orquesta-test-yaql-path-functions'
        expected_output = {'basename': 'file.txt', 'dirname': '/path/to/some'}
        expected_result = {'output': expected_output}
        self._execute_workflow(wf_name, execute_async=False, expected_result=expected_result)

    def test_path_functions_in_jinja(self):
        if False:
            i = 10
            return i + 15
        wf_name = 'examples.orquesta-test-jinja-path-functions'
        expected_output = {'basename': 'file.txt', 'dirname': '/path/to/some'}
        expected_result = {'output': expected_output}
        self._execute_workflow(wf_name, execute_async=False, expected_result=expected_result)

    def test_regex_functions_in_yaql(self):
        if False:
            print('Hello World!')
        wf_name = 'examples.orquesta-test-yaql-regex-functions'
        expected_output = {'match': True, 'replace': 'wxyz', 'search': True, 'substring': '668 Infinite Dr'}
        expected_result = {'output': expected_output}
        self._execute_workflow(wf_name, execute_async=False, expected_result=expected_result)

    def test_regex_functions_in_jinja(self):
        if False:
            i = 10
            return i + 15
        wf_name = 'examples.orquesta-test-jinja-regex-functions'
        expected_output = {'match': True, 'replace': 'wxyz', 'search': True, 'substring': '668 Infinite Dr'}
        expected_result = {'output': expected_output}
        self._execute_workflow(wf_name, execute_async=False, expected_result=expected_result)

    def test_time_functions_in_yaql(self):
        if False:
            while True:
                i = 10
        wf_name = 'examples.orquesta-test-yaql-time-functions'
        expected_output = {'time': '3h25m45s'}
        expected_result = {'output': expected_output}
        self._execute_workflow(wf_name, execute_async=False, expected_result=expected_result)

    def test_time_functions_in_jinja(self):
        if False:
            for i in range(10):
                print('nop')
        wf_name = 'examples.orquesta-test-jinja-time-functions'
        expected_output = {'time': '3h25m45s'}
        expected_result = {'output': expected_output}
        self._execute_workflow(wf_name, execute_async=False, expected_result=expected_result)

    def test_version_functions_in_yaql(self):
        if False:
            i = 10
            return i + 15
        wf_name = 'examples.orquesta-test-yaql-version-functions'
        expected_output = {'compare_equal': 0, 'compare_more_than': -1, 'compare_less_than': 1, 'equal': True, 'more_than': False, 'less_than': False, 'match': True, 'bump_major': '1.0.0', 'bump_minor': '0.11.0', 'bump_patch': '0.10.1', 'strip_patch': '0.10'}
        expected_result = {'output': expected_output}
        self._execute_workflow(wf_name, execute_async=False, expected_result=expected_result)

    def test_version_functions_in_jinja(self):
        if False:
            while True:
                i = 10
        wf_name = 'examples.orquesta-test-jinja-version-functions'
        expected_output = {'compare_equal': 0, 'compare_more_than': -1, 'compare_less_than': 1, 'equal': True, 'more_than': False, 'less_than': False, 'match': True, 'bump_major': '1.0.0', 'bump_minor': '0.11.0', 'bump_patch': '0.10.1', 'strip_patch': '0.10'}
        expected_result = {'output': expected_output}
        self._execute_workflow(wf_name, execute_async=False, expected_result=expected_result)