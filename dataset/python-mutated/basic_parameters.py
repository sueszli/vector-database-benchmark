from metaflow_test import MetaflowTest, ExpectationFailed, steps

class BasicParameterTest(MetaflowTest):
    PRIORITY = 1
    PARAMETERS = {'no_default_param': {'default': None}, 'bool_param': {'default': False}, 'bool_true_param': {'default': True}, 'int_param': {'default': 123}, 'str_param': {'default': "'foobar'"}, 'list_param': {'separator': "','", 'default': '"a,b,c"'}, 'json_param': {'default': '\'{"a": [1,2,3]}\'', 'type': 'JSONType'}}
    HEADER = "\nimport os\nos.environ['METAFLOW_RUN_NO_DEFAULT_PARAM'] = 'test_str'\nos.environ['METAFLOW_RUN_BOOL_PARAM'] = 'False'\n"

    @steps(0, ['all'])
    def step_all(self):
        if False:
            i = 10
            return i + 15
        assert_equals('test_str', self.no_default_param)
        assert_equals(False, self.bool_param)
        assert_equals(True, self.bool_true_param)
        assert_equals(123, self.int_param)
        assert_equals('foobar', self.str_param)
        assert_equals(['a', 'b', 'c'], self.list_param)
        assert_equals({'a': [1, 2, 3]}, self.json_param)
        try:
            self.int_param = 5
            raise ExpectationFailed(AttributeError, 'nothing')
        except AttributeError:
            pass

    def check_results(self, flow, checker):
        if False:
            i = 10
            return i + 15
        for step in flow:
            checker.assert_artifact(step.name, 'no_default_param', 'test_str')
            checker.assert_artifact(step.name, 'bool_param', False)
            checker.assert_artifact(step.name, 'bool_true_param', True)
            checker.assert_artifact(step.name, 'int_param', 123)
            checker.assert_artifact(step.name, 'str_param', 'foobar')
            checker.assert_artifact(step.name, 'list_param', ['a', 'b', 'c'])
            checker.assert_artifact(step.name, 'json_param', {'a': [1, 2, 3]})