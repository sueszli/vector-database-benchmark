from metaflow_test import MetaflowTest, ExpectationFailed, steps

class DynamicParameterTest(MetaflowTest):
    PRIORITY = 3
    PARAMETERS = {'str_param': {'default': 'str_func'}, 'json_param': {'default': 'json_func', 'type': 'JSONType'}, 'nondefault_param': {'default': 'lambda _: True', 'type': 'bool'}}
    HEADER = '\nimport os\nos.environ[\'METAFLOW_RUN_NONDEFAULT_PARAM\'] = \'False\'\n\ndef str_func(ctx):\n    import os\n    from metaflow import current\n    assert_equals(current.project_name, \'dynamic_parameters_project\')\n    assert_equals(ctx.parameter_name, \'str_param\')\n    assert_equals(ctx.flow_name, \'DynamicParameterTestFlow\')\n    assert_equals(ctx.user_name, os.environ[\'METAFLOW_USER\'])\n\n    if os.path.exists(\'str_func.only_once\'):\n        raise Exception("Dynamic parameter function invoked multiple times!")\n\n    with open(\'str_func.only_once\', \'w\') as f:\n        f.write(\'foo\')\n\n    return \'does this work?\'\n\ndef json_func(ctx):\n    import json\n    return json.dumps({\'a\': [8]})\n\n@project(name=\'dynamic_parameters_project\')\n'

    @steps(0, ['singleton'], required=True)
    def step_single(self):
        if False:
            i = 10
            return i + 15
        assert_equals(self.str_param, 'does this work?')
        assert_equals(self.nondefault_param, False)
        assert_equals(self.json_param, {'a': [8]})

    @steps(1, ['all'])
    def step_all(self):
        if False:
            print('Hello World!')
        pass

    def check_results(self, flow, checker):
        if False:
            while True:
                i = 10
        for step in flow:
            checker.assert_artifact(step.name, 'nondefault_param', False)
            checker.assert_artifact(step.name, 'str_param', 'does this work?')
            checker.assert_artifact(step.name, 'json_param', {'a': [8]})