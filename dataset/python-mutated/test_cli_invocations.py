import os
import runpy
from urllib.parse import urlencode
import pytest
from dagster_webserver.app import create_app_from_workspace_process_context
from starlette.testclient import TestClient
from dagster._cli.workspace import get_workspace_process_context_from_kwargs
from dagster._core.test_utils import instance_for_test
from dagster._utils import check_script, pushd, script_relative_path
PIPELINES_OR_ERROR_QUERY = '\n{\n    repositoriesOrError {\n        ... on PythonError {\n            message\n            stack\n        }\n        ... on RepositoryConnection {\n            nodes {\n                pipelines {\n                    name\n                }\n            }\n        }\n    }\n}\n'
cli_args = [('basics/single_op_job/', 'hello.py', 'file_sizes_job', None, None, None, 0, None), ('basics/connecting_ops/', 'serial_job.py', 'serial', None, None, None, 0, None), ('basics/connecting_ops/', 'complex_job.py', 'diamond', None, None, None, 0, None), ('basics/testing/', 'inputs_typed.py', 'inputs_job', None, None, None, 0, None), ('basics/testing/', 'custom_types.py', 'custom_type_job', None, None, None, 0, None), ('basics/testing/', 'custom_types_2.py', 'custom_type_job', None, None, None, 1, None)]

def path_to_tutorial_file(path):
    if False:
        while True:
            i = 10
    return script_relative_path(os.path.join('../../docs_snippets/intro_tutorial/', path))

def load_dagster_webserver_for_workspace_cli_args(n_pipelines=1, **kwargs):
    if False:
        while True:
            i = 10
    with instance_for_test() as instance:
        with get_workspace_process_context_from_kwargs(instance, version='', read_only=False, kwargs=kwargs) as workspace_process_context:
            client = TestClient(create_app_from_workspace_process_context(workspace_process_context))
            url_query = urlencode({'query': PIPELINES_OR_ERROR_QUERY})
            res = client.get(f'/graphql?{url_query}')
            json_res = res.json()
            assert 'data' in json_res
            assert 'repositoriesOrError' in json_res['data']
            assert 'nodes' in json_res['data']['repositoriesOrError']
            assert len(json_res['data']['repositoriesOrError']['nodes'][0]['pipelines']) == n_pipelines
        return res

@pytest.mark.parametrize('dirname,filename,fn_name,_env_yaml,_mode,_preset,_return_code,_exception', cli_args)
def test_load_pipeline(dirname, filename, fn_name, _env_yaml, _mode, _preset, _return_code, _exception):
    if False:
        print('Hello World!')
    with pushd(path_to_tutorial_file(dirname)):
        filepath = path_to_tutorial_file(os.path.join(dirname, filename))
        load_dagster_webserver_for_workspace_cli_args(python_file=(filepath,), fn_name=fn_name)

@pytest.mark.parametrize('dirname,filename,_fn_name,_env_yaml,_mode,_preset,return_code,_exception', cli_args)
def test_script(dirname, filename, _fn_name, _env_yaml, _mode, _preset, return_code, _exception):
    if False:
        while True:
            i = 10
    with pushd(path_to_tutorial_file(dirname)):
        filepath = path_to_tutorial_file(os.path.join(dirname, filename))
        check_script(filepath, return_code)

@pytest.mark.parametrize('dirname,filename,_fn_name,_env_yaml,_mode,_preset,_return_code,exception', cli_args)
def test_runpy(dirname, filename, _fn_name, _env_yaml, _mode, _preset, _return_code, exception):
    if False:
        return 10
    with pushd(path_to_tutorial_file(dirname)):
        filepath = path_to_tutorial_file(os.path.join(dirname, filename))
        if exception:
            with pytest.raises(exception):
                runpy.run_path(filepath, run_name='__main__')
        else:
            runpy.run_path(filepath, run_name='__main__')