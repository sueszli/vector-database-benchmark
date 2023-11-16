import json
import os
import subprocess
import time
from contextlib import contextmanager
import requests
from dagster import file_relative_path
from dagster._core.storage.dagster_run import DagsterRunStatus
IS_BUILDKITE = os.getenv('BUILDKITE') is not None

@contextmanager
def docker_service_up(docker_compose_file):
    if False:
        print('Hello World!')
    if IS_BUILDKITE:
        yield
        return
    try:
        subprocess.check_output(['docker-compose', '-f', docker_compose_file, 'stop'])
        subprocess.check_output(['docker-compose', '-f', docker_compose_file, 'rm', '-f'])
    except subprocess.CalledProcessError:
        pass
    build_process = subprocess.Popen([file_relative_path(docker_compose_file, './build.sh')])
    build_process.wait()
    assert build_process.returncode == 0
    up_process = subprocess.Popen(['docker-compose', '-f', docker_compose_file, 'up', '--no-start'])
    up_process.wait()
    assert up_process.returncode == 0
    start_process = subprocess.Popen(['docker-compose', '-f', docker_compose_file, 'start'])
    start_process.wait()
    assert start_process.returncode == 0
    try:
        yield
    finally:
        subprocess.check_output(['docker-compose', '-f', docker_compose_file, 'stop'])
        subprocess.check_output(['docker-compose', '-f', docker_compose_file, 'rm', '-f'])
PIPELINES_OR_ERROR_QUERY = '\n{\n    repositoriesOrError {\n        ... on PythonError {\n            message\n            stack\n        }\n        ... on RepositoryConnection {\n            nodes {\n                pipelines {\n                    name\n                }\n            }\n        }\n    }\n}\n'
RUN_QUERY = '\nquery RunQuery($runId: ID!) {\n  pipelineRunOrError(runId: $runId) {\n    __typename\n    ... on PipelineRun {\n      status\n    }\n  }\n}\n'
LAUNCH_PIPELINE_MUTATION = '\nmutation($executionParams: ExecutionParams!) {\n  launchPipelineExecution(executionParams: $executionParams) {\n    __typename\n    ... on LaunchRunSuccess {\n      run {\n        runId\n        status\n      }\n    }\n    ... on PipelineNotFoundError {\n      message\n      pipelineName\n    }\n    ... on PythonError {\n      message\n    }\n  }\n}\n'
TERMINATE_MUTATION = '\nmutation($runId: String!) {\n  terminatePipelineExecution(runId: $runId){\n    __typename\n    ... on TerminateRunSuccess{\n      run {\n        runId\n      }\n    }\n    ... on TerminateRunFailure {\n      run {\n        runId\n      }\n      message\n    }\n    ... on RunNotFoundError {\n      runId\n    }\n    ... on PythonError {\n      message\n      stack\n    }\n  }\n}\n'

def test_deploy_docker():
    if False:
        while True:
            i = 10
    with docker_service_up(file_relative_path(__file__, '../from_source/docker-compose.yml')):
        start_time = time.time()
        webserver_host = os.environ.get('DEPLOY_DOCKER_WEBSERVER_HOST', 'localhost')
        while True:
            if time.time() - start_time > 15:
                raise Exception('Timed out waiting for webserver to be available')
            try:
                sanity_check = requests.get(f'http://{webserver_host}:3000/server_info')
                assert 'dagster_webserver' in sanity_check.text
                break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        res = requests.get(f'http://{webserver_host}:3000/graphql?query={PIPELINES_OR_ERROR_QUERY}').json()
        data = res.get('data')
        assert data
        repositoriesOrError = data.get('repositoriesOrError')
        assert repositoriesOrError
        nodes = repositoriesOrError.get('nodes')
        assert nodes
        names = {node['name'] for node in nodes[0]['pipelines']}
        assert names == {'my_job', 'hanging_job', 'my_step_isolated_job'}
        variables = {'executionParams': {'selector': {'repositoryLocationName': 'example_user_code', 'repositoryName': 'deploy_docker_repository', 'pipelineName': 'my_job'}, 'mode': 'default'}}
        launch_res = requests.post('http://{webserver_host}:3000/graphql?query={query_string}&variables={variables}'.format(webserver_host=webserver_host, query_string=LAUNCH_PIPELINE_MUTATION, variables=json.dumps(variables))).json()
        assert launch_res['data']['launchPipelineExecution']['__typename'] == 'LaunchRunSuccess'
        run = launch_res['data']['launchPipelineExecution']['run']
        run_id = run['runId']
        assert run['status'] == 'QUEUED'
        _wait_for_run_status(run_id, webserver_host, DagsterRunStatus.SUCCESS)
        variables = {'executionParams': {'selector': {'repositoryLocationName': 'example_user_code', 'repositoryName': 'deploy_docker_repository', 'pipelineName': 'my_step_isolated_job'}, 'mode': 'default'}}
        launch_res = requests.post('http://{webserver_host}:3000/graphql?query={query_string}&variables={variables}'.format(webserver_host=webserver_host, query_string=LAUNCH_PIPELINE_MUTATION, variables=json.dumps(variables))).json()
        assert launch_res['data']['launchPipelineExecution']['__typename'] == 'LaunchRunSuccess'
        run = launch_res['data']['launchPipelineExecution']['run']
        run_id = run['runId']
        assert run['status'] == 'QUEUED'
        _wait_for_run_status(run_id, webserver_host, DagsterRunStatus.SUCCESS)
        variables = {'executionParams': {'selector': {'repositoryLocationName': 'example_user_code', 'repositoryName': 'deploy_docker_repository', 'pipelineName': 'hanging_job'}, 'mode': 'default'}}
        launch_res = requests.post('http://{webserver_host}:3000/graphql?query={query_string}&variables={variables}'.format(webserver_host=webserver_host, query_string=LAUNCH_PIPELINE_MUTATION, variables=json.dumps(variables))).json()
        assert launch_res['data']['launchPipelineExecution']['__typename'] == 'LaunchRunSuccess'
        run = launch_res['data']['launchPipelineExecution']['run']
        hanging_run_id = run['runId']
        _wait_for_run_status(hanging_run_id, webserver_host, DagsterRunStatus.STARTED)
        terminate_res = requests.post('http://{webserver_host}:3000/graphql?query={query_string}&variables={variables}'.format(webserver_host=webserver_host, query_string=TERMINATE_MUTATION, variables=json.dumps({'runId': hanging_run_id}))).json()
        assert terminate_res['data']['terminatePipelineExecution']['__typename'] == 'TerminateRunSuccess', str(terminate_res)
        _wait_for_run_status(hanging_run_id, webserver_host, DagsterRunStatus.CANCELED)

def _wait_for_run_status(run_id, webserver_host, desired_status):
    if False:
        while True:
            i = 10
    start_time = time.time()
    while True:
        if time.time() - start_time > 60:
            raise Exception(f'Timed out waiting for run to reach status {desired_status}')
        run_res = requests.get('http://{webserver_host}:3000/graphql?query={query_string}&variables={variables}'.format(webserver_host=webserver_host, query_string=RUN_QUERY, variables=json.dumps({'runId': run_id}))).json()
        status = run_res['data']['pipelineRunOrError']['status']
        assert status and status != 'FAILED'
        if status == desired_status.value:
            break
        time.sleep(1)