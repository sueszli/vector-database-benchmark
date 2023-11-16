import time
import pytest
from dagster._core.storage.dagster_run import DagsterRunStatus
from dagster_graphql import DagsterGraphQLClientError
from dagster_graphql.client.query import LAUNCH_PIPELINE_EXECUTION_MUTATION
from dagster_graphql.test.utils import execute_dagster_graphql, infer_job_selector
from ..graphql.graphql_context_test_suite import ExecutingGraphQLContextTestMatrix
from ..graphql.repo import csv_hello_world_ops_config
from .conftest import MockClient, python_client_test_suite

@python_client_test_suite
def test_get_run_status_success(mock_client: MockClient):
    if False:
        while True:
            i = 10
    expected_result = DagsterRunStatus.SUCCESS
    response = {'pipelineRunOrError': {'__typename': 'PipelineRun', 'status': expected_result}}
    mock_client.mock_gql_client.execute.return_value = response
    actual_result = mock_client.python_client.get_run_status('foo')
    assert actual_result == expected_result

@python_client_test_suite
def test_get_run_status_fails_with_python_error(mock_client: MockClient):
    if False:
        i = 10
        return i + 15
    (error_type, error_msg) = ('PythonError', 'something exploded')
    response = {'pipelineRunOrError': {'__typename': error_type, 'message': error_msg}}
    mock_client.mock_gql_client.execute.return_value = response
    with pytest.raises(DagsterGraphQLClientError) as exc_info:
        mock_client.python_client.get_run_status('foo')
    assert exc_info.value.args == (error_type, error_msg)

@python_client_test_suite
def test_get_run_status_fails_with_pipeline_run_not_found_error(mock_client: MockClient):
    if False:
        for i in range(10):
            print('nop')
    (error_type, error_msg) = ('RunNotFoundError', 'The specified pipeline run does not exist')
    response = {'pipelineRunOrError': {'__typename': error_type, 'message': error_msg}}
    mock_client.mock_gql_client.execute.return_value = response
    with pytest.raises(DagsterGraphQLClientError) as exc_info:
        mock_client.python_client.get_run_status('foo')
    assert exc_info.value.args == (error_type, error_msg)

@python_client_test_suite
def test_get_run_status_fails_with_query_error(mock_client: MockClient):
    if False:
        print('Hello World!')
    mock_client.mock_gql_client.execute.side_effect = Exception('foo')
    with pytest.raises(DagsterGraphQLClientError) as _:
        mock_client.python_client.get_run_status('foo')

class TestGetRunStatusWithClient(ExecutingGraphQLContextTestMatrix):

    def test_get_run_status(self, graphql_context, graphql_client):
        if False:
            for i in range(10):
                print('nop')
        selector = infer_job_selector(graphql_context, 'csv_hello_world')
        result = execute_dagster_graphql(graphql_context, LAUNCH_PIPELINE_EXECUTION_MUTATION, variables={'executionParams': {'selector': selector, 'runConfigData': csv_hello_world_ops_config(), 'mode': 'default'}})
        assert not result.errors
        assert result.data
        run_id = result.data['launchPipelineExecution']['run']['runId']
        start_time = time.time()
        while True:
            if time.time() - start_time > 30:
                raise Exception('Timed out waiting for get_run_status to return SUCCESS')
            status = graphql_client.get_run_status(run_id)
            if status == DagsterRunStatus.SUCCESS:
                break
            time.sleep(3)