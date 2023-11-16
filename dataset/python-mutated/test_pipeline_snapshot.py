from unittest import mock
import pytest
from dagster._core.workspace.context import WorkspaceRequestContext
from dagster._seven import json
from dagster_graphql.implementation.fetch_pipelines import _get_job_snapshot_from_instance
from dagster_graphql.implementation.utils import UserFacingGraphQLError
from dagster_graphql.test.utils import execute_dagster_graphql, main_repo_location_name, main_repo_name
from .repo import noop_job
SNAPSHOT_OR_ERROR_QUERY_BY_SNAPSHOT_ID = '\nquery PipelineSnapshotQueryBySnapshotID($snapshotId: String!) {\n    pipelineSnapshotOrError(snapshotId: $snapshotId) {\n        __typename\n        ... on PipelineSnapshot {\n            name\n            pipelineSnapshotId\n            description\n            dagsterTypes { key }\n            solids { name }\n            modes { name }\n            solidHandles { handleID }\n            tags { key value }\n        }\n        ... on PipelineSnapshotNotFoundError {\n            snapshotId\n        }\n    }\n}\n'
SNAPSHOT_OR_ERROR_QUERY_BY_PIPELINE_NAME = '\nquery PipelineSnapshotQueryByActivePipelineName($activePipelineSelector: PipelineSelector!) {\n    pipelineSnapshotOrError(activePipelineSelector: $activePipelineSelector) {\n        __typename\n        ... on PipelineSnapshot {\n            name\n            pipelineSnapshotId\n            description\n            dagsterTypes { key }\n            solids { name }\n            modes { name }\n            solidHandles { handleID }\n            tags { key value }\n        }\n        ... on PipelineSnapshotNotFoundError {\n            snapshotId\n        }\n    }\n}\n'

def pretty_dump(data) -> str:
    if False:
        while True:
            i = 10
    return json.dumps(data, indent=2, separators=(',', ': '))

def test_fetch_snapshot_or_error_by_snapshot_id_success(graphql_context: WorkspaceRequestContext, snapshot):
    if False:
        return 10
    instance = graphql_context.instance
    result = noop_job.execute_in_process(instance=instance)
    assert result.success
    run = instance.get_run_by_id(result.run_id)
    assert run and run.job_snapshot_id
    result = execute_dagster_graphql(graphql_context, SNAPSHOT_OR_ERROR_QUERY_BY_SNAPSHOT_ID, {'snapshotId': run.job_snapshot_id})
    assert not result.errors
    assert result.data
    assert result.data['pipelineSnapshotOrError']['__typename'] == 'PipelineSnapshot'
    snapshot.assert_match(pretty_dump(result.data))

def test_fetch_snapshot_or_error_by_snapshot_id_snapshot_not_found(graphql_context: WorkspaceRequestContext, snapshot):
    if False:
        for i in range(10):
            print('nop')
    result = execute_dagster_graphql(graphql_context, SNAPSHOT_OR_ERROR_QUERY_BY_SNAPSHOT_ID, {'snapshotId': 'notthere'})
    assert not result.errors
    assert result.data
    assert result.data['pipelineSnapshotOrError']['__typename'] == 'PipelineSnapshotNotFoundError'
    assert result.data['pipelineSnapshotOrError']['snapshotId'] == 'notthere'
    snapshot.assert_match(pretty_dump(result.data))

def test_fetch_snapshot_or_error_by_active_pipeline_name_success(graphql_context: WorkspaceRequestContext, snapshot):
    if False:
        return 10
    result = execute_dagster_graphql(graphql_context, SNAPSHOT_OR_ERROR_QUERY_BY_PIPELINE_NAME, {'activePipelineSelector': {'pipelineName': 'csv_hello_world', 'repositoryName': main_repo_name(), 'repositoryLocationName': main_repo_location_name()}})
    assert not result.errors
    assert result.data
    assert result.data['pipelineSnapshotOrError']['__typename'] == 'PipelineSnapshot'
    assert result.data['pipelineSnapshotOrError']['name'] == 'csv_hello_world'
    snapshot.assert_match(pretty_dump(result.data))

def test_fetch_snapshot_or_error_by_active_pipeline_name_not_found(graphql_context: WorkspaceRequestContext, snapshot):
    if False:
        i = 10
        return i + 15
    result = execute_dagster_graphql(graphql_context, SNAPSHOT_OR_ERROR_QUERY_BY_PIPELINE_NAME, {'activePipelineSelector': {'pipelineName': 'jkdjfkdj', 'repositoryName': main_repo_name(), 'repositoryLocationName': main_repo_location_name()}})
    assert not result.errors
    assert result.data
    assert result.data['pipelineSnapshotOrError']['__typename'] == 'PipelineNotFoundError'
    snapshot.assert_match(pretty_dump(result.data))

def test_temporary_error_or_deletion_after_instance_check():
    if False:
        while True:
            i = 10
    instance = mock.MagicMock()
    instance.has_historical_job.return_value = True
    instance.get_historical_job.return_value = None
    with pytest.raises(UserFacingGraphQLError):
        _get_job_snapshot_from_instance(instance, 'kjdkfjd')