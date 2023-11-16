import os
import time
from typing import Any
import pytest
from dagster._core.definitions.reconstruct import ReconstructableRepository
from dagster._core.execution.api import execute_job
from dagster._core.storage.dagster_run import DagsterRunStatus
from dagster._core.test_utils import create_run_for_test
from dagster._core.workspace.context import WorkspaceRequestContext
from dagster._grpc.types import CancelExecutionRequest
from dagster._utils import file_relative_path, safe_tempfile_path
from dagster_graphql.client.query import LAUNCH_PIPELINE_EXECUTION_MUTATION
from dagster_graphql.test.utils import execute_dagster_graphql, infer_job_selector
from .graphql_context_test_suite import GraphQLContextVariant, ReadonlyGraphQLContextTestMatrix, make_graphql_context_test_suite
RUN_CANCELLATION_QUERY = '\nmutation($runId: String!, $terminatePolicy: TerminateRunPolicy) {\n  terminatePipelineExecution(runId: $runId, terminatePolicy:$terminatePolicy){\n    __typename\n    ... on TerminateRunSuccess{\n      run {\n        runId\n      }\n    }\n    ... on TerminateRunFailure {\n      run {\n        runId\n      }\n      message\n    }\n    ... on RunNotFoundError {\n      runId\n    }\n    ... on PythonError {\n      message\n      stack\n    }\n  }\n}\n'
RUNS_CANCELLATION_QUERY = '\nmutation TerminateRuns($runIds: [String!]!) {\n  terminateRuns(runIds: $runIds) {\n    __typename\n    ... on PythonError{\n      message\n    }\n    ... on TerminateRunsResult {\n      terminateRunResults {\n        ... on TerminateRunSuccess {\n            __typename\n            run {\n                runId\n            }\n        }\n        ... on RunNotFoundError {\n            __typename\n            runId\n        }\n        ... on TerminateRunFailure {\n            __typename\n            run {\n                runId\n            }\n            message\n        }\n      }\n    }\n  }\n}\n'
BULK_TERMINATION_PERMISSIONS_QUERY = '\nquery canBulkTerminate {\n    canBulkTerminate\n}\n'
BACKCOMPAT_LEGACY_TERMINATE_PIPELINE = '\nmutation TerminatePipeline($runId: String!) {\n  terminatePipelineExecution(runId: $runId){\n    __typename\n    ... on TerminatePipelineExecutionSuccess{\n      run {\n        runId\n      }\n    }\n    ... on TerminatePipelineExecutionFailure {\n      message\n    }\n    ... on PipelineRunNotFoundError {\n      runId\n    }\n    ... on PythonError {\n      message\n      stack\n    }\n  }\n}\n'
QueuedRunCoordinatorTestSuite: Any = make_graphql_context_test_suite(context_variants=[GraphQLContextVariant.sqlite_with_queued_run_coordinator_managed_grpc_env()])

class TestQueuedRunTermination(QueuedRunCoordinatorTestSuite):

    def test_cancel_queued_run(self, graphql_context: WorkspaceRequestContext):
        if False:
            for i in range(10):
                print('nop')
        selector = infer_job_selector(graphql_context, 'infinite_loop_job')
        with safe_tempfile_path() as path:
            result = execute_dagster_graphql(graphql_context, LAUNCH_PIPELINE_EXECUTION_MUTATION, variables={'executionParams': {'selector': selector, 'mode': 'default', 'runConfigData': {'ops': {'loop': {'config': {'file': path}}}}}})
            assert not result.errors
            assert result.data
            assert result.data['launchPipelineExecution']['__typename'] == 'LaunchRunSuccess'
            run_id = result.data['launchPipelineExecution']['run']['runId']
            run = graphql_context.instance.get_run_by_id(run_id)
            assert run and run.status == DagsterRunStatus.QUEUED
            result = execute_dagster_graphql(graphql_context, RUN_CANCELLATION_QUERY, variables={'runId': run_id})
            assert result.data['terminatePipelineExecution']['__typename'] == 'TerminateRunSuccess', str(result.data)

    def test_cancel_runs(self, graphql_context: WorkspaceRequestContext):
        if False:
            while True:
                i = 10
        selector = infer_job_selector(graphql_context, 'infinite_loop_job')
        with safe_tempfile_path() as path:
            result = execute_dagster_graphql(graphql_context, LAUNCH_PIPELINE_EXECUTION_MUTATION, variables={'executionParams': {'selector': selector, 'mode': 'default', 'runConfigData': {'ops': {'loop': {'config': {'file': path}}}}}})
            assert not result.errors
            assert result.data
            assert result.data['launchPipelineExecution']['__typename'] == 'LaunchRunSuccess'
            run_id = result.data['launchPipelineExecution']['run']['runId']
            run = graphql_context.instance.get_run_by_id(run_id)
            assert run and run.status == DagsterRunStatus.QUEUED
            result = execute_dagster_graphql(graphql_context, RUNS_CANCELLATION_QUERY, variables={'runIds': [run_id, 'nonexistent_id']})
            assert result.data['terminateRuns']['__typename'] == 'TerminateRunsResult'
            assert result.data['terminateRuns']['terminateRunResults'][0]['__typename'] == 'TerminateRunSuccess'
            assert result.data['terminateRuns']['terminateRunResults'][1]['__typename'] == 'RunNotFoundError'
            assert result.data['terminateRuns']['terminateRunResults'][1]['runId'] == 'nonexistent_id'

    def test_force_cancel_queued_run(self, graphql_context: WorkspaceRequestContext):
        if False:
            return 10
        selector = infer_job_selector(graphql_context, 'infinite_loop_job')
        with safe_tempfile_path() as path:
            result = execute_dagster_graphql(graphql_context, LAUNCH_PIPELINE_EXECUTION_MUTATION, variables={'executionParams': {'selector': selector, 'mode': 'default', 'runConfigData': {'ops': {'loop': {'config': {'file': path}}}}}})
            assert not result.errors
            assert result.data
            assert result.data['launchPipelineExecution']['__typename'] == 'LaunchRunSuccess'
            run_id = result.data['launchPipelineExecution']['run']['runId']
            run = graphql_context.instance.get_run_by_id(run_id)
            assert run and run.status == DagsterRunStatus.QUEUED
            result = execute_dagster_graphql(graphql_context, RUN_CANCELLATION_QUERY, variables={'runId': run_id, 'terminatePolicy': 'MARK_AS_CANCELED_IMMEDIATELY'})
            assert result.data['terminatePipelineExecution']['__typename'] == 'TerminateRunSuccess'
RunTerminationTestSuite: Any = make_graphql_context_test_suite(context_variants=[GraphQLContextVariant.sqlite_with_default_run_launcher_managed_grpc_env()])

def _exception_terminate(_run_id):
    if False:
        while True:
            i = 10
    raise Exception('FAILED TO TERMINATE')

def _return_fail_terminate(_run_id):
    if False:
        print('Hello World!')
    return False

class TestTerminationReadonly(ReadonlyGraphQLContextTestMatrix):

    def test_termination_permission_failure(self, graphql_context: WorkspaceRequestContext):
        if False:
            return 10
        run_id = create_run_for_test(graphql_context.instance).run_id
        result = execute_dagster_graphql(graphql_context, RUN_CANCELLATION_QUERY, variables={'runId': run_id})
        assert not result.errors
        assert result.data
        assert result.data['terminatePipelineExecution']['__typename'] == 'TerminateRunFailure'
        assert 'do not have permission' in result.data['terminatePipelineExecution']['message']

    def test_cancel_runs_permission_failure(self, graphql_context: WorkspaceRequestContext):
        if False:
            i = 10
            return i + 15
        run_id = create_run_for_test(graphql_context.instance).run_id
        result = execute_dagster_graphql(graphql_context, RUNS_CANCELLATION_QUERY, variables={'runIds': [run_id]})
        assert not result.errors
        assert result.data
        assert result.data['terminateRuns']['__typename'] == 'TerminateRunsResult'
        assert len(result.data['terminateRuns']['terminateRunResults']) == 1
        assert result.data['terminateRuns']['terminateRunResults'][0]['__typename'] == 'TerminateRunFailure'
        assert 'do not have permission' in result.data['terminateRuns']['terminateRunResults'][0]['message']

    def test_no_bulk_terminate_permission(self, graphql_context: WorkspaceRequestContext):
        if False:
            i = 10
            return i + 15
        result = execute_dagster_graphql(graphql_context, BULK_TERMINATION_PERMISSIONS_QUERY)
        assert not result.errors
        assert result.data
        assert result.data['canBulkTerminate'] is False

class TestRunVariantTermination(RunTerminationTestSuite):

    def test_basic_termination(self, graphql_context: WorkspaceRequestContext):
        if False:
            return 10
        selector = infer_job_selector(graphql_context, 'infinite_loop_job')
        with safe_tempfile_path() as path:
            result = execute_dagster_graphql(graphql_context, LAUNCH_PIPELINE_EXECUTION_MUTATION, variables={'executionParams': {'selector': selector, 'mode': 'default', 'runConfigData': {'ops': {'loop': {'config': {'file': path}}}}}})
            assert not result.errors
            assert result.data
            assert result.data['launchPipelineExecution']['__typename'] == 'LaunchRunSuccess'
            run_id = result.data['launchPipelineExecution']['run']['runId']
            assert run_id
            while not os.path.exists(path):
                time.sleep(0.1)
            result = execute_dagster_graphql(graphql_context, RUN_CANCELLATION_QUERY, variables={'runId': run_id})
            assert result.data['terminatePipelineExecution']['__typename'] == 'TerminateRunSuccess'

    def test_force_termination(self, graphql_context: WorkspaceRequestContext):
        if False:
            for i in range(10):
                print('nop')
        selector = infer_job_selector(graphql_context, 'infinite_loop_job')
        with safe_tempfile_path() as path:
            result = execute_dagster_graphql(graphql_context, LAUNCH_PIPELINE_EXECUTION_MUTATION, variables={'executionParams': {'selector': selector, 'mode': 'default', 'runConfigData': {'ops': {'loop': {'config': {'file': path}}}}}})
            assert not result.errors
            assert result.data
            assert result.data['launchPipelineExecution']['__typename'] == 'LaunchRunSuccess'
            run_id = result.data['launchPipelineExecution']['run']['runId']
            assert run_id
            while not os.path.exists(path):
                time.sleep(0.1)
            result = execute_dagster_graphql(graphql_context, RUN_CANCELLATION_QUERY, variables={'runId': run_id, 'terminatePolicy': 'MARK_AS_CANCELED_IMMEDIATELY'})
            assert result.data['terminatePipelineExecution']['__typename'] == 'TerminateRunSuccess'
            instance = graphql_context.instance
            run = instance.get_run_by_id(run_id)
            assert run and run.status == DagsterRunStatus.CANCELED

    def test_run_not_found(self, graphql_context: WorkspaceRequestContext):
        if False:
            print('Hello World!')
        result = execute_dagster_graphql(graphql_context, RUN_CANCELLATION_QUERY, variables={'runId': 'nope'})
        assert result.data['terminatePipelineExecution']['__typename'] == 'RunNotFoundError'

    @pytest.mark.parametrize(argnames=['new_terminate_method', 'terminate_result'], argvalues=[[_return_fail_terminate, 'TerminateRunFailure'], [_exception_terminate, 'PythonError']])
    def test_terminate_failed(self, graphql_context: WorkspaceRequestContext, new_terminate_method, terminate_result):
        if False:
            return 10
        selector = infer_job_selector(graphql_context, 'infinite_loop_job')
        with safe_tempfile_path() as path:
            old_terminate = graphql_context.instance.run_launcher.terminate
            graphql_context.instance.run_launcher.terminate = new_terminate_method
            result = execute_dagster_graphql(graphql_context, LAUNCH_PIPELINE_EXECUTION_MUTATION, variables={'executionParams': {'selector': selector, 'mode': 'default', 'runConfigData': {'ops': {'loop': {'config': {'file': path}}}}}})
            assert not result.errors
            assert result.data
            assert result.data['launchPipelineExecution']['__typename'] == 'LaunchRunSuccess'
            run_id = result.data['launchPipelineExecution']['run']['runId']
            while not os.path.exists(path):
                time.sleep(0.1)
            result = execute_dagster_graphql(graphql_context, RUN_CANCELLATION_QUERY, variables={'runId': run_id})
            assert result.data['terminatePipelineExecution']['__typename'] == terminate_result, str(result.data)
            result = execute_dagster_graphql(graphql_context, RUN_CANCELLATION_QUERY, variables={'runId': run_id, 'terminatePolicy': 'MARK_AS_CANCELED_IMMEDIATELY'})
            assert result.data['terminatePipelineExecution']['__typename'] == 'TerminateRunSuccess'
            assert result.data['terminatePipelineExecution']['run']['runId'] == run_id
            graphql_context.instance.run_launcher.terminate = old_terminate
            code_location = graphql_context.code_locations[0]
            code_location.client.cancel_execution(CancelExecutionRequest(run_id=run_id))
            run = graphql_context.instance.get_run_by_id(run_id)
            assert run and run.status == DagsterRunStatus.CANCELED

    def test_run_finished(self, graphql_context: WorkspaceRequestContext):
        if False:
            return 10
        instance = graphql_context.instance
        recon_job = ReconstructableRepository.for_file(file_relative_path(__file__, 'repo.py'), 'test_repo').get_reconstructable_job('noop_job')
        with execute_job(recon_job, instance=instance) as exec_result:
            assert exec_result.success
            assert exec_result.run_id
            time.sleep(0.05)
            result = execute_dagster_graphql(graphql_context, RUN_CANCELLATION_QUERY, variables={'runId': exec_result.run_id})
            assert result.data['terminatePipelineExecution']['__typename'] == 'TerminateRunFailure'
            assert 'could not be terminated due to having status SUCCESS.' in result.data['terminatePipelineExecution']['message']
            result = execute_dagster_graphql(graphql_context, RUN_CANCELLATION_QUERY, variables={'runId': exec_result.run_id, 'terminatePolicy': 'MARK_AS_CANCELED_IMMEDIATELY'})
            assert result.data['terminatePipelineExecution']['__typename'] == 'TerminateRunFailure'
            assert 'could not be terminated due to having status SUCCESS.' in result.data['terminatePipelineExecution']['message']

    def test_backcompat_termination(self, graphql_context: WorkspaceRequestContext):
        if False:
            print('Hello World!')
        selector = infer_job_selector(graphql_context, 'infinite_loop_job')
        with safe_tempfile_path() as path:
            result = execute_dagster_graphql(graphql_context, LAUNCH_PIPELINE_EXECUTION_MUTATION, variables={'executionParams': {'selector': selector, 'mode': 'default', 'runConfigData': {'ops': {'loop': {'config': {'file': path}}}}}})
            assert not result.errors
            assert result.data
            assert result.data['launchPipelineExecution']['__typename'] == 'LaunchRunSuccess'
            run_id = result.data['launchPipelineExecution']['run']['runId']
            assert run_id
            while not os.path.exists(path):
                time.sleep(0.1)
            result = execute_dagster_graphql(graphql_context, BACKCOMPAT_LEGACY_TERMINATE_PIPELINE, variables={'runId': run_id})
            assert result.data['terminatePipelineExecution']['run']['runId'] == run_id

    def test_has_bulk_terminate_permission(self, graphql_context: WorkspaceRequestContext):
        if False:
            i = 10
            return i + 15
        result = execute_dagster_graphql(graphql_context, BULK_TERMINATION_PERMISSIONS_QUERY)
        assert not result.errors
        assert result.data
        assert result.data['canBulkTerminate'] is True