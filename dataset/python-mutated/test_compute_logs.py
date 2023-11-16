from dagster._core.events import DagsterEventType
from dagster_graphql.test.utils import execute_dagster_graphql, execute_dagster_graphql_subscription, infer_job_selector
from .graphql_context_test_suite import ExecutingGraphQLContextTestMatrix
from .utils import sync_execute_get_run_log_data
COMPUTE_LOGS_QUERY = '\n  query ComputeLogsQuery($runId: ID!, $stepKey: String!) {\n    pipelineRunOrError(runId: $runId) {\n      ... on PipelineRun {\n        runId\n        computeLogs(stepKey: $stepKey) {\n          stdout {\n            data\n          }\n        }\n      }\n    }\n  }\n'
COMPUTE_LOGS_SUBSCRIPTION = '\n  subscription ComputeLogsSubscription($runId: ID!, $stepKey: String!, $ioType: ComputeIOType!, $cursor: String!) {\n    computeLogs(runId: $runId, stepKey: $stepKey, ioType: $ioType, cursor: $cursor) {\n      data\n    }\n  }\n'

class TestComputeLogs(ExecutingGraphQLContextTestMatrix):

    def test_get_compute_logs_over_graphql(self, graphql_context, snapshot):
        if False:
            for i in range(10):
                print('nop')
        selector = infer_job_selector(graphql_context, 'spew_job')
        payload = sync_execute_get_run_log_data(context=graphql_context, variables={'executionParams': {'selector': selector, 'mode': 'default'}})
        run_id = payload['run']['runId']
        logs = graphql_context.instance.all_logs(run_id, of_type=DagsterEventType.LOGS_CAPTURED)
        assert len(logs) == 1
        entry = logs[0]
        file_key = entry.dagster_event.logs_captured_data.file_key
        result = execute_dagster_graphql(graphql_context, COMPUTE_LOGS_QUERY, variables={'runId': run_id, 'stepKey': file_key})
        compute_logs = result.data['pipelineRunOrError']['computeLogs']
        snapshot.assert_match(compute_logs)

    def test_compute_logs_subscription_graphql(self, graphql_context, snapshot):
        if False:
            print('Hello World!')
        selector = infer_job_selector(graphql_context, 'spew_job')
        payload = sync_execute_get_run_log_data(context=graphql_context, variables={'executionParams': {'selector': selector, 'mode': 'default'}})
        run_id = payload['run']['runId']
        logs = graphql_context.instance.all_logs(run_id, of_type=DagsterEventType.LOGS_CAPTURED)
        assert len(logs) == 1
        entry = logs[0]
        file_key = entry.dagster_event.logs_captured_data.file_key
        results = execute_dagster_graphql_subscription(graphql_context, COMPUTE_LOGS_SUBSCRIPTION, variables={'runId': run_id, 'stepKey': file_key, 'ioType': 'STDOUT', 'cursor': '0'})
        assert len(results) == 1
        result = results[0]
        assert result.data['computeLogs']['data'] == 'HELLO WORLD\n'
        snapshot.assert_match([result.data])