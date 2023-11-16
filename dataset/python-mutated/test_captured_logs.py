from dagster._core.events import DagsterEventType
from dagster_graphql.test.utils import execute_dagster_graphql, execute_dagster_graphql_subscription, infer_job_selector
from .graphql_context_test_suite import ExecutingGraphQLContextTestMatrix
from .utils import sync_execute_get_run_log_data
CAPTURED_LOGS_QUERY = '\n  query CapturedLogsQuery($logKey: [String!]!) {\n    capturedLogs(logKey: $logKey) {\n      stdout\n      stderr\n      cursor\n    }\n  }\n'
CAPTURED_LOGS_SUBSCRIPTION = '\n  subscription CapturedLogsSubscription($logKey: [String!]!) {\n    capturedLogs(logKey: $logKey) {\n      stdout\n      stderr\n      cursor\n    }\n  }\n'
CAPTURED_LOGS_EVENT_QUERY = '\n  query CapturedLogsEventQuery($runId: ID!) {\n    runOrError(runId: $runId) {\n      __typename\n      ... on Run {\n        eventConnection {\n          events {\n            ... on LogsCapturedEvent {\n              message\n              timestamp\n              fileKey\n              stepKeys\n              externalStdoutUrl\n              externalStderrUrl\n            }\n          }\n        }\n      }\n    }\n  }\n'

class TestCapturedLogs(ExecutingGraphQLContextTestMatrix):

    def test_get_captured_logs_over_graphql(self, graphql_context):
        if False:
            while True:
                i = 10
        selector = infer_job_selector(graphql_context, 'spew_job')
        payload = sync_execute_get_run_log_data(context=graphql_context, variables={'executionParams': {'selector': selector, 'mode': 'default'}})
        run_id = payload['run']['runId']
        logs = graphql_context.instance.all_logs(run_id, of_type=DagsterEventType.LOGS_CAPTURED)
        assert len(logs) == 1
        entry = logs[0]
        log_key = [run_id, 'compute_logs', entry.dagster_event.logs_captured_data.file_key]
        result = execute_dagster_graphql(graphql_context, CAPTURED_LOGS_QUERY, variables={'logKey': log_key})
        stdout = result.data['capturedLogs']['stdout']
        assert stdout == 'HELLO WORLD\n'

    def test_captured_logs_subscription_graphql(self, graphql_context):
        if False:
            print('Hello World!')
        selector = infer_job_selector(graphql_context, 'spew_job')
        payload = sync_execute_get_run_log_data(context=graphql_context, variables={'executionParams': {'selector': selector, 'mode': 'default'}})
        run_id = payload['run']['runId']
        logs = graphql_context.instance.all_logs(run_id, of_type=DagsterEventType.LOGS_CAPTURED)
        assert len(logs) == 1
        entry = logs[0]
        log_key = [run_id, 'compute_logs', entry.dagster_event.logs_captured_data.file_key]
        results = execute_dagster_graphql_subscription(graphql_context, CAPTURED_LOGS_SUBSCRIPTION, variables={'logKey': log_key})
        assert len(results) == 1
        stdout = results[0].data['capturedLogs']['stdout']
        assert stdout == 'HELLO WORLD\n'

    def test_captured_logs_event_graphql(self, graphql_context):
        if False:
            while True:
                i = 10
        selector = infer_job_selector(graphql_context, 'spew_job')
        payload = sync_execute_get_run_log_data(context=graphql_context, variables={'executionParams': {'selector': selector, 'mode': 'default'}})
        run_id = payload['run']['runId']
        result = execute_dagster_graphql(graphql_context, CAPTURED_LOGS_EVENT_QUERY, variables={'runId': run_id})
        assert result.data['runOrError']['__typename'] == 'Run'
        events = result.data['runOrError']['eventConnection']['events']
        assert len(events) > 0