from dagster._core.test_utils import wait_for_runs_to_finish
from dagster._core.utils import make_new_run_id
from dagster_graphql.client.query import (
    LAUNCH_PIPELINE_EXECUTION_MUTATION,
    LAUNCH_PIPELINE_REEXECUTION_MUTATION,
)
from dagster_graphql.test.utils import execute_dagster_graphql, infer_job_selector

from .graphql_context_test_suite import ExecutingGraphQLContextTestMatrix
from .repo import csv_hello_world_ops_config

RUN_QUERY = """
query RunQuery($runId: ID!) {
  pipelineRunOrError(runId: $runId) {
    __typename
    ... on Run {
        status
      }
    }
  }
"""


class TestReexecution(ExecutingGraphQLContextTestMatrix):
    def test_full_pipeline_reexecution_fs_storage(self, graphql_context, snapshot):
        selector = infer_job_selector(graphql_context, "csv_hello_world")
        run_id = make_new_run_id()
        result_one = execute_dagster_graphql(
            graphql_context,
            LAUNCH_PIPELINE_EXECUTION_MUTATION,
            variables={
                "executionParams": {
                    "selector": selector,
                    "runConfigData": csv_hello_world_ops_config(),
                    "executionMetadata": {"runId": run_id},
                    "mode": "default",
                }
            },
        )

        assert result_one.data["launchPipelineExecution"]["__typename"] == "LaunchRunSuccess"

        result_one.data["launchPipelineExecution"]["run"]["runId"] = "<runId dummy value>"
        result_one.data["launchPipelineExecution"]["run"][
            "runConfigYaml"
        ] = "<runConfigYaml dummy value>"

        snapshot.assert_match(result_one.data)

        # reexecution
        new_run_id = make_new_run_id()

        result_two = execute_dagster_graphql(
            graphql_context,
            LAUNCH_PIPELINE_REEXECUTION_MUTATION,
            variables={
                "executionParams": {
                    "selector": selector,
                    "runConfigData": csv_hello_world_ops_config(),
                    "executionMetadata": {
                        "runId": new_run_id,
                        "rootRunId": run_id,
                        "parentRunId": run_id,
                    },
                    "mode": "default",
                }
            },
        )

        query_result = result_two.data["launchPipelineReexecution"]
        assert query_result["__typename"] == "LaunchRunSuccess"
        assert query_result["run"]["rootRunId"] == run_id
        assert query_result["run"]["parentRunId"] == run_id

    def test_full_pipeline_reexecution_in_memory_storage(self, graphql_context, snapshot):
        run_id = make_new_run_id()
        selector = infer_job_selector(graphql_context, "csv_hello_world")
        result_one = execute_dagster_graphql(
            graphql_context,
            LAUNCH_PIPELINE_EXECUTION_MUTATION,
            variables={
                "executionParams": {
                    "selector": selector,
                    "runConfigData": csv_hello_world_ops_config(),
                    "executionMetadata": {"runId": run_id},
                    "mode": "default",
                }
            },
        )

        assert result_one.data["launchPipelineExecution"]["__typename"] == "LaunchRunSuccess"

        result_one.data["launchPipelineExecution"]["run"]["runId"] = "<runId dummy value>"
        result_one.data["launchPipelineExecution"]["run"][
            "runConfigYaml"
        ] = "<runConfigYaml dummy value>"

        snapshot.assert_match(result_one.data)

        # reexecution
        new_run_id = make_new_run_id()

        result_two = execute_dagster_graphql(
            graphql_context,
            LAUNCH_PIPELINE_REEXECUTION_MUTATION,
            variables={
                "executionParams": {
                    "selector": selector,
                    "runConfigData": csv_hello_world_ops_config(),
                    "executionMetadata": {
                        "runId": new_run_id,
                        "rootRunId": run_id,
                        "parentRunId": run_id,
                    },
                    "mode": "default",
                }
            },
        )

        query_result = result_two.data["launchPipelineReexecution"]
        assert query_result["__typename"] == "LaunchRunSuccess"
        assert query_result["run"]["rootRunId"] == run_id
        assert query_result["run"]["parentRunId"] == run_id

    def test_pipeline_reexecution_successful_launch(self, graphql_context):
        selector = infer_job_selector(graphql_context, "no_config_job")
        run_id = make_new_run_id()
        result = execute_dagster_graphql(
            context=graphql_context,
            query=LAUNCH_PIPELINE_EXECUTION_MUTATION,
            variables={
                "executionParams": {
                    "selector": selector,
                    "executionMetadata": {"runId": run_id},
                    "mode": "default",
                }
            },
        )

        assert result.data["launchPipelineExecution"]["__typename"] == "LaunchRunSuccess"
        assert result.data["launchPipelineExecution"]["run"]["status"] == "STARTING"

        wait_for_runs_to_finish(graphql_context.instance)

        result = execute_dagster_graphql(
            context=graphql_context, query=RUN_QUERY, variables={"runId": run_id}
        )
        assert result.data["pipelineRunOrError"]["__typename"] == "Run"
        assert result.data["pipelineRunOrError"]["status"] == "SUCCESS"

        # reexecution
        new_run_id = make_new_run_id()
        result = execute_dagster_graphql(
            context=graphql_context,
            query=LAUNCH_PIPELINE_REEXECUTION_MUTATION,
            variables={
                "executionParams": {
                    "selector": selector,
                    "executionMetadata": {
                        "runId": new_run_id,
                        "rootRunId": run_id,
                        "parentRunId": run_id,
                    },
                    "mode": "default",
                }
            },
        )
        assert result.data["launchPipelineReexecution"]["__typename"] == "LaunchRunSuccess"

        wait_for_runs_to_finish(graphql_context.instance)

        result = execute_dagster_graphql(
            context=graphql_context, query=RUN_QUERY, variables={"runId": new_run_id}
        )
        assert result.data["pipelineRunOrError"]["__typename"] == "Run"
        assert result.data["pipelineRunOrError"]["status"] == "SUCCESS"
