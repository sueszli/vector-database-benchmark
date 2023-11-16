import time

from dagster import job, op, repository
from dagster._core.storage.dagster_run import DagsterRunStatus
from dagster._core.test_utils import instance_for_test
from dagster_graphql.test.utils import (
    define_out_of_process_context,
    execute_dagster_graphql,
)

RUNS_QUERY = """
query RunsQuery {
  pipelineRunsOrError {
    __typename
    ... on PipelineRuns {
      results {
        runId
        pipelineName
        status
        runConfigYaml
        stats {
          ... on PipelineRunStatsSnapshot {
            startTime
            endTime
            stepsFailed
          }
        }
      }
    }
  }
}
"""

PAGINATED_RUNS_QUERY = """
query PaginatedRunsQuery($cursor: String!, $limit: Int) {
  pipelineRunsOrError(
    cursor: $cursor
    limit: $limit
  ) {
    __typename
    ... on PipelineRuns {
      results {
        runId
        pipelineName
        status
        runConfigYaml
        stats {
          ... on PipelineRunStatsSnapshot {
            startTime
            endTime
            stepsFailed
          }
        }
      }
    }
  }
}
"""

FILTERED_RUNS_QUERY = """
query FilteredRunsQuery {
  pipelineRunsOrError(filter: { statuses: [FAILURE] }) {
    __typename
    ... on PipelineRuns {
      results {
        runId
        pipelineName
        status
        runConfigYaml
        stats {
          ... on PipelineRunStatsSnapshot {
            startTime
            endTime
            stepsFailed
          }
        }
      }
    }
  }
}
"""

REPOSITORIES_QUERY = """
query RepositoriesQuery {
  repositoriesOrError {
    ... on RepositoryConnection {
      nodes {
        name
        location {
          name
        }
      }
    }
  }
}
"""

PIPELINES_QUERY = """
query PipelinesQuery(
  $repositoryLocationName: String!
  $repositoryName: String!
) {
  repositoryOrError(
    repositorySelector: {
      repositoryLocationName: $repositoryLocationName
      repositoryName: $repositoryName
    }
  ) {
    ... on Repository {
      pipelines {
        name
      }
    }
  }
}
"""

LAUNCH_PIPELINE = """
mutation ExecutePipeline(
  $repositoryLocationName: String!
  $repositoryName: String!
  $pipelineName: String!
  $runConfigData: RunConfigData!
  $mode: String!
) {
  launchPipelineExecution(
    executionParams: {
      selector: {
        repositoryLocationName: $repositoryLocationName
        repositoryName: $repositoryName
        pipelineName: $pipelineName
      }
      runConfigData: $runConfigData
      mode: $mode
    }
  ) {
    __typename
    ... on LaunchPipelineRunSuccess {
      run {
        runId
      }
    }
    ... on PipelineConfigValidationInvalid {
      errors {
        message
        reason
      }
    }
    ... on PythonError {
      message
    }
  }
}
"""


def get_repo():
    @op
    def my_op():
        pass

    @op
    def loop():
        while True:
            time.sleep(0.1)

    @job
    def infinite_loop_job():
        loop()

    @job
    def foo_job():
        my_op()

    @repository
    def my_repo():
        return [infinite_loop_job, foo_job]

    return my_repo


def test_runs_query():
    with instance_for_test() as instance:
        repo = get_repo()
        run_id_1 = instance.create_run_for_job(
            repo.get_job("foo_job"), status=DagsterRunStatus.STARTED
        ).run_id
        run_id_2 = instance.create_run_for_job(
            repo.get_job("foo_job"), status=DagsterRunStatus.FAILURE
        ).run_id
        with define_out_of_process_context(__file__, "get_repo", instance) as context:
            result = execute_dagster_graphql(context, RUNS_QUERY)
            assert result.data
            run_ids = [run["runId"] for run in result.data["pipelineRunsOrError"]["results"]]
            assert len(run_ids) == 2
            assert run_ids[0] == run_id_2
            assert run_ids[1] == run_id_1


def test_paginated_runs_query():
    with instance_for_test() as instance:
        repo = get_repo()
        _ = instance.create_run_for_job(
            repo.get_job("foo_job"), status=DagsterRunStatus.STARTED
        ).run_id
        run_id_2 = instance.create_run_for_job(
            repo.get_job("foo_job"), status=DagsterRunStatus.FAILURE
        ).run_id
        run_id_3 = instance.create_run_for_job(
            repo.get_job("foo_job"), status=DagsterRunStatus.SUCCESS
        ).run_id
        with define_out_of_process_context(__file__, "get_repo", instance) as context:
            result = execute_dagster_graphql(
                context,
                PAGINATED_RUNS_QUERY,
                variables={"cursor": run_id_3, "limit": 1},
            )
            assert result.data
            run_ids = [run["runId"] for run in result.data["pipelineRunsOrError"]["results"]]
            assert len(run_ids) == 1
            assert run_ids[0] == run_id_2


def test_filtered_runs_query():
    with instance_for_test() as instance:
        repo = get_repo()
        _ = instance.create_run_for_job(
            repo.get_job("foo_job"), status=DagsterRunStatus.STARTED
        ).run_id
        run_id_2 = instance.create_run_for_job(
            repo.get_job("foo_job"), status=DagsterRunStatus.FAILURE
        ).run_id
        _ = instance.create_run_for_job(
            repo.get_job("foo_job"), status=DagsterRunStatus.SUCCESS
        ).run_id
        with define_out_of_process_context(__file__, "get_repo", instance) as context:
            result = execute_dagster_graphql(context, FILTERED_RUNS_QUERY)
            assert result.data
            run_ids = [run["runId"] for run in result.data["pipelineRunsOrError"]["results"]]
            assert len(run_ids) == 1
            assert run_ids[0] == run_id_2


def test_repositories_query():
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, "get_repo", instance) as context:
            result = execute_dagster_graphql(context, REPOSITORIES_QUERY)
            assert not result.errors
            assert result.data
            repositories = result.data["repositoriesOrError"]["nodes"]
            assert len(repositories) == 1
            assert repositories[0]["name"] == "my_repo"


def test_pipelines_query():
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, "get_repo", instance) as context:
            result = execute_dagster_graphql(
                context,
                PIPELINES_QUERY,
                variables={
                    "repositoryLocationName": "test_location",
                    "repositoryName": "my_repo",
                },
            )
            assert not result.errors
            assert result.data
            pipelines = result.data["repositoryOrError"]["pipelines"]
            assert len(pipelines) == 2


def test_launch_mutation():
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, "get_repo", instance) as context:
            result = execute_dagster_graphql(
                context,
                LAUNCH_PIPELINE,
                variables={
                    "repositoryLocationName": "test_location",
                    "repositoryName": "my_repo",
                    "pipelineName": "foo_job",
                    "runConfigData": {},
                    "mode": "default",
                },
            )
            assert not result.errors
            assert result.data
            run = result.data["launchPipelineExecution"]["run"]
            assert run and run["runId"]


def test_launch_mutation_error():
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, "get_repo", instance) as context:
            result = execute_dagster_graphql(
                context,
                LAUNCH_PIPELINE,
                variables={
                    "repositoryLocationName": "test_location",
                    "repositoryName": "my_repo",
                    "pipelineName": "foo_job",
                    "runConfigData": {"invalid": "config"},
                    "mode": "default",
                },
            )
            assert not result.errors
            assert result.data
            errors = result.data["launchPipelineExecution"]["errors"]
            assert len(errors) == 1
            message = errors[0]["message"]
            assert message
