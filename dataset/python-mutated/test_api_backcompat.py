import time
from dagster import job, op, repository
from dagster._core.storage.dagster_run import DagsterRunStatus
from dagster._core.test_utils import instance_for_test
from dagster_graphql.test.utils import define_out_of_process_context, execute_dagster_graphql
RUNS_QUERY = '\nquery RunsQuery {\n  pipelineRunsOrError {\n    __typename\n    ... on PipelineRuns {\n      results {\n        runId\n        pipelineName\n        status\n        runConfigYaml\n        stats {\n          ... on PipelineRunStatsSnapshot {\n            startTime\n            endTime\n            stepsFailed\n          }\n        }\n      }\n    }\n  }\n}\n'
PAGINATED_RUNS_QUERY = '\nquery PaginatedRunsQuery($cursor: String!, $limit: Int) {\n  pipelineRunsOrError(\n    cursor: $cursor\n    limit: $limit\n  ) {\n    __typename\n    ... on PipelineRuns {\n      results {\n        runId\n        pipelineName\n        status\n        runConfigYaml\n        stats {\n          ... on PipelineRunStatsSnapshot {\n            startTime\n            endTime\n            stepsFailed\n          }\n        }\n      }\n    }\n  }\n}\n'
FILTERED_RUNS_QUERY = '\nquery FilteredRunsQuery {\n  pipelineRunsOrError(filter: { statuses: [FAILURE] }) {\n    __typename\n    ... on PipelineRuns {\n      results {\n        runId\n        pipelineName\n        status\n        runConfigYaml\n        stats {\n          ... on PipelineRunStatsSnapshot {\n            startTime\n            endTime\n            stepsFailed\n          }\n        }\n      }\n    }\n  }\n}\n'
REPOSITORIES_QUERY = '\nquery RepositoriesQuery {\n  repositoriesOrError {\n    ... on RepositoryConnection {\n      nodes {\n        name\n        location {\n          name\n        }\n      }\n    }\n  }\n}\n'
PIPELINES_QUERY = '\nquery PipelinesQuery(\n  $repositoryLocationName: String!\n  $repositoryName: String!\n) {\n  repositoryOrError(\n    repositorySelector: {\n      repositoryLocationName: $repositoryLocationName\n      repositoryName: $repositoryName\n    }\n  ) {\n    ... on Repository {\n      pipelines {\n        name\n      }\n    }\n  }\n}\n'
LAUNCH_PIPELINE = '\nmutation ExecutePipeline(\n  $repositoryLocationName: String!\n  $repositoryName: String!\n  $pipelineName: String!\n  $runConfigData: RunConfigData!\n  $mode: String!\n) {\n  launchPipelineExecution(\n    executionParams: {\n      selector: {\n        repositoryLocationName: $repositoryLocationName\n        repositoryName: $repositoryName\n        pipelineName: $pipelineName\n      }\n      runConfigData: $runConfigData\n      mode: $mode\n    }\n  ) {\n    __typename\n    ... on LaunchPipelineRunSuccess {\n      run {\n        runId\n      }\n    }\n    ... on PipelineConfigValidationInvalid {\n      errors {\n        message\n        reason\n      }\n    }\n    ... on PythonError {\n      message\n    }\n  }\n}\n'

def get_repo():
    if False:
        i = 10
        return i + 15

    @op
    def my_op():
        if False:
            return 10
        pass

    @op
    def loop():
        if False:
            while True:
                i = 10
        while True:
            time.sleep(0.1)

    @job
    def infinite_loop_job():
        if False:
            while True:
                i = 10
        loop()

    @job
    def foo_job():
        if False:
            print('Hello World!')
        my_op()

    @repository
    def my_repo():
        if False:
            for i in range(10):
                print('nop')
        return [infinite_loop_job, foo_job]
    return my_repo

def test_runs_query():
    if False:
        return 10
    with instance_for_test() as instance:
        repo = get_repo()
        run_id_1 = instance.create_run_for_job(repo.get_job('foo_job'), status=DagsterRunStatus.STARTED).run_id
        run_id_2 = instance.create_run_for_job(repo.get_job('foo_job'), status=DagsterRunStatus.FAILURE).run_id
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            result = execute_dagster_graphql(context, RUNS_QUERY)
            assert result.data
            run_ids = [run['runId'] for run in result.data['pipelineRunsOrError']['results']]
            assert len(run_ids) == 2
            assert run_ids[0] == run_id_2
            assert run_ids[1] == run_id_1

def test_paginated_runs_query():
    if False:
        print('Hello World!')
    with instance_for_test() as instance:
        repo = get_repo()
        _ = instance.create_run_for_job(repo.get_job('foo_job'), status=DagsterRunStatus.STARTED).run_id
        run_id_2 = instance.create_run_for_job(repo.get_job('foo_job'), status=DagsterRunStatus.FAILURE).run_id
        run_id_3 = instance.create_run_for_job(repo.get_job('foo_job'), status=DagsterRunStatus.SUCCESS).run_id
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            result = execute_dagster_graphql(context, PAGINATED_RUNS_QUERY, variables={'cursor': run_id_3, 'limit': 1})
            assert result.data
            run_ids = [run['runId'] for run in result.data['pipelineRunsOrError']['results']]
            assert len(run_ids) == 1
            assert run_ids[0] == run_id_2

def test_filtered_runs_query():
    if False:
        print('Hello World!')
    with instance_for_test() as instance:
        repo = get_repo()
        _ = instance.create_run_for_job(repo.get_job('foo_job'), status=DagsterRunStatus.STARTED).run_id
        run_id_2 = instance.create_run_for_job(repo.get_job('foo_job'), status=DagsterRunStatus.FAILURE).run_id
        _ = instance.create_run_for_job(repo.get_job('foo_job'), status=DagsterRunStatus.SUCCESS).run_id
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            result = execute_dagster_graphql(context, FILTERED_RUNS_QUERY)
            assert result.data
            run_ids = [run['runId'] for run in result.data['pipelineRunsOrError']['results']]
            assert len(run_ids) == 1
            assert run_ids[0] == run_id_2

def test_repositories_query():
    if False:
        for i in range(10):
            print('nop')
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            result = execute_dagster_graphql(context, REPOSITORIES_QUERY)
            assert not result.errors
            assert result.data
            repositories = result.data['repositoriesOrError']['nodes']
            assert len(repositories) == 1
            assert repositories[0]['name'] == 'my_repo'

def test_pipelines_query():
    if False:
        print('Hello World!')
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            result = execute_dagster_graphql(context, PIPELINES_QUERY, variables={'repositoryLocationName': 'test_location', 'repositoryName': 'my_repo'})
            assert not result.errors
            assert result.data
            pipelines = result.data['repositoryOrError']['pipelines']
            assert len(pipelines) == 2

def test_launch_mutation():
    if False:
        print('Hello World!')
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            result = execute_dagster_graphql(context, LAUNCH_PIPELINE, variables={'repositoryLocationName': 'test_location', 'repositoryName': 'my_repo', 'pipelineName': 'foo_job', 'runConfigData': {}, 'mode': 'default'})
            assert not result.errors
            assert result.data
            run = result.data['launchPipelineExecution']['run']
            assert run and run['runId']

def test_launch_mutation_error():
    if False:
        while True:
            i = 10
    with instance_for_test() as instance:
        with define_out_of_process_context(__file__, 'get_repo', instance) as context:
            result = execute_dagster_graphql(context, LAUNCH_PIPELINE, variables={'repositoryLocationName': 'test_location', 'repositoryName': 'my_repo', 'pipelineName': 'foo_job', 'runConfigData': {'invalid': 'config'}, 'mode': 'default'})
            assert not result.errors
            assert result.data
            errors = result.data['launchPipelineExecution']['errors']
            assert len(errors) == 1
            message = errors[0]['message']
            assert message