from dagster._core.test_utils import SingleThreadPoolExecutor, wait_for_futures
from dagster._daemon import get_default_daemon_logger
from dagster._daemon.sensor import execute_sensor_iteration
from dagster_graphql.test.utils import (
    execute_dagster_graphql,
    infer_instigation_selector,
    infer_repository_selector,
)

from .graphql_context_test_suite import NonLaunchableGraphQLContextTestMatrix

INSTIGATION_QUERY = """
query JobQuery($instigationSelector: InstigationSelector!) {
  instigationStateOrError(instigationSelector: $instigationSelector) {
    __typename
    ... on PythonError {
      message
      stack
    }
    ... on InstigationState {
        id
        nextTick {
            timestamp
        }
    }
  }
}
"""


def _create_sensor_tick(graphql_context):
    logger = get_default_daemon_logger("SensorDaemon")
    futures = {}
    list(
        execute_sensor_iteration(
            graphql_context.process_context,
            logger,
            threadpool_executor=SingleThreadPoolExecutor(),
            sensor_tick_futures=futures,
        )
    )
    wait_for_futures(futures)


class TestNextTickRepository(NonLaunchableGraphQLContextTestMatrix):
    def test_schedule_next_tick(self, graphql_context):
        repository_selector = infer_repository_selector(graphql_context)
        external_repository = graphql_context.get_code_location(
            repository_selector["repositoryLocationName"]
        ).get_repository(repository_selector["repositoryName"])

        schedule_name = "no_config_job_hourly_schedule"
        external_schedule = external_repository.get_external_schedule(schedule_name)
        selector = infer_instigation_selector(graphql_context, schedule_name)

        # need to be running in order to generate a future tick
        graphql_context.instance.start_schedule(external_schedule)
        result = execute_dagster_graphql(
            graphql_context, INSTIGATION_QUERY, variables={"instigationSelector": selector}
        )

        assert result.data
        assert result.data["instigationStateOrError"]["__typename"] == "InstigationState"
        next_tick = result.data["instigationStateOrError"]["nextTick"]
        assert next_tick

    def test_sensor_next_tick(self, graphql_context):
        repository_selector = infer_repository_selector(graphql_context)
        external_repository = graphql_context.get_code_location(
            repository_selector["repositoryLocationName"]
        ).get_repository(repository_selector["repositoryName"])

        sensor_name = "always_no_config_sensor"
        external_sensor = external_repository.get_external_sensor(sensor_name)
        selector = infer_instigation_selector(graphql_context, sensor_name)

        # need to be running and create a sensor tick in the last 30 seconds in order to generate a
        # future tick
        graphql_context.instance.start_sensor(external_sensor)
        _create_sensor_tick(graphql_context)

        result = execute_dagster_graphql(
            graphql_context, INSTIGATION_QUERY, variables={"instigationSelector": selector}
        )

        assert result.data
        assert result.data["instigationStateOrError"]["__typename"] == "InstigationState"
        next_tick = result.data["instigationStateOrError"]["nextTick"]
        assert next_tick
