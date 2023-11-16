import pendulum
from dagster import materialize
from dagster._core.scheduler.instigation import TickStatus
from dagster._seven.compat.pendulum import create_pendulum_time, to_timezone

from .test_run_status_sensors import (
    instance_with_single_code_location_multiple_repos_with_sensors,
)
from .test_sensor_run import (
    a_source_asset,
    evaluate_sensors,
    validate_tick,
)


def test_monitor_source_asset_sensor(executor):
    """Tests a multi asset sensor that monitors an asset in another repo."""
    freeze_datetime = to_timezone(
        create_pendulum_time(year=2019, month=2, day=27, tz="UTC"),
        "US/Central",
    )
    with instance_with_single_code_location_multiple_repos_with_sensors() as (
        instance,
        workspace_ctx,
        repos,
    ):
        asset_sensor_repo = repos["asset_sensor_repo"]
        with pendulum.test(freeze_datetime):
            the_sensor = asset_sensor_repo.get_external_sensor("monitor_source_asset_sensor")
            instance.start_sensor(the_sensor)

            evaluate_sensors(workspace_ctx, executor)

            ticks = instance.get_ticks(the_sensor.get_external_origin_id(), the_sensor.selector_id)
            assert len(ticks) == 1
            validate_tick(
                ticks[0],
                the_sensor,
                freeze_datetime,
                TickStatus.SKIPPED,
            )

            freeze_datetime = freeze_datetime.add(seconds=60)
        with pendulum.test(freeze_datetime):
            materialize([a_source_asset], instance=instance)

            evaluate_sensors(workspace_ctx, executor)

            ticks = instance.get_ticks(the_sensor.get_external_origin_id(), the_sensor.selector_id)
            assert len(ticks) == 2
            validate_tick(
                ticks[0],
                the_sensor,
                freeze_datetime,
                TickStatus.SUCCESS,
            )
            run_request = instance.get_runs(limit=1)[0]
            assert run_request.job_name == "the_job"
