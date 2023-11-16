import datetime
from .schedules import configurable_job_schedule, configurable_job
from dagster import build_schedule_context, validate_run_config

def test_configurable_job_schedule():
    if False:
        return 10
    context = build_schedule_context(scheduled_execution_time=datetime.datetime(2020, 1, 1))
    run_request = configurable_job_schedule(context)
    assert validate_run_config(configurable_job, run_request.run_config)