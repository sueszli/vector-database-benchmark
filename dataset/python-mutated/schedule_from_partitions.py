from .partitioned_job import my_partitioned_config
from dagster import build_schedule_from_partitioned_job, job

@job(config=my_partitioned_config)
def do_stuff_partitioned():
    if False:
        print('Hello World!')
    ...
do_stuff_partitioned_schedule = build_schedule_from_partitioned_job(do_stuff_partitioned)
from dagster import asset, build_schedule_from_partitioned_job, define_asset_job, HourlyPartitionsDefinition

@asset(partitions_def=HourlyPartitionsDefinition(start_date='2020-01-01-00:00'))
def hourly_asset():
    if False:
        print('Hello World!')
    ...
partitioned_asset_job = define_asset_job('partitioned_job', selection=[hourly_asset])
asset_partitioned_schedule = build_schedule_from_partitioned_job(partitioned_asset_job)
from .static_partitioned_job import continent_job, CONTINENTS
from dagster import schedule, RunRequest

@schedule(cron_schedule='0 0 * * *', job=continent_job)
def continent_schedule():
    if False:
        i = 10
        return i + 15
    for c in CONTINENTS:
        yield RunRequest(run_key=c, partition_key=c)

@schedule(cron_schedule='0 0 * * *', job=continent_job)
def antarctica_schedule():
    if False:
        i = 10
        return i + 15
    return RunRequest(partition_key='Antarctica')