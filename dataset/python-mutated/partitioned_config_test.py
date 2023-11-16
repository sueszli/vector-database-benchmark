from docs_snippets.concepts.partitions_schedules_sensors.partitioned_job import do_stuff_partitioned
from dagster import job, op
from dagster import validate_run_config, daily_partitioned_config
from datetime import datetime

@daily_partitioned_config(start_date=datetime(2020, 1, 1))
def my_partitioned_config(start: datetime, _end: datetime):
    if False:
        i = 10
        return i + 15
    return {'ops': {'process_data_for_date': {'config': {'date': start.strftime('%Y-%m-%d')}}}}

def test_my_partitioned_config():
    if False:
        while True:
            i = 10
    run_config = my_partitioned_config(datetime(2020, 1, 3), datetime(2020, 1, 4))
    assert run_config == {'ops': {'process_data_for_date': {'config': {'date': '2020-01-03'}}}}
    assert validate_run_config(do_stuff_partitioned, run_config)
from dagster import Config, OpExecutionContext

@daily_partitioned_config(start_date=datetime(2020, 1, 1), minute_offset=15)
def my_offset_partitioned_config(start: datetime, _end: datetime):
    if False:
        return 10
    return {'ops': {'process_data': {'config': {'start': start.strftime('%Y-%m-%d-%H:%M'), 'end': _end.strftime('%Y-%m-%d-%H:%M')}}}}

class ProcessDataConfig(Config):
    start: str
    end: str

@op
def process_data(context: OpExecutionContext, config: ProcessDataConfig):
    if False:
        return 10
    s = config.start
    e = config.end
    context.log.info(f'processing data for {s} - {e}')

@job(config=my_offset_partitioned_config)
def do_more_stuff_partitioned():
    if False:
        print('Hello World!')
    process_data()

def test_my_offset_partitioned_config():
    if False:
        for i in range(10):
            print('nop')
    keys = my_offset_partitioned_config.get_partition_keys()
    assert keys[0] == '2020-01-01'
    assert keys[1] == '2020-01-02'
    run_config = my_offset_partitioned_config.get_run_config_for_partition_key(keys[0])
    assert validate_run_config(do_more_stuff_partitioned, run_config)
    assert run_config == {'ops': {'process_data': {'config': {'start': '2020-01-01-00:15', 'end': '2020-01-02-00:15'}}}}