import json
import pytest
from dagster import DagsterUnknownPartitionError, StaticPartitionsDefinition, daily_partitioned_config, dynamic_partitioned_config, job, op, static_partitioned_config
from dagster._core.definitions.partition import partitioned_config
from dagster._seven.compat.pendulum import create_pendulum_time

@op
def my_op(context):
    if False:
        while True:
            i = 10
    context.log.info(context.op_config)
RUN_CONFIG = {'ops': {'my_op': {'config': 'hello'}}}

def test_static_partitioned_job():
    if False:
        while True:
            i = 10

    @static_partitioned_config(['blah'], tags_for_partition_key_fn=lambda key: {'foo': key})
    def my_static_partitioned_config(_partition_key: str):
        if False:
            i = 10
            return i + 15
        return RUN_CONFIG
    assert my_static_partitioned_config('') == RUN_CONFIG

    @job(config=my_static_partitioned_config)
    def my_job():
        if False:
            i = 10
            return i + 15
        my_op()
    partition_keys = my_static_partitioned_config.get_partition_keys()
    assert partition_keys == ['blah']
    result = my_job.execute_in_process(partition_key='blah')
    assert result.success
    assert result.dagster_run.tags['foo'] == 'blah'
    with pytest.raises(DagsterUnknownPartitionError, match='Could not find a partition'):
        result = my_job.execute_in_process(partition_key='doesnotexist')

def test_time_based_partitioned_job():
    if False:
        while True:
            i = 10

    @daily_partitioned_config(start_date='2021-05-05', tags_for_partition_fn=lambda start, end: {'foo': start.strftime('%Y-%m-%d')})
    def my_daily_partitioned_config(_start, _end):
        if False:
            print('Hello World!')
        return RUN_CONFIG
    assert my_daily_partitioned_config(None, None) == RUN_CONFIG

    @job(config=my_daily_partitioned_config)
    def my_job():
        if False:
            while True:
                i = 10
        my_op()
    freeze_datetime = create_pendulum_time(year=2021, month=5, day=6, hour=23, minute=59, second=59, tz='UTC')
    partition_keys = my_daily_partitioned_config.get_partition_keys(freeze_datetime)
    assert len(partition_keys) == 1
    partition_key = partition_keys[0]
    result = my_job.execute_in_process(partition_key=partition_key)
    assert result.success
    assert result.dagster_run.tags['foo'] == '2021-05-05'
    with pytest.raises(DagsterUnknownPartitionError, match='Could not find a partition'):
        result = my_job.execute_in_process(partition_key='doesnotexist')

def test_general_partitioned_config():
    if False:
        while True:
            i = 10
    partitions_def = StaticPartitionsDefinition(['blah'])

    @partitioned_config(partitions_def, tags_for_partition_key_fn=lambda key: {'foo': key})
    def my_partitioned_config(_partition_key):
        if False:
            while True:
                i = 10
        return {'ops': {'my_op': {'config': _partition_key}}}
    assert my_partitioned_config('blah') == {'ops': {'my_op': {'config': 'blah'}}}

    @job(config=my_partitioned_config)
    def my_job():
        if False:
            i = 10
            return i + 15
        my_op()
    partition_keys = my_partitioned_config.get_partition_keys()
    assert partition_keys == ['blah']
    result = my_job.execute_in_process(partition_key='blah')
    assert result.success
    assert result.dagster_run.tags['foo'] == 'blah'
    with pytest.raises(DagsterUnknownPartitionError, match='Could not find a partition'):
        result = my_job.execute_in_process(partition_key='doesnotexist')

def test_dynamic_partitioned_config():
    if False:
        while True:
            i = 10

    def partition_fn(_current_time=None):
        if False:
            return 10
        return ['blah']

    @dynamic_partitioned_config(partition_fn, tags_for_partition_key_fn=lambda key: {'foo': key})
    def my_dynamic_partitioned_config(_partition_key):
        if False:
            for i in range(10):
                print('nop')
        return RUN_CONFIG
    assert my_dynamic_partitioned_config('') == RUN_CONFIG

    @job(config=my_dynamic_partitioned_config)
    def my_job():
        if False:
            print('Hello World!')
        my_op()
    partition_keys = my_dynamic_partitioned_config.get_partition_keys()
    assert partition_keys == ['blah']
    result = my_job.execute_in_process(partition_key='blah')
    assert result.success
    assert result.dagster_run.tags['foo'] == 'blah'
    with pytest.raises(DagsterUnknownPartitionError, match='Could not find a partition'):
        result = my_job.execute_in_process(partition_key='doesnotexist')

def test_dict_partitioned_config_tags():
    if False:
        while True:
            i = 10

    def partition_fn(_current_time=None):
        if False:
            i = 10
            return i + 15
        return ['blah']

    @dynamic_partitioned_config(partition_fn, tags_for_partition_key_fn=lambda key: {'foo': {'bar': key}})
    def my_dynamic_partitioned_config(_partition_key):
        if False:
            return 10
        return RUN_CONFIG
    assert my_dynamic_partitioned_config('') == RUN_CONFIG

    @job(config=my_dynamic_partitioned_config)
    def my_job():
        if False:
            for i in range(10):
                print('nop')
        my_op()
    partition_keys = my_dynamic_partitioned_config.get_partition_keys()
    assert partition_keys == ['blah']
    result = my_job.execute_in_process(partition_key='blah')
    assert result.success
    assert result.dagster_run.tags['foo'] == json.dumps({'bar': 'blah'})