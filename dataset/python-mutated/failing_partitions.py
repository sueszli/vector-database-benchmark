from random import random
from dagster import DailyPartitionsDefinition, MultiPartitionsDefinition, StaticPartitionsDefinition, asset
from .dynamic_asset_partitions import multipartition_w_dynamic_partitions_def
FAILURE_RATE = 0.5

@asset(partitions_def=DailyPartitionsDefinition(start_date='2023-02-02'))
def failing_time_partitioned():
    if False:
        for i in range(10):
            print('nop')
    if random() < FAILURE_RATE:
        raise ValueError('Failed')

@asset(partitions_def=StaticPartitionsDefinition(['a', 'b', 'c']))
def failing_static_partitioned():
    if False:
        return 10
    if random() < FAILURE_RATE:
        raise ValueError('Failed')

@asset(partitions_def=StaticPartitionsDefinition(['a', 'b', 'c']))
def downstream_of_failing_partitioned(failing_static_partitioned):
    if False:
        while True:
            i = 10
    ...
time_window_partitions = DailyPartitionsDefinition(start_date='2022-01-01')
static_partitions = StaticPartitionsDefinition(['a', 'b', 'c', 'd'])
composite = MultiPartitionsDefinition({'abc': static_partitions, 'date': time_window_partitions})

@asset(partitions_def=composite)
def failing_multi_partitioned(context):
    if False:
        i = 10
        return i + 15
    if random() < FAILURE_RATE:
        raise ValueError('Failed')

@asset(partitions_def=composite)
def failing_pattern_multi_partitioned(context):
    if False:
        while True:
            i = 10
    'Fail in different patterns for different partitions.\n\n    2022-01: fail d\n    2022-03: fail even days\n    2022-05: fail even days and d\n    2022-07: fail randomly\n    2023: fail all\n    '
    partition = context.partition_key.keys_by_dimension
    abc = partition['abc']
    date = partition['date']
    if date.startswith('2023'):
        raise ValueError('Failed')
    if date.startswith('2022-01') and abc == 'd':
        raise ValueError('Failed')
    if date.startswith('2022-03'):
        day = int(date[-2:])
        if day % 2 == 0:
            raise ValueError('Failed')
    if date.startswith('2022-05'):
        day = int(date[-2:])
        if day % 2 == 0:
            raise ValueError('Failed')
        elif abc == 'd':
            raise ValueError('Failed')
    if date.startswith('2022-07'):
        if random() < FAILURE_RATE:
            raise ValueError('Failed')

@asset(partitions_def=multipartition_w_dynamic_partitions_def)
def multipartitioned_with_dynamic_dimension_random_failures():
    if False:
        return 10
    if random() < FAILURE_RATE:
        raise ValueError('Failed')