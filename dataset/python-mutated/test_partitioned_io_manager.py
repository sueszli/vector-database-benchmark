import datetime
from typing import Any, Dict
import pytest
from dagster import DagsterTypeCheckDidNotPass, DailyPartitionsDefinition, HourlyPartitionsDefinition, asset, materialize
from dagster._core.instance_for_test import instance_for_test
from pytest import fixture

@fixture
def start():
    if False:
        print('Hello World!')
    return datetime.datetime(2022, 1, 1)

@fixture
def hourly(start):
    if False:
        for i in range(10):
            print('nop')
    return HourlyPartitionsDefinition(start_date=f'{start:%Y-%m-%d-%H:%M}')

@fixture
def daily(start):
    if False:
        return 10
    return DailyPartitionsDefinition(start_date=f'{start:%Y-%m-%d}')

def test_partitioned_io_manager(hourly, daily):
    if False:
        while True:
            i = 10

    @asset(partitions_def=hourly)
    def hourly_asset():
        if False:
            i = 10
            return i + 15
        return 42

    @asset(partitions_def=daily)
    def daily_asset(hourly_asset: Dict[str, Any]):
        if False:
            return 10
        return hourly_asset
    with instance_for_test() as instance:
        hourly_keys = [f'2022-01-01-{hour:02d}:00' for hour in range(0, 24)]
        for key in hourly_keys:
            materialize([hourly_asset], partition_key=key, instance=instance)
        result = materialize([*hourly_asset.to_source_assets(), daily_asset], partition_key='2022-01-01', instance=instance)
        expected = {k: 42 for k in hourly_keys}
        assert result.output_for_node('daily_asset') == expected

def test_partitioned_io_manager_preserves_single_partition_dependency(daily):
    if False:
        for i in range(10):
            print('nop')

    @asset(partitions_def=daily)
    def upstream_asset():
        if False:
            for i in range(10):
                print('nop')
        return 42

    @asset(partitions_def=daily)
    def daily_asset(upstream_asset: int):
        if False:
            return 10
        return upstream_asset
    result = materialize([upstream_asset, daily_asset], partition_key='2022-01-01')
    assert result.output_for_node('daily_asset') == 42

def test_partitioned_io_manager_single_partition_dependency_errors_with_wrong_typing(daily):
    if False:
        i = 10
        return i + 15

    @asset(partitions_def=daily)
    def upstream_asset():
        if False:
            return 10
        return 42

    @asset(partitions_def=daily)
    def daily_asset(upstream_asset: Dict[str, Any]):
        if False:
            print('Hello World!')
        return upstream_asset
    with pytest.raises(DagsterTypeCheckDidNotPass):
        materialize([upstream_asset, daily_asset], partition_key='2022-01-01')