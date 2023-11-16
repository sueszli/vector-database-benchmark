import datetime
import random
from collections import defaultdict
from typing import List, NamedTuple, Optional
import mock
import pendulum
import pytest
from dagster import AssetKey, AssetOut, AssetSelection, DagsterEventType, DagsterInstance, Output, asset, multi_asset, repository
from dagster._core.definitions.asset_graph import AssetGraph
from dagster._core.definitions.asset_layer import build_asset_selection_job
from dagster._core.definitions.data_time import CachingDataTimeResolver
from dagster._core.definitions.data_version import DataVersion
from dagster._core.definitions.decorators.source_asset_decorator import observable_source_asset
from dagster._core.definitions.events import AssetKeyPartitionKey
from dagster._core.definitions.materialize import materialize_to_memory
from dagster._core.definitions.observe import observe
from dagster._core.definitions.time_window_partitions import DailyPartitionsDefinition
from dagster._core.event_api import EventRecordsFilter
from dagster._seven.compat.pendulum import create_pendulum_time
from dagster._utils.caching_instance_queryer import CachingInstanceQueryer

@pytest.mark.parametrize('ignore_asset_tags', [True, False])
@pytest.mark.parametrize(['runs_to_expected_data_times_index'], [([('ace', {'ace': {'a': 0}}), ('abd', {'ab': {'a': 1}, 'cde': {'a': 0}}), ('ac', {'ac': {'a': 2}, 'b': {'a': 1}, 'ed': {'a': 0}}), ('e', {'ace': {'a': 2}, 'b': {'a': 1}, 'd': {'a': 0}})],), ([('abcf', {'abc': {'a': 0}}), ('bd', {'abd': {'a': 0}}), ('a', {'a': {'a': 2}, 'bcd': {'a': 0}}), ('f', {'a': {'a': 2}, 'bcdf': {'a': 0}}), ('bdf', {'ab': {'a': 2}, 'cdf': {'a': 0}}), ('c', {'abc': {'a': 2}, 'df': {'a': 0}}), ('df', {'abcdf': {'a': 2}})],)])
def test_calculate_data_time_unpartitioned(ignore_asset_tags, runs_to_expected_data_times_index):
    if False:
        while True:
            i = 10
    'A = B = D = F\n     \\\\  //\n       C = E\n    B,C,D share an op.\n    '

    @asset
    def a():
        if False:
            return 10
        return 1

    @multi_asset(deps=[AssetKey('a')], outs={'b': AssetOut(is_required=False), 'c': AssetOut(is_required=False), 'd': AssetOut(is_required=False)}, can_subset=True, internal_asset_deps={'b': {AssetKey('a')}, 'c': {AssetKey('a')}, 'd': {AssetKey('b'), AssetKey('c')}})
    def bcd(context):
        if False:
            i = 10
            return i + 15
        for output_name in sorted(context.selected_output_names):
            yield Output(output_name, output_name)

    @asset(deps=[AssetKey('c')])
    def e():
        if False:
            while True:
                i = 10
        return 1

    @asset(deps=[AssetKey('d')])
    def f():
        if False:
            while True:
                i = 10
        return 1
    all_assets = [a, bcd, e, f]
    asset_graph = AssetGraph.from_assets(all_assets)
    with DagsterInstance.ephemeral() as instance:
        materialization_times_index = defaultdict(dict)
        for (idx, (to_materialize, expected_index_mapping)) in enumerate(runs_to_expected_data_times_index):
            result = build_asset_selection_job('materialize_job', assets=all_assets, source_assets=[], asset_selection=AssetSelection.keys(*(AssetKey(c) for c in to_materialize)).resolve(all_assets), asset_checks=[]).execute_in_process(instance=instance)
            assert result.success
            data_time_queryer = CachingDataTimeResolver(instance_queryer=CachingInstanceQueryer(instance, asset_graph))
            for entry in instance.all_logs(result.run_id, of_type=DagsterEventType.ASSET_MATERIALIZATION):
                asset_key = entry.dagster_event.event_specific_data.materialization.asset_key
                materialization_times_index[asset_key][idx] = datetime.datetime.fromtimestamp(entry.timestamp, tz=datetime.timezone.utc)
            for (asset_keys, expected_data_times) in expected_index_mapping.items():
                for ak in asset_keys:
                    latest_asset_record = data_time_queryer.instance_queryer.get_latest_materialization_or_observation_record(AssetKeyPartitionKey(AssetKey(ak)))
                    if ignore_asset_tags:
                        with mock.patch('dagster.AssetMaterialization.tags', new_callable=mock.PropertyMock) as tags_property:
                            tags_property.return_value = None
                            upstream_data_times = data_time_queryer.get_data_time_by_key_for_record(record=latest_asset_record)
                    else:
                        upstream_data_times = data_time_queryer.get_data_time_by_key_for_record(record=latest_asset_record)
                    assert upstream_data_times == {AssetKey(k): materialization_times_index[AssetKey(k)][v] for (k, v) in expected_data_times.items()}

@asset(partitions_def=DailyPartitionsDefinition(start_date='2023-01-01'))
def partitioned_asset():
    if False:
        print('Hello World!')
    pass

@asset(deps=[AssetKey('partitioned_asset')])
def unpartitioned_asset():
    if False:
        return 10
    pass

@repository
def partition_repo():
    if False:
        print('Hello World!')
    return [partitioned_asset, unpartitioned_asset]

def _materialize_partitions(instance, partitions):
    if False:
        print('Hello World!')
    for partition in partitions:
        result = materialize_to_memory(assets=[partitioned_asset], instance=instance, partition_key=partition)
        assert result.success

def _get_record(instance):
    if False:
        print('Hello World!')
    result = materialize_to_memory(assets=[unpartitioned_asset, *partitioned_asset.to_source_assets()], instance=instance)
    assert result.success
    return next(iter(instance.get_event_records(EventRecordsFilter(event_type=DagsterEventType.ASSET_MATERIALIZATION, asset_key=AssetKey('unpartitioned_asset')), ascending=False, limit=1)))

class PartitionedDataTimeScenario(NamedTuple):
    before_partitions: List[str]
    after_partitions: List[str]
    expected_time: Optional[datetime.datetime]
scenarios = {'empty': PartitionedDataTimeScenario(before_partitions=[], after_partitions=[], expected_time=None), 'first_missing': PartitionedDataTimeScenario(before_partitions=['2023-01-02', '2023-01-03'], after_partitions=[], expected_time=None), 'some_filled': PartitionedDataTimeScenario(before_partitions=['2023-01-01', '2023-01-02', '2023-01-03'], after_partitions=[], expected_time=datetime.datetime(2023, 1, 4, tzinfo=datetime.timezone.utc)), 'middle_missing': PartitionedDataTimeScenario(before_partitions=['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-05', '2023-01-06'], after_partitions=[], expected_time=datetime.datetime(2023, 1, 4, tzinfo=datetime.timezone.utc)), 'new_duplicate_partitions': PartitionedDataTimeScenario(before_partitions=['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'], after_partitions=['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-03'], expected_time=datetime.datetime(2023, 1, 5, tzinfo=datetime.timezone.utc)), 'new_duplicate_partitions2': PartitionedDataTimeScenario(before_partitions=['2023-01-01', '2023-01-02'], after_partitions=['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-01'], expected_time=datetime.datetime(2023, 1, 3, tzinfo=datetime.timezone.utc)), 'net_new_partitions': PartitionedDataTimeScenario(before_partitions=['2023-01-01', '2023-01-02', '2023-01-03'], after_partitions=['2023-01-04', '2023-01-05', '2023-01-06'], expected_time=datetime.datetime(2023, 1, 4, tzinfo=datetime.timezone.utc)), 'net_new_partitions2': PartitionedDataTimeScenario(before_partitions=['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'], after_partitions=['2023-01-01', '2023-01-01', '2023-01-01', '2023-01-06', '2023-01-06', '2023-01-06'], expected_time=datetime.datetime(2023, 1, 5, tzinfo=datetime.timezone.utc)), 'net_new_partitions_with_middle_missing': PartitionedDataTimeScenario(before_partitions=['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-05', '2023-01-06'], after_partitions=['2023-01-04', '2023-01-04'], expected_time=datetime.datetime(2023, 1, 4, tzinfo=datetime.timezone.utc))}

@pytest.mark.parametrize('scenario', list(scenarios.values()), ids=list(scenarios.keys()))
def test_partitioned_data_time(scenario):
    if False:
        for i in range(10):
            print('nop')
    with DagsterInstance.ephemeral() as instance, pendulum.test(create_pendulum_time(2023, 1, 7)):
        _materialize_partitions(instance, scenario.before_partitions)
        record = _get_record(instance=instance)
        _materialize_partitions(instance, scenario.after_partitions)
        data_time_queryer = CachingDataTimeResolver(instance_queryer=CachingInstanceQueryer(instance, partition_repo.asset_graph))
        data_time = data_time_queryer.get_data_time_by_key_for_record(record=record)
        if scenario.expected_time is None:
            assert data_time == {} or data_time == {AssetKey('partitioned_asset'): None}
        else:
            assert data_time == {AssetKey('partitioned_asset'): scenario.expected_time}

@observable_source_asset
def sA():
    if False:
        print('Hello World!')
    return DataVersion(str(random.random()))

@observable_source_asset
def sB():
    if False:
        print('Hello World!')
    return DataVersion(str(random.random()))

@asset(deps=[sA])
def A():
    if False:
        while True:
            i = 10
    pass

@asset(deps=[sB])
def B():
    if False:
        i = 10
        return i + 15
    pass

@asset(deps=[B])
def B2():
    if False:
        i = 10
        return i + 15
    pass

@asset(deps=[sA, sB])
def AB():
    if False:
        return 10
    pass

@repository
def versioned_repo():
    if False:
        return 10
    return [sA, sB, A, B, AB, B2]

def observe_sources(*args):
    if False:
        print('Hello World!')

    def observe_sources_fn(*, instance, times_by_key, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        for arg in args:
            key = AssetKey(arg)
            observe(source_assets=[versioned_repo.source_assets_by_key[key]], instance=instance)
            latest_record = instance.get_latest_data_version_record(key, is_source=True)
            latest_timestamp = latest_record.timestamp
            times_by_key[key].append(datetime.datetime.fromtimestamp(latest_timestamp, tz=datetime.timezone.utc))
    return observe_sources_fn

def run_assets(*args):
    if False:
        i = 10
        return i + 15

    def run_assets_fn(*, instance, **kwargs):
        if False:
            print('Hello World!')
        assets = [versioned_repo.assets_defs_by_key[AssetKey(arg)] for arg in args]
        materialize_to_memory(assets=assets, instance=instance)
    return run_assets_fn

def assert_has_current_time(key_str):
    if False:
        print('Hello World!')

    def assert_has_current_time_fn(*, instance, evaluation_time, **kwargs):
        if False:
            return 10
        resolver = CachingDataTimeResolver(instance_queryer=CachingInstanceQueryer(instance, versioned_repo.asset_graph))
        data_time = resolver.get_current_data_time(AssetKey(key_str), current_time=evaluation_time)
        assert data_time == evaluation_time
    return assert_has_current_time_fn

def assert_has_index_time(key_str, source_key_str, index):
    if False:
        return 10

    def assert_has_index_time_fn(*, instance, times_by_key, evaluation_time, **kwargs):
        if False:
            return 10
        resolver = CachingDataTimeResolver(instance_queryer=CachingInstanceQueryer(instance, versioned_repo.asset_graph))
        data_time = resolver.get_current_data_time(AssetKey(key_str), current_time=evaluation_time)
        if index is None:
            assert data_time is None
        else:
            assert data_time == times_by_key[AssetKey(source_key_str)][index]
    return assert_has_index_time_fn
timelines = {'basic_one_parent': [observe_sources('sA'), assert_has_index_time('A', None, None), run_assets('A'), assert_has_current_time('A'), observe_sources('sA'), assert_has_index_time('A', 'sA', 1), run_assets('A'), assert_has_current_time('A')], 'basic_two_parents': [observe_sources('sA', 'sB'), assert_has_index_time('AB', None, None), run_assets('AB'), assert_has_current_time('AB'), observe_sources('sA'), assert_has_index_time('AB', 'sA', 1), run_assets('AB'), assert_has_current_time('AB'), observe_sources('sA'), assert_has_index_time('AB', 'sA', 2), observe_sources('sB'), assert_has_index_time('AB', 'sA', 2), run_assets('AB'), assert_has_current_time('AB')], 'chained': [observe_sources('sA', 'sB'), run_assets('B'), assert_has_current_time('B'), run_assets('B2'), assert_has_current_time('B2'), observe_sources('sA'), assert_has_current_time('B'), assert_has_current_time('B2'), observe_sources('sB'), assert_has_index_time('B', 'sB', 1), assert_has_index_time('B2', 'sB', 1), run_assets('B'), assert_has_current_time('B'), assert_has_index_time('B2', 'sB', 1), run_assets('B2'), assert_has_current_time('B2')], 'chained_multiple_observations': [observe_sources('sB'), run_assets('B', 'B2'), assert_has_current_time('B'), assert_has_current_time('B2'), observe_sources('sB'), observe_sources('sB'), observe_sources('sB'), observe_sources('sB'), observe_sources('sB'), assert_has_index_time('B', 'sB', 1), assert_has_index_time('B2', 'sB', 1), run_assets('B'), assert_has_current_time('B'), observe_sources('sB'), observe_sources('sB'), observe_sources('sB'), observe_sources('sB'), observe_sources('sB'), assert_has_index_time('B', 'sB', 6), assert_has_index_time('B2', 'sB', 1)]}

@pytest.mark.parametrize('timeline', list(timelines.values()), ids=list(timelines.keys()))
def test_non_volatile_data_time(timeline):
    if False:
        for i in range(10):
            print('nop')
    with DagsterInstance.ephemeral() as instance:
        times_by_key = defaultdict(list)
        for action in timeline:
            action(instance=instance, times_by_key=times_by_key, evaluation_time=pendulum.now('UTC'))