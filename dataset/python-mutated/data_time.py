"""The "data time" of an asset materialization is the timestamp on its earliest ancestor
materialization.

An asset materialization is parent of another asset materialization if both:
- The child asset depends on the parent asset.
- The parent materialization is the latest materialization of the parent asset that occurred before
    the child materialization.

The idea of data time is: if an asset is downstream of another asset, then the freshness of the data
in the downstream asset depends on the freshness of the data in the upstream asset. No matter how
recently you've materialized the downstream asset, it can't be fresher than the upstream
materialization it was derived from.
"""
import datetime
from typing import AbstractSet, Dict, Mapping, Optional, Sequence, Tuple, cast
import pendulum
import dagster._check as check
from dagster._core.definitions.asset_graph import AssetGraph
from dagster._core.definitions.asset_selection import AssetSelection
from dagster._core.definitions.data_version import DATA_VERSION_TAG, DataVersion, get_input_event_pointer_tag
from dagster._core.definitions.events import AssetKey, AssetKeyPartitionKey
from dagster._core.definitions.freshness_policy import FreshnessMinutes
from dagster._core.definitions.time_window_partitions import BaseTimeWindowPartitionsSubset, TimeWindowPartitionsDefinition
from dagster._core.errors import DagsterInvariantViolationError
from dagster._core.event_api import EventLogRecord
from dagster._core.storage.dagster_run import FINISHED_STATUSES, DagsterRunStatus, RunsFilter
from dagster._utils import datetime_as_float, make_hashable
from dagster._utils.cached_method import cached_method
from dagster._utils.caching_instance_queryer import CachingInstanceQueryer

class CachingDataTimeResolver:
    _instance_queryer: CachingInstanceQueryer
    _asset_graph: AssetGraph

    def __init__(self, instance_queryer: CachingInstanceQueryer):
        if False:
            print('Hello World!')
        self._instance_queryer = instance_queryer

    @property
    def instance_queryer(self) -> CachingInstanceQueryer:
        if False:
            return 10
        return self._instance_queryer

    @property
    def asset_graph(self) -> AssetGraph:
        if False:
            i = 10
            return i + 15
        return self.instance_queryer.asset_graph

    def _calculate_data_time_partitioned(self, asset_key: AssetKey, cursor: int, partitions_def: TimeWindowPartitionsDefinition) -> Optional[datetime.datetime]:
        if False:
            i = 10
            return i + 15
        'Returns the time up until which all available data has been consumed for this asset.\n\n        At a high level, this algorithm works as follows:\n\n        First, calculate the subset of partitions that have been materialized up until this point\n        in time (ignoring the cursor). This is done using the get_materialized_partitions query,\n\n        Next, we calculate the set of partitions that are net-new since the cursor. This is done by\n        comparing the count of materializations before after the cursor to the total count of\n        materializations.\n\n        Finally, we calculate the minimum time window of the net-new partitions. This time window\n        did not exist at the time of the cursor, so we know that we have all data up until the\n        beginning of that time window, or all data up until the end of the first filled time window\n        in the total set, whichever is less.\n        '
        partition_subset = partitions_def.empty_subset().with_partition_keys((partition_key for partition_key in self._instance_queryer.get_materialized_partitions(asset_key) if partitions_def.has_partition_key(partition_key, current_time=self._instance_queryer.evaluation_time)))
        if not isinstance(partition_subset, BaseTimeWindowPartitionsSubset):
            check.failed(f'Invalid partition subset {type(partition_subset)}')
        sorted_time_windows = sorted(partition_subset.included_time_windows)
        if len(sorted_time_windows) == 0:
            return None
        first_filled_time_window = sorted_time_windows[0]
        first_available_time_window = partitions_def.get_first_partition_window()
        if first_available_time_window is None:
            return None
        if first_available_time_window.start < first_filled_time_window.start:
            return None
        asset_record = self._instance_queryer.get_asset_record(asset_key)
        if asset_record is not None and asset_record.asset_entry is not None and (asset_record.asset_entry.last_materialization_record is not None) and (asset_record.asset_entry.last_materialization_record.storage_id <= cursor):
            return first_filled_time_window.end
        partitions = self._instance_queryer.get_materialized_partitions(asset_key)
        prev_partitions = self._instance_queryer.get_materialized_partitions(asset_key, before_cursor=cursor + 1)
        net_new_partitions = {partition_key for partition_key in partitions - prev_partitions if partitions_def.has_partition_key(partition_key, current_time=self._instance_queryer.evaluation_time)}
        if not net_new_partitions:
            return first_filled_time_window.end
        oldest_net_new_time_window = min((partitions_def.time_window_for_partition_key(partition_key) for partition_key in net_new_partitions))
        return min(oldest_net_new_time_window.start, first_filled_time_window.end)

    def _calculate_data_time_by_key_time_partitioned(self, asset_key: AssetKey, cursor: int, partitions_def: TimeWindowPartitionsDefinition) -> Mapping[AssetKey, Optional[datetime.datetime]]:
        if False:
            print('Hello World!')
        'Returns the data time (i.e. the time up to which the asset has incorporated all available\n        data) for a time-partitioned asset. This method takes into account all partitions that were\n        materialized for this asset up to the provided cursor.\n        '
        partition_data_time = self._calculate_data_time_partitioned(asset_key=asset_key, cursor=cursor, partitions_def=partitions_def)
        root_keys = AssetSelection.keys(asset_key).upstream().sources().resolve(self.asset_graph)
        return {key: partition_data_time for key in root_keys}

    def _upstream_records_by_key(self, asset_key: AssetKey, record_id: int, record_tags_dict: Mapping[str, str]) -> Mapping[AssetKey, 'EventLogRecord']:
        if False:
            i = 10
            return i + 15
        upstream_records: Dict[AssetKey, EventLogRecord] = {}
        for parent_key in self.asset_graph.get_parents(asset_key):
            if parent_key in self.asset_graph.source_asset_keys and (not self.asset_graph.is_observable(parent_key)):
                continue
            input_event_pointer_tag = get_input_event_pointer_tag(parent_key)
            if input_event_pointer_tag not in record_tags_dict:
                before_cursor = record_id
            elif record_tags_dict[input_event_pointer_tag] != 'NULL':
                before_cursor = int(record_tags_dict[input_event_pointer_tag]) + 1
            else:
                before_cursor = None
            if before_cursor is not None:
                parent_record = self._instance_queryer.get_latest_materialization_or_observation_record(AssetKeyPartitionKey(parent_key), before_cursor=before_cursor)
                if parent_record is not None:
                    upstream_records[parent_key] = parent_record
        return upstream_records

    @cached_method
    def _calculate_data_time_by_key_unpartitioned(self, *, asset_key: AssetKey, record_id: int, record_timestamp: float, record_tags: Tuple[Tuple[str, str]], current_time: datetime.datetime) -> Mapping[AssetKey, Optional[datetime.datetime]]:
        if False:
            print('Hello World!')
        record_tags_dict = dict(record_tags)
        upstream_records_by_key = self._upstream_records_by_key(asset_key, record_id, record_tags_dict)
        if not upstream_records_by_key:
            if not self.asset_graph.has_non_source_parents(asset_key):
                return {asset_key: datetime.datetime.fromtimestamp(record_timestamp, tz=datetime.timezone.utc)}
            else:
                return {}
        data_time_by_key: Dict[AssetKey, Optional[datetime.datetime]] = {}
        for (parent_key, parent_record) in upstream_records_by_key.items():
            for (upstream_key, data_time) in self._calculate_data_time_by_key(asset_key=parent_key, record_id=parent_record.storage_id, record_timestamp=parent_record.event_log_entry.timestamp, record_tags=make_hashable((parent_record.asset_materialization.tags if parent_record.asset_materialization else parent_record.event_log_entry.asset_observation.tags if parent_record.event_log_entry.asset_observation else None) or {}), current_time=current_time).items():
                if data_time is None:
                    data_time_by_key[upstream_key] = None
                else:
                    cur_data_time = data_time_by_key.get(upstream_key, data_time)
                    data_time_by_key[upstream_key] = min(cur_data_time, data_time) if cur_data_time is not None else None
        return data_time_by_key

    @cached_method
    def _calculate_data_time_by_key_observable_source(self, *, asset_key: AssetKey, record_id: int, record_tags: Tuple[Tuple[str, str]], current_time: datetime.datetime) -> Mapping[AssetKey, Optional[datetime.datetime]]:
        if False:
            i = 10
            return i + 15
        data_version_value = dict(record_tags).get(DATA_VERSION_TAG)
        if data_version_value is None:
            return {asset_key: None}
        data_version = DataVersion(data_version_value)
        next_version_record = self._instance_queryer.next_version_record(asset_key=asset_key, data_version=data_version, after_cursor=record_id)
        if next_version_record is None:
            return {asset_key: current_time}
        next_version_timestamp = next_version_record.event_log_entry.timestamp
        return {asset_key: datetime.datetime.fromtimestamp(next_version_timestamp, tz=datetime.timezone.utc)}

    @cached_method
    def _calculate_data_time_by_key(self, *, asset_key: AssetKey, record_id: Optional[int], record_timestamp: Optional[float], record_tags: Tuple[Tuple[str, str]], current_time: datetime.datetime) -> Mapping[AssetKey, Optional[datetime.datetime]]:
        if False:
            print('Hello World!')
        if record_id is None:
            return {key: None for key in self.asset_graph.get_non_source_roots(asset_key)}
        record_timestamp = check.not_none(record_timestamp)
        partitions_def = self.asset_graph.get_partitions_def(asset_key)
        if isinstance(partitions_def, TimeWindowPartitionsDefinition):
            return self._calculate_data_time_by_key_time_partitioned(asset_key=asset_key, cursor=record_id, partitions_def=partitions_def)
        elif self.asset_graph.is_observable(asset_key):
            return self._calculate_data_time_by_key_observable_source(asset_key=asset_key, record_id=record_id, record_tags=record_tags, current_time=current_time)
        else:
            return self._calculate_data_time_by_key_unpartitioned(asset_key=asset_key, record_id=record_id, record_timestamp=record_timestamp, record_tags=record_tags, current_time=current_time)

    @cached_method
    def _get_in_progress_run_ids(self, current_time: datetime.datetime) -> Sequence[str]:
        if False:
            return 10
        return [record.dagster_run.run_id for record in self.instance_queryer.instance.get_run_records(filters=RunsFilter(statuses=[status for status in DagsterRunStatus if status not in FINISHED_STATUSES], created_after=current_time - datetime.timedelta(days=1)), limit=25)]

    @cached_method
    def _get_in_progress_data_time_in_run(self, *, run_id: str, asset_key: AssetKey, current_time: datetime.datetime) -> Optional[datetime.datetime]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the upstream data times that a given asset key will be expected to have at the\n        completion of the given run.\n        '
        planned_keys = self._instance_queryer.get_planned_materializations_for_run(run_id=run_id)
        materialized_keys = self._instance_queryer.get_current_materializations_for_run(run_id=run_id)
        if asset_key not in planned_keys or asset_key in materialized_keys:
            return self.get_current_data_time(asset_key, current_time=current_time)
        if not self.asset_graph.has_non_source_parents(asset_key):
            return current_time
        data_time = current_time
        for parent_key in self.asset_graph.get_parents(asset_key):
            if parent_key not in self.asset_graph.materializable_asset_keys:
                continue
            parent_data_time = self._get_in_progress_data_time_in_run(run_id=run_id, asset_key=parent_key, current_time=current_time)
            if parent_data_time is None:
                return None
            data_time = min(data_time, parent_data_time)
        return data_time

    def get_in_progress_data_time(self, asset_key: AssetKey, current_time: datetime.datetime) -> Optional[datetime.datetime]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a mapping containing the maximum upstream data time that the input asset will\n        have once all in-progress runs complete.\n        '
        data_time: Optional[datetime.datetime] = None
        for run_id in self._get_in_progress_run_ids(current_time=current_time):
            if not self._instance_queryer.is_asset_planned_for_run(run_id=run_id, asset=asset_key):
                continue
            run_data_time = self._get_in_progress_data_time_in_run(run_id=run_id, asset_key=asset_key, current_time=current_time)
            if run_data_time is not None:
                data_time = max(run_data_time, data_time or run_data_time)
        return data_time

    def get_ignored_failure_data_time(self, asset_key: AssetKey, current_time: datetime.datetime) -> Optional[datetime.datetime]:
        if False:
            print('Hello World!')
        'Returns the data time that this asset would have if the most recent run successfully\n        completed. If the most recent run did not fail, then this will return the current data time\n        for this asset.\n        '
        current_data_time = self.get_current_data_time(asset_key, current_time=current_time)
        asset_record = self._instance_queryer.get_asset_record(asset_key)
        if asset_record is None or asset_record.asset_entry.last_run_id is None:
            return current_data_time
        run_id = asset_record.asset_entry.last_run_id
        latest_run_record = self._instance_queryer._get_run_record_by_id(run_id=run_id)
        if latest_run_record is None or latest_run_record.dagster_run.status != DagsterRunStatus.FAILURE:
            return current_data_time
        latest_materialization = asset_record.asset_entry.last_materialization
        if latest_materialization is not None and latest_materialization.run_id == latest_run_record.dagster_run.run_id:
            return current_data_time
        run_failure_time = datetime.datetime.utcfromtimestamp(latest_run_record.end_time or datetime_as_float(latest_run_record.create_timestamp)).replace(tzinfo=datetime.timezone.utc)
        return self._get_in_progress_data_time_in_run(run_id=run_id, asset_key=asset_key, current_time=run_failure_time)

    def get_data_time_by_key_for_record(self, record: EventLogRecord, current_time: Optional[datetime.datetime]=None) -> Mapping[AssetKey, Optional[datetime.datetime]]:
        if False:
            while True:
                i = 10
        'Method to enable calculating the timestamps of materializations or observations of\n        upstream assets which were relevant to a given AssetMaterialization. These timestamps can\n        be calculated relative to any upstream asset keys.\n\n        The heart of this functionality is a recursive method which takes a given asset materialization\n        and finds the most recent materialization of each of its parents which happened *before* that\n        given materialization event.\n        '
        event = record.asset_materialization or record.asset_observation
        if record.asset_key is None or event is None:
            raise DagsterInvariantViolationError('Can only calculate data times for records with a materialization / observation event and an asset_key.')
        return self._calculate_data_time_by_key(asset_key=record.asset_key, record_id=record.storage_id, record_timestamp=record.event_log_entry.timestamp, record_tags=make_hashable(event.tags or {}), current_time=current_time or pendulum.now('UTC'))

    def get_current_data_time(self, asset_key: AssetKey, current_time: datetime.datetime) -> Optional[datetime.datetime]:
        if False:
            return 10
        latest_record = self.instance_queryer.get_latest_materialization_or_observation_record(AssetKeyPartitionKey(asset_key))
        if latest_record is None:
            return None
        data_times = set(self.get_data_time_by_key_for_record(latest_record, current_time).values())
        if None in data_times or not data_times:
            return None
        return min(cast(AbstractSet[datetime.datetime], data_times), default=None)

    def get_minutes_overdue(self, asset_key: AssetKey, evaluation_time: datetime.datetime) -> Optional[FreshnessMinutes]:
        if False:
            for i in range(10):
                print('nop')
        freshness_policy = self.asset_graph.freshness_policies_by_key.get(asset_key)
        if freshness_policy is None:
            raise DagsterInvariantViolationError('Cannot calculate minutes late for asset without a FreshnessPolicy')
        return freshness_policy.minutes_overdue(data_time=self.get_current_data_time(asset_key, current_time=evaluation_time), evaluation_time=evaluation_time)