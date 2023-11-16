import logging
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, AbstractSet, Callable, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Set, Tuple, Union, cast
import pendulum
import dagster._check as check
from dagster._core.definitions.asset_graph import AssetGraph
from dagster._core.definitions.asset_graph_subset import AssetGraphSubset
from dagster._core.definitions.data_version import DATA_VERSION_TAG, DataVersion, extract_data_version_from_entry
from dagster._core.definitions.events import AssetKey, AssetKeyPartitionKey
from dagster._core.definitions.partition import PartitionsSubset
from dagster._core.definitions.time_window_partitions import TimeWindowPartitionsDefinition, get_time_partition_key, get_time_partitions_def
from dagster._core.errors import DagsterDefinitionChangedDeserializationError, DagsterInvalidDefinitionError
from dagster._core.events import DagsterEventType
from dagster._core.instance import DagsterInstance, DynamicPartitionsStore
from dagster._core.storage.dagster_run import DagsterRun, RunRecord
from dagster._core.storage.tags import PARTITION_NAME_TAG
from dagster._utils.cached_method import cached_method
if TYPE_CHECKING:
    from dagster._core.storage.event_log import EventLogRecord
    from dagster._core.storage.event_log.base import AssetRecord

class CachingInstanceQueryer(DynamicPartitionsStore):
    """Provides utility functions for querying for asset-materialization related data from the
    instance which will attempt to limit redundant expensive calls. Intended for use within the
    scope of a single "request" (e.g. GQL request, sensor tick).

    Args:
        instance (DagsterInstance): The instance to query.
    """

    def __init__(self, instance: DagsterInstance, asset_graph: AssetGraph, evaluation_time: Optional[datetime]=None, logger: Optional[logging.Logger]=None):
        if False:
            for i in range(10):
                print('nop')
        self._instance = instance
        self._asset_graph = asset_graph
        self._logger = logger or logging.getLogger('dagster')
        self._asset_record_cache: Dict[AssetKey, Optional[AssetRecord]] = {}
        self._asset_partitions_cache: Dict[Optional[int], Dict[AssetKey, Set[str]]] = defaultdict(dict)
        self._asset_partition_versions_updated_after_cursor_cache: Dict[AssetKeyPartitionKey, int] = {}
        self._dynamic_partitions_cache: Dict[str, Sequence[str]] = {}
        self._evaluation_time = evaluation_time if evaluation_time else pendulum.now('UTC')
        self._respect_materialization_data_versions = self._instance.auto_materialize_respect_materialization_data_versions

    @property
    def instance(self) -> DagsterInstance:
        if False:
            i = 10
            return i + 15
        return self._instance

    @property
    def asset_graph(self) -> AssetGraph:
        if False:
            return 10
        return self._asset_graph

    @property
    def evaluation_time(self) -> datetime:
        if False:
            print('Hello World!')
        return self._evaluation_time

    def prefetch_asset_records(self, asset_keys: Iterable[AssetKey]):
        if False:
            while True:
                i = 10
        'For performance, batches together queries for selected assets.'
        keys_to_fetch = set(asset_keys) - set(self._asset_record_cache.keys())
        if len(keys_to_fetch) == 0:
            return
        asset_records = self.instance.get_asset_records(list(keys_to_fetch))
        for asset_record in asset_records:
            self._asset_record_cache[asset_record.asset_entry.asset_key] = asset_record
        for key in asset_keys:
            if key not in self._asset_record_cache:
                self._asset_record_cache[key] = None

    @cached_method
    def get_failed_or_in_progress_subset(self, *, asset_key: AssetKey) -> PartitionsSubset:
        if False:
            i = 10
            return i + 15
        'Returns a PartitionsSubset representing the set of partitions that are either in progress\n        or whose last materialization attempt failed.\n        '
        from dagster._core.storage.partition_status_cache import get_and_update_asset_status_cache_value
        partitions_def = check.not_none(self.asset_graph.get_partitions_def(asset_key))
        asset_record = self.get_asset_record(asset_key)
        cache_value = get_and_update_asset_status_cache_value(instance=self.instance, asset_key=asset_key, partitions_def=partitions_def, dynamic_partitions_loader=self, asset_record=asset_record)
        if cache_value is None:
            return partitions_def.empty_subset()
        return cache_value.deserialize_failed_partition_subsets(partitions_def) | cache_value.deserialize_in_progress_partition_subsets(partitions_def)

    def has_cached_asset_record(self, asset_key: AssetKey) -> bool:
        if False:
            print('Hello World!')
        return asset_key in self._asset_record_cache

    def get_asset_record(self, asset_key: AssetKey) -> Optional['AssetRecord']:
        if False:
            print('Hello World!')
        if asset_key not in self._asset_record_cache:
            self._asset_record_cache[asset_key] = next(iter(self.instance.get_asset_records([asset_key])), None)
        return self._asset_record_cache[asset_key]

    def _event_type_for_key(self, asset_key: AssetKey) -> DagsterEventType:
        if False:
            return 10
        if self.asset_graph.is_source(asset_key):
            return DagsterEventType.ASSET_OBSERVATION
        else:
            return DagsterEventType.ASSET_MATERIALIZATION

    @cached_method
    def _get_latest_materialization_or_observation_record(self, *, asset_partition: AssetKeyPartitionKey, before_cursor: Optional[int]=None) -> Optional['EventLogRecord']:
        if False:
            while True:
                i = 10
        'Returns the latest event log record for the given asset partition of an asset. For\n        observable source assets, this will be an AssetObservation, otherwise it will be an\n        AssetMaterialization.\n        '
        from dagster._core.event_api import EventRecordsFilter
        if before_cursor is None and asset_partition.partition_key is None and (not self.asset_graph.is_observable(asset_partition.asset_key)):
            asset_record = self.get_asset_record(asset_partition.asset_key)
            if asset_record is None:
                return None
            return asset_record.asset_entry.last_materialization_record
        records = self.instance.get_event_records(EventRecordsFilter(event_type=self._event_type_for_key(asset_partition.asset_key), asset_key=asset_partition.asset_key, asset_partitions=[asset_partition.partition_key] if asset_partition.partition_key else None, before_cursor=before_cursor), ascending=False, limit=1)
        return next(iter(records), None)

    @cached_method
    def _get_latest_materialization_or_observation_storage_ids_by_asset_partition(self, *, asset_key: AssetKey) -> Mapping[AssetKeyPartitionKey, Optional[int]]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a mapping from asset partition to the latest storage id for that asset partition\n        for all asset partitions associated with the given asset key.\n\n        Note that for partitioned assets, an asset partition with a None partition key will be\n        present in the mapping, representing the latest storage id for the asset as a whole.\n        '
        asset_partition = AssetKeyPartitionKey(asset_key)
        latest_record = self._get_latest_materialization_or_observation_record(asset_partition=asset_partition)
        latest_storage_ids = {asset_partition: latest_record.storage_id if latest_record is not None else None}
        if self.asset_graph.is_partitioned(asset_key):
            latest_storage_ids.update({AssetKeyPartitionKey(asset_key, partition_key): storage_id for (partition_key, storage_id) in self.instance.get_latest_storage_id_by_partition(asset_key, event_type=self._event_type_for_key(asset_key)).items()})
        return latest_storage_ids

    def get_latest_materialization_or_observation_storage_id(self, asset_partition: AssetKeyPartitionKey) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        'Returns the latest storage id for the given asset partition. If the asset has never been\n        materialized, returns None.\n\n        Args:\n            asset_partition (AssetKeyPartitionKey): The asset partition to query.\n        '
        return self._get_latest_materialization_or_observation_storage_ids_by_asset_partition(asset_key=asset_partition.asset_key).get(asset_partition)

    def asset_partition_has_materialization_or_observation(self, asset_partition: AssetKeyPartitionKey, after_cursor: Optional[int]=None) -> bool:
        if False:
            print('Hello World!')
        'Returns True if there is a materialization record for the given asset partition after\n        the specified cursor.\n\n        Args:\n            asset_partition (AssetKeyPartitionKey): The asset partition to query.\n            after_cursor (Optional[int]): Filter parameter such that only records with a storage_id\n                greater than this value will be considered.\n        '
        if not self.asset_graph.is_source(asset_partition.asset_key):
            asset_record = self.get_asset_record(asset_partition.asset_key)
            if asset_record is None or asset_record.asset_entry.last_materialization_record is None or (after_cursor and asset_record.asset_entry.last_materialization_record.storage_id <= after_cursor):
                return False
        return (self.get_latest_materialization_or_observation_storage_id(asset_partition) or 0) > (after_cursor or 0)

    def get_latest_materialization_or_observation_record(self, asset_partition: AssetKeyPartitionKey, after_cursor: Optional[int]=None, before_cursor: Optional[int]=None) -> Optional['EventLogRecord']:
        if False:
            return 10
        'Returns the latest record for the given asset partition given the specified cursors.\n\n        Args:\n            asset_partition (AssetKeyPartitionKey): The asset partition to query.\n            after_cursor (Optional[int]): Filter parameter such that only records with a storage_id\n                greater than this value will be considered.\n            before_cursor (Optional[int]): Filter parameter such that only records with a storage_id\n                less than this value will be considered.\n        '
        check.param_invariant(not (after_cursor and before_cursor), 'before_cursor', 'Cannot set both before_cursor and after_cursor')
        if not self.asset_partition_has_materialization_or_observation(asset_partition, after_cursor):
            return None
        elif (before_cursor or 0) > (self.get_latest_materialization_or_observation_storage_id(asset_partition) or 0):
            return self._get_latest_materialization_or_observation_record(asset_partition=asset_partition)
        return self._get_latest_materialization_or_observation_record(asset_partition=asset_partition, before_cursor=before_cursor)

    @cached_method
    def next_version_record(self, *, asset_key: AssetKey, after_cursor: Optional[int], data_version: Optional[DataVersion]) -> Optional['EventLogRecord']:
        if False:
            return 10
        from dagster._core.event_api import EventRecordsFilter
        for record in self.instance.get_event_records(EventRecordsFilter(event_type=DagsterEventType.ASSET_OBSERVATION, asset_key=asset_key, after_cursor=after_cursor), ascending=True):
            record_version = extract_data_version_from_entry(record.event_log_entry)
            if record_version is not None and record_version != data_version:
                return record
        return None

    @cached_method
    def _get_run_record_by_id(self, *, run_id: str) -> Optional[RunRecord]:
        if False:
            print('Hello World!')
        return self.instance.get_run_record_by_id(run_id)

    def _get_run_by_id(self, run_id: str) -> Optional[DagsterRun]:
        if False:
            return 10
        run_record = self._get_run_record_by_id(run_id=run_id)
        if run_record is not None:
            return run_record.dagster_run
        return None

    def run_has_tag(self, run_id: str, tag_key: str, tag_value: Optional[str]) -> bool:
        if False:
            i = 10
            return i + 15
        run_tags = cast(DagsterRun, self._get_run_by_id(run_id)).tags
        if tag_value is None:
            return tag_key in run_tags
        else:
            return run_tags.get(tag_key) == tag_value

    @cached_method
    def _get_planned_materializations_for_run_from_events(self, *, run_id: str) -> AbstractSet[AssetKey]:
        if False:
            for i in range(10):
                print('nop')
        'Provides a fallback for fetching the planned materializations for a run from\n        the ASSET_MATERIALIZATION_PLANNED events in the event log, in cases where this information\n        is not available on the DagsterRun object.\n\n        Args:\n            run_id (str): The run id\n        '
        materializations_planned = self.instance.get_records_for_run(run_id=run_id, of_type=DagsterEventType.ASSET_MATERIALIZATION_PLANNED).records
        return set((cast(AssetKey, record.asset_key) for record in materializations_planned))

    def get_planned_materializations_for_run(self, run_id: str) -> AbstractSet[AssetKey]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the set of asset keys that are planned to be materialized by the run.\n\n        Args:\n            run_id (str): The run id\n        '
        run = self._get_run_by_id(run_id=run_id)
        if run is None:
            return set()
        elif run.asset_selection:
            return run.asset_selection
        else:
            return self._get_planned_materializations_for_run_from_events(run_id=run_id)

    def is_asset_planned_for_run(self, run_id: str, asset: Union[AssetKey, AssetKeyPartitionKey]) -> bool:
        if False:
            return 10
        'Returns True if the asset is planned to be materialized by the run.'
        run = self._get_run_by_id(run_id=run_id)
        if not run:
            return False
        if isinstance(asset, AssetKeyPartitionKey):
            asset_key = asset.asset_key
            if run.tags.get(PARTITION_NAME_TAG) != asset.partition_key:
                return False
        else:
            asset_key = asset
        return asset_key in self.get_planned_materializations_for_run(run_id=run_id)

    @cached_method
    def get_current_materializations_for_run(self, *, run_id: str) -> AbstractSet[AssetKey]:
        if False:
            print('Hello World!')
        'Returns the set of asset keys that have been materialized by a given run.\n\n        Args:\n            run_id (str): The run id\n        '
        materializations = self.instance.get_records_for_run(run_id=run_id, of_type=DagsterEventType.ASSET_MATERIALIZATION).records
        return set((cast(AssetKey, record.asset_key) for record in materializations))

    @cached_method
    def get_active_backfill_target_asset_graph_subset(self) -> AssetGraphSubset:
        if False:
            for i in range(10):
                print('nop')
        'Returns an AssetGraphSubset representing the set of assets that are currently targeted by\n        an active asset backfill.\n        '
        from dagster._core.execution.asset_backfill import AssetBackfillData
        from dagster._core.execution.backfill import BulkActionStatus
        asset_backfills = [backfill for backfill in self.instance.get_backfills(status=BulkActionStatus.REQUESTED) if backfill.is_asset_backfill]
        result = AssetGraphSubset(self.asset_graph)
        for asset_backfill in asset_backfills:
            if asset_backfill.serialized_asset_backfill_data is None:
                check.failed('Asset backfill missing serialized_asset_backfill_data')
            try:
                asset_backfill_data = AssetBackfillData.from_serialized(asset_backfill.serialized_asset_backfill_data, self.asset_graph, asset_backfill.backfill_timestamp)
            except DagsterDefinitionChangedDeserializationError:
                self._logger.warning(f'Not considering assets in backfill {asset_backfill.backfill_id} since its data could not be deserialized')
                continue
            result |= asset_backfill_data.target_subset
        return result

    def get_materialized_partitions(self, asset_key: AssetKey, before_cursor: Optional[int]=None) -> Set[str]:
        if False:
            i = 10
            return i + 15
        'Returns a list of the partitions that have been materialized for the given asset key.\n\n        Args:\n            asset_key (AssetKey): The asset key.\n            before_cursor (Optional[int]): The cursor before which to look for materialized\n                partitions. If not provided, will look at all materializations.\n        '
        if before_cursor not in self._asset_partitions_cache or asset_key not in self._asset_partitions_cache[before_cursor]:
            self._asset_partitions_cache[before_cursor][asset_key] = self.instance.get_materialized_partitions(asset_key=asset_key, before_cursor=before_cursor)
        return self._asset_partitions_cache[before_cursor][asset_key]

    def get_dynamic_partitions(self, partitions_def_name: str) -> Sequence[str]:
        if False:
            i = 10
            return i + 15
        'Returns a list of partitions for a partitions definition.'
        if partitions_def_name not in self._dynamic_partitions_cache:
            self._dynamic_partitions_cache[partitions_def_name] = self.instance.get_dynamic_partitions(partitions_def_name)
        return self._dynamic_partitions_cache[partitions_def_name]

    def has_dynamic_partition(self, partitions_def_name: str, partition_key: str) -> bool:
        if False:
            i = 10
            return i + 15
        return partition_key in self.get_dynamic_partitions(partitions_def_name)

    def asset_partitions_with_newly_updated_parents_and_new_latest_storage_id(self, latest_storage_id: Optional[int], target_asset_keys: FrozenSet[AssetKey], target_asset_keys_and_parents: FrozenSet[AssetKey], can_reconcile_fn: Callable[[AssetKeyPartitionKey], bool]=lambda _: True, map_old_time_partitions: bool=True) -> Tuple[AbstractSet[AssetKeyPartitionKey], Optional[int]]:
        if False:
            print('Hello World!')
        'Finds asset partitions in the given selection whose parents have been materialized since\n        latest_storage_id.\n\n        Returns:\n            - A set of asset partitions.\n            - The latest observed storage_id across all relevant assets. Can be used to avoid scanning\n                the same events the next time this function is called.\n        '
        result_asset_partitions: Set[AssetKeyPartitionKey] = set()
        result_latest_storage_id = latest_storage_id
        for asset_key in target_asset_keys_and_parents:
            if self.asset_graph.is_source(asset_key) and (not self.asset_graph.is_observable(asset_key)):
                continue
            new_asset_partitions = self.get_asset_partitions_updated_after_cursor(asset_key=asset_key, asset_partitions=None, after_cursor=latest_storage_id, respect_materialization_data_versions=False)
            if not new_asset_partitions:
                continue
            partitions_def = self.asset_graph.get_partitions_def(asset_key)
            if partitions_def is None:
                latest_record = check.not_none(self.get_latest_materialization_or_observation_record(AssetKeyPartitionKey(asset_key)))
                for child in self.asset_graph.get_children_partitions(dynamic_partitions_store=self, current_time=self.evaluation_time, asset_key=asset_key):
                    child_partitions_def = self.asset_graph.get_partitions_def(child.asset_key)
                    child_time_partitions_def = get_time_partitions_def(child_partitions_def)
                    if child.asset_key in target_asset_keys and (not (not map_old_time_partitions and child_time_partitions_def is not None and (get_time_partition_key(child_partitions_def, child.partition_key) != child_time_partitions_def.get_last_partition_key(current_time=self.evaluation_time)))) and (not self.is_asset_planned_for_run(latest_record.run_id, child.asset_key)):
                        result_asset_partitions.add(child)
            else:
                partitions_subset = partitions_def.empty_subset().with_partition_keys([asset_partition.partition_key for asset_partition in new_asset_partitions if asset_partition.partition_key is not None and partitions_def.has_partition_key(asset_partition.partition_key, dynamic_partitions_store=self, current_time=self.evaluation_time)])
                for child in self.asset_graph.get_children(asset_key):
                    child_partitions_def = self.asset_graph.get_partitions_def(child)
                    if child not in target_asset_keys:
                        continue
                    elif not child_partitions_def:
                        result_asset_partitions.add(AssetKeyPartitionKey(child, None))
                    else:
                        partition_mapping = self.asset_graph.get_partition_mapping(child, asset_key)
                        try:
                            child_partitions_subset = partition_mapping.get_downstream_partitions_for_partitions(partitions_subset, downstream_partitions_def=child_partitions_def, dynamic_partitions_store=self, current_time=self.evaluation_time)
                        except DagsterInvalidDefinitionError as e:
                            raise DagsterInvalidDefinitionError(f'Could not map partitions between parent {asset_key.to_string()} and child {child.to_string()}.') from e
                        for child_partition in child_partitions_subset.get_partition_keys():
                            child_asset_partition = AssetKeyPartitionKey(child, child_partition)
                            if not can_reconcile_fn(child_asset_partition):
                                continue
                            elif child_partitions_def != partitions_def or child_partition not in partitions_subset or child_partition not in self.get_failed_or_in_progress_subset(asset_key=child):
                                result_asset_partitions.add(child_asset_partition)
                            else:
                                latest_partition_record = check.not_none(self.get_latest_materialization_or_observation_record(AssetKeyPartitionKey(asset_key, child_partition), after_cursor=latest_storage_id))
                                if not self.is_asset_planned_for_run(latest_partition_record.run_id, child):
                                    result_asset_partitions.add(child_asset_partition)
            asset_latest_storage_id = self.get_latest_materialization_or_observation_storage_id(AssetKeyPartitionKey(asset_key))
            if result_latest_storage_id is None or (asset_latest_storage_id or 0) > result_latest_storage_id:
                result_latest_storage_id = asset_latest_storage_id
        return (result_asset_partitions, result_latest_storage_id)

    def _asset_partitions_data_versions(self, asset_key: AssetKey, asset_partitions: Optional[AbstractSet[AssetKeyPartitionKey]], after_cursor: Optional[int]=None, before_cursor: Optional[int]=None) -> Mapping[AssetKeyPartitionKey, Optional[DataVersion]]:
        if False:
            i = 10
            return i + 15
        if not self.asset_graph.is_partitioned(asset_key):
            asset_partition = AssetKeyPartitionKey(asset_key)
            latest_record = self.get_latest_materialization_or_observation_record(asset_partition, after_cursor=after_cursor, before_cursor=before_cursor)
            return {asset_partition: extract_data_version_from_entry(latest_record.event_log_entry)} if latest_record is not None else {}
        else:
            query_result = self.instance._event_storage.get_latest_tags_by_partition(asset_key, event_type=self._event_type_for_key(asset_key), tag_keys=[DATA_VERSION_TAG], after_cursor=after_cursor, before_cursor=before_cursor, asset_partitions=[asset_partition.partition_key for asset_partition in asset_partitions if asset_partition.partition_key is not None] if asset_partitions is not None else None)
            return {AssetKeyPartitionKey(asset_key, partition_key): DataVersion(tags[DATA_VERSION_TAG]) if tags.get(DATA_VERSION_TAG) else None for (partition_key, tags) in query_result.items()}

    def _asset_partition_versions_updated_after_cursor(self, asset_key: AssetKey, asset_partitions: AbstractSet[AssetKeyPartitionKey], after_cursor: int) -> AbstractSet[AssetKeyPartitionKey]:
        if False:
            i = 10
            return i + 15
        updated_asset_partitions = {ap for ap in asset_partitions if ap in self._asset_partition_versions_updated_after_cursor_cache and self._asset_partition_versions_updated_after_cursor_cache[ap] <= after_cursor}
        to_query_asset_partitions = asset_partitions - updated_asset_partitions
        if not to_query_asset_partitions:
            return updated_asset_partitions
        latest_versions = self._asset_partitions_data_versions(asset_key, to_query_asset_partitions, after_cursor=after_cursor)
        previous_versions = self._asset_partitions_data_versions(asset_key, to_query_asset_partitions, before_cursor=after_cursor + 1)
        queryed_updated_asset_partitions = {ap for (ap, version) in latest_versions.items() if previous_versions.get(ap) != version}
        for asset_partition in queryed_updated_asset_partitions:
            self._asset_partition_versions_updated_after_cursor_cache[asset_partition] = after_cursor
        return {*updated_asset_partitions, *queryed_updated_asset_partitions}

    def get_asset_partitions_updated_after_cursor(self, asset_key: AssetKey, asset_partitions: Optional[AbstractSet[AssetKeyPartitionKey]], after_cursor: Optional[int], respect_materialization_data_versions: bool) -> AbstractSet[AssetKeyPartitionKey]:
        if False:
            return 10
        'Returns the set of asset partitions that have been updated after the given cursor.\n\n        Args:\n            asset_key (AssetKey): The asset key to check.\n            asset_partitions (Optional[Sequence[AssetKeyPartitionKey]]): If supplied, will filter\n                the set of checked partitions to the given partitions.\n            after_cursor (Optional[int]): The cursor after which to look for updates.\n            respect_materialization_data_versions (bool): If True, will use data versions to filter\n                out asset partitions which were materialized, but not have not had their data\n                versions changed since the given cursor.\n                NOTE: This boolean has been temporarily disabled\n        '
        if not self.asset_partition_has_materialization_or_observation(AssetKeyPartitionKey(asset_key), after_cursor=after_cursor):
            return set()
        last_storage_id_by_asset_partition = self._get_latest_materialization_or_observation_storage_ids_by_asset_partition(asset_key=asset_key)
        if asset_partitions is None:
            updated_after_cursor = {asset_partition for (asset_partition, latest_storage_id) in last_storage_id_by_asset_partition.items() if (latest_storage_id or 0) > (after_cursor or 0)}
        else:
            updated_after_cursor = set()
            for asset_partition in asset_partitions:
                latest_storage_id = last_storage_id_by_asset_partition.get(asset_partition)
                if latest_storage_id is not None and latest_storage_id > (after_cursor or 0):
                    updated_after_cursor.add(asset_partition)
        if not updated_after_cursor:
            return set()
        if after_cursor is None or (not self.asset_graph.is_source(asset_key) and (not respect_materialization_data_versions)):
            return updated_after_cursor
        return self._asset_partition_versions_updated_after_cursor(asset_key, updated_after_cursor, after_cursor)

    def get_parent_asset_partitions_updated_after_child(self, asset_partition: AssetKeyPartitionKey, parent_asset_partitions: AbstractSet[AssetKeyPartitionKey], respect_materialization_data_versions: bool, ignored_parent_keys: AbstractSet[AssetKey]) -> AbstractSet[AssetKeyPartitionKey]:
        if False:
            print('Hello World!')
        'Returns values inside parent_asset_partitions that correspond to asset partitions that\n        have been updated since the latest materialization of asset_partition.\n        '
        parent_asset_partitions_by_key: Dict[AssetKey, Set[AssetKeyPartitionKey]] = defaultdict(set)
        for parent in parent_asset_partitions:
            parent_asset_partitions_by_key[parent.asset_key].add(parent)
        partitions_def = self.asset_graph.get_partitions_def(asset_partition.asset_key)
        updated_parents = set()
        for (parent_key, parent_asset_partitions) in parent_asset_partitions_by_key.items():
            if parent_key in ignored_parent_keys:
                continue
            if self.asset_graph.is_source(parent_key) and (not self.asset_graph.is_observable(parent_key)):
                continue
            if isinstance(partitions_def, TimeWindowPartitionsDefinition) and (not self.asset_graph.is_partitioned(parent_key)) and (asset_partition.partition_key != partitions_def.get_last_partition_key(current_time=self.evaluation_time, dynamic_partitions_store=self)):
                continue
            updated_parents.update(self.get_asset_partitions_updated_after_cursor(asset_key=parent_key, asset_partitions=parent_asset_partitions, after_cursor=self.get_latest_materialization_or_observation_storage_id(asset_partition), respect_materialization_data_versions=respect_materialization_data_versions))
        return updated_parents

    def have_ignorable_partition_mapping_for_outdated(self, asset_key: AssetKey, upstream_asset_key: AssetKey) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "Returns whether the given assets have a partition mapping between them which can be\n        ignored in the context of calculating if an asset is outdated or not.\n\n        These mappings are ignored in cases where respecting them would require an unrealistic\n        number of upstream partitions to be in a 'good' state before allowing a downstream asset\n        to be considered up to date.\n        "
        return asset_key == upstream_asset_key

    @cached_method
    def get_outdated_ancestors(self, *, asset_partition: AssetKeyPartitionKey) -> AbstractSet[AssetKey]:
        if False:
            while True:
                i = 10
        if self.asset_graph.is_source(asset_partition.asset_key):
            return set()
        parent_asset_partitions = self.asset_graph.get_parents_partitions(dynamic_partitions_store=self, current_time=self._evaluation_time, asset_key=asset_partition.asset_key, partition_key=asset_partition.partition_key).parent_partitions
        ignored_parent_keys = {parent for parent in self.asset_graph.get_parents(asset_partition.asset_key) if self.have_ignorable_partition_mapping_for_outdated(asset_partition.asset_key, parent)}
        updated_parents = self.get_parent_asset_partitions_updated_after_child(asset_partition=asset_partition, parent_asset_partitions=parent_asset_partitions, respect_materialization_data_versions=self._respect_materialization_data_versions, ignored_parent_keys=ignored_parent_keys)
        root_unreconciled_ancestors = {asset_partition.asset_key} if updated_parents else set()
        for parent in set(parent_asset_partitions) - updated_parents:
            if parent.asset_key in ignored_parent_keys:
                continue
            root_unreconciled_ancestors.update(self.get_outdated_ancestors(asset_partition=parent))
        return root_unreconciled_ancestors