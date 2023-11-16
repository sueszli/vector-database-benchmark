import dataclasses
import datetime
import itertools
import logging
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, AbstractSet, Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, cast
import pendulum
import dagster._check as check
from dagster._core.definitions.auto_materialize_policy import AutoMaterializePolicy
from dagster._core.definitions.data_time import CachingDataTimeResolver
from dagster._core.definitions.events import AssetKey, AssetKeyPartitionKey
from dagster._core.definitions.run_request import RunRequest
from dagster._core.definitions.time_window_partitions import get_time_partitions_def
from dagster._core.instance import DynamicPartitionsStore
from dagster._utils.cached_method import cached_method
from ... import PartitionKeyRange
from ..storage.tags import ASSET_PARTITION_RANGE_END_TAG, ASSET_PARTITION_RANGE_START_TAG
from .asset_daemon_cursor import AssetDaemonCursor
from .asset_graph import AssetGraph
from .auto_materialize_rule import AutoMaterializeRule, DiscardOnMaxMaterializationsExceededRule, RuleEvaluationContext
from .auto_materialize_rule_evaluation import AutoMaterializeAssetEvaluation, AutoMaterializeRuleEvaluation
from .backfill_policy import BackfillPolicy, BackfillPolicyType
from .freshness_based_auto_materialize import get_expected_data_time_for_asset_key
from .partition import PartitionsDefinition, ScheduleType
if TYPE_CHECKING:
    from dagster._core.instance import DagsterInstance
    from dagster._utils.caching_instance_queryer import CachingInstanceQueryer

def get_implicit_auto_materialize_policy(asset_key: AssetKey, asset_graph: AssetGraph) -> Optional[AutoMaterializePolicy]:
    if False:
        return 10
    'For backcompat with pre-auto materialize policy graphs, assume a default scope of 1 day.'
    auto_materialize_policy = asset_graph.get_auto_materialize_policy(asset_key)
    if auto_materialize_policy is None:
        time_partitions_def = get_time_partitions_def(asset_graph.get_partitions_def(asset_key))
        if time_partitions_def is None:
            max_materializations_per_minute = None
        elif time_partitions_def.schedule_type == ScheduleType.HOURLY:
            max_materializations_per_minute = 24
        else:
            max_materializations_per_minute = 1
        rules = {AutoMaterializeRule.materialize_on_missing(), AutoMaterializeRule.materialize_on_required_for_freshness(), AutoMaterializeRule.skip_on_parent_outdated(), AutoMaterializeRule.skip_on_parent_missing(), AutoMaterializeRule.skip_on_required_but_nonexistent_parents(), AutoMaterializeRule.skip_on_backfill_in_progress()}
        if not bool(asset_graph.get_downstream_freshness_policies(asset_key=asset_key)):
            rules.add(AutoMaterializeRule.materialize_on_parent_updated())
        return AutoMaterializePolicy(rules=rules, max_materializations_per_minute=max_materializations_per_minute)
    return auto_materialize_policy

class AssetDaemonContext:

    def __init__(self, evaluation_id: int, instance: 'DagsterInstance', asset_graph: AssetGraph, cursor: AssetDaemonCursor, materialize_run_tags: Optional[Mapping[str, str]], observe_run_tags: Optional[Mapping[str, str]], auto_observe: bool, target_asset_keys: Optional[AbstractSet[AssetKey]], respect_materialization_data_versions: bool, logger: logging.Logger, evaluation_time: Optional[datetime.datetime]=None):
        if False:
            print('Hello World!')
        from dagster._utils.caching_instance_queryer import CachingInstanceQueryer
        self._evaluation_id = evaluation_id
        self._instance_queryer = CachingInstanceQueryer(instance, asset_graph, evaluation_time=evaluation_time, logger=logger)
        self._data_time_resolver = CachingDataTimeResolver(self.instance_queryer)
        self._cursor = cursor
        self._target_asset_keys = target_asset_keys or {key for (key, policy) in self.asset_graph.auto_materialize_policies_by_key.items() if policy is not None}
        self._materialize_run_tags = materialize_run_tags
        self._observe_run_tags = observe_run_tags
        self._auto_observe = auto_observe
        self._respect_materialization_data_versions = respect_materialization_data_versions
        self._logger = logger
        self.instance_queryer.prefetch_asset_records([key for key in self.target_asset_keys_and_parents if not self.asset_graph.is_source(key)])
        self._verbose_log_fn = self._logger.info if os.getenv('ASSET_DAEMON_VERBOSE_LOGS') else self._logger.debug

    @property
    def instance_queryer(self) -> 'CachingInstanceQueryer':
        if False:
            for i in range(10):
                print('nop')
        return self._instance_queryer

    @property
    def data_time_resolver(self) -> CachingDataTimeResolver:
        if False:
            i = 10
            return i + 15
        return self._data_time_resolver

    @property
    def cursor(self) -> AssetDaemonCursor:
        if False:
            i = 10
            return i + 15
        return self._cursor

    @property
    def asset_graph(self) -> AssetGraph:
        if False:
            while True:
                i = 10
        return self.instance_queryer.asset_graph

    @property
    def latest_storage_id(self) -> Optional[int]:
        if False:
            print('Hello World!')
        return self.cursor.latest_storage_id

    @property
    def target_asset_keys(self) -> AbstractSet[AssetKey]:
        if False:
            while True:
                i = 10
        return self._target_asset_keys

    @property
    def target_asset_keys_and_parents(self) -> AbstractSet[AssetKey]:
        if False:
            for i in range(10):
                print('nop')
        return {parent for asset_key in self.target_asset_keys for parent in self.asset_graph.get_parents(asset_key)} | self.target_asset_keys

    @property
    def respect_materialization_data_versions(self) -> bool:
        if False:
            return 10
        return self._respect_materialization_data_versions

    @cached_method
    def _get_never_handled_and_newly_handled_root_asset_partitions(self) -> Tuple[Mapping[AssetKey, AbstractSet[AssetKeyPartitionKey]], AbstractSet[AssetKey], Mapping[AssetKey, AbstractSet[str]]]:
        if False:
            print('Hello World!')
        'Finds asset partitions that have never been materialized or requested and that have no\n        parents.\n\n        Returns:\n        - Asset (partition)s that have never been materialized or requested.\n        - Non-partitioned assets that had never been materialized or requested up to the previous cursor\n            but are now materialized.\n        - Asset (partition)s that had never been materialized or requested up to the previous cursor but\n            are now materialized.\n        '
        never_handled = defaultdict(set)
        newly_materialized_root_asset_keys = set()
        newly_materialized_root_partitions_by_asset_key = defaultdict(set)
        for asset_key in self.target_asset_keys & self.asset_graph.root_materializable_or_observable_asset_keys:
            if self.asset_graph.is_partitioned(asset_key):
                for partition_key in self.cursor.get_unhandled_partitions(asset_key, self.asset_graph, dynamic_partitions_store=self.instance_queryer, current_time=self.instance_queryer.evaluation_time):
                    asset_partition = AssetKeyPartitionKey(asset_key, partition_key)
                    if self.instance_queryer.asset_partition_has_materialization_or_observation(asset_partition):
                        newly_materialized_root_partitions_by_asset_key[asset_key].add(partition_key)
                    else:
                        never_handled[asset_key].add(asset_partition)
            elif not self.cursor.was_previously_handled(asset_key):
                asset_partition = AssetKeyPartitionKey(asset_key)
                if self.instance_queryer.asset_partition_has_materialization_or_observation(asset_partition):
                    newly_materialized_root_asset_keys.add(asset_key)
                else:
                    never_handled[asset_key].add(asset_partition)
        return (never_handled, newly_materialized_root_asset_keys, newly_materialized_root_partitions_by_asset_key)

    def get_never_handled_root_asset_partitions_for_key(self, asset_key: AssetKey) -> AbstractSet[AssetKeyPartitionKey]:
        if False:
            print('Hello World!')
        'Returns the set of root asset partitions that have never been handled for a given asset\n        key. If the input asset key is not a root asset, this will always be an empty set.\n        '
        (never_handled, _, _) = self._get_never_handled_and_newly_handled_root_asset_partitions()
        return never_handled.get(asset_key, set())

    def get_newly_updated_roots(self) -> Tuple[AbstractSet[AssetKey], Mapping[AssetKey, AbstractSet[str]]]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the set of unpartitioned root asset keys that have been updated since the last\n        tick, and a mapping from partitioned root asset keys to the set of partition keys that have\n        been materialized since the last tick.\n        '
        (_, newly_handled_keys, newly_handled_partitions_by_key) = self._get_never_handled_and_newly_handled_root_asset_partitions()
        return (newly_handled_keys, newly_handled_partitions_by_key)

    @cached_method
    def _get_asset_partitions_with_newly_updated_parents_by_key_and_new_latest_storage_id(self) -> Tuple[Mapping[AssetKey, AbstractSet[AssetKeyPartitionKey]], Optional[int]]:
        if False:
            i = 10
            return i + 15
        'Returns a mapping from asset keys to the set of asset partitions that have newly updated\n        parents, and the new latest storage ID.\n        '
        (asset_partitions, new_latest_storage_id) = self.instance_queryer.asset_partitions_with_newly_updated_parents_and_new_latest_storage_id(latest_storage_id=self.latest_storage_id, target_asset_keys=frozenset(self.target_asset_keys), target_asset_keys_and_parents=frozenset(self.target_asset_keys_and_parents), map_old_time_partitions=False)
        ret = defaultdict(set)
        for asset_partition in asset_partitions:
            ret[asset_partition.asset_key].add(asset_partition)
        return (ret, new_latest_storage_id)

    def get_new_latest_storage_id(self) -> Optional[int]:
        if False:
            return 10
        'Returns the latest storage of all target asset keys since the last tick.'
        (_, new_latest_storage_id) = self._get_asset_partitions_with_newly_updated_parents_by_key_and_new_latest_storage_id()
        return new_latest_storage_id

    def get_asset_partitions_with_newly_updated_parents_for_key(self, asset_key: AssetKey) -> AbstractSet[AssetKeyPartitionKey]:
        if False:
            print('Hello World!')
        'Returns the set of asset partitions whose parents have been updated since the last tick\n        for a given asset key.\n        '
        (updated_parent_mapping, _) = self._get_asset_partitions_with_newly_updated_parents_by_key_and_new_latest_storage_id()
        return updated_parent_mapping.get(asset_key, set())

    def evaluate_asset(self, asset_key: AssetKey, will_materialize_mapping: Mapping[AssetKey, AbstractSet[AssetKeyPartitionKey]], expected_data_time_mapping: Mapping[AssetKey, Optional[datetime.datetime]]) -> Tuple[AutoMaterializeAssetEvaluation, AbstractSet[AssetKeyPartitionKey], AbstractSet[AssetKeyPartitionKey]]:
        if False:
            while True:
                i = 10
        'Evaluates the auto materialize policy of a given asset key.\n\n        Params:\n            - asset_key: The asset key to evaluate.\n            - will_materialize_mapping: A mapping of AssetKey to the set of AssetKeyPartitionKeys\n                that will be materialized this tick. As this function is called in topological order,\n                this mapping will contain the expected materializations of all upstream assets.\n            - expected_data_time_mapping: A mapping of AssetKey to the expected data time of the\n                asset after this tick. As this function is called in topological order, this mapping\n                will contain the expected data times of all upstream assets.\n\n        Returns:\n            - An AutoMaterializeAssetEvaluation object representing serializable information about\n                this evaluation.\n            - The set of AssetKeyPartitionKeys that should be materialized.\n            - The set of AssetKeyPartitionKeys that should be discarded.\n        '
        auto_materialize_policy = check.not_none(self.asset_graph.auto_materialize_policies_by_key.get(asset_key))
        all_results: List[Tuple[AutoMaterializeRuleEvaluation, AbstractSet[AssetKeyPartitionKey]]] = []
        to_materialize: Set[AssetKeyPartitionKey] = set()
        to_skip: Set[AssetKeyPartitionKey] = set()
        to_discard: Set[AssetKeyPartitionKey] = set()
        materialize_context = RuleEvaluationContext(asset_key=asset_key, cursor=self.cursor, instance_queryer=self.instance_queryer, data_time_resolver=self.data_time_resolver, will_materialize_mapping=will_materialize_mapping, expected_data_time_mapping=expected_data_time_mapping, candidates=set(), daemon_context=self)
        for materialize_rule in auto_materialize_policy.materialize_rules:
            rule_snapshot = materialize_rule.to_snapshot()
            self._verbose_log_fn(f'Evaluating materialize rule: {rule_snapshot}')
            for (evaluation_data, asset_partitions) in materialize_rule.evaluate_for_asset(materialize_context):
                all_results.append((AutoMaterializeRuleEvaluation(rule_snapshot=rule_snapshot, evaluation_data=evaluation_data), asset_partitions))
                self._verbose_log_fn(f'Rule returned {len(asset_partitions)} partitions')
                to_materialize.update(asset_partitions)
            self._verbose_log_fn('Done evaluating materialize rule')
        skip_context = dataclasses.replace(materialize_context, candidates=to_materialize)
        for skip_rule in auto_materialize_policy.skip_rules:
            rule_snapshot = skip_rule.to_snapshot()
            self._verbose_log_fn(f'Evaluating skip rule: {rule_snapshot}')
            for (evaluation_data, asset_partitions) in skip_rule.evaluate_for_asset(skip_context):
                all_results.append((AutoMaterializeRuleEvaluation(rule_snapshot=rule_snapshot, evaluation_data=evaluation_data), asset_partitions))
                self._verbose_log_fn(f'Rule returned {len(asset_partitions)} partitions')
                to_skip.update(asset_partitions)
            self._verbose_log_fn('Done evaluating skip rule')
        to_materialize.difference_update(to_skip)
        if auto_materialize_policy.max_materializations_per_minute is not None:
            rule = DiscardOnMaxMaterializationsExceededRule(limit=auto_materialize_policy.max_materializations_per_minute)
            rule_snapshot = rule.to_snapshot()
            self._verbose_log_fn(f'Evaluating discard rule: {rule_snapshot}')
            for (evaluation_data, asset_partitions) in rule.evaluate_for_asset(dataclasses.replace(skip_context, candidates=to_materialize)):
                all_results.append((AutoMaterializeRuleEvaluation(rule_snapshot=rule_snapshot, evaluation_data=evaluation_data), asset_partitions))
                self._verbose_log_fn(f'Discard rule returned {len(asset_partitions)} partitions')
                to_discard.update(asset_partitions)
            self._verbose_log_fn('Done evaluating discard rule')
        to_materialize.difference_update(to_discard)
        to_skip.difference_update(to_discard)
        return (AutoMaterializeAssetEvaluation.from_rule_evaluation_results(asset_key=asset_key, asset_graph=self.asset_graph, asset_partitions_by_rule_evaluation=all_results, num_requested=len(to_materialize), num_skipped=len(to_skip), num_discarded=len(to_discard), dynamic_partitions_store=self.instance_queryer), to_materialize, to_discard)

    def get_auto_materialize_asset_evaluations(self) -> Tuple[Mapping[AssetKey, AutoMaterializeAssetEvaluation], AbstractSet[AssetKeyPartitionKey], AbstractSet[AssetKeyPartitionKey]]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a mapping from asset key to the AutoMaterializeAssetEvaluation for that key, as\n        well as sets of all asset partitions that should be materialized or discarded this tick.\n        '
        evaluations_by_key: Dict[AssetKey, AutoMaterializeAssetEvaluation] = {}
        will_materialize_mapping: Dict[AssetKey, AbstractSet[AssetKeyPartitionKey]] = defaultdict(set)
        to_discard: Set[AssetKeyPartitionKey] = set()
        expected_data_time_mapping: Dict[AssetKey, Optional[datetime.datetime]] = defaultdict()
        visited_multi_asset_keys = set()
        num_checked_assets = 0
        num_target_asset_keys = len(self.target_asset_keys)
        for asset_key in itertools.chain(*self.asset_graph.toposort_asset_keys()):
            if asset_key not in self.target_asset_keys:
                continue
            num_checked_assets = num_checked_assets + 1
            start_time = time.time()
            self._verbose_log_fn(f'Evaluating asset {asset_key.to_user_string()} ({num_checked_assets}/{num_target_asset_keys})')
            if asset_key in visited_multi_asset_keys:
                self._verbose_log_fn(f'Asset {asset_key.to_user_string()} already visited')
                continue
            (evaluation, to_materialize_for_asset, to_discard_for_asset) = self.evaluate_asset(asset_key, will_materialize_mapping, expected_data_time_mapping)
            log_fn = self._logger.info if evaluation.num_requested or evaluation.num_skipped or evaluation.num_discarded else self._logger.debug
            to_materialize_str = ','.join([to_materialize.partition_key or 'No partition' for to_materialize in to_materialize_for_asset])
            log_fn(f"Asset {asset_key.to_user_string()} evaluation result: {evaluation.num_requested} requested ({to_materialize_str}), {evaluation.num_skipped} skipped, {evaluation.num_discarded} discarded ({format(time.time() - start_time, '.3f')} seconds)")
            evaluations_by_key[asset_key] = evaluation
            will_materialize_mapping[asset_key] = to_materialize_for_asset
            to_discard.update(to_discard_for_asset)
            expected_data_time = get_expected_data_time_for_asset_key(self.asset_graph, asset_key, will_materialize_mapping=will_materialize_mapping, expected_data_time_mapping=expected_data_time_mapping, data_time_resolver=self.data_time_resolver, current_time=self.instance_queryer.evaluation_time, will_materialize=bool(to_materialize_for_asset))
            expected_data_time_mapping[asset_key] = expected_data_time
            if to_materialize_for_asset:
                for neighbor_key in self.asset_graph.get_required_multi_asset_keys(asset_key):
                    auto_materialize_policy = self.asset_graph.auto_materialize_policies_by_key.get(neighbor_key)
                    if auto_materialize_policy is None:
                        check.failed(f'Expected auto materialize policy on asset {asset_key}')
                    to_materialize_for_neighbor = {ap._replace(asset_key=neighbor_key) for ap in to_materialize_for_asset}
                    to_discard_for_neighbor = {ap._replace(asset_key=neighbor_key) for ap in to_discard_for_asset}
                    evaluations_by_key[neighbor_key] = evaluation._replace(asset_key=neighbor_key, rule_snapshots=auto_materialize_policy.rule_snapshots)
                    will_materialize_mapping[neighbor_key] = to_materialize_for_neighbor
                    to_discard.update(to_discard_for_neighbor)
                    expected_data_time_mapping[neighbor_key] = expected_data_time
                    visited_multi_asset_keys.add(neighbor_key)
        to_materialize = set().union(*will_materialize_mapping.values())
        return (evaluations_by_key, to_materialize, to_discard)

    def evaluate(self) -> Tuple[Sequence[RunRequest], AssetDaemonCursor, Sequence[AutoMaterializeAssetEvaluation]]:
        if False:
            for i in range(10):
                print('nop')
        observe_request_timestamp = pendulum.now().timestamp()
        auto_observe_run_requests = get_auto_observe_run_requests(asset_graph=self.asset_graph, last_observe_request_timestamp_by_asset_key=self.cursor.last_observe_request_timestamp_by_asset_key, current_timestamp=observe_request_timestamp, run_tags=self._observe_run_tags) if self._auto_observe else []
        (evaluations_by_asset_key, to_materialize, to_discard) = self.get_auto_materialize_asset_evaluations()
        run_requests = [*build_run_requests(asset_partitions=to_materialize, asset_graph=self.asset_graph, run_tags=self._materialize_run_tags), *auto_observe_run_requests]
        (newly_materialized_root_asset_keys, newly_materialized_root_partitions_by_asset_key) = self.get_newly_updated_roots()
        return (run_requests, self.cursor.with_updates(latest_storage_id=self.get_new_latest_storage_id(), to_materialize=to_materialize, to_discard=to_discard, asset_graph=self.asset_graph, newly_materialized_root_asset_keys=newly_materialized_root_asset_keys, newly_materialized_root_partitions_by_asset_key=newly_materialized_root_partitions_by_asset_key, evaluation_id=self._evaluation_id, newly_observe_requested_asset_keys=[asset_key for run_request in auto_observe_run_requests for asset_key in cast(Sequence[AssetKey], run_request.asset_selection)], observe_request_timestamp=observe_request_timestamp, evaluations=list(evaluations_by_asset_key.values()), evaluation_time=self.instance_queryer.evaluation_time), [evaluation for evaluation in evaluations_by_asset_key.values() if not evaluation.equivalent_to_stored_evaluation(self.cursor.latest_evaluation_by_asset_key.get(evaluation.asset_key), self.asset_graph)])

def build_run_requests(asset_partitions: Iterable[AssetKeyPartitionKey], asset_graph: AssetGraph, run_tags: Optional[Mapping[str, str]]) -> Sequence[RunRequest]:
    if False:
        print('Hello World!')
    assets_to_reconcile_by_partitions_def_partition_key: Mapping[Tuple[Optional[PartitionsDefinition], Optional[str]], Set[AssetKey]] = defaultdict(set)
    for asset_partition in asset_partitions:
        assets_to_reconcile_by_partitions_def_partition_key[asset_graph.get_partitions_def(asset_partition.asset_key), asset_partition.partition_key].add(asset_partition.asset_key)
    run_requests = []
    for ((partitions_def, partition_key), asset_keys) in assets_to_reconcile_by_partitions_def_partition_key.items():
        tags = {**(run_tags or {})}
        if partition_key is not None:
            if partitions_def is None:
                check.failed('Partition key provided for unpartitioned asset')
            tags.update({**partitions_def.get_tags_for_partition_key(partition_key)})
        for asset_keys_in_repo in asset_graph.split_asset_keys_by_repository(asset_keys):
            run_requests.append(RunRequest(asset_selection=list(asset_keys_in_repo), partition_key=partition_key, tags=tags))
    return run_requests

def build_run_requests_with_backfill_policies(asset_partitions: Iterable[AssetKeyPartitionKey], asset_graph: AssetGraph, run_tags: Optional[Mapping[str, str]], dynamic_partitions_store: DynamicPartitionsStore) -> Sequence[RunRequest]:
    if False:
        print('Hello World!')
    'If all assets have backfill policies, we should respect them and materialize them according\n    to their backfill policies.\n    '
    run_requests = []
    asset_partition_keys: Mapping[AssetKey, Set[str]] = {asset_key_partition.asset_key: set() for asset_key_partition in asset_partitions}
    for asset_partition in asset_partitions:
        if asset_partition.partition_key:
            asset_partition_keys[asset_partition.asset_key].add(asset_partition.partition_key)
    assets_to_reconcile_by_partitions_def_partition_keys: Mapping[Tuple[Optional[PartitionsDefinition], Optional[FrozenSet[str]]], Set[AssetKey]] = defaultdict(set)
    for (asset_key, partition_keys) in asset_partition_keys.items():
        assets_to_reconcile_by_partitions_def_partition_keys[asset_graph.get_partitions_def(asset_key), frozenset(partition_keys) if partition_keys else None].add(asset_key)
    for ((partitions_def, partition_keys), asset_keys) in assets_to_reconcile_by_partitions_def_partition_keys.items():
        tags = {**(run_tags or {})}
        if partitions_def is None and partition_keys is not None:
            check.failed('Partition key provided for unpartitioned asset')
        if partitions_def is not None and partition_keys is None:
            check.failed('Partition key missing for partitioned asset')
        if partitions_def is None and partition_keys is None:
            run_requests.append(RunRequest(asset_selection=list(asset_keys), tags=tags))
        else:
            backfill_policies = {check.not_none(asset_graph.get_backfill_policy(asset_key)) for asset_key in asset_keys}
            if len(backfill_policies) == 1:
                backfill_policy = backfill_policies.pop()
                run_requests.extend(_build_run_requests_with_backfill_policy(list(asset_keys), check.not_none(backfill_policy), check.not_none(partition_keys), check.not_none(partitions_def), tags, dynamic_partitions_store=dynamic_partitions_store))
            else:
                for asset_key in asset_keys:
                    backfill_policy = asset_graph.get_backfill_policy(asset_key)
                    run_requests.extend(_build_run_requests_with_backfill_policy([asset_key], check.not_none(backfill_policy), check.not_none(partition_keys), check.not_none(partitions_def), tags, dynamic_partitions_store=dynamic_partitions_store))
    return run_requests

def _build_run_requests_with_backfill_policy(asset_keys: Sequence[AssetKey], backfill_policy: BackfillPolicy, partition_keys: FrozenSet[str], partitions_def: PartitionsDefinition, tags: Dict[str, Any], dynamic_partitions_store: DynamicPartitionsStore) -> Sequence[RunRequest]:
    if False:
        while True:
            i = 10
    run_requests = []
    partition_subset = partitions_def.subset_with_partition_keys(partition_keys)
    partition_key_ranges = partition_subset.get_partition_key_ranges(dynamic_partitions_store=dynamic_partitions_store)
    for partition_key_range in partition_key_ranges:
        if backfill_policy.policy_type == BackfillPolicyType.SINGLE_RUN:
            run_requests.append(_build_run_request_for_partition_key_range(asset_keys=list(asset_keys), partition_range_start=partition_key_range.start, partition_range_end=partition_key_range.end, run_tags=tags))
        else:
            run_requests.extend(_build_run_requests_for_partition_key_range(asset_keys=list(asset_keys), partitions_def=partitions_def, partition_key_range=partition_key_range, max_partitions_per_run=check.int_param(backfill_policy.max_partitions_per_run, 'max_partitions_per_run'), run_tags=tags))
    return run_requests

def _build_run_requests_for_partition_key_range(asset_keys: Sequence[AssetKey], partitions_def: PartitionsDefinition, partition_key_range: PartitionKeyRange, max_partitions_per_run: int, run_tags: Dict[str, str]) -> Sequence[RunRequest]:
    if False:
        return 10
    'Builds multiple run requests for the given partition key range. Each run request will have at most\n    max_partitions_per_run partitions.\n    '
    partition_keys = partitions_def.get_partition_keys_in_range(partition_key_range)
    partition_range_start_index = partition_keys.index(partition_key_range.start)
    partition_range_end_index = partition_keys.index(partition_key_range.end)
    partition_chunk_start_index = partition_range_start_index
    run_requests = []
    while partition_chunk_start_index <= partition_range_end_index:
        partition_chunk_end_index = partition_chunk_start_index + max_partitions_per_run - 1
        if partition_chunk_end_index > partition_range_end_index:
            partition_chunk_end_index = partition_range_end_index
        partition_chunk_start_key = partition_keys[partition_chunk_start_index]
        partition_chunk_end_key = partition_keys[partition_chunk_end_index]
        run_requests.append(_build_run_request_for_partition_key_range(asset_keys, partition_chunk_start_key, partition_chunk_end_key, run_tags))
        partition_chunk_start_index = partition_chunk_end_index + 1
    return run_requests

def _build_run_request_for_partition_key_range(asset_keys: Sequence[AssetKey], partition_range_start: str, partition_range_end: str, run_tags: Dict[str, str]) -> RunRequest:
    if False:
        return 10
    'Builds a single run request for the given asset key and partition key range.'
    tags = {**(run_tags or {}), ASSET_PARTITION_RANGE_START_TAG: partition_range_start, ASSET_PARTITION_RANGE_END_TAG: partition_range_end}
    return RunRequest(asset_selection=asset_keys, tags=tags)

def get_auto_observe_run_requests(last_observe_request_timestamp_by_asset_key: Mapping[AssetKey, float], current_timestamp: float, asset_graph: AssetGraph, run_tags: Optional[Mapping[str, str]]) -> Sequence[RunRequest]:
    if False:
        return 10
    assets_to_auto_observe: Set[AssetKey] = set()
    for asset_key in asset_graph.source_asset_keys:
        last_observe_request_timestamp = last_observe_request_timestamp_by_asset_key.get(asset_key)
        auto_observe_interval_minutes = asset_graph.get_auto_observe_interval_minutes(asset_key)
        if auto_observe_interval_minutes and (last_observe_request_timestamp is None or last_observe_request_timestamp + auto_observe_interval_minutes * 60 < current_timestamp):
            assets_to_auto_observe.add(asset_key)
    partitions_def_and_asset_key_groups: List[Sequence[AssetKey]] = []
    for repository_asset_keys in asset_graph.split_asset_keys_by_repository(assets_to_auto_observe):
        asset_keys_by_partitions_def = defaultdict(list)
        for asset_key in repository_asset_keys:
            partitions_def = asset_graph.get_partitions_def(asset_key)
            asset_keys_by_partitions_def[partitions_def].append(asset_key)
        partitions_def_and_asset_key_groups.extend(asset_keys_by_partitions_def.values())
    return [RunRequest(asset_selection=list(asset_keys), tags=run_tags) for asset_keys in partitions_def_and_asset_key_groups if len(asset_keys) > 0]