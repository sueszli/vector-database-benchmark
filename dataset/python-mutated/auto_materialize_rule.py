import datetime
import functools
from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, AbstractSet, Callable, Dict, Iterable, Mapping, NamedTuple, Optional, Sequence, Set
import pytz
import dagster._check as check
from dagster._annotations import experimental, public
from dagster._core.definitions.auto_materialize_rule_evaluation import AutoMaterializeDecisionType, AutoMaterializeRuleEvaluationData, AutoMaterializeRuleSnapshot, ParentUpdatedRuleEvaluationData, RuleEvaluationResults, WaitingOnAssetsRuleEvaluationData
from dagster._core.definitions.data_time import CachingDataTimeResolver
from dagster._core.definitions.events import AssetKey, AssetKeyPartitionKey
from dagster._core.definitions.freshness_based_auto_materialize import freshness_evaluation_results_for_asset_key
from dagster._core.definitions.multi_dimensional_partitions import MultiPartitionsDefinition
from dagster._core.definitions.partition_mapping import IdentityPartitionMapping
from dagster._core.definitions.time_window_partition_mapping import TimeWindowPartitionMapping
from dagster._core.definitions.time_window_partitions import get_time_partitions_def
from dagster._core.storage.dagster_run import RunsFilter
from dagster._core.storage.tags import AUTO_MATERIALIZE_TAG
from dagster._serdes.serdes import whitelist_for_serdes
from dagster._utils.caching_instance_queryer import CachingInstanceQueryer
from dagster._utils.schedules import cron_string_iterator, is_valid_cron_string, reverse_cron_string_iterator
from .asset_graph import AssetGraph, sort_key_for_asset_partition
if TYPE_CHECKING:
    from dagster._core.definitions.asset_daemon_context import AssetDaemonContext
    from dagster._core.definitions.asset_daemon_cursor import AssetDaemonCursor
    from dagster._core.definitions.auto_materialize_rule_evaluation import AutoMaterializeAssetEvaluation

@dataclass(frozen=True)
class RuleEvaluationContext:
    asset_key: AssetKey
    cursor: 'AssetDaemonCursor'
    instance_queryer: CachingInstanceQueryer
    data_time_resolver: CachingDataTimeResolver
    will_materialize_mapping: Mapping[AssetKey, AbstractSet[AssetKeyPartitionKey]]
    expected_data_time_mapping: Mapping[AssetKey, Optional[datetime.datetime]]
    candidates: AbstractSet[AssetKeyPartitionKey]
    daemon_context: 'AssetDaemonContext'

    @property
    def asset_graph(self) -> AssetGraph:
        if False:
            i = 10
            return i + 15
        return self.instance_queryer.asset_graph

    @property
    def previous_tick_evaluation(self) -> Optional['AutoMaterializeAssetEvaluation']:
        if False:
            return 10
        'Returns the evaluation of the asset on the previous tick.'
        return self.cursor.latest_evaluation_by_asset_key.get(self.asset_key)

    @property
    def evaluation_time(self) -> datetime.datetime:
        if False:
            print('Hello World!')
        'Returns the time at which this rule is being evaluated.'
        return self.instance_queryer.evaluation_time

    @property
    def auto_materialize_run_tags(self) -> Mapping[str, str]:
        if False:
            return 10
        return {AUTO_MATERIALIZE_TAG: 'true', **self.instance_queryer.instance.auto_materialize_run_tags}

    @functools.cached_property
    def previous_tick_requested_or_discarded_asset_partitions(self) -> AbstractSet[AssetKeyPartitionKey]:
        if False:
            return 10
        'Returns the set of asset partitions that were requested or discarded on the previous tick.'
        if not self.previous_tick_evaluation:
            return set()
        return self.previous_tick_evaluation.get_requested_or_discarded_asset_partitions(asset_graph=self.asset_graph)

    @functools.cached_property
    def previous_tick_evaluated_asset_partitions(self) -> AbstractSet[AssetKeyPartitionKey]:
        if False:
            while True:
                i = 10
        'Returns the set of asset partitions that were evaluated on the previous tick.'
        if not self.previous_tick_evaluation:
            return set()
        return self.previous_tick_evaluation.get_evaluated_asset_partitions(asset_graph=self.asset_graph)

    def get_previous_tick_results(self, rule: 'AutoMaterializeRule') -> 'RuleEvaluationResults':
        if False:
            for i in range(10):
                print('nop')
        'Returns the results that were calculated for a given rule on the previous tick.'
        if not self.previous_tick_evaluation:
            return []
        return self.previous_tick_evaluation.get_rule_evaluation_results(rule_snapshot=rule.to_snapshot(), asset_graph=self.asset_graph)

    def get_candidates_not_evaluated_by_rule_on_previous_tick(self) -> AbstractSet[AssetKeyPartitionKey]:
        if False:
            print('Hello World!')
        'Returns the set of candidates that were not evaluated by the rule that is currently being\n        evaluated on the previous tick.\n\n        Any asset partition that was evaluated by any rule on the previous tick must have been\n        evaluated by *all* skip rules.\n        '
        return self.candidates - self.previous_tick_evaluated_asset_partitions

    def get_candidates_with_updated_or_will_update_parents(self) -> AbstractSet[AssetKeyPartitionKey]:
        if False:
            while True:
                i = 10
        "Returns the set of candidate asset partitions whose parents have been updated since the\n        last tick or will be requested on this tick.\n\n        Many rules depend on the state of the asset's parents, so this function is useful for\n        finding asset partitions that should be re-evaluated.\n        "
        updated_parents = self.get_asset_partitions_with_updated_parents_since_previous_tick()
        will_update_parents = set(self.get_will_update_parent_mapping().keys())
        return self.candidates & (updated_parents | will_update_parents)

    def materialized_requested_or_discarded_since_previous_tick(self, asset_partition: AssetKeyPartitionKey) -> bool:
        if False:
            print('Hello World!')
        'Returns whether an asset partition has been materialized, requested, or discarded since\n        the last tick.\n        '
        if asset_partition in self.previous_tick_requested_or_discarded_asset_partitions:
            return True
        return self.instance_queryer.asset_partition_has_materialization_or_observation(asset_partition, after_cursor=self.cursor.latest_storage_id)

    def materializable_in_same_run(self, child_key: AssetKey, parent_key: AssetKey) -> bool:
        if False:
            return 10
        'Returns whether a child asset can be materialized in the same run as a parent asset.'
        from dagster._core.definitions.external_asset_graph import ExternalAssetGraph
        return child_key in self.asset_graph.materializable_asset_keys and parent_key in self.asset_graph.materializable_asset_keys and self.asset_graph.have_same_partitioning(child_key, parent_key) and (not self.asset_graph.is_partitioned(parent_key) or isinstance(self.asset_graph.get_partition_mapping(child_key, parent_key), (TimeWindowPartitionMapping, IdentityPartitionMapping))) and (not isinstance(self.asset_graph, ExternalAssetGraph) or self.asset_graph.get_repository_handle(child_key) == self.asset_graph.get_repository_handle(parent_key))

    def get_parents_that_will_not_be_materialized_on_current_tick(self, *, asset_partition: AssetKeyPartitionKey) -> AbstractSet[AssetKeyPartitionKey]:
        if False:
            for i in range(10):
                print('nop')
        'Returns the set of parent asset partitions that will not be updated in the same run of\n        this asset partition if a run is launched for this asset partition on this tick.\n        '
        return {parent for parent in self.asset_graph.get_parents_partitions(dynamic_partitions_store=self.instance_queryer, current_time=self.instance_queryer.evaluation_time, asset_key=asset_partition.asset_key, partition_key=asset_partition.partition_key).parent_partitions if parent not in self.will_materialize_mapping.get(parent.asset_key, set()) or not self.materializable_in_same_run(asset_partition.asset_key, parent.asset_key)}

    def get_asset_partitions_with_updated_parents_since_previous_tick(self) -> AbstractSet[AssetKeyPartitionKey]:
        if False:
            print('Hello World!')
        'Returns the set of asset partitions for the current key which have parents that updated\n        since the last tick.\n        '
        return self.daemon_context.get_asset_partitions_with_newly_updated_parents_for_key(self.asset_key)

    def get_will_update_parent_mapping(self) -> Mapping[AssetKeyPartitionKey, AbstractSet[AssetKey]]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a mapping from asset partitions of the current asset to the set of parent keys\n        which will be requested this tick and can execute in the same run as the current asset.\n        '
        will_update_parents_by_asset_partition = defaultdict(set)
        for parent_key in self.asset_graph.get_parents(self.asset_key):
            if not self.materializable_in_same_run(self.asset_key, parent_key):
                continue
            for parent_partition in self.will_materialize_mapping.get(parent_key, set()):
                asset_partition = AssetKeyPartitionKey(self.asset_key, parent_partition.partition_key)
                will_update_parents_by_asset_partition[asset_partition].add(parent_key)
        return will_update_parents_by_asset_partition

    def will_update_asset_partition(self, asset_partition: AssetKeyPartitionKey) -> bool:
        if False:
            i = 10
            return i + 15
        return asset_partition in self.will_materialize_mapping.get(asset_partition.asset_key, set())

    def get_asset_partitions_by_asset_key(self, asset_partitions: AbstractSet[AssetKeyPartitionKey]) -> Mapping[AssetKey, Set[AssetKeyPartitionKey]]:
        if False:
            print('Hello World!')
        asset_partitions_by_asset_key: Dict[AssetKey, Set[AssetKeyPartitionKey]] = defaultdict(set)
        for parent in asset_partitions:
            asset_partitions_by_asset_key[parent.asset_key].add(parent)
        return asset_partitions_by_asset_key

class AutoMaterializeRule(ABC):
    """An AutoMaterializeRule defines a bit of logic which helps determine if a materialization
    should be kicked off for a given asset partition.

    Each rule can have one of two decision types, `MATERIALIZE` (indicating that an asset partition
    should be materialized) or `SKIP` (indicating that the asset partition should not be
    materialized).

    Materialize rules are evaluated first, and skip rules operate over the set of candidates that
    are produced by the materialize rules. Other than that, there is no ordering between rules.
    """

    @abstractproperty
    def decision_type(self) -> AutoMaterializeDecisionType:
        if False:
            print('Hello World!')
        'The decision type of the rule (either `MATERIALIZE` or `SKIP`).'
        ...

    @abstractproperty
    def description(self) -> str:
        if False:
            return 10
        "A human-readable description of this rule. As a basic guideline, this string should\n        complete the sentence: 'Indicates an asset should be (materialize/skipped) when ____'.\n        "
        ...

    def add_evaluation_data_from_previous_tick(self, context: RuleEvaluationContext, asset_partitions_by_evaluation_data: Mapping[Optional[AutoMaterializeRuleEvaluationData], Set[AssetKeyPartitionKey]], should_use_past_data_fn: Callable[[AssetKeyPartitionKey], bool]) -> 'RuleEvaluationResults':
        if False:
            print('Hello World!')
        'Combines a given set of evaluation data with evaluation data from the previous tick. The\n        returned value will include the union of the evaluation data contained within\n        `asset_partitions_by_evaluation_data` and the evaluation data calculated for asset\n        partitions on the previous tick for which `should_use_past_data_fn` evaluates to `True`.\n\n        Args:\n            context: The current RuleEvaluationContext.\n            asset_partitions_by_evaluation_data: A mapping from evaluation data to the set of asset\n                partitions that the rule applies to.\n            should_use_past_data_fn: A function that returns whether a given asset partition from the\n                previous tick should be included in the results of this tick.\n        '
        asset_partitions_by_evaluation_data = defaultdict(set, asset_partitions_by_evaluation_data)
        evaluated_asset_partitions = set().union(*asset_partitions_by_evaluation_data.values())
        for (evaluation_data, asset_partitions) in context.get_previous_tick_results(self):
            for ap in asset_partitions:
                if ap in evaluated_asset_partitions:
                    continue
                elif should_use_past_data_fn(ap):
                    asset_partitions_by_evaluation_data[evaluation_data].add(ap)
        return list(asset_partitions_by_evaluation_data.items())

    @abstractmethod
    def evaluate_for_asset(self, context: RuleEvaluationContext) -> RuleEvaluationResults:
        if False:
            for i in range(10):
                print('nop')
        'The core evaluation function for the rule. This function takes in a context object and\n        returns a mapping from evaluated rules to the set of asset partitions that the rule applies\n        to.\n        '
        ...

    @public
    @staticmethod
    def materialize_on_required_for_freshness() -> 'MaterializeOnRequiredForFreshnessRule':
        if False:
            while True:
                i = 10
        'Materialize an asset partition if it is required to satisfy a freshness policy of this\n        asset or one of its downstream assets.\n\n        Note: This rule has no effect on partitioned assets.\n        '
        return MaterializeOnRequiredForFreshnessRule()

    @public
    @staticmethod
    def materialize_on_cron(cron_schedule: str, timezone: str='UTC', all_partitions: bool=False) -> 'MaterializeOnCronRule':
        if False:
            for i in range(10):
                print('nop')
        'Materialize an asset partition if it has not been materialized since the latest cron\n        schedule tick. For assets with a time component to their partitions_def, this rule will\n        request all partitions that have been missed since the previous tick.\n\n        Args:\n            cron_schedule (str): A cron schedule string (e.g. "`0 * * * *`") indicating the ticks for\n                which this rule should fire.\n            timezone (str): The timezone in which this cron schedule should be evaluated. Defaults\n                to "UTC".\n            all_partitions (bool): If True, this rule fires for all partitions of this asset on each\n                cron tick. If False, this rule fires only for the last partition of this asset.\n                Defaults to False.\n        '
        check.param_invariant(is_valid_cron_string(cron_schedule), 'cron_schedule', 'must be a valid cron string')
        check.param_invariant(timezone in pytz.all_timezones_set, 'timezone', 'must be a valid timezone')
        return MaterializeOnCronRule(cron_schedule=cron_schedule, timezone=timezone, all_partitions=all_partitions)

    @public
    @staticmethod
    def materialize_on_parent_updated(updated_parent_filter: Optional['AutoMaterializeAssetPartitionsFilter']=None) -> 'MaterializeOnParentUpdatedRule':
        if False:
            i = 10
            return i + 15
        "Materialize an asset partition if one of its parents has been updated more recently\n        than it has.\n\n        Note: For time-partitioned or dynamic-partitioned assets downstream of an unpartitioned\n        asset, this rule will only fire for the most recent partition of the downstream.\n\n        Args:\n            updated_parent_filter (Optional[AutoMaterializeAssetPartitionsFilter]): Filter to apply\n                to updated parents. If a parent was updated but does not pass the filter criteria,\n                then it won't count as updated for the sake of this rule.\n        "
        return MaterializeOnParentUpdatedRule(updated_parent_filter=updated_parent_filter)

    @public
    @staticmethod
    def materialize_on_missing() -> 'MaterializeOnMissingRule':
        if False:
            while True:
                i = 10
        "Materialize an asset partition if it has never been materialized before. This rule will\n        not fire for non-root assets unless that asset's parents have been updated.\n        "
        return MaterializeOnMissingRule()

    @public
    @staticmethod
    def skip_on_parent_missing() -> 'SkipOnParentMissingRule':
        if False:
            print('Hello World!')
        'Skip materializing an asset partition if one of its parent asset partitions has never\n        been materialized (for regular assets) or observed (for observable source assets).\n        '
        return SkipOnParentMissingRule()

    @public
    @staticmethod
    def skip_on_parent_outdated() -> 'SkipOnParentOutdatedRule':
        if False:
            return 10
        'Skip materializing an asset partition if any of its parents has not incorporated the\n        latest data from its ancestors.\n        '
        return SkipOnParentOutdatedRule()

    @public
    @staticmethod
    def skip_on_not_all_parents_updated(require_update_for_all_parent_partitions: bool=False) -> 'SkipOnNotAllParentsUpdatedRule':
        if False:
            while True:
                i = 10
        "Skip materializing an asset partition if any of its parents have not been updated since\n        the asset's last materialization.\n\n        Args:\n            require_update_for_all_parent_partitions (Optional[bool]): Applies only to an unpartitioned\n                asset or an asset partition that depends on more than one partition in any upstream asset.\n                If true, requires all upstream partitions in each upstream asset to be materialized since\n                the downstream asset's last materialization in order to update it. If false, requires at\n                least one upstream partition in each upstream asset to be materialized since the downstream\n                asset's last materialization in order to update it. Defaults to false.\n        "
        return SkipOnNotAllParentsUpdatedRule(require_update_for_all_parent_partitions)

    @public
    @staticmethod
    def skip_on_required_but_nonexistent_parents() -> 'SkipOnRequiredButNonexistentParentsRule':
        if False:
            return 10
        'Skip an asset partition if it depends on parent partitions that do not exist.\n\n        For example, imagine a downstream asset is time-partitioned, starting in 2022, but has a\n        time-partitioned parent which starts in 2023. This rule will skip attempting to materialize\n        downstream partitions from before 2023, since the parent partitions do not exist.\n        '
        return SkipOnRequiredButNonexistentParentsRule()

    @public
    @staticmethod
    def skip_on_backfill_in_progress(all_partitions: bool=False) -> 'SkipOnBackfillInProgressRule':
        if False:
            print('Hello World!')
        "Skip an asset's partitions if targeted by an in-progress backfill.\n\n        Args:\n            all_partitions (bool): If True, skips all partitions of the asset being backfilled,\n                regardless of whether the specific partition is targeted by a backfill.\n                If False, skips only partitions targeted by a backfill. Defaults to False.\n        "
        return SkipOnBackfillInProgressRule(all_partitions)

    def to_snapshot(self) -> AutoMaterializeRuleSnapshot:
        if False:
            while True:
                i = 10
        'Returns a serializable snapshot of this rule for historical evaluations.'
        return AutoMaterializeRuleSnapshot(class_name=self.__class__.__name__, description=self.description, decision_type=self.decision_type)

    def __eq__(self, other) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return type(self) == type(other) and super().__eq__(other)

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return hash(hash(type(self)) + super().__hash__())

@whitelist_for_serdes
class MaterializeOnRequiredForFreshnessRule(AutoMaterializeRule, NamedTuple('_MaterializeOnRequiredForFreshnessRule', [])):

    @property
    def decision_type(self) -> AutoMaterializeDecisionType:
        if False:
            while True:
                i = 10
        return AutoMaterializeDecisionType.MATERIALIZE

    @property
    def description(self) -> str:
        if False:
            while True:
                i = 10
        return "required to meet this or downstream asset's freshness policy"

    def evaluate_for_asset(self, context: RuleEvaluationContext) -> RuleEvaluationResults:
        if False:
            return 10
        freshness_conditions = freshness_evaluation_results_for_asset_key(asset_key=context.asset_key, data_time_resolver=context.data_time_resolver, asset_graph=context.asset_graph, current_time=context.instance_queryer.evaluation_time, will_materialize_mapping=context.will_materialize_mapping, expected_data_time_mapping=context.expected_data_time_mapping)
        return freshness_conditions

@whitelist_for_serdes
class MaterializeOnCronRule(AutoMaterializeRule, NamedTuple('_MaterializeOnCronRule', [('cron_schedule', str), ('timezone', str), ('all_partitions', bool)])):

    @property
    def decision_type(self) -> AutoMaterializeDecisionType:
        if False:
            print('Hello World!')
        return AutoMaterializeDecisionType.MATERIALIZE

    @property
    def description(self) -> str:
        if False:
            i = 10
            return i + 15
        return f"not materialized since last cron schedule tick of '{self.cron_schedule}' (timezone: {self.timezone})"

    def missed_cron_ticks(self, context: RuleEvaluationContext) -> Sequence[datetime.datetime]:
        if False:
            while True:
                i = 10
        'Returns the cron ticks which have been missed since the previous cursor was generated.'
        if not context.cursor.latest_evaluation_timestamp:
            previous_dt = next(reverse_cron_string_iterator(end_timestamp=context.evaluation_time.timestamp(), cron_string=self.cron_schedule, execution_timezone=self.timezone))
            return [previous_dt]
        missed_ticks = []
        for dt in cron_string_iterator(start_timestamp=context.cursor.latest_evaluation_timestamp, cron_string=self.cron_schedule, execution_timezone=self.timezone):
            if dt > context.evaluation_time:
                break
            missed_ticks.append(dt)
        return missed_ticks

    def get_asset_partitions_to_request(self, context: RuleEvaluationContext) -> AbstractSet[AssetKeyPartitionKey]:
        if False:
            while True:
                i = 10
        missed_ticks = self.missed_cron_ticks(context)
        if not missed_ticks:
            return set()
        partitions_def = context.asset_graph.get_partitions_def(context.asset_key)
        if partitions_def is None:
            return {AssetKeyPartitionKey(context.asset_key)}
        if self.all_partitions:
            return {AssetKeyPartitionKey(context.asset_key, partition_key) for partition_key in partitions_def.get_partition_keys(current_time=context.evaluation_time)}
        time_partitions_def = get_time_partitions_def(partitions_def)
        if time_partitions_def is None:
            return {AssetKeyPartitionKey(context.asset_key, partitions_def.get_last_partition_key())}
        missed_time_partition_keys = filter(None, [time_partitions_def.get_last_partition_key(current_time=missed_tick) for missed_tick in missed_ticks])
        if isinstance(partitions_def, MultiPartitionsDefinition):
            return {AssetKeyPartitionKey(context.asset_key, partition_key) for time_partition_key in missed_time_partition_keys for partition_key in partitions_def.get_multipartition_keys_with_dimension_value(partitions_def.time_window_dimension.name, time_partition_key, dynamic_partitions_store=context.instance_queryer)}
        else:
            return {AssetKeyPartitionKey(context.asset_key, time_partition_key) for time_partition_key in missed_time_partition_keys}

    def evaluate_for_asset(self, context: RuleEvaluationContext) -> RuleEvaluationResults:
        if False:
            i = 10
            return i + 15
        asset_partitions_to_request = self.get_asset_partitions_to_request(context)
        asset_partitions_by_evaluation_data = defaultdict(set)
        if asset_partitions_to_request:
            asset_partitions_by_evaluation_data[None].update(asset_partitions_to_request)
        return self.add_evaluation_data_from_previous_tick(context, asset_partitions_by_evaluation_data, should_use_past_data_fn=lambda ap: not context.materialized_requested_or_discarded_since_previous_tick(ap))

@whitelist_for_serdes
@experimental
class AutoMaterializeAssetPartitionsFilter(NamedTuple('_AutoMaterializeAssetPartitionsFilter', [('latest_run_required_tags', Optional[Mapping[str, str]])])):
    """A filter that can be applied to an asset partition, during auto-materialize evaluation, and
    returns a boolean for whether it passes.

    Attributes:
        latest_run_required_tags (Optional[Sequence[str]]): `passes` returns
            True if the run responsible for the latest materialization of the asset partition does
            not have all of these tags.
    """

    @property
    def description(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'latest run includes required tags: {self.latest_run_required_tags}'

    def passes(self, context: RuleEvaluationContext, asset_partitions: Iterable[AssetKeyPartitionKey]) -> Iterable[AssetKeyPartitionKey]:
        if False:
            print('Hello World!')
        if self.latest_run_required_tags is None:
            return asset_partitions
        will_update_asset_partitions: Set[AssetKeyPartitionKey] = set()
        asset_partitions_by_latest_run_id: Dict[str, Set[AssetKeyPartitionKey]] = defaultdict(set)
        for asset_partition in asset_partitions:
            if context.will_update_asset_partition(asset_partition):
                will_update_asset_partitions.add(asset_partition)
            else:
                record = context.instance_queryer.get_latest_materialization_or_observation_record(asset_partition)
                if record is None:
                    raise RuntimeError(f'No materialization record found for asset partition {asset_partition}')
                asset_partitions_by_latest_run_id[record.run_id].add(asset_partition)
        if len(asset_partitions_by_latest_run_id) > 0:
            run_ids_with_required_tags = context.instance_queryer.instance.get_run_ids(filters=RunsFilter(run_ids=list(asset_partitions_by_latest_run_id.keys()), tags=self.latest_run_required_tags))
        else:
            run_ids_with_required_tags = set()
        updated_partitions_with_required_tags = {asset_partition for (run_id, run_id_asset_partitions) in asset_partitions_by_latest_run_id.items() if run_id in run_ids_with_required_tags for asset_partition in run_id_asset_partitions}
        if self.latest_run_required_tags.items() <= context.auto_materialize_run_tags.items():
            return will_update_asset_partitions | updated_partitions_with_required_tags
        else:
            return updated_partitions_with_required_tags

    def __hash__(self):
        if False:
            return 10
        return hash(frozenset((self.latest_run_required_tags or {}).items()))

@whitelist_for_serdes
class MaterializeOnParentUpdatedRule(AutoMaterializeRule, NamedTuple('_MaterializeOnParentUpdatedRule', [('updated_parent_filter', Optional[AutoMaterializeAssetPartitionsFilter])])):

    def __new__(cls, updated_parent_filter: Optional[AutoMaterializeAssetPartitionsFilter]=None):
        if False:
            i = 10
            return i + 15
        return super().__new__(cls, updated_parent_filter=updated_parent_filter)

    @property
    def decision_type(self) -> AutoMaterializeDecisionType:
        if False:
            for i in range(10):
                print('nop')
        return AutoMaterializeDecisionType.MATERIALIZE

    @property
    def description(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        base = 'upstream data has changed since latest materialization'
        if self.updated_parent_filter is not None:
            return f"{base} and matches filter '{self.updated_parent_filter.description}'"
        else:
            return base

    def evaluate_for_asset(self, context: RuleEvaluationContext) -> RuleEvaluationResults:
        if False:
            print('Hello World!')
        'Evaluates the set of asset partitions of this asset whose parents have been updated,\n        or will update on this tick.\n        '
        will_update_parents_by_asset_partition = context.get_will_update_parent_mapping()
        has_or_will_update = context.get_asset_partitions_with_updated_parents_since_previous_tick() | set(will_update_parents_by_asset_partition.keys())
        asset_partitions_by_updated_parents: Mapping[AssetKeyPartitionKey, Set[AssetKeyPartitionKey]] = defaultdict(set)
        asset_partitions_by_will_update_parents: Mapping[AssetKeyPartitionKey, Set[AssetKeyPartitionKey]] = defaultdict(set)
        for asset_partition in has_or_will_update:
            parent_asset_partitions = context.asset_graph.get_parents_partitions(dynamic_partitions_store=context.instance_queryer, current_time=context.instance_queryer.evaluation_time, asset_key=asset_partition.asset_key, partition_key=asset_partition.partition_key).parent_partitions
            updated_parent_asset_partitions = context.instance_queryer.get_parent_asset_partitions_updated_after_child(asset_partition, parent_asset_partitions, respect_materialization_data_versions=context.daemon_context.respect_materialization_data_versions and len(parent_asset_partitions | has_or_will_update) < 100, ignored_parent_keys={context.asset_key})
            for parent in updated_parent_asset_partitions:
                asset_partitions_by_updated_parents[parent].add(asset_partition)
            for parent in parent_asset_partitions:
                if context.will_update_asset_partition(parent):
                    asset_partitions_by_will_update_parents[parent].add(asset_partition)
        updated_and_will_update_parents = asset_partitions_by_updated_parents.keys() | asset_partitions_by_will_update_parents.keys()
        filtered_updated_and_will_update_parents = self.updated_parent_filter.passes(context, updated_and_will_update_parents) if self.updated_parent_filter else updated_and_will_update_parents
        updated_parent_assets_by_asset_partition: Dict[AssetKeyPartitionKey, Set[AssetKey]] = defaultdict(set)
        will_update_parent_assets_by_asset_partition: Dict[AssetKeyPartitionKey, Set[AssetKey]] = defaultdict(set)
        for updated_or_will_update_parent in filtered_updated_and_will_update_parents:
            for child in asset_partitions_by_updated_parents.get(updated_or_will_update_parent, []):
                updated_parent_assets_by_asset_partition[child].add(updated_or_will_update_parent.asset_key)
            for child in asset_partitions_by_will_update_parents.get(updated_or_will_update_parent, []):
                will_update_parent_assets_by_asset_partition[child].add(updated_or_will_update_parent.asset_key)
        asset_partitions_by_evaluation_data = defaultdict(set)
        for asset_partition in updated_parent_assets_by_asset_partition.keys() | will_update_parent_assets_by_asset_partition.keys():
            asset_partitions_by_evaluation_data[ParentUpdatedRuleEvaluationData(updated_asset_keys=frozenset(updated_parent_assets_by_asset_partition.get(asset_partition, [])), will_update_asset_keys=frozenset(will_update_parent_assets_by_asset_partition.get(asset_partition, [])))].add(asset_partition)
        return self.add_evaluation_data_from_previous_tick(context, asset_partitions_by_evaluation_data, should_use_past_data_fn=lambda ap: not context.materialized_requested_or_discarded_since_previous_tick(ap))

@whitelist_for_serdes
class MaterializeOnMissingRule(AutoMaterializeRule, NamedTuple('_MaterializeOnMissingRule', [])):

    @property
    def decision_type(self) -> AutoMaterializeDecisionType:
        if False:
            return 10
        return AutoMaterializeDecisionType.MATERIALIZE

    @property
    def description(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'materialization is missing'

    def evaluate_for_asset(self, context: RuleEvaluationContext) -> RuleEvaluationResults:
        if False:
            print('Hello World!')
        'Evaluates the set of asset partitions for this asset which are missing and were not\n        previously discarded. Currently only applies to root asset partitions and asset partitions\n        with updated parents.\n        '
        asset_partitions_by_evaluation_data = defaultdict(set)
        missing_asset_partitions = set(context.daemon_context.get_never_handled_root_asset_partitions_for_key(context.asset_key))
        for candidate in context.daemon_context.get_asset_partitions_with_newly_updated_parents_for_key(context.asset_key):
            if not context.instance_queryer.asset_partition_has_materialization_or_observation(candidate):
                missing_asset_partitions |= {candidate}
        if missing_asset_partitions:
            asset_partitions_by_evaluation_data[None] = missing_asset_partitions
        return self.add_evaluation_data_from_previous_tick(context, asset_partitions_by_evaluation_data, should_use_past_data_fn=lambda ap: ap not in missing_asset_partitions and (not context.materialized_requested_or_discarded_since_previous_tick(ap)))

@whitelist_for_serdes
class SkipOnParentOutdatedRule(AutoMaterializeRule, NamedTuple('_SkipOnParentOutdatedRule', [])):

    @property
    def decision_type(self) -> AutoMaterializeDecisionType:
        if False:
            while True:
                i = 10
        return AutoMaterializeDecisionType.SKIP

    @property
    def description(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'waiting on upstream data to be up to date'

    def evaluate_for_asset(self, context: RuleEvaluationContext) -> RuleEvaluationResults:
        if False:
            for i in range(10):
                print('nop')
        asset_partitions_by_evaluation_data = defaultdict(set)
        candidates_to_evaluate = context.get_candidates_not_evaluated_by_rule_on_previous_tick() | context.get_candidates_with_updated_or_will_update_parents()
        for candidate in candidates_to_evaluate:
            outdated_ancestors = set()
            for parent in context.get_parents_that_will_not_be_materialized_on_current_tick(asset_partition=candidate):
                if context.instance_queryer.have_ignorable_partition_mapping_for_outdated(candidate.asset_key, parent.asset_key):
                    continue
                outdated_ancestors.update(context.instance_queryer.get_outdated_ancestors(asset_partition=parent))
            if outdated_ancestors:
                asset_partitions_by_evaluation_data[WaitingOnAssetsRuleEvaluationData(frozenset(outdated_ancestors))].add(candidate)
        return self.add_evaluation_data_from_previous_tick(context, asset_partitions_by_evaluation_data, should_use_past_data_fn=lambda ap: ap not in candidates_to_evaluate)

@whitelist_for_serdes
class SkipOnParentMissingRule(AutoMaterializeRule, NamedTuple('_SkipOnParentMissingRule', [])):

    @property
    def decision_type(self) -> AutoMaterializeDecisionType:
        if False:
            return 10
        return AutoMaterializeDecisionType.SKIP

    @property
    def description(self) -> str:
        if False:
            print('Hello World!')
        return 'waiting on upstream data to be present'

    def evaluate_for_asset(self, context: RuleEvaluationContext) -> RuleEvaluationResults:
        if False:
            for i in range(10):
                print('nop')
        asset_partitions_by_evaluation_data = defaultdict(set)
        candidates_to_evaluate = context.get_candidates_not_evaluated_by_rule_on_previous_tick() | context.get_candidates_with_updated_or_will_update_parents()
        for candidate in candidates_to_evaluate:
            missing_parent_asset_keys = set()
            for parent in context.get_parents_that_will_not_be_materialized_on_current_tick(asset_partition=candidate):
                if context.asset_graph.is_source(parent.asset_key) and (not context.asset_graph.is_observable(parent.asset_key)):
                    continue
                if not context.instance_queryer.asset_partition_has_materialization_or_observation(parent):
                    missing_parent_asset_keys.add(parent.asset_key)
            if missing_parent_asset_keys:
                asset_partitions_by_evaluation_data[WaitingOnAssetsRuleEvaluationData(frozenset(missing_parent_asset_keys))].add(candidate)
        return self.add_evaluation_data_from_previous_tick(context, asset_partitions_by_evaluation_data, should_use_past_data_fn=lambda ap: ap not in candidates_to_evaluate)

@whitelist_for_serdes
class SkipOnNotAllParentsUpdatedRule(AutoMaterializeRule, NamedTuple('_SkipOnNotAllParentsUpdatedRule', [('require_update_for_all_parent_partitions', bool)])):
    """An auto-materialize rule that enforces that an asset can only be materialized if all parents
    have been materialized since the asset's last materialization.

    Attributes:
        require_update_for_all_parent_partitions (Optional[bool]): Applies only to an unpartitioned
            asset or an asset partition that depends on more than one partition in any upstream asset.
            If true, requires all upstream partitions in each upstream asset to be materialized since
            the downstream asset's last materialization in order to update it. If false, requires at
            least one upstream partition in each upstream asset to be materialized since the downstream
            asset's last materialization in order to update it. Defaults to false.
    """

    @property
    def decision_type(self) -> AutoMaterializeDecisionType:
        if False:
            while True:
                i = 10
        return AutoMaterializeDecisionType.SKIP

    @property
    def description(self) -> str:
        if False:
            while True:
                i = 10
        if self.require_update_for_all_parent_partitions is False:
            return 'waiting on upstream data to be updated'
        else:
            return 'waiting until all upstream partitions are updated'

    def evaluate_for_asset(self, context: RuleEvaluationContext) -> RuleEvaluationResults:
        if False:
            while True:
                i = 10
        asset_partitions_by_evaluation_data = defaultdict(set)
        candidates_to_evaluate = context.get_candidates_not_evaluated_by_rule_on_previous_tick() | context.get_candidates_with_updated_or_will_update_parents()
        for candidate in candidates_to_evaluate:
            parent_partitions = context.asset_graph.get_parents_partitions(context.instance_queryer, context.instance_queryer.evaluation_time, context.asset_key, candidate.partition_key).parent_partitions
            updated_parent_partitions = context.instance_queryer.get_parent_asset_partitions_updated_after_child(candidate, parent_partitions, context.daemon_context.respect_materialization_data_versions, ignored_parent_keys=set()) | set().union(*[context.will_materialize_mapping.get(parent, set()) for parent in context.asset_graph.get_parents(context.asset_key)])
            if self.require_update_for_all_parent_partitions:
                non_updated_parent_keys = {parent.asset_key for parent in parent_partitions - updated_parent_partitions}
            else:
                parent_asset_keys = context.asset_graph.get_parents(context.asset_key)
                updated_parent_partitions_by_asset_key = context.get_asset_partitions_by_asset_key(updated_parent_partitions)
                non_updated_parent_keys = {parent for parent in parent_asset_keys if not updated_parent_partitions_by_asset_key.get(parent)}
            non_updated_parent_keys -= {context.asset_key}
            if non_updated_parent_keys:
                asset_partitions_by_evaluation_data[WaitingOnAssetsRuleEvaluationData(frozenset(non_updated_parent_keys))].add(candidate)
        return self.add_evaluation_data_from_previous_tick(context, asset_partitions_by_evaluation_data, should_use_past_data_fn=lambda ap: ap not in candidates_to_evaluate)

@whitelist_for_serdes
class SkipOnRequiredButNonexistentParentsRule(AutoMaterializeRule, NamedTuple('_SkipOnRequiredButNonexistentParentsRule', [])):

    @property
    def decision_type(self) -> AutoMaterializeDecisionType:
        if False:
            while True:
                i = 10
        return AutoMaterializeDecisionType.SKIP

    @property
    def description(self) -> str:
        if False:
            return 10
        return 'required parent partitions do not exist'

    def evaluate_for_asset(self, context: RuleEvaluationContext) -> RuleEvaluationResults:
        if False:
            for i in range(10):
                print('nop')
        asset_partitions_by_evaluation_data = defaultdict(set)
        candidates_to_evaluate = context.get_candidates_not_evaluated_by_rule_on_previous_tick()
        for candidate in candidates_to_evaluate:
            nonexistent_parent_partitions = context.asset_graph.get_parents_partitions(context.instance_queryer, context.instance_queryer.evaluation_time, candidate.asset_key, candidate.partition_key).required_but_nonexistent_parents_partitions
            nonexistent_parent_keys = {parent.asset_key for parent in nonexistent_parent_partitions}
            if nonexistent_parent_keys:
                asset_partitions_by_evaluation_data[WaitingOnAssetsRuleEvaluationData(frozenset(nonexistent_parent_keys))].add(candidate)
        return self.add_evaluation_data_from_previous_tick(context, asset_partitions_by_evaluation_data, should_use_past_data_fn=lambda ap: ap not in candidates_to_evaluate)

@whitelist_for_serdes
class SkipOnBackfillInProgressRule(AutoMaterializeRule, NamedTuple('_SkipOnBackfillInProgressRule', [('all_partitions', bool)])):

    @property
    def decision_type(self) -> AutoMaterializeDecisionType:
        if False:
            while True:
                i = 10
        return AutoMaterializeDecisionType.SKIP

    @property
    def description(self) -> str:
        if False:
            print('Hello World!')
        if self.all_partitions:
            return 'part of an asset targeted by an in-progress backfill'
        else:
            return 'targeted by an in-progress backfill'

    def evaluate_for_asset(self, context: RuleEvaluationContext) -> RuleEvaluationResults:
        if False:
            i = 10
            return i + 15
        backfill_in_progress_candidates: AbstractSet[AssetKeyPartitionKey] = set()
        backfilling_subset = context.instance_queryer.get_active_backfill_target_asset_graph_subset()
        if self.all_partitions:
            backfill_in_progress_candidates = {candidate for candidate in context.candidates if candidate.asset_key in backfilling_subset.asset_keys}
        else:
            backfill_in_progress_candidates = {candidate for candidate in context.candidates if candidate in backfilling_subset}
        if backfill_in_progress_candidates:
            return [(None, backfill_in_progress_candidates)]
        return []

@whitelist_for_serdes
class DiscardOnMaxMaterializationsExceededRule(AutoMaterializeRule, NamedTuple('_DiscardOnMaxMaterializationsExceededRule', [('limit', int)])):

    @property
    def decision_type(self) -> AutoMaterializeDecisionType:
        if False:
            while True:
                i = 10
        return AutoMaterializeDecisionType.DISCARD

    @property
    def description(self) -> str:
        if False:
            return 10
        return f'exceeds {self.limit} materialization(s) per minute'

    def evaluate_for_asset(self, context: RuleEvaluationContext) -> RuleEvaluationResults:
        if False:
            i = 10
            return i + 15
        rate_limited_asset_partitions = set(sorted(context.candidates, key=lambda x: sort_key_for_asset_partition(context.asset_graph, x))[self.limit:])
        if rate_limited_asset_partitions:
            return [(None, rate_limited_asset_partitions)]
        return []