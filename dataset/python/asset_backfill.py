import json
import logging
import os
from datetime import datetime
from enum import Enum
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import pendulum

from dagster import (
    PartitionKeyRange,
    _check as check,
)
from dagster._core.definitions.asset_daemon_context import (
    build_run_requests,
    build_run_requests_with_backfill_policies,
)
from dagster._core.definitions.asset_graph import AssetGraph
from dagster._core.definitions.asset_graph_subset import AssetGraphSubset
from dagster._core.definitions.asset_selection import AssetSelection
from dagster._core.definitions.assets_job import is_base_asset_job_name
from dagster._core.definitions.events import AssetKey, AssetKeyPartitionKey
from dagster._core.definitions.external_asset_graph import ExternalAssetGraph
from dagster._core.definitions.partition import PartitionsDefinition, PartitionsSubset
from dagster._core.definitions.run_request import RunRequest
from dagster._core.definitions.selector import JobSubsetSelector, PartitionsByAssetSelector
from dagster._core.errors import (
    DagsterAssetBackfillDataLoadError,
    DagsterBackfillFailedError,
    DagsterDefinitionChangedDeserializationError,
    DagsterInvariantViolationError,
)
from dagster._core.event_api import EventRecordsFilter
from dagster._core.events import DagsterEventType
from dagster._core.host_representation import (
    ExternalExecutionPlan,
    ExternalJob,
)
from dagster._core.instance import DagsterInstance, DynamicPartitionsStore
from dagster._core.storage.dagster_run import (
    CANCELABLE_RUN_STATUSES,
    DagsterRunStatus,
    RunsFilter,
)
from dagster._core.storage.tags import (
    ASSET_PARTITION_RANGE_END_TAG,
    ASSET_PARTITION_RANGE_START_TAG,
    BACKFILL_ID_TAG,
    PARTITION_NAME_TAG,
)
from dagster._core.workspace.context import (
    BaseWorkspaceRequestContext,
    IWorkspaceProcessContext,
)
from dagster._core.workspace.workspace import IWorkspace
from dagster._utils import hash_collection, utc_datetime_from_timestamp
from dagster._utils.caching_instance_queryer import CachingInstanceQueryer

if TYPE_CHECKING:
    from .backfill import PartitionBackfill

RUN_CHUNK_SIZE = 25


MAX_RUNS_CANCELED_PER_ITERATION = 50


class AssetBackfillStatus(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    MATERIALIZED = "MATERIALIZED"
    FAILED = "FAILED"


class PartitionedAssetBackfillStatus(
    NamedTuple(
        "_PartitionedAssetBackfillStatus",
        [
            ("asset_key", AssetKey),
            ("num_targeted_partitions", int),
            ("partitions_counts_by_status", Mapping[AssetBackfillStatus, int]),
        ],
    )
):
    def __new__(
        cls,
        asset_key: AssetKey,
        num_targeted_partitions: int,
        partitions_counts_by_status: Mapping[AssetBackfillStatus, int],
    ):
        return super(PartitionedAssetBackfillStatus, cls).__new__(
            cls,
            check.inst_param(asset_key, "asset_key", AssetKey),
            check.int_param(num_targeted_partitions, "num_targeted_partitions"),
            check.mapping_param(
                partitions_counts_by_status,
                "partitions_counts_by_status",
                key_type=AssetBackfillStatus,
                value_type=int,
            ),
        )


class UnpartitionedAssetBackfillStatus(
    NamedTuple(
        "_UnpartitionedAssetBackfillStatus",
        [("asset_key", AssetKey), ("backfill_status", Optional[AssetBackfillStatus])],
    )
):
    def __new__(cls, asset_key: AssetKey, asset_backfill_status: Optional[AssetBackfillStatus]):
        return super(UnpartitionedAssetBackfillStatus, cls).__new__(
            cls,
            check.inst_param(asset_key, "asset_key", AssetKey),
            check.opt_inst_param(
                asset_backfill_status, "asset_backfill_status", AssetBackfillStatus
            ),
        )


class AssetBackfillData(NamedTuple):
    """Has custom serialization instead of standard Dagster NamedTuple serialization because the
    asset graph is required to build the AssetGraphSubset objects.
    """

    target_subset: AssetGraphSubset
    requested_runs_for_target_roots: bool
    latest_storage_id: Optional[int]
    materialized_subset: AssetGraphSubset
    requested_subset: AssetGraphSubset
    failed_and_downstream_subset: AssetGraphSubset
    backfill_start_time: datetime

    def replace_requested_subset(self, requested_subset: AssetGraphSubset) -> "AssetBackfillData":
        return AssetBackfillData(
            target_subset=self.target_subset,
            latest_storage_id=self.latest_storage_id,
            requested_runs_for_target_roots=self.requested_runs_for_target_roots,
            materialized_subset=self.materialized_subset,
            failed_and_downstream_subset=self.failed_and_downstream_subset,
            requested_subset=requested_subset,
            backfill_start_time=self.backfill_start_time,
        )

    def is_complete(self) -> bool:
        """The asset backfill is complete when all runs to be requested have finished (success,
        failure, or cancellation). Since the AssetBackfillData object stores materialization states
        per asset partition, the daemon continues to update the backfill data until all runs have
        finished in order to display the final partition statuses in the UI.
        """
        return (
            (
                self.materialized_subset | self.failed_and_downstream_subset
            ).num_partitions_and_non_partitioned_assets
            == self.target_subset.num_partitions_and_non_partitioned_assets
        )

    def have_all_requested_runs_finished(self) -> bool:
        for partition in self.requested_subset.iterate_asset_partitions():
            if (
                partition not in self.materialized_subset
                and partition not in self.failed_and_downstream_subset
            ):
                return False

        return True

    def get_target_root_asset_partitions(
        self, instance_queryer: CachingInstanceQueryer
    ) -> Iterable[AssetKeyPartitionKey]:
        def _get_self_and_downstream_targeted_subset(
            initial_subset: AssetGraphSubset,
        ) -> AssetGraphSubset:
            self_and_downstream = initial_subset
            for asset_key in initial_subset.asset_keys:
                self_and_downstream = self_and_downstream | (
                    self.target_subset.asset_graph.bfs_filter_subsets(
                        instance_queryer,
                        lambda asset_key, _: asset_key in self.target_subset,
                        initial_subset.filter_asset_keys({asset_key}),
                        current_time=instance_queryer.evaluation_time,
                    )
                    & self.target_subset
                )
            return self_and_downstream

        assets_with_no_parents_in_target_subset = {
            asset_key
            for asset_key in self.target_subset.asset_keys
            if all(
                parent not in self.target_subset.asset_keys
                for parent in self.target_subset.asset_graph.get_parents(asset_key)
                - {asset_key}  # Do not include an asset as its own parent
            )
        }

        # The partitions that do not have any parents in the target subset
        root_subset = self.target_subset.filter_asset_keys(assets_with_no_parents_in_target_subset)

        # Partitions in root_subset and their downstreams within the target subset
        root_and_downstream_partitions = _get_self_and_downstream_targeted_subset(root_subset)

        # The result of the root_and_downstream_partitions on the previous iteration, used to
        # determine when no new partitions are targeted so we can early exit
        previous_root_and_downstream_partitions = None

        while (
            root_and_downstream_partitions != self.target_subset
            and root_and_downstream_partitions
            != previous_root_and_downstream_partitions  # Check against previous iteration result to exit if no new partitions are targeted
        ):
            # Find the asset graph subset is not yet targeted by the backfill
            unreachable_targets = self.target_subset - root_and_downstream_partitions

            # Find the root assets of the unreachable targets. Any targeted partition in these
            # assets becomes part of the root subset
            unreachable_target_root_subset = unreachable_targets.filter_asset_keys(
                AssetSelection.keys(*unreachable_targets.asset_keys)
                .sources()
                .resolve(unreachable_targets.asset_graph)
            )
            root_subset = root_subset | unreachable_target_root_subset

            # Track the previous value of root_and_downstream_partitions.
            # If the values are the same, we know no new partitions have been targeted.
            previous_root_and_downstream_partitions = root_and_downstream_partitions

            # Update root_and_downstream_partitions to include downstreams of the new root subset
            root_and_downstream_partitions = (
                root_and_downstream_partitions
                | _get_self_and_downstream_targeted_subset(unreachable_target_root_subset)
            )

        if root_and_downstream_partitions == previous_root_and_downstream_partitions:
            raise DagsterInvariantViolationError(
                "Unable to determine root partitions for backfill. The following asset partitions"
                " are not targeted:"
                f" \n\n{list((self.target_subset - root_and_downstream_partitions).iterate_asset_partitions())} \n\n"
                " This is likely a system error. Please report this issue to the Dagster team."
            )

        return list(root_subset.iterate_asset_partitions())

    def get_target_partitions_subset(self, asset_key: AssetKey) -> PartitionsSubset:
        # Return the targeted partitions for the root partitioned asset keys
        return self.target_subset.get_partitions_subset(asset_key)

    def get_target_root_partitions_subset(self) -> PartitionsSubset:
        """Returns the most upstream partitions subset that was targeted by the backfill."""
        partitioned_asset_keys = {
            asset_key
            for asset_key in self.target_subset.asset_keys
            if self.target_subset.asset_graph.get_partitions_def(asset_key) is not None
        }

        root_partitioned_asset_keys = (
            AssetSelection.keys(*partitioned_asset_keys)
            .sources()
            .resolve(self.target_subset.asset_graph)
        )

        # Return the targeted partitions for the root partitioned asset keys
        return self.target_subset.get_partitions_subset(next(iter(root_partitioned_asset_keys)))

    def get_num_partitions(self) -> Optional[int]:
        """Only valid when the same number of partitions are targeted in every asset.

        When not valid, returns None.
        """
        asset_partition_nums = {
            len(subset) for subset in self.target_subset.partitions_subsets_by_asset_key.values()
        }
        if len(asset_partition_nums) == 0:
            return 0
        elif len(asset_partition_nums) == 1:
            return next(iter(asset_partition_nums))
        else:
            return None

    def get_targeted_asset_keys_topological_order(self) -> Sequence[AssetKey]:
        """Returns a topological ordering of asset keys targeted by the backfill
        that exist in the asset graph.

        Orders keys in the same topological level alphabetically.
        """
        toposorted_keys = self.target_subset.asset_graph.toposort_asset_keys()

        targeted_toposorted_keys = []
        for level_keys in toposorted_keys:
            for key in sorted(level_keys):
                if key in self.target_subset.asset_keys:
                    targeted_toposorted_keys.append(key)

        return targeted_toposorted_keys

    def get_backfill_status_per_asset_key(
        self,
    ) -> Sequence[Union[PartitionedAssetBackfillStatus, UnpartitionedAssetBackfillStatus]]:
        """Returns a list containing each targeted asset key's backfill status.
        This list orders assets topologically and only contains statuses for assets that are
        currently existent in the asset graph.
        """

        def _get_status_for_asset_key(
            asset_key: AssetKey,
        ) -> Union[PartitionedAssetBackfillStatus, UnpartitionedAssetBackfillStatus]:
            if self.target_subset.asset_graph.get_partitions_def(asset_key) is not None:
                materialized_subset = self.materialized_subset.get_partitions_subset(asset_key)
                failed_subset = self.failed_and_downstream_subset.get_partitions_subset(asset_key)
                requested_subset = self.requested_subset.get_partitions_subset(asset_key)

                # The failed subset includes partitions that failed and their downstream partitions.
                # The downstream partitions are not included in the requested subset, so we determine
                # the in progress subset by subtracting partitions that are failed and requested.
                requested_and_failed_subset = failed_subset & requested_subset
                in_progress_subset = requested_subset - (
                    requested_and_failed_subset | materialized_subset
                )

                return PartitionedAssetBackfillStatus(
                    asset_key,
                    len(self.target_subset.get_partitions_subset(asset_key)),
                    {
                        AssetBackfillStatus.MATERIALIZED: len(materialized_subset),
                        AssetBackfillStatus.FAILED: len(failed_subset - materialized_subset),
                        AssetBackfillStatus.IN_PROGRESS: len(in_progress_subset),
                    },
                )
            else:
                failed = bool(
                    asset_key in self.failed_and_downstream_subset.non_partitioned_asset_keys
                )
                materialized = bool(
                    asset_key in self.materialized_subset.non_partitioned_asset_keys
                )
                in_progress = bool(asset_key in self.requested_subset.non_partitioned_asset_keys)

                if failed:
                    return UnpartitionedAssetBackfillStatus(asset_key, AssetBackfillStatus.FAILED)
                if materialized:
                    return UnpartitionedAssetBackfillStatus(
                        asset_key, AssetBackfillStatus.MATERIALIZED
                    )
                if in_progress:
                    return UnpartitionedAssetBackfillStatus(
                        asset_key, AssetBackfillStatus.IN_PROGRESS
                    )
                return UnpartitionedAssetBackfillStatus(asset_key, None)

        # Only return back statuses for the assets that still exist in the workspace
        topological_order = self.get_targeted_asset_keys_topological_order()
        return [_get_status_for_asset_key(asset_key) for asset_key in topological_order]

    def get_partition_names(self) -> Optional[Sequence[str]]:
        """Only valid when the same number of partitions are targeted in every asset.

        When not valid, returns None.
        """
        subsets = self.target_subset.partitions_subsets_by_asset_key.values()
        if len(subsets) == 0:
            return []

        first_subset = next(iter(subsets))
        if any(subset != first_subset for subset in subsets):
            return None

        return list(first_subset.get_partition_keys())

    @classmethod
    def empty(
        cls, target_subset: AssetGraphSubset, backfill_start_time: datetime
    ) -> "AssetBackfillData":
        asset_graph = target_subset.asset_graph
        return cls(
            target_subset=target_subset,
            requested_runs_for_target_roots=False,
            requested_subset=AssetGraphSubset(asset_graph),
            materialized_subset=AssetGraphSubset(asset_graph),
            failed_and_downstream_subset=AssetGraphSubset(asset_graph),
            latest_storage_id=None,
            backfill_start_time=backfill_start_time,
        )

    @classmethod
    def is_valid_serialization(cls, serialized: str, asset_graph: AssetGraph) -> bool:
        storage_dict = json.loads(serialized)
        return AssetGraphSubset.can_deserialize(
            storage_dict["serialized_target_subset"], asset_graph
        )

    @classmethod
    def from_serialized(
        cls, serialized: str, asset_graph: AssetGraph, backfill_start_timestamp: float
    ) -> "AssetBackfillData":
        storage_dict = json.loads(serialized)

        return cls(
            target_subset=AssetGraphSubset.from_storage_dict(
                storage_dict["serialized_target_subset"], asset_graph
            ),
            requested_runs_for_target_roots=storage_dict["requested_runs_for_target_roots"],
            requested_subset=AssetGraphSubset.from_storage_dict(
                storage_dict["serialized_requested_subset"], asset_graph
            ),
            materialized_subset=AssetGraphSubset.from_storage_dict(
                storage_dict["serialized_materialized_subset"], asset_graph
            ),
            failed_and_downstream_subset=AssetGraphSubset.from_storage_dict(
                storage_dict["serialized_failed_subset"], asset_graph
            ),
            latest_storage_id=storage_dict["latest_storage_id"],
            backfill_start_time=utc_datetime_from_timestamp(backfill_start_timestamp),
        )

    @classmethod
    def from_partitions_by_assets(
        cls,
        asset_graph: AssetGraph,
        dynamic_partitions_store: DynamicPartitionsStore,
        backfill_start_time: datetime,
        partitions_by_assets: Sequence[PartitionsByAssetSelector],
    ) -> "AssetBackfillData":
        """Create an AssetBackfillData object from a list of PartitionsByAssetSelector objects.
        Accepts a list of asset partitions selections, used to determine the target partitions to backfill.
        For targeted assets, if partitioned and no partitions selections are provided, targets all partitions.
        """
        check.sequence_param(partitions_by_assets, "partitions_by_asset", PartitionsByAssetSelector)

        non_partitioned_asset_keys = set()
        partitions_subsets_by_asset_key = dict()
        for partitions_by_asset_selector in partitions_by_assets:
            asset_key = partitions_by_asset_selector.asset_key
            partitions = partitions_by_asset_selector.partitions
            partition_def = asset_graph.get_partitions_def(asset_key)
            if partitions and partition_def:
                if partitions.partition_range:
                    # a range of partitions is selected
                    partition_keys_in_range = partition_def.get_partition_keys_in_range(
                        partition_key_range=PartitionKeyRange(
                            start=partitions.partition_range.start,
                            end=partitions.partition_range.end,
                        ),
                        dynamic_partitions_store=dynamic_partitions_store,
                    )
                    partition_subset_in_range = partition_def.subset_with_partition_keys(
                        partition_keys_in_range
                    )
                    partitions_subsets_by_asset_key.update({asset_key: partition_subset_in_range})
                else:
                    raise DagsterBackfillFailedError(
                        "partitions_by_asset_selector does not have a partition range selected"
                    )
            elif partition_def:
                # no partitions selected for partitioned asset, we will select all partitions
                all_partitions = partition_def.subset_with_all_partitions()
                partitions_subsets_by_asset_key.update({asset_key: all_partitions})
            else:
                # asset is not partitioned
                non_partitioned_asset_keys.add(asset_key)

        target_subset = AssetGraphSubset(
            asset_graph,
            partitions_subsets_by_asset_key=partitions_subsets_by_asset_key,
            non_partitioned_asset_keys=non_partitioned_asset_keys,
        )
        return cls.empty(target_subset, backfill_start_time)

    @classmethod
    def from_asset_partitions(
        cls,
        asset_graph: AssetGraph,
        partition_names: Optional[Sequence[str]],
        asset_selection: Sequence[AssetKey],
        dynamic_partitions_store: DynamicPartitionsStore,
        backfill_start_time: datetime,
        all_partitions: bool,
    ) -> "AssetBackfillData":
        check.invariant(
            partition_names is None or all_partitions is False,
            "Can't provide both a set of partitions and all_partitions=True",
        )

        if all_partitions:
            target_subset = AssetGraphSubset.from_asset_keys(
                asset_selection, asset_graph, dynamic_partitions_store, backfill_start_time
            )
        elif partition_names is not None:
            partitioned_asset_keys = {
                asset_key
                for asset_key in asset_selection
                if asset_graph.get_partitions_def(asset_key) is not None
            }

            root_partitioned_asset_keys = (
                AssetSelection.keys(*partitioned_asset_keys).sources().resolve(asset_graph)
            )
            root_partitions_defs = {
                asset_graph.get_partitions_def(asset_key)
                for asset_key in root_partitioned_asset_keys
            }
            if len(root_partitions_defs) > 1:
                raise DagsterBackfillFailedError(
                    "All the assets at the root of the backfill must have the same"
                    " PartitionsDefinition"
                )

            root_partitions_def = next(iter(root_partitions_defs))
            if not root_partitions_def:
                raise DagsterBackfillFailedError(
                    "If assets within the backfill have different partitionings, then root assets"
                    " must be partitioned"
                )

            root_partitions_subset = root_partitions_def.subset_with_partition_keys(partition_names)
            target_subset = AssetGraphSubset(
                asset_graph,
                non_partitioned_asset_keys=set(asset_selection) - partitioned_asset_keys,
            )
            for root_asset_key in root_partitioned_asset_keys:
                target_subset |= asset_graph.bfs_filter_subsets(
                    dynamic_partitions_store,
                    lambda asset_key, _: asset_key in partitioned_asset_keys,
                    AssetGraphSubset(
                        asset_graph,
                        partitions_subsets_by_asset_key={root_asset_key: root_partitions_subset},
                    ),
                    current_time=backfill_start_time,
                )
        else:
            check.failed("Either partition_names must not be None or all_partitions must be True")

        return cls.empty(target_subset, backfill_start_time)

    def serialize(self, dynamic_partitions_store: DynamicPartitionsStore) -> str:
        storage_dict = {
            "requested_runs_for_target_roots": self.requested_runs_for_target_roots,
            "serialized_target_subset": self.target_subset.to_storage_dict(
                dynamic_partitions_store=dynamic_partitions_store
            ),
            "latest_storage_id": self.latest_storage_id,
            "serialized_requested_subset": self.requested_subset.to_storage_dict(
                dynamic_partitions_store=dynamic_partitions_store
            ),
            "serialized_materialized_subset": self.materialized_subset.to_storage_dict(
                dynamic_partitions_store=dynamic_partitions_store
            ),
            "serialized_failed_subset": self.failed_and_downstream_subset.to_storage_dict(
                dynamic_partitions_store=dynamic_partitions_store
            ),
        }
        return json.dumps(storage_dict)


def create_asset_backfill_data_from_asset_partitions(
    asset_graph: ExternalAssetGraph,
    asset_selection: Sequence[AssetKey],
    partition_names: Sequence[str],
    dynamic_partitions_store: DynamicPartitionsStore,
) -> AssetBackfillData:
    backfill_timestamp = pendulum.now("UTC").timestamp()
    return AssetBackfillData.from_asset_partitions(
        asset_graph=asset_graph,
        partition_names=partition_names,
        asset_selection=asset_selection,
        dynamic_partitions_store=dynamic_partitions_store,
        all_partitions=False,
        backfill_start_time=utc_datetime_from_timestamp(backfill_timestamp),
    )


def _get_unloadable_location_names(context: IWorkspace, logger: logging.Logger) -> Sequence[str]:
    location_entries_by_name = {
        location_entry.origin.location_name: location_entry
        for location_entry in context.get_workspace_snapshot().values()
    }
    unloadable_location_names = []

    for location_name, location_entry in location_entries_by_name.items():
        if location_entry.load_error:
            logger.warning(
                f"Failure loading location {location_name} due to error:"
                f" {location_entry.load_error}"
            )
            unloadable_location_names.append(location_name)

    return unloadable_location_names


class AssetBackfillIterationResult(NamedTuple):
    run_requests: Sequence[RunRequest]
    backfill_data: AssetBackfillData


def _get_requested_asset_partitions_from_run_requests(
    run_requests: Sequence[RunRequest],
    asset_graph: ExternalAssetGraph,
    instance_queryer: CachingInstanceQueryer,
) -> AbstractSet[AssetKeyPartitionKey]:
    requested_partitions = set()
    for run_request in run_requests:
        # Run request targets a range of partitions
        range_start = run_request.tags.get(ASSET_PARTITION_RANGE_START_TAG)
        range_end = run_request.tags.get(ASSET_PARTITION_RANGE_END_TAG)
        if range_start and range_end:
            # When a run request targets a range of partitions, each asset is expected to
            # have the same partitions def
            selected_assets = cast(Sequence[AssetKey], run_request.asset_selection)
            check.invariant(len(selected_assets) > 0)
            partitions_defs = set(
                asset_graph.get_partitions_def(asset_key) for asset_key in selected_assets
            )
            check.invariant(
                len(partitions_defs) == 1,
                "Expected all assets selected in partition range run request to have the same"
                " partitions def",
            )

            partitions_def = cast(PartitionsDefinition, next(iter(partitions_defs)))
            partitions_in_range = partitions_def.get_partition_keys_in_range(
                PartitionKeyRange(range_start, range_end), instance_queryer
            )
            requested_partitions = requested_partitions | {
                AssetKeyPartitionKey(asset_key, partition_key)
                for asset_key in selected_assets
                for partition_key in partitions_in_range
            }
        else:
            requested_partitions = requested_partitions | {
                AssetKeyPartitionKey(asset_key, run_request.partition_key)
                for asset_key in cast(Sequence[AssetKey], run_request.asset_selection)
            }

    return requested_partitions


def _submit_runs_and_update_backfill_in_chunks(
    instance: DagsterInstance,
    workspace_process_context: IWorkspaceProcessContext,
    backfill_id: str,
    asset_backfill_iteration_result: AssetBackfillIterationResult,
    previous_asset_backfill_data: AssetBackfillData,
    asset_graph: ExternalAssetGraph,
    instance_queryer: CachingInstanceQueryer,
) -> Iterable[Optional[AssetBackfillData]]:
    from dagster._core.execution.backfill import BulkActionStatus, PartitionBackfill

    run_requests = asset_backfill_iteration_result.run_requests
    submitted_partitions = previous_asset_backfill_data.requested_subset

    # Initially, the only requested partitions are the partitions requested during the last
    # backfill iteration
    backfill_data_with_submitted_runs = (
        asset_backfill_iteration_result.backfill_data.replace_requested_subset(submitted_partitions)
    )

    mid_iteration_cancel_requested = False

    # Iterate through runs to request, submitting runs in chunks.
    # In between each chunk, check that the backfill is still marked as 'requested',
    # to ensure that no more runs are requested if the backfill is marked as canceled/canceling.
    unsubmitted_run_request_idx = 0
    pipeline_and_execution_plan_cache: Dict[int, Tuple[ExternalJob, ExternalExecutionPlan]] = {}
    while unsubmitted_run_request_idx < len(run_requests):
        chunk_end_idx = min(unsubmitted_run_request_idx + RUN_CHUNK_SIZE, len(run_requests))
        run_requests_chunk = run_requests[unsubmitted_run_request_idx:chunk_end_idx]

        # Refetch, in case the backfill was requested for cancellation in the meantime
        backfill = cast(PartitionBackfill, instance.get_backfill(backfill_id))
        if backfill.status != BulkActionStatus.REQUESTED:
            mid_iteration_cancel_requested = True
            break

        # Submit runs in the chunk
        for run_request in run_requests_chunk:
            yield None
            submit_run_request(
                run_request=run_request,
                asset_graph=asset_graph,
                # create a new request context for each run in case the code location server
                # is swapped out in the middle of the backfill
                workspace=workspace_process_context.create_request_context(),
                instance=instance,
                pipeline_and_execution_plan_cache=pipeline_and_execution_plan_cache,
            )

        unsubmitted_run_request_idx = chunk_end_idx

        requested_partitions_in_chunk = _get_requested_asset_partitions_from_run_requests(
            run_requests_chunk, asset_graph, instance_queryer
        )
        submitted_partitions = submitted_partitions | requested_partitions_in_chunk

        # AssetBackfillIterationResult contains the requested subset after all runs are submitted.
        # Replace this value with just the partitions that have been submitted so far.
        backfill_data_with_submitted_runs = (
            asset_backfill_iteration_result.backfill_data.replace_requested_subset(
                submitted_partitions
            )
        )

        # Refetch, in case the backfill was requested for cancellation in the meantime
        backfill = cast(PartitionBackfill, instance.get_backfill(backfill_id))
        updated_backfill = backfill.with_asset_backfill_data(
            backfill_data_with_submitted_runs, dynamic_partitions_store=instance
        )
        instance.update_backfill(updated_backfill)

    if not mid_iteration_cancel_requested:
        if submitted_partitions != asset_backfill_iteration_result.backfill_data.requested_subset:
            missing_partitions = list(
                (
                    asset_backfill_iteration_result.backfill_data.requested_subset
                    - submitted_partitions
                ).iterate_asset_partitions()
            )
            check.failed(
                "Did not submit run requests for all expected partitions. \n\nPartitions not"
                f" submitted: {missing_partitions}",
            )

    yield backfill_data_with_submitted_runs


def execute_asset_backfill_iteration(
    backfill: "PartitionBackfill",
    logger: logging.Logger,
    workspace_process_context: IWorkspaceProcessContext,
    instance: DagsterInstance,
) -> Iterable[None]:
    """Runs an iteration of the backfill, including submitting runs and updating the backfill object
    in the DB.

    This is a generator so that we can return control to the daemon and let it heartbeat during
    expensive operations.
    """
    from dagster._core.execution.backfill import BulkActionStatus, PartitionBackfill

    workspace_context = workspace_process_context.create_request_context()
    unloadable_locations = _get_unloadable_location_names(workspace_context, logger)
    asset_graph = ExternalAssetGraph.from_workspace(workspace_context)

    if backfill.serialized_asset_backfill_data is None:
        check.failed("Asset backfill missing serialized_asset_backfill_data")

    try:
        previous_asset_backfill_data = AssetBackfillData.from_serialized(
            backfill.serialized_asset_backfill_data, asset_graph, backfill.backfill_timestamp
        )
    except DagsterDefinitionChangedDeserializationError as ex:
        unloadable_locations_error = (
            "This could be because it's inside a code location that's failing to load:"
            f" {unloadable_locations}"
            if unloadable_locations
            else ""
        )
        if os.environ.get("DAGSTER_BACKFILL_RETRY_DEFINITION_CHANGED_ERROR"):
            logger.warning(
                f"Backfill {backfill.backfill_id} was unable to continue due to a missing asset or"
                " partition in the asset graph. The backfill will resume once it is available"
                f" again.\n{ex}. {unloadable_locations_error}"
            )
            yield None
            return
        else:
            raise DagsterAssetBackfillDataLoadError(f"{ex}. {unloadable_locations_error}")

    backfill_start_time = utc_datetime_from_timestamp(backfill.backfill_timestamp)

    instance_queryer = CachingInstanceQueryer(
        instance=instance, asset_graph=asset_graph, evaluation_time=backfill_start_time
    )

    if backfill.status == BulkActionStatus.REQUESTED:
        result = None
        for result in execute_asset_backfill_iteration_inner(
            backfill_id=backfill.backfill_id,
            asset_backfill_data=previous_asset_backfill_data,
            instance_queryer=instance_queryer,
            asset_graph=asset_graph,
            run_tags=backfill.tags,
            backfill_start_time=backfill_start_time,
        ):
            yield None

        if not isinstance(result, AssetBackfillIterationResult):
            check.failed(
                "Expected execute_asset_backfill_iteration_inner to return an"
                " AssetBackfillIterationResult"
            )

        updated_asset_backfill_data = result.backfill_data

        if result.run_requests:
            for updated_asset_backfill_data in _submit_runs_and_update_backfill_in_chunks(
                instance,
                workspace_process_context,
                backfill.backfill_id,
                result,
                previous_asset_backfill_data,
                asset_graph,
                instance_queryer,
            ):
                yield None

            if not isinstance(updated_asset_backfill_data, AssetBackfillData):
                check.failed(
                    "Expected _submit_runs_and_update_backfill_in_chunks to return an"
                    " AssetBackfillData object"
                )

        # Update the backfill with new asset backfill data
        # Refetch, in case the backfill was canceled in the meantime
        backfill = cast(PartitionBackfill, instance.get_backfill(backfill.backfill_id))
        updated_backfill = backfill.with_asset_backfill_data(
            updated_asset_backfill_data, dynamic_partitions_store=instance
        )
        if updated_asset_backfill_data.is_complete():
            # The asset backfill is complete when all runs to be requested have finished (success,
            # failure, or cancellation). Since the AssetBackfillData object stores materialization states
            # per asset partition, the daemon continues to update the backfill data until all runs have
            # finished in order to display the final partition statuses in the UI.
            updated_backfill = updated_backfill.with_status(BulkActionStatus.COMPLETED)

        instance.update_backfill(updated_backfill)

    elif backfill.status == BulkActionStatus.CANCELING:
        if not instance.run_coordinator:
            check.failed("The instance must have a run coordinator in order to cancel runs")

        # Query for cancelable runs, enforcing a limit on the number of runs to cancel in an iteration
        # as canceling runs incurs cost
        runs_to_cancel_in_iteration = instance.run_storage.get_run_ids(
            filters=RunsFilter(
                statuses=CANCELABLE_RUN_STATUSES,
                tags={
                    BACKFILL_ID_TAG: backfill.backfill_id,
                },
            ),
            limit=MAX_RUNS_CANCELED_PER_ITERATION,
        )

        yield None

        if runs_to_cancel_in_iteration:
            for run_id in runs_to_cancel_in_iteration:
                instance.run_coordinator.cancel_run(run_id)
                yield None

        # Update the asset backfill data to contain the newly materialized/failed partitions.
        updated_asset_backfill_data = None
        for updated_asset_backfill_data in get_canceling_asset_backfill_iteration_data(
            backfill.backfill_id,
            previous_asset_backfill_data,
            instance_queryer,
            asset_graph,
            backfill_start_time,
        ):
            yield None

        if not isinstance(updated_asset_backfill_data, AssetBackfillData):
            check.failed(
                "Expected get_canceling_asset_backfill_iteration_data to return a PartitionBackfill"
            )

        updated_backfill = backfill.with_asset_backfill_data(
            updated_asset_backfill_data, dynamic_partitions_store=instance
        )
        # The asset backfill is successfully canceled when all requested runs have finished (success,
        # failure, or cancellation). Since the AssetBackfillData object stores materialization states
        # per asset partition, the daemon continues to update the backfill data until all runs have
        # finished in order to display the final partition statuses in the UI.
        if updated_asset_backfill_data.have_all_requested_runs_finished():
            updated_backfill = updated_backfill.with_status(BulkActionStatus.CANCELED)

        instance.update_backfill(updated_backfill)
    else:
        check.failed(f"Unexpected backfill status: {backfill.status}")


def get_canceling_asset_backfill_iteration_data(
    backfill_id: str,
    asset_backfill_data: AssetBackfillData,
    instance_queryer: CachingInstanceQueryer,
    asset_graph: ExternalAssetGraph,
    backfill_start_time: datetime,
) -> Iterable[Optional[AssetBackfillData]]:
    """For asset backfills in the "canceling" state, fetch the asset backfill data with the updated
    materialized and failed subsets.
    """
    updated_materialized_subset = None
    for updated_materialized_subset in get_asset_backfill_iteration_materialized_partitions(
        backfill_id, asset_backfill_data, asset_graph, instance_queryer
    ):
        yield None

    if not isinstance(updated_materialized_subset, AssetGraphSubset):
        check.failed(
            "Expected get_asset_backfill_iteration_materialized_partitions to return an"
            " AssetGraphSubset"
        )

    failed_and_downstream_subset = _get_failed_and_downstream_asset_partitions(
        backfill_id,
        asset_backfill_data,
        asset_graph,
        instance_queryer,
        backfill_start_time,
    )
    updated_backfill_data = AssetBackfillData(
        target_subset=asset_backfill_data.target_subset,
        latest_storage_id=asset_backfill_data.latest_storage_id,
        requested_runs_for_target_roots=asset_backfill_data.requested_runs_for_target_roots,
        materialized_subset=updated_materialized_subset,
        failed_and_downstream_subset=failed_and_downstream_subset,
        requested_subset=asset_backfill_data.requested_subset,
        backfill_start_time=backfill_start_time,
    )

    yield updated_backfill_data


def submit_run_request(
    asset_graph: ExternalAssetGraph,
    run_request: RunRequest,
    instance: DagsterInstance,
    workspace: BaseWorkspaceRequestContext,
    pipeline_and_execution_plan_cache: Dict[int, Tuple[ExternalJob, ExternalExecutionPlan]],
) -> None:
    """Creates and submits a run for the given run request."""
    repo_handle = asset_graph.get_repository_handle(
        cast(Sequence[AssetKey], run_request.asset_selection)[0]
    )
    location_name = repo_handle.code_location_origin.location_name
    job_name = _get_implicit_job_name_for_assets(
        asset_graph, cast(Sequence[AssetKey], run_request.asset_selection)
    )
    if job_name is None:
        check.failed(
            "Could not find an implicit asset job for the given assets:"
            f" {run_request.asset_selection}"
        )

    if not run_request.asset_selection:
        check.failed("Expected RunRequest to have an asset selection")

    pipeline_selector = JobSubsetSelector(
        location_name=location_name,
        repository_name=repo_handle.repository_name,
        job_name=job_name,
        asset_selection=run_request.asset_selection,
        op_selection=None,
    )

    selector_id = hash_collection(pipeline_selector)

    if selector_id not in pipeline_and_execution_plan_cache:
        code_location = workspace.get_code_location(repo_handle.code_location_origin.location_name)

        external_job = code_location.get_external_job(pipeline_selector)

        external_execution_plan = code_location.get_external_execution_plan(
            external_job,
            {},
            step_keys_to_execute=None,
            known_state=None,
            instance=instance,
        )
        pipeline_and_execution_plan_cache[selector_id] = (
            external_job,
            external_execution_plan,
        )

    external_job, external_execution_plan = pipeline_and_execution_plan_cache[selector_id]

    run = instance.create_run(
        job_snapshot=external_job.job_snapshot,
        execution_plan_snapshot=external_execution_plan.execution_plan_snapshot,
        parent_job_snapshot=external_job.parent_job_snapshot,
        job_name=external_job.name,
        run_id=None,
        resolved_op_selection=None,
        op_selection=None,
        run_config={},
        step_keys_to_execute=None,
        tags=run_request.tags,
        root_run_id=None,
        parent_run_id=None,
        status=DagsterRunStatus.NOT_STARTED,
        external_job_origin=external_job.get_external_origin(),
        job_code_origin=external_job.get_python_origin(),
        asset_selection=frozenset(run_request.asset_selection),
        asset_check_selection=None,
    )

    instance.submit_run(run.run_id, workspace)


def _get_implicit_job_name_for_assets(
    asset_graph: ExternalAssetGraph, asset_keys: Sequence[AssetKey]
) -> Optional[str]:
    job_names = set(asset_graph.get_materialization_job_names(asset_keys[0]))
    for asset_key in asset_keys[1:]:
        job_names &= set(asset_graph.get_materialization_job_names(asset_key))

    return next(job_name for job_name in job_names if is_base_asset_job_name(job_name))


def get_asset_backfill_iteration_materialized_partitions(
    backfill_id: str,
    asset_backfill_data: AssetBackfillData,
    asset_graph: ExternalAssetGraph,
    instance_queryer: CachingInstanceQueryer,
) -> Iterable[Optional[AssetGraphSubset]]:
    """Returns the partitions that have been materialized by the backfill.

    This function is a generator so we can return control to the daemon and let it heartbeat
    during expensive operations.
    """
    recently_materialized_asset_partitions = AssetGraphSubset(asset_graph)
    for asset_key in asset_backfill_data.target_subset.asset_keys:
        records = instance_queryer.instance.get_event_records(
            EventRecordsFilter(
                event_type=DagsterEventType.ASSET_MATERIALIZATION,
                asset_key=asset_key,
                after_cursor=asset_backfill_data.latest_storage_id,
            )
        )
        records_in_backfill = [
            record
            for record in records
            if instance_queryer.run_has_tag(
                run_id=record.run_id, tag_key=BACKFILL_ID_TAG, tag_value=backfill_id
            )
        ]
        recently_materialized_asset_partitions |= {
            AssetKeyPartitionKey(asset_key, record.partition_key) for record in records_in_backfill
        }

        yield None

    updated_materialized_subset = (
        asset_backfill_data.materialized_subset | recently_materialized_asset_partitions
    )

    yield updated_materialized_subset


def _get_failed_and_downstream_asset_partitions(
    backfill_id: str,
    asset_backfill_data: AssetBackfillData,
    asset_graph: ExternalAssetGraph,
    instance_queryer: CachingInstanceQueryer,
    backfill_start_time: datetime,
) -> AssetGraphSubset:
    failed_and_downstream_subset = AssetGraphSubset.from_asset_partition_set(
        asset_graph.bfs_filter_asset_partitions(
            instance_queryer,
            lambda asset_partitions, _: any(
                asset_partition in asset_backfill_data.target_subset
                for asset_partition in asset_partitions
            ),
            _get_failed_asset_partitions(instance_queryer, backfill_id, asset_graph),
            evaluation_time=backfill_start_time,
        ),
        asset_graph,
    )
    return failed_and_downstream_subset


def execute_asset_backfill_iteration_inner(
    backfill_id: str,
    asset_backfill_data: AssetBackfillData,
    asset_graph: ExternalAssetGraph,
    instance_queryer: CachingInstanceQueryer,
    run_tags: Mapping[str, str],
    backfill_start_time: datetime,
) -> Iterable[Optional[AssetBackfillIterationResult]]:
    """Core logic of a backfill iteration. Has no side effects.

    Computes which runs should be requested, if any, as well as updated bookkeeping about the status
    of asset partitions targeted by the backfill.

    This is a generator so that we can return control to the daemon and let it heartbeat during
    expensive operations.
    """
    initial_candidates: Set[AssetKeyPartitionKey] = set()
    request_roots = not asset_backfill_data.requested_runs_for_target_roots
    if request_roots:
        initial_candidates.update(
            asset_backfill_data.get_target_root_asset_partitions(instance_queryer)
        )

        yield None

        next_latest_storage_id = instance_queryer.instance.event_log_storage.get_maximum_record_id()

        updated_materialized_subset = AssetGraphSubset(asset_graph)
        failed_and_downstream_subset = AssetGraphSubset(asset_graph)
    else:
        target_parent_asset_keys = {
            parent
            for target_asset_key in asset_backfill_data.target_subset.asset_keys
            for parent in asset_graph.get_parents(target_asset_key)
        }
        target_asset_keys_and_parents = (
            asset_backfill_data.target_subset.asset_keys | target_parent_asset_keys
        )
        (
            parent_materialized_asset_partitions,
            next_latest_storage_id,
        ) = instance_queryer.asset_partitions_with_newly_updated_parents_and_new_latest_storage_id(
            target_asset_keys=frozenset(asset_backfill_data.target_subset.asset_keys),
            target_asset_keys_and_parents=frozenset(target_asset_keys_and_parents),
            latest_storage_id=asset_backfill_data.latest_storage_id,
        )
        initial_candidates.update(parent_materialized_asset_partitions)

        yield None

        updated_materialized_subset = None
        for updated_materialized_subset in get_asset_backfill_iteration_materialized_partitions(
            backfill_id, asset_backfill_data, asset_graph, instance_queryer
        ):
            yield None

        if not isinstance(updated_materialized_subset, AssetGraphSubset):
            check.failed(
                "Expected get_asset_backfill_iteration_materialized_partitions to return an"
                " AssetGraphSubset"
            )

        failed_and_downstream_subset = _get_failed_and_downstream_asset_partitions(
            backfill_id, asset_backfill_data, asset_graph, instance_queryer, backfill_start_time
        )

        yield None

    asset_partitions_to_request = asset_graph.bfs_filter_asset_partitions(
        instance_queryer,
        lambda unit, visited: should_backfill_atomic_asset_partitions_unit(
            candidates_unit=unit,
            asset_partitions_to_request=visited,
            asset_graph=asset_graph,
            materialized_subset=updated_materialized_subset,
            requested_subset=asset_backfill_data.requested_subset,
            target_subset=asset_backfill_data.target_subset,
            failed_and_downstream_subset=failed_and_downstream_subset,
            dynamic_partitions_store=instance_queryer,
            current_time=backfill_start_time,
        ),
        initial_asset_partitions=initial_candidates,
        evaluation_time=backfill_start_time,
    )

    # check if all assets have backfill policies if any of them do, otherwise, raise error
    asset_backfill_policies = [
        asset_graph.get_backfill_policy(asset_key)
        for asset_key in {
            asset_partition.asset_key for asset_partition in asset_partitions_to_request
        }
    ]
    all_assets_have_backfill_policies = all(
        backfill_policy is not None for backfill_policy in asset_backfill_policies
    )
    if all_assets_have_backfill_policies:
        run_requests = build_run_requests_with_backfill_policies(
            asset_partitions=asset_partitions_to_request,
            asset_graph=asset_graph,
            run_tags={**run_tags, BACKFILL_ID_TAG: backfill_id},
            dynamic_partitions_store=instance_queryer,
        )
    else:
        if not all(backfill_policy is None for backfill_policy in asset_backfill_policies):
            # if some assets have backfill policies, but not all of them, raise error
            raise DagsterBackfillFailedError(
                "Either all assets must have backfill policies or none of them must have backfill"
                " policies. To backfill these assets together, either add backfill policies to all"
                " assets, or remove backfill policies from all assets."
            )
        # When any of the assets do not have backfill policies, we fall back to the default behavior of
        # backfilling them partition by partition.
        run_requests = build_run_requests(
            asset_partitions=asset_partitions_to_request,
            asset_graph=asset_graph,
            run_tags={**run_tags, BACKFILL_ID_TAG: backfill_id},
        )

    if request_roots:
        check.invariant(
            len(run_requests) > 0,
            "At least one run should be requested on first backfill iteration",
        )

    updated_asset_backfill_data = AssetBackfillData(
        target_subset=asset_backfill_data.target_subset,
        latest_storage_id=next_latest_storage_id or asset_backfill_data.latest_storage_id,
        requested_runs_for_target_roots=asset_backfill_data.requested_runs_for_target_roots
        or request_roots,
        materialized_subset=updated_materialized_subset,
        failed_and_downstream_subset=failed_and_downstream_subset,
        requested_subset=asset_backfill_data.requested_subset | asset_partitions_to_request,
        backfill_start_time=backfill_start_time,
    )
    yield AssetBackfillIterationResult(run_requests, updated_asset_backfill_data)


def should_backfill_atomic_asset_partitions_unit(
    asset_graph: ExternalAssetGraph,
    candidates_unit: Iterable[AssetKeyPartitionKey],
    asset_partitions_to_request: AbstractSet[AssetKeyPartitionKey],
    target_subset: AssetGraphSubset,
    requested_subset: AssetGraphSubset,
    materialized_subset: AssetGraphSubset,
    failed_and_downstream_subset: AssetGraphSubset,
    dynamic_partitions_store: DynamicPartitionsStore,
    current_time: datetime,
) -> bool:
    """Args:
    candidates_unit: A set of asset partitions that must all be materialized if any is
        materialized.
    """
    for candidate in candidates_unit:
        if (
            candidate not in target_subset
            or candidate in failed_and_downstream_subset
            or candidate in materialized_subset
            or candidate in requested_subset
        ):
            return False

        parent_partitions_result = asset_graph.get_parents_partitions(
            dynamic_partitions_store, current_time, *candidate
        )

        if parent_partitions_result.required_but_nonexistent_parents_partitions:
            raise DagsterInvariantViolationError(
                f"Asset partition {candidate}"
                " depends on invalid partition keys"
                f" {parent_partitions_result.required_but_nonexistent_parents_partitions}"
            )

        for parent in parent_partitions_result.parent_partitions:
            can_run_with_parent = (
                parent in asset_partitions_to_request
                and asset_graph.have_same_partitioning(parent.asset_key, candidate.asset_key)
                and parent.partition_key == candidate.partition_key
                and asset_graph.get_repository_handle(candidate.asset_key)
                is asset_graph.get_repository_handle(parent.asset_key)
                and asset_graph.get_backfill_policy(parent.asset_key)
                == asset_graph.get_backfill_policy(candidate.asset_key)
            )

            if (
                parent in target_subset
                and not can_run_with_parent
                and parent not in materialized_subset
            ):
                return False

    return True


def _get_failed_asset_partitions(
    instance_queryer: CachingInstanceQueryer, backfill_id: str, asset_graph: ExternalAssetGraph
) -> Sequence[AssetKeyPartitionKey]:
    """Returns asset partitions that materializations were requested for as part of the backfill, but
    will not be materialized.

    Includes canceled asset partitions. Implementation assumes that successful runs won't have any
    failed partitions.
    """
    runs = instance_queryer.instance.get_runs(
        filters=RunsFilter(
            tags={BACKFILL_ID_TAG: backfill_id},
            statuses=[DagsterRunStatus.CANCELED, DagsterRunStatus.FAILURE],
        )
    )

    result: List[AssetKeyPartitionKey] = []

    for run in runs:
        if (
            run.tags.get(ASSET_PARTITION_RANGE_START_TAG)
            and run.tags.get(ASSET_PARTITION_RANGE_END_TAG)
            and run.tags.get(PARTITION_NAME_TAG) is None
        ):
            # it was a chunked backfill run previously, so we need to reconstruct the partition keys
            planned_asset_keys = instance_queryer.get_planned_materializations_for_run(
                run_id=run.run_id
            )
            completed_asset_keys = instance_queryer.get_current_materializations_for_run(
                run_id=run.run_id
            )
            failed_asset_keys = planned_asset_keys - completed_asset_keys

            if failed_asset_keys:
                partition_range = PartitionKeyRange(
                    start=check.not_none(run.tags.get(ASSET_PARTITION_RANGE_START_TAG)),
                    end=check.not_none(run.tags.get(ASSET_PARTITION_RANGE_END_TAG)),
                )
                for asset_key in failed_asset_keys:
                    result.extend(
                        asset_graph.get_asset_partitions_in_range(
                            asset_key, partition_range, instance_queryer
                        )
                    )
        else:
            # a regular backfill run that run on a single partition
            partition_key = run.tags.get(PARTITION_NAME_TAG)
            planned_asset_keys = instance_queryer.get_planned_materializations_for_run(
                run_id=run.run_id
            )
            completed_asset_keys = instance_queryer.get_current_materializations_for_run(
                run_id=run.run_id
            )
            result.extend(
                AssetKeyPartitionKey(asset_key, partition_key)
                for asset_key in planned_asset_keys - completed_asset_keys
            )

    return result
