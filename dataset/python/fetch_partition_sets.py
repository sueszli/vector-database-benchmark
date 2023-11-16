from collections import defaultdict
from typing import TYPE_CHECKING, Optional, Sequence, Union

import dagster._check as check
from dagster._core.definitions.selector import RepositorySelector
from dagster._core.errors import DagsterUserCodeProcessError
from dagster._core.host_representation import (
    ExternalPartitionSet,
    RepositoryHandle,
)
from dagster._core.host_representation.external_data import (
    ExternalPartitionExecutionErrorData,
    ExternalPartitionNamesData,
)
from dagster._core.storage.dagster_run import DagsterRunStatus, RunPartitionData, RunsFilter
from dagster._core.storage.tags import (
    PARTITION_NAME_TAG,
    PARTITION_SET_TAG,
    REPOSITORY_LABEL_TAG,
    TagType,
    get_tag_type,
)
from dagster._utils.yaml_utils import dump_run_config_yaml

from dagster_graphql.schema.util import ResolveInfo

if TYPE_CHECKING:
    from dagster_graphql.schema.errors import GraphenePartitionSetNotFoundError
    from dagster_graphql.schema.partition_sets import (
        GraphenePartition,
        GraphenePartitionRun,
        GraphenePartitionRunConfig,
        GraphenePartitions,
        GraphenePartitionSet,
        GraphenePartitionSets,
        GraphenePartitionStatus,
        GraphenePartitionStatusCounts,
        GraphenePartitionTags,
    )


def get_partition_sets_or_error(
    graphene_info: ResolveInfo, repository_selector: RepositorySelector, pipeline_name: str
) -> "GraphenePartitionSets":
    from ..schema.partition_sets import GraphenePartitionSet, GraphenePartitionSets

    check.inst_param(repository_selector, "repository_selector", RepositorySelector)
    check.str_param(pipeline_name, "pipeline_name")
    location = graphene_info.context.get_code_location(repository_selector.location_name)
    repository = location.get_repository(repository_selector.repository_name)
    partition_sets = [
        partition_set
        for partition_set in repository.get_external_partition_sets()
        if partition_set.job_name == pipeline_name
    ]

    return GraphenePartitionSets(
        results=[
            GraphenePartitionSet(
                external_repository_handle=repository.handle,
                external_partition_set=partition_set,
            )
            for partition_set in sorted(
                partition_sets,
                key=lambda partition_set: (
                    partition_set.job_name,
                    partition_set.mode,
                    partition_set.name,
                ),
            )
        ]
    )


def get_partition_set(
    graphene_info: ResolveInfo, repository_selector: RepositorySelector, partition_set_name: str
) -> Union["GraphenePartitionSet", "GraphenePartitionSetNotFoundError"]:
    from ..schema.partition_sets import GraphenePartitionSet, GraphenePartitionSetNotFoundError

    check.inst_param(repository_selector, "repository_selector", RepositorySelector)
    check.str_param(partition_set_name, "partition_set_name")
    location = graphene_info.context.get_code_location(repository_selector.location_name)
    repository = location.get_repository(repository_selector.repository_name)
    partition_sets = repository.get_external_partition_sets()
    for partition_set in partition_sets:
        if partition_set.name == partition_set_name:
            return GraphenePartitionSet(
                external_repository_handle=repository.handle,
                external_partition_set=partition_set,
            )

    return GraphenePartitionSetNotFoundError(partition_set_name)


def get_partition_by_name(
    graphene_info: ResolveInfo,
    repository_handle: RepositoryHandle,
    partition_set: ExternalPartitionSet,
    partition_name: str,
) -> "GraphenePartition":
    from ..schema.partition_sets import GraphenePartition

    check.inst_param(repository_handle, "repository_handle", RepositoryHandle)
    check.inst_param(partition_set, "partition_set", ExternalPartitionSet)
    check.str_param(partition_name, "partition_name")
    return GraphenePartition(
        external_repository_handle=repository_handle,
        external_partition_set=partition_set,
        partition_name=partition_name,
    )


def get_partition_config(
    graphene_info: ResolveInfo,
    repository_handle: RepositoryHandle,
    partition_set_name: str,
    partition_name: str,
) -> "GraphenePartitionRunConfig":
    from ..schema.partition_sets import GraphenePartitionRunConfig

    check.inst_param(repository_handle, "repository_handle", RepositoryHandle)
    check.str_param(partition_set_name, "partition_set_name")
    check.str_param(partition_name, "partition_name")

    result = graphene_info.context.get_external_partition_config(
        repository_handle,
        partition_set_name,
        partition_name,
        graphene_info.context.instance,
    )

    if isinstance(result, ExternalPartitionExecutionErrorData):
        raise DagsterUserCodeProcessError.from_error_info(result.error)

    return GraphenePartitionRunConfig(yaml=dump_run_config_yaml(result.run_config))


def get_partition_tags(
    graphene_info: ResolveInfo,
    repository_handle: RepositoryHandle,
    partition_set_name: str,
    partition_name: str,
) -> "GraphenePartitionTags":
    from ..schema.partition_sets import GraphenePartitionTags
    from ..schema.tags import GraphenePipelineTag

    check.inst_param(repository_handle, "repository_handle", RepositoryHandle)
    check.str_param(partition_set_name, "partition_set_name")
    check.str_param(partition_name, "partition_name")

    result = graphene_info.context.get_external_partition_tags(
        repository_handle, partition_set_name, partition_name, graphene_info.context.instance
    )

    if isinstance(result, ExternalPartitionExecutionErrorData):
        raise DagsterUserCodeProcessError.from_error_info(result.error)

    return GraphenePartitionTags(
        results=[
            GraphenePipelineTag(key=key, value=value)
            for key, value in result.tags.items()
            if get_tag_type(key) != TagType.HIDDEN
        ]
    )


def get_partitions(
    graphene_info: ResolveInfo,
    repository_handle: RepositoryHandle,
    partition_set: ExternalPartitionSet,
    cursor: Optional[str] = None,
    limit: Optional[int] = None,
    reverse: bool = False,
) -> "GraphenePartitions":
    from ..schema.partition_sets import GraphenePartition, GraphenePartitions

    check.inst_param(repository_handle, "repository_handle", RepositoryHandle)
    check.inst_param(partition_set, "partition_set", ExternalPartitionSet)
    result = graphene_info.context.get_external_partition_names(
        partition_set, instance=graphene_info.context.instance
    )
    assert isinstance(result, ExternalPartitionNamesData)

    partition_names = _apply_cursor_limit_reverse(result.partition_names, cursor, limit, reverse)

    return GraphenePartitions(
        results=[
            GraphenePartition(
                external_partition_set=partition_set,
                external_repository_handle=repository_handle,
                partition_name=partition_name,
            )
            for partition_name in partition_names
        ]
    )


def _apply_cursor_limit_reverse(
    items: Sequence[str], cursor: Optional[str], limit: Optional[int], reverse: Optional[bool]
) -> Sequence[str]:
    start = 0
    end = len(items)
    index = 0

    if cursor:
        index = next((idx for (idx, item) in enumerate(items) if item == cursor))

        if reverse:
            end = index
        else:
            start = index + 1

    if limit:
        if reverse:
            start = end - limit
        else:
            end = start + limit

    return items[max(start, 0) : end]


def get_partition_set_partition_statuses(
    graphene_info: ResolveInfo, external_partition_set: ExternalPartitionSet
) -> Sequence["GraphenePartitionStatus"]:
    check.inst_param(external_partition_set, "external_partition_set", ExternalPartitionSet)

    repository_handle = external_partition_set.repository_handle
    partition_set_name = external_partition_set.name

    run_partition_data = graphene_info.context.instance.run_storage.get_run_partition_data(
        runs_filter=RunsFilter(
            statuses=[status for status in DagsterRunStatus if status != DagsterRunStatus.CANCELED],
            tags={
                PARTITION_SET_TAG: partition_set_name,
                REPOSITORY_LABEL_TAG: repository_handle.get_external_origin().get_label(),
            },
        )
    )
    names_result = graphene_info.context.get_external_partition_names(
        external_partition_set, graphene_info.context.instance
    )

    if isinstance(names_result, ExternalPartitionExecutionErrorData):
        raise DagsterUserCodeProcessError.from_error_info(names_result.error)

    return partition_statuses_from_run_partition_data(
        partition_set_name, run_partition_data, names_result.partition_names
    )


def partition_statuses_from_run_partition_data(
    partition_set_name: Optional[str],
    run_partition_data: Sequence[RunPartitionData],
    partition_names: Sequence[str],
    backfill_id: Optional[str] = None,
) -> Sequence["GraphenePartitionStatus"]:
    from ..schema.partition_sets import GraphenePartitionStatus, GraphenePartitionStatuses

    partition_data_by_name = {
        partition_data.partition: partition_data for partition_data in run_partition_data
    }

    suffix = f":{backfill_id}" if backfill_id else ""

    results = []
    for name in partition_names:
        partition_id = f'{partition_set_name or "__NO_PARTITION_SET__"}:{name}{suffix}'
        if not partition_data_by_name.get(name):
            results.append(
                GraphenePartitionStatus(
                    id=partition_id,
                    partitionName=name,
                )
            )
            continue
        partition_data = partition_data_by_name[name]
        results.append(
            GraphenePartitionStatus(
                id=partition_id,
                partitionName=name,
                runId=partition_data.run_id,
                runStatus=partition_data.status.value,
                runDuration=(
                    partition_data.end_time - partition_data.start_time
                    if partition_data.end_time and partition_data.start_time
                    else None
                ),
            )
        )

    return GraphenePartitionStatuses(results=results)


def partition_status_counts_from_run_partition_data(
    run_partition_data: Sequence[RunPartitionData], partition_names: Sequence[str]
) -> Sequence["GraphenePartitionStatusCounts"]:
    from ..schema.partition_sets import GraphenePartitionStatusCounts

    partition_data_by_name = {
        partition_data.partition: partition_data for partition_data in run_partition_data
    }

    count_by_status = defaultdict(int)
    for name in partition_names:
        if not partition_data_by_name.get(name):
            count_by_status["NOT_STARTED"] += 1
            continue
        partition_data = partition_data_by_name[name]
        count_by_status[partition_data.status.value] += 1

    return [GraphenePartitionStatusCounts(runStatus=k, count=v) for k, v in count_by_status.items()]


def get_partition_set_partition_runs(
    graphene_info: ResolveInfo, partition_set: ExternalPartitionSet
) -> Sequence["GraphenePartitionRun"]:
    from ..schema.partition_sets import GraphenePartitionRun
    from ..schema.pipelines.pipeline import GrapheneRun

    result = graphene_info.context.get_external_partition_names(
        partition_set, instance=graphene_info.context.instance
    )
    assert isinstance(result, ExternalPartitionNamesData)
    run_records = graphene_info.context.instance.get_run_records(
        RunsFilter(tags={PARTITION_SET_TAG: partition_set.name})
    )

    by_partition = {}
    for record in run_records:
        partition_name = record.dagster_run.tags.get(PARTITION_NAME_TAG)
        if not partition_name or partition_name in by_partition:
            # all_partition_set_runs is in descending order by creation time, we should ignore
            # runs for the same partition if we've already considered the partition
            continue
        by_partition[partition_name] = record

    return [
        GraphenePartitionRun(
            id=f"{partition_set.name}:{partition_name}",
            partitionName=partition_name,
            run=(
                GrapheneRun(by_partition[partition_name])
                if partition_name in by_partition
                else None
            ),
        )
        # for partition_name, run_record in by_partition.items()
        for partition_name in result.partition_names
    ]
