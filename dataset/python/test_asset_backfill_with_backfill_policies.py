import math
from unittest.mock import MagicMock

import pendulum
import pytest
from dagster import (
    BackfillPolicy,
    DagsterInstance,
    DailyPartitionsDefinition,
    DynamicPartitionsDefinition,
    WeeklyPartitionsDefinition,
    asset,
)
from dagster._core.errors import DagsterBackfillFailedError
from dagster._core.execution.asset_backfill import AssetBackfillData, AssetBackfillStatus
from dagster._core.storage.tags import (
    ASSET_PARTITION_RANGE_END_TAG,
    ASSET_PARTITION_RANGE_START_TAG,
)

from dagster_tests.core_tests.execution_tests.test_asset_backfill import (
    execute_asset_backfill_iteration_consume_generator,
    get_asset_graph,
    run_backfill_to_completion,
)


def test_asset_backfill_not_all_asset_have_backfill_policy():
    @asset(backfill_policy=None)
    def unpartitioned_upstream_of_partitioned():
        return 1

    @asset(
        partitions_def=DailyPartitionsDefinition("2023-01-01"),
        backfill_policy=BackfillPolicy.single_run(),
    )
    def upstream_daily_partitioned_asset():
        return 1

    assets_by_repo_name = {
        "repo": [
            unpartitioned_upstream_of_partitioned,
            upstream_daily_partitioned_asset,
        ]
    }
    asset_graph = get_asset_graph(assets_by_repo_name)

    backfill_data = AssetBackfillData.from_asset_partitions(
        partition_names=None,
        asset_graph=asset_graph,
        asset_selection=[
            unpartitioned_upstream_of_partitioned.key,
            upstream_daily_partitioned_asset.key,
        ],
        dynamic_partitions_store=MagicMock(),
        all_partitions=True,
        backfill_start_time=pendulum.now("UTC"),
    )

    with pytest.raises(
        DagsterBackfillFailedError,
        match=(
            "Either all assets must have backfill policies or none of them must have backfill"
            " policies"
        ),
    ):
        execute_asset_backfill_iteration_consume_generator(
            backfill_id="test_backfill_id",
            asset_backfill_data=backfill_data,
            asset_graph=asset_graph,
            instance=DagsterInstance.ephemeral(),
        )


def test_asset_backfill_parent_and_children_have_different_backfill_policy():
    time_now = pendulum.now("UTC")
    daily_partitions_def: DailyPartitionsDefinition = DailyPartitionsDefinition("2023-01-01")

    @asset(partitions_def=daily_partitions_def, backfill_policy=BackfillPolicy.single_run())
    def upstream_daily_partitioned_asset():
        return 1

    @asset(partitions_def=daily_partitions_def, backfill_policy=BackfillPolicy.multi_run())
    def downstream_daily_partitioned_asset(upstream_daily_partitioned_asset):
        return upstream_daily_partitioned_asset + 1

    assets_by_repo_name = {
        "repo": [
            upstream_daily_partitioned_asset,
            downstream_daily_partitioned_asset,
        ]
    }
    asset_graph = get_asset_graph(assets_by_repo_name)

    backfill_id = "test_backfill_id"
    backfill_data = AssetBackfillData.from_asset_partitions(
        partition_names=None,
        asset_graph=asset_graph,
        asset_selection=[
            upstream_daily_partitioned_asset.key,
            downstream_daily_partitioned_asset.key,
        ],
        dynamic_partitions_store=MagicMock(),
        all_partitions=True,
        backfill_start_time=time_now,
    )

    result1 = execute_asset_backfill_iteration_consume_generator(
        backfill_id=backfill_id,
        asset_backfill_data=backfill_data,
        asset_graph=asset_graph,
        instance=DagsterInstance.ephemeral(),
    )
    assert result1.backfill_data != backfill_data
    assert len(result1.run_requests) == 1
    # The first iteration of backfill should only create run request for the upstream asset since
    # the downstream does not have same backfill policy as the upstream.
    assert result1.run_requests[0].asset_selection == [upstream_daily_partitioned_asset.key]


def test_asset_backfill_parent_and_children_have_same_backfill_policy():
    time_now = pendulum.now("UTC")
    daily_partitions_def: DailyPartitionsDefinition = DailyPartitionsDefinition("2023-01-01")

    @asset(backfill_policy=BackfillPolicy.single_run())
    def upstream_non_partitioned_asset():
        return 1

    @asset(partitions_def=daily_partitions_def, backfill_policy=BackfillPolicy.single_run())
    def upstream_daily_partitioned_asset():
        return 1

    @asset(partitions_def=daily_partitions_def, backfill_policy=BackfillPolicy.single_run())
    def downstream_daily_partitioned_asset(upstream_daily_partitioned_asset):
        return upstream_daily_partitioned_asset + 1

    assets_by_repo_name = {
        "repo": [
            upstream_non_partitioned_asset,
            upstream_daily_partitioned_asset,
            downstream_daily_partitioned_asset,
        ]
    }
    asset_graph = get_asset_graph(assets_by_repo_name)

    backfill_data = AssetBackfillData.from_asset_partitions(
        partition_names=None,
        asset_graph=asset_graph,
        asset_selection=[
            upstream_daily_partitioned_asset.key,
            downstream_daily_partitioned_asset.key,
            upstream_non_partitioned_asset.key,
        ],
        dynamic_partitions_store=MagicMock(),
        all_partitions=True,
        backfill_start_time=time_now,
    )

    result = execute_asset_backfill_iteration_consume_generator(
        backfill_id="test_backfill_id",
        asset_backfill_data=backfill_data,
        asset_graph=asset_graph,
        instance=DagsterInstance.ephemeral(),
    )
    assert result.backfill_data != backfill_data
    assert len(result.run_requests) == 2

    for run_request in result.run_requests:
        if run_request.tags.__contains__(ASSET_PARTITION_RANGE_START_TAG):
            # single run request for partitioned asset, both parent and the children somce they share same
            # partitions def and backfill policy
            assert run_request.partition_key is None
            assert upstream_daily_partitioned_asset.key in run_request.asset_selection
            assert downstream_daily_partitioned_asset.key in run_request.asset_selection
            assert run_request.tags.get(ASSET_PARTITION_RANGE_START_TAG) == "2023-01-01"
            assert (
                run_request.tags.get(ASSET_PARTITION_RANGE_END_TAG)
                == daily_partitions_def.get_partition_keys(time_now)[-1]
            )
        else:
            assert run_request.partition_key is None
            assert run_request.asset_selection == [upstream_non_partitioned_asset.key]
            assert run_request.tags.get(ASSET_PARTITION_RANGE_START_TAG) is None
            assert run_request.tags.get(ASSET_PARTITION_RANGE_END_TAG) is None


def test_asset_backfill_return_single_run_request_for_non_partitioned():
    @asset(backfill_policy=BackfillPolicy.single_run())
    def unpartitioned_upstream_of_partitioned():
        return 1

    assets_by_repo_name = {
        "repo": [
            unpartitioned_upstream_of_partitioned,
        ]
    }
    asset_graph = get_asset_graph(assets_by_repo_name)

    backfill_data = AssetBackfillData.from_asset_partitions(
        partition_names=None,
        asset_graph=asset_graph,
        asset_selection=[
            unpartitioned_upstream_of_partitioned.key,
        ],
        dynamic_partitions_store=MagicMock(),
        all_partitions=True,
        backfill_start_time=pendulum.now("UTC"),
    )
    backfill_id = "test_backfill_id"
    result = execute_asset_backfill_iteration_consume_generator(
        backfill_id=backfill_id,
        asset_backfill_data=backfill_data,
        asset_graph=asset_graph,
        instance=DagsterInstance.ephemeral(),
    )
    assert result.backfill_data != backfill_data
    assert len(result.run_requests) == 1
    assert result.run_requests[0].partition_key is None
    assert result.run_requests[0].tags == {"dagster/backfill": backfill_id}


def test_asset_backfill_return_single_run_request_for_partitioned():
    time_now = pendulum.now("UTC")
    daily_partitions_def: DailyPartitionsDefinition = DailyPartitionsDefinition("2023-01-01")

    @asset(partitions_def=daily_partitions_def, backfill_policy=BackfillPolicy.single_run())
    def upstream_daily_partitioned_asset():
        return 1

    assets_by_repo_name = {
        "repo": [
            upstream_daily_partitioned_asset,
        ]
    }
    asset_graph = get_asset_graph(assets_by_repo_name)

    backfill_data = AssetBackfillData.from_asset_partitions(
        partition_names=None,
        asset_graph=asset_graph,
        asset_selection=[
            upstream_daily_partitioned_asset.key,
        ],
        dynamic_partitions_store=MagicMock(),
        all_partitions=True,
        backfill_start_time=time_now,
    )

    result = execute_asset_backfill_iteration_consume_generator(
        backfill_id="test_backfill_id",
        asset_backfill_data=backfill_data,
        asset_graph=asset_graph,
        instance=DagsterInstance.ephemeral(),
    )
    assert result.backfill_data != backfill_data
    assert len(result.run_requests) == 1
    assert result.run_requests[0].partition_key is None
    assert result.run_requests[0].tags.get(ASSET_PARTITION_RANGE_START_TAG) == "2023-01-01"
    assert (
        result.run_requests[0].tags.get(ASSET_PARTITION_RANGE_END_TAG)
        == daily_partitions_def.get_partition_keys(time_now)[-1]
    )


def test_asset_backfill_return_multiple_run_request_for_partitioned():
    time_now = pendulum.now("UTC")
    daily_partitions_def: DailyPartitionsDefinition = DailyPartitionsDefinition(
        "2023-01-01", end_date="2023-08-11"
    )
    num_of_daily_partitions = daily_partitions_def.get_num_partitions(time_now)

    @asset(partitions_def=daily_partitions_def, backfill_policy=BackfillPolicy.multi_run(7))
    def upstream_daily_partitioned_asset():
        return 1

    assets_by_repo_name = {
        "repo": [
            upstream_daily_partitioned_asset,
        ]
    }
    asset_graph = get_asset_graph(assets_by_repo_name)

    backfill_data = AssetBackfillData.from_asset_partitions(
        partition_names=None,
        asset_graph=asset_graph,
        asset_selection=[
            upstream_daily_partitioned_asset.key,
        ],
        dynamic_partitions_store=MagicMock(),
        all_partitions=True,
        backfill_start_time=time_now,
    )

    result = execute_asset_backfill_iteration_consume_generator(
        backfill_id="test_backfill_id",
        asset_backfill_data=backfill_data,
        asset_graph=asset_graph,
        instance=DagsterInstance.ephemeral(),
    )
    assert result.backfill_data != backfill_data
    assert len(result.run_requests) == math.ceil(num_of_daily_partitions / 7)
    assert result.run_requests[0].partition_key is None
    assert result.run_requests[0].tags.get(ASSET_PARTITION_RANGE_START_TAG) == "2023-01-01"
    assert (
        result.run_requests[-1].tags.get(ASSET_PARTITION_RANGE_END_TAG)
        == daily_partitions_def.get_partition_keys(time_now)[-1]
    )


def test_asset_backfill_status_count_with_backfill_policies():
    daily_partitions_def: DailyPartitionsDefinition = DailyPartitionsDefinition("2023-01-01")
    weekly_partitions_def = WeeklyPartitionsDefinition("2023-01-01")

    time_now = pendulum.now("UTC")
    num_of_daily_partitions = daily_partitions_def.get_num_partitions(time_now)
    num_of_weekly_partitions = weekly_partitions_def.get_num_partitions(time_now)

    @asset(backfill_policy=BackfillPolicy.single_run())
    def unpartitioned_upstream_of_partitioned():
        return 1

    @asset(
        partitions_def=daily_partitions_def,
        backfill_policy=BackfillPolicy.single_run(),
        deps={unpartitioned_upstream_of_partitioned},
    )
    def upstream_daily_partitioned_asset():
        return 2

    @asset(
        partitions_def=weekly_partitions_def,
        backfill_policy=BackfillPolicy.single_run(),
        deps={upstream_daily_partitioned_asset},
    )
    def downstream_weekly_partitioned_asset():
        return 3

    assets_by_repo_name = {
        "repo": [
            unpartitioned_upstream_of_partitioned,
            upstream_daily_partitioned_asset,
            downstream_weekly_partitioned_asset,
        ]
    }
    asset_graph = get_asset_graph(assets_by_repo_name)
    instance = DagsterInstance.ephemeral()

    # Construct a backfill data with all_partitions=True on assets with single run backfill policies.
    backfill_data = AssetBackfillData.from_asset_partitions(
        partition_names=None,
        asset_graph=asset_graph,
        asset_selection=[
            unpartitioned_upstream_of_partitioned.key,
            upstream_daily_partitioned_asset.key,
            downstream_weekly_partitioned_asset.key,
        ],
        dynamic_partitions_store=MagicMock(),
        all_partitions=True,
        backfill_start_time=time_now,
    )

    (
        completed_backfill_data,
        requested_asset_partitions,
        fail_and_downstream_asset_partitions,
    ) = run_backfill_to_completion(
        instance=instance,
        asset_graph=asset_graph,
        assets_by_repo_name=assets_by_repo_name,
        backfill_data=backfill_data,
        fail_asset_partitions=set(),
    )

    counts = completed_backfill_data.get_backfill_status_per_asset_key()

    assert counts[0].asset_key == unpartitioned_upstream_of_partitioned.key
    assert counts[0].backfill_status == AssetBackfillStatus.MATERIALIZED

    assert counts[1].asset_key == upstream_daily_partitioned_asset.key
    assert (
        counts[1].partitions_counts_by_status[AssetBackfillStatus.MATERIALIZED]
        == num_of_daily_partitions
    )
    assert counts[1].num_targeted_partitions == num_of_daily_partitions

    assert counts[2].asset_key == downstream_weekly_partitioned_asset.key
    assert (
        counts[2].partitions_counts_by_status[AssetBackfillStatus.MATERIALIZED]
        == num_of_weekly_partitions
    )
    assert counts[2].num_targeted_partitions == num_of_weekly_partitions


def test_backfill_run_contains_more_than_one_asset():
    upstream_partitions_def: DailyPartitionsDefinition = DailyPartitionsDefinition("2023-01-01")
    downstream_partitions_def: DailyPartitionsDefinition = DailyPartitionsDefinition("2023-01-02")

    time_now = pendulum.now("UTC")
    upstream_num_of_partitions = upstream_partitions_def.get_num_partitions(time_now)
    downstream_num_of_partitions = downstream_partitions_def.get_num_partitions(time_now)

    @asset(partitions_def=upstream_partitions_def, backfill_policy=BackfillPolicy.single_run())
    def upstream_a():
        return 1

    @asset(partitions_def=upstream_partitions_def, backfill_policy=BackfillPolicy.single_run())
    def upstream_b():
        return 2

    @asset(
        partitions_def=downstream_partitions_def,
        backfill_policy=BackfillPolicy.single_run(),
        deps={"upstream_a"},
    )
    def downstream_a():
        return 1

    @asset(
        partitions_def=downstream_partitions_def,
        backfill_policy=BackfillPolicy.single_run(),
        deps={"upstream_b"},
    )
    def downstream_b():
        return 2

    assets_by_repo_name = {"repo": [upstream_a, upstream_b, downstream_a, downstream_b]}
    asset_graph = get_asset_graph(assets_by_repo_name)
    instance = DagsterInstance.ephemeral()

    backfill_data = AssetBackfillData.from_asset_partitions(
        partition_names=None,
        asset_graph=asset_graph,
        asset_selection=[
            upstream_a.key,
            upstream_b.key,
            downstream_a.key,
            downstream_b.key,
        ],
        dynamic_partitions_store=MagicMock(),
        all_partitions=True,
        backfill_start_time=time_now,
    )

    (
        completed_backfill_data,
        _,
        _,
    ) = run_backfill_to_completion(
        instance=instance,
        asset_graph=asset_graph,
        assets_by_repo_name=assets_by_repo_name,
        backfill_data=backfill_data,
        fail_asset_partitions=set(),
    )

    counts = completed_backfill_data.get_backfill_status_per_asset_key()

    assert counts[0].asset_key == upstream_a.key
    assert (
        counts[0].partitions_counts_by_status[AssetBackfillStatus.MATERIALIZED]
        == upstream_num_of_partitions
    )
    assert counts[0].partitions_counts_by_status[AssetBackfillStatus.FAILED] == 0
    assert counts[0].partitions_counts_by_status[AssetBackfillStatus.IN_PROGRESS] == 0
    assert counts[0].num_targeted_partitions == upstream_num_of_partitions

    assert counts[1].asset_key == upstream_b.key
    assert (
        counts[1].partitions_counts_by_status[AssetBackfillStatus.MATERIALIZED]
        == upstream_num_of_partitions
    )
    assert counts[1].partitions_counts_by_status[AssetBackfillStatus.FAILED] == 0
    assert counts[1].partitions_counts_by_status[AssetBackfillStatus.IN_PROGRESS] == 0
    assert counts[1].num_targeted_partitions == upstream_num_of_partitions

    assert counts[2].asset_key == downstream_a.key
    assert (
        counts[2].partitions_counts_by_status[AssetBackfillStatus.MATERIALIZED]
        == downstream_num_of_partitions
    )
    assert counts[2].partitions_counts_by_status[AssetBackfillStatus.FAILED] == 0
    assert counts[2].partitions_counts_by_status[AssetBackfillStatus.IN_PROGRESS] == 0
    assert counts[2].num_targeted_partitions == downstream_num_of_partitions

    assert counts[3].asset_key == downstream_b.key
    assert (
        counts[3].partitions_counts_by_status[AssetBackfillStatus.MATERIALIZED]
        == downstream_num_of_partitions
    )
    assert counts[3].partitions_counts_by_status[AssetBackfillStatus.FAILED] == 0
    assert counts[3].partitions_counts_by_status[AssetBackfillStatus.IN_PROGRESS] == 0
    assert counts[3].num_targeted_partitions == downstream_num_of_partitions


def test_dynamic_partitions():
    @asset(
        backfill_policy=BackfillPolicy.single_run(),
        partitions_def=DynamicPartitionsDefinition(name="apple"),
    )
    def asset1() -> None:
        ...

    assets_by_repo_name = {"repo": [asset1]}
    asset_graph = get_asset_graph(assets_by_repo_name)

    instance = DagsterInstance.ephemeral()
    instance.add_dynamic_partitions("apple", ["foo", "bar"])

    backfill_data = AssetBackfillData.from_asset_partitions(
        asset_graph=asset_graph,
        asset_selection=[asset1.key],
        dynamic_partitions_store=instance,
        partition_names=["foo", "bar"],
        backfill_start_time=pendulum.now("UTC"),
        all_partitions=False,
    )

    result = execute_asset_backfill_iteration_consume_generator(
        backfill_id="test_backfill_id",
        asset_backfill_data=backfill_data,
        asset_graph=asset_graph,
        instance=instance,
    )
    assert result.backfill_data != backfill_data
    assert len(result.run_requests) == 1
    assert result.run_requests[0].tags.get(ASSET_PARTITION_RANGE_START_TAG) == "foo"
    assert result.run_requests[0].tags.get(ASSET_PARTITION_RANGE_END_TAG) == "bar"
