from datetime import datetime, timezone
from typing import Optional, Sequence

import pytest
from dagster import (
    DailyPartitionsDefinition,
    HourlyPartitionsDefinition,
    MonthlyPartitionsDefinition,
    TimeWindowPartitionMapping,
    TimeWindowPartitionsDefinition,
    WeeklyPartitionsDefinition,
)
from dagster._core.definitions.partition_key_range import PartitionKeyRange
from dagster._core.definitions.time_window_partitions import BaseTimeWindowPartitionsSubset


def subset_with_keys(partitions_def: TimeWindowPartitionsDefinition, keys: Sequence[str]):
    return partitions_def.empty_subset().with_partition_keys(keys)


def subset_with_key_range(partitions_def: TimeWindowPartitionsDefinition, start: str, end: str):
    return partitions_def.empty_subset().with_partition_keys(
        partitions_def.get_partition_keys_in_range(PartitionKeyRange(start, end))
    )


def test_get_upstream_partitions_for_partition_range_same_partitioning():
    downstream_partitions_def = DailyPartitionsDefinition(start_date="2021-05-05")
    upstream_partitions_def = DailyPartitionsDefinition(start_date="2021-05-05")
    # single partition key
    result = TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
        subset_with_keys(downstream_partitions_def, ["2021-05-07"]),
        upstream_partitions_def,
    )
    assert result.partitions_subset == upstream_partitions_def.empty_subset().with_partition_keys(
        ["2021-05-07"]
    )

    # range of partition keys
    result = TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-07", "2021-05-09"),
        upstream_partitions_def,
    )
    assert result.partitions_subset == subset_with_key_range(
        upstream_partitions_def, "2021-05-07", "2021-05-09"
    )


def test_get_upstream_partitions_for_partition_range_same_partitioning_different_formats():
    downstream_partitions_def = DailyPartitionsDefinition(start_date="2021-05-05")
    upstream_partitions_def = DailyPartitionsDefinition(start_date="2021/05/05", fmt="%Y/%m/%d")

    result = TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-07", "2021-05-09"),
        upstream_partitions_def,
    )
    assert result.partitions_subset == subset_with_key_range(
        upstream_partitions_def, "2021/05/07", "2021/05/09"
    )
    assert (
        result.partitions_subset.get_partition_keys()
        == upstream_partitions_def.get_partition_keys_in_range(
            PartitionKeyRange("2021/05/07", "2021/05/09")
        )
    )


def test_get_upstream_partitions_for_partition_range_hourly_downstream_daily_upstream():
    downstream_partitions_def = HourlyPartitionsDefinition(start_date="2021-05-05-00:00")
    upstream_partitions_def = DailyPartitionsDefinition(start_date="2021-05-05")
    result = TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
        subset_with_keys(downstream_partitions_def, ["2021-05-07-05:00"]),
        upstream_partitions_def,
    )
    assert result.partitions_subset == upstream_partitions_def.empty_subset().with_partition_keys(
        ["2021-05-07"]
    )

    result = TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-07-05:00", "2021-05-09-09:00"),
        upstream_partitions_def,
    )
    assert (
        result.partitions_subset.get_partition_keys()
        == upstream_partitions_def.get_partition_keys_in_range(
            PartitionKeyRange("2021-05-07", "2021-05-09")
        )
    )


def test_get_upstream_partitions_for_partition_range_daily_downstream_hourly_upstream():
    downstream_partitions_def = DailyPartitionsDefinition(start_date="2021-05-05")
    upstream_partitions_def = HourlyPartitionsDefinition(start_date="2021-05-05-00:00")
    result = TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
        subset_with_keys(downstream_partitions_def, ["2021-05-07"]),
        upstream_partitions_def,
    )
    assert (
        result.partitions_subset.get_partition_keys()
        == upstream_partitions_def.get_partition_keys_in_range(
            PartitionKeyRange("2021-05-07-00:00", "2021-05-07-23:00")
        )
    )

    result = TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-07", "2021-05-09"),
        upstream_partitions_def,
    )
    assert (
        result.partitions_subset.get_partition_keys()
        == upstream_partitions_def.get_partition_keys_in_range(
            PartitionKeyRange("2021-05-07-00:00", "2021-05-09-23:00")
        )
    )


def test_get_upstream_partitions_for_partition_range_monthly_downstream_daily_upstream():
    downstream_partitions_def = MonthlyPartitionsDefinition(start_date="2021-05-01")
    upstream_partitions_def = DailyPartitionsDefinition(start_date="2021-05-01")
    result = TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-01", "2021-07-01"),
        upstream_partitions_def,
    )
    assert (
        result.partitions_subset.get_partition_keys()
        == upstream_partitions_def.get_partition_keys_in_range(
            PartitionKeyRange("2021-05-01", "2021-07-31")
        )
    )


def test_get_upstream_partitions_for_partition_range_twice_daily_downstream_daily_upstream():
    start = datetime(year=2020, month=1, day=5)
    downstream_partitions_def = TimeWindowPartitionsDefinition(
        cron_schedule="0 0 * * *", start=start, fmt="%Y-%m-%d"
    )
    upstream_partitions_def = TimeWindowPartitionsDefinition(
        cron_schedule="0 0,11 * * *", start=start, fmt="%Y-%m-%d %H:%M"
    )
    result = TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-01", "2021-05-03"),
        upstream_partitions_def,
    )
    assert (
        result.partitions_subset.get_partition_keys()
        == upstream_partitions_def.get_partition_keys_in_range(
            PartitionKeyRange("2021-05-01 00:00", "2021-05-03 11:00")
        )
    )


def test_get_upstream_partitions_for_partition_range_daily_downstream_twice_daily_upstream():
    start = datetime(year=2020, month=1, day=5)
    downstream_partitions_def = TimeWindowPartitionsDefinition(
        cron_schedule="0 0,11 * * *", start=start, fmt="%Y-%m-%d %H:%M"
    )
    upstream_partitions_def = TimeWindowPartitionsDefinition(
        cron_schedule="0 0 * * *", start=start, fmt="%Y-%m-%d"
    )
    result = TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-01 00:00", "2021-05-03 00:00"),
        upstream_partitions_def,
    )
    assert (
        result.partitions_subset.get_partition_keys()
        == upstream_partitions_def.get_partition_keys_in_range(
            PartitionKeyRange("2021-05-01", "2021-05-03")
        )
    )


def test_get_upstream_partitions_for_partition_range_daily_non_aligned():
    start = datetime(year=2020, month=1, day=5)
    downstream_partitions_def = TimeWindowPartitionsDefinition(
        cron_schedule="0 0 * * *", start=start, fmt="%Y-%m-%d"
    )
    upstream_partitions_def = TimeWindowPartitionsDefinition(
        cron_schedule="0 11 * * *", start=start, fmt="%Y-%m-%d"
    )
    result = TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-02", "2021-05-04"),
        upstream_partitions_def,
    )
    assert (
        result.partitions_subset.get_partition_keys()
        == upstream_partitions_def.get_partition_keys_in_range(
            PartitionKeyRange("2021-05-01", "2021-05-04")
        )
    )


def test_get_upstream_partitions_for_partition_range_weekly_with_offset():
    partitions_def = WeeklyPartitionsDefinition(
        start_date="2022-09-04", day_offset=0, hour_offset=10
    )

    result = TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(partitions_def, "2022-09-11", "2022-09-11"),
        partitions_def,
    )
    assert result.partitions_subset.get_partition_keys() == (
        partitions_def.get_partition_keys_in_range(PartitionKeyRange("2022-09-11", "2022-09-11"))
    )


def test_daily_to_daily_lag():
    downstream_partitions_def = upstream_partitions_def = DailyPartitionsDefinition(
        start_date="2021-05-05"
    )
    mapping = TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)

    # single partition key
    assert mapping.get_upstream_mapped_partitions_result_for_partitions(
        subset_with_keys(downstream_partitions_def, ["2021-05-07"]), upstream_partitions_def
    ).partitions_subset.get_partition_keys() == ["2021-05-06"]

    assert mapping.get_downstream_partitions_for_partitions(
        subset_with_keys(upstream_partitions_def, ["2021-05-06"]), downstream_partitions_def
    ).get_partition_keys() == ["2021-05-07"]

    # first partition key
    assert (
        mapping.get_upstream_mapped_partitions_result_for_partitions(
            subset_with_keys(downstream_partitions_def, ["2021-05-05"]), upstream_partitions_def
        ).partitions_subset.get_partition_keys()
        == []
    )

    # range of partition keys
    assert mapping.get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-07", "2021-05-09"),
        upstream_partitions_def,
    ).partitions_subset.get_partition_keys() == ["2021-05-06", "2021-05-07", "2021-05-08"]

    assert mapping.get_downstream_partitions_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-06", "2021-05-08"),
        downstream_partitions_def,
    ).get_partition_keys() == ["2021-05-07", "2021-05-08", "2021-05-09"]

    # range overlaps start
    assert mapping.get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-05", "2021-05-07"),
        upstream_partitions_def,
    ).partitions_subset.get_partition_keys() == ["2021-05-05", "2021-05-06"]


def test_exotic_cron_schedule_lag():
    # every 4 hours
    downstream_partitions_def = upstream_partitions_def = TimeWindowPartitionsDefinition(
        start="2021-05-05_00", cron_schedule="0 */4 * * *", fmt="%Y-%m-%d_%H"
    )
    mapping = TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)
    # single partition key
    assert mapping.get_upstream_mapped_partitions_result_for_partitions(
        subset_with_keys(downstream_partitions_def, ["2021-05-06_04"]), upstream_partitions_def
    ).partitions_subset.get_partition_keys() == ["2021-05-06_00"]

    assert mapping.get_downstream_partitions_for_partitions(
        subset_with_keys(upstream_partitions_def, ["2021-05-06_00"]), downstream_partitions_def
    ).get_partition_keys() == ["2021-05-06_04"]

    # first partition key
    assert (
        mapping.get_upstream_mapped_partitions_result_for_partitions(
            subset_with_keys(downstream_partitions_def, ["2021-05-05_00"]), upstream_partitions_def
        ).partitions_subset.get_partition_keys()
        == []
    )

    # range of partition keys
    assert mapping.get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-07_04", "2021-05-07_12"),
        upstream_partitions_def,
    ).partitions_subset.get_partition_keys() == ["2021-05-07_00", "2021-05-07_04", "2021-05-07_08"]

    assert mapping.get_downstream_partitions_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-07_04", "2021-05-07_12"),
        downstream_partitions_def,
    ).get_partition_keys() == ["2021-05-07_08", "2021-05-07_12", "2021-05-07_16"]

    # range overlaps start
    assert mapping.get_upstream_mapped_partitions_result_for_partitions(
        subset_with_key_range(downstream_partitions_def, "2021-05-05_00", "2021-05-05_08"),
        upstream_partitions_def,
    ).partitions_subset.get_partition_keys() == ["2021-05-05_00", "2021-05-05_04"]


def test_daily_to_daily_lag_different_start_date():
    upstream_partitions_def = DailyPartitionsDefinition(start_date="2021-05-05")
    downstream_partitions_def = DailyPartitionsDefinition(start_date="2021-05-06")
    mapping = TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)

    assert mapping.get_upstream_mapped_partitions_result_for_partitions(
        subset_with_keys(downstream_partitions_def, ["2021-05-06"]), upstream_partitions_def
    ).partitions_subset.get_partition_keys() == ["2021-05-05"]

    assert mapping.get_downstream_partitions_for_partitions(
        subset_with_keys(upstream_partitions_def, ["2021-05-05"]), downstream_partitions_def
    ).get_partition_keys() == ["2021-05-06"]


def test_daily_to_daily_many_to_one():
    upstream_partitions_def = DailyPartitionsDefinition(start_date="2021-05-05")
    downstream_partitions_def = DailyPartitionsDefinition(start_date="2021-05-06")
    mapping = TimeWindowPartitionMapping(start_offset=-1)

    assert mapping.get_upstream_mapped_partitions_result_for_partitions(
        subset_with_keys(downstream_partitions_def, ["2022-07-04"]), upstream_partitions_def
    ).partitions_subset.get_partition_keys() == ["2022-07-03", "2022-07-04"]

    assert mapping.get_upstream_mapped_partitions_result_for_partitions(
        subset_with_keys(downstream_partitions_def, ["2022-07-04", "2022-07-05"]),
        upstream_partitions_def,
    ).partitions_subset.get_partition_keys() == ["2022-07-03", "2022-07-04", "2022-07-05"]

    assert mapping.get_downstream_partitions_for_partitions(
        subset_with_keys(upstream_partitions_def, ["2022-07-03", "2022-07-04"]),
        downstream_partitions_def,
    ).get_partition_keys() == ["2022-07-03", "2022-07-04", "2022-07-05"]

    assert mapping.get_downstream_partitions_for_partitions(
        subset_with_keys(upstream_partitions_def, ["2022-07-03"]), downstream_partitions_def
    ).get_partition_keys() == ["2022-07-03", "2022-07-04"]

    assert mapping.get_downstream_partitions_for_partitions(
        subset_with_keys(upstream_partitions_def, ["2022-07-04"]), downstream_partitions_def
    ).get_partition_keys() == ["2022-07-04", "2022-07-05"]


@pytest.mark.parametrize(
    "upstream_partitions_def,downstream_partitions_def,upstream_keys,expected_downstream_keys,current_time",
    [
        (
            HourlyPartitionsDefinition(start_date="2021-05-05-00:00"),
            DailyPartitionsDefinition(start_date="2021-05-05"),
            ["2021-05-05-00:00"],
            [],
            datetime(2021, 5, 5, 1),
        ),
        (
            HourlyPartitionsDefinition(start_date="2021-05-05-00:00"),
            DailyPartitionsDefinition(start_date="2021-05-05"),
            ["2021-05-05-23:00", "2021-05-06-00:00", "2021-05-06-01:00"],
            ["2021-05-05"],
            datetime(2021, 5, 6, 1),
        ),
        (
            HourlyPartitionsDefinition(start_date="2021-05-05-00:00"),
            DailyPartitionsDefinition(start_date="2021-05-05"),
            ["2021-05-05-23:00", "2021-05-06-00:00", "2021-05-06-01:00"],
            ["2021-05-05", "2021-05-06"],
            None,
        ),
        (
            HourlyPartitionsDefinition(start_date="2021-05-05-00:00", timezone="US/Central"),
            DailyPartitionsDefinition(start_date="2021-05-05", timezone="US/Central"),
            ["2021-05-05-23:00", "2021-05-06-00:00", "2021-05-06-01:00"],
            ["2021-05-05"],
            datetime(2021, 5, 6, 6, tzinfo=timezone.utc),
        ),
        (
            HourlyPartitionsDefinition(start_date="2021-05-05-00:00"),
            DailyPartitionsDefinition(start_date="2021-05-05", end_offset=1),
            ["2021-05-05-23:00", "2021-05-06-00:00", "2021-05-06-01:00"],
            ["2021-05-05", "2021-05-06"],
            datetime(2021, 5, 6, 1),
        ),
        (
            DailyPartitionsDefinition(start_date="2022-01-01"),
            DailyPartitionsDefinition(start_date="2021-01-01"),
            ["2022-12-30"],
            ["2022-12-30"],
            datetime(2022, 12, 31, 1),
        ),
    ],
)
def test_get_downstream_with_current_time(
    upstream_partitions_def: TimeWindowPartitionsDefinition,
    downstream_partitions_def: TimeWindowPartitionsDefinition,
    upstream_keys: Sequence[str],
    expected_downstream_keys: Sequence[str],
    current_time: Optional[datetime],
):
    mapping = TimeWindowPartitionMapping()
    assert (
        mapping.get_downstream_partitions_for_partitions(
            subset_with_keys(upstream_partitions_def, upstream_keys),
            downstream_partitions_def,
            current_time=current_time,
        ).get_partition_keys()
        == expected_downstream_keys
    )


@pytest.mark.parametrize(
    "upstream_partitions_def,downstream_partitions_def,expected_upstream_keys,downstream_keys,current_time,invalid_upstream_keys",
    [
        (
            DailyPartitionsDefinition(start_date="2021-05-05"),
            HourlyPartitionsDefinition(start_date="2021-05-05-00:00"),
            [],
            ["2021-06-01-00:00"],
            datetime(2021, 6, 1, 1),
            ["2021-06-01"],
        ),
        (
            DailyPartitionsDefinition(start_date="2021-05-05"),
            HourlyPartitionsDefinition(start_date="2021-05-05-00:00"),
            ["2021-05-05"],
            ["2021-05-05-23:00", "2021-05-06-00:00", "2021-05-06-01:00"],
            datetime(2021, 5, 6, 1),
            ["2021-05-06"],
        ),
        (
            DailyPartitionsDefinition(start_date="2021-05-05"),
            HourlyPartitionsDefinition(start_date="2021-05-05-00:00"),
            ["2021-05-05"],
            ["2021-05-05-23:00"],
            datetime(2021, 5, 6, 1),
            [],
        ),
        (
            DailyPartitionsDefinition(start_date="2021-05-05", timezone="US/Central"),
            HourlyPartitionsDefinition(start_date="2021-05-05-00:00", timezone="US/Central"),
            ["2021-05-05"],
            ["2021-05-05-23:00"],
            datetime(2021, 5, 6, 5, tzinfo=timezone.utc),  # 2021-05-06-00:00 in US/Central
            [],
        ),
        (
            DailyPartitionsDefinition(start_date="2021-05-05", timezone="US/Central"),
            HourlyPartitionsDefinition(start_date="2021-05-05-00:00", timezone="US/Central"),
            [],
            ["2021-05-05-22:00"],
            datetime(2021, 5, 6, 4, tzinfo=timezone.utc),  # 2021-05-05-23:00 in US/Central
            ["2021-05-05"],
        ),
        (
            DailyPartitionsDefinition(start_date="2021-05-05", end_offset=1),
            HourlyPartitionsDefinition(start_date="2021-05-05-00:00"),
            ["2021-05-05", "2021-05-06"],
            ["2021-05-05-23:00", "2021-05-06-00:00", "2021-05-06-01:00"],
            datetime(2021, 5, 6, 1),
            [],
        ),
        (
            DailyPartitionsDefinition(start_date="2022-01-01"),
            DailyPartitionsDefinition(start_date="2021-01-01"),
            [],
            ["2021-06-06"],
            datetime(2022, 1, 6, 1),
            ["2021-06-06"],
        ),
        (
            DailyPartitionsDefinition(start_date="2022-01-01"),
            DailyPartitionsDefinition(start_date="2021-01-01"),
            ["2022-01-01"],
            ["2022-01-01"],
            datetime(2022, 1, 6, 1),
            [],
        ),
        (
            DailyPartitionsDefinition(start_date="2022-01-01"),
            DailyPartitionsDefinition(start_date="2021-01-01"),
            [],
            ["2021-12-31"],
            datetime(2022, 1, 6, 1),
            ["2021-12-31"],
        ),
        (
            DailyPartitionsDefinition(start_date="2022-01-01"),
            DailyPartitionsDefinition(start_date="2021-01-01"),
            [],
            ["2021-12-30"],
            datetime(2021, 12, 31, 1),
            ["2021-12-30"],
        ),
    ],
)
def test_get_upstream_with_current_time(
    upstream_partitions_def: TimeWindowPartitionsDefinition,
    downstream_partitions_def: TimeWindowPartitionsDefinition,
    expected_upstream_keys: Sequence[str],
    downstream_keys: Sequence[str],
    current_time: Optional[datetime],
    invalid_upstream_keys: Sequence[str],
):
    mapping = TimeWindowPartitionMapping()

    upstream_partitions_result = mapping.get_upstream_mapped_partitions_result_for_partitions(
        subset_with_keys(downstream_partitions_def, downstream_keys),
        upstream_partitions_def,
        current_time=current_time,
    )
    assert (
        upstream_partitions_result.partitions_subset.get_partition_keys() == expected_upstream_keys
    )
    assert (
        upstream_partitions_result.required_but_nonexistent_partition_keys == invalid_upstream_keys
    )


def test_different_start_time_partitions_defs():
    jan_start = DailyPartitionsDefinition("2023-01-01")
    feb_start = DailyPartitionsDefinition("2023-02-01")

    assert (
        TimeWindowPartitionMapping()
        .get_downstream_partitions_for_partitions(
            upstream_partitions_subset=subset_with_keys(jan_start, ["2023-01-15"]),
            downstream_partitions_def=feb_start,
        )
        .get_partition_keys()
        == []
    )

    upstream_partitions_result = (
        TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
            downstream_partitions_subset=subset_with_keys(jan_start, ["2023-01-15"]),
            upstream_partitions_def=feb_start,
        )
    )
    assert upstream_partitions_result.partitions_subset.get_partition_keys() == []
    assert upstream_partitions_result.required_but_nonexistent_partition_keys == ["2023-01-15"]


def test_different_end_time_partitions_defs():
    jan_partitions_def = DailyPartitionsDefinition("2023-01-01", end_date="2023-01-31")
    jan_feb_partitions_def = DailyPartitionsDefinition("2023-01-01", end_date="2023-02-28")

    assert TimeWindowPartitionMapping().get_downstream_partitions_for_partitions(
        upstream_partitions_subset=subset_with_keys(jan_partitions_def, ["2023-01-15"]),
        downstream_partitions_def=jan_feb_partitions_def,
    ).get_partition_keys() == ["2023-01-15"]

    assert (
        TimeWindowPartitionMapping()
        .get_downstream_partitions_for_partitions(
            upstream_partitions_subset=subset_with_keys(jan_feb_partitions_def, ["2023-02-15"]),
            downstream_partitions_def=jan_partitions_def,
        )
        .get_partition_keys()
        == []
    )

    upstream_partitions_result = (
        TimeWindowPartitionMapping().get_upstream_mapped_partitions_result_for_partitions(
            downstream_partitions_subset=subset_with_keys(jan_feb_partitions_def, ["2023-02-15"]),
            upstream_partitions_def=jan_partitions_def,
        )
    )
    assert upstream_partitions_result.partitions_subset.get_partition_keys() == []
    assert upstream_partitions_result.required_but_nonexistent_partition_keys == ["2023-02-15"]


def test_daily_upstream_of_yearly():
    daily = DailyPartitionsDefinition("2023-01-01")
    yearly = TimeWindowPartitionsDefinition(
        cron_schedule="0 0 1 1 *",
        fmt="%Y-%m-%d",
        start="2023-01-01",
        end_offset=1,
    )  # Partition exists for current year

    assert TimeWindowPartitionMapping(
        allow_nonexistent_upstream_partitions=True
    ).get_upstream_mapped_partitions_result_for_partitions(
        downstream_partitions_subset=subset_with_keys(yearly, ["2023-01-01"]),
        upstream_partitions_def=daily,
        current_time=datetime(2023, 1, 5, 0),
    ).partitions_subset.get_partition_keys() == [
        "2023-01-01",
        "2023-01-02",
        "2023-01-03",
        "2023-01-04",
    ]


@pytest.mark.parametrize(
    "downstream_partitions_subset,upstream_partitions_def,allow_nonexistent_upstream_partitions,current_time,valid_partitions_mapped_to,required_but_nonexistent_partition_keys",
    [
        (
            DailyPartitionsDefinition(start_date="2023-05-01")
            .empty_subset()
            .with_partition_keys(["2023-05-10", "2023-05-30", "2023-06-01"]),
            DailyPartitionsDefinition("2023-06-01"),
            False,
            datetime(2023, 6, 5, 0),
            ["2023-06-01"],
            ["2023-05-10", "2023-05-30"],
        ),
        (
            DailyPartitionsDefinition(start_date="2023-05-01")
            .empty_subset()
            .with_partition_keys(["2023-05-09", "2023-05-10"]),
            DailyPartitionsDefinition("2023-05-01", end_date="2023-05-10"),
            False,
            datetime(2023, 5, 12, 0),
            ["2023-05-09"],
            ["2023-05-10"],
        ),
        (
            DailyPartitionsDefinition(start_date="2023-05-01")
            .empty_subset()
            .with_partition_keys(["2023-05-10", "2023-05-30", "2023-06-01"]),
            DailyPartitionsDefinition("2023-06-01"),
            True,
            datetime(2023, 6, 5, 0),
            ["2023-06-01"],
            [],
        ),
        (
            DailyPartitionsDefinition(start_date="2023-05-01")
            .empty_subset()
            .with_partition_keys(["2023-05-09", "2023-05-10"]),
            DailyPartitionsDefinition("2023-05-01", end_date="2023-05-10"),
            True,
            datetime(2023, 5, 12, 0),
            ["2023-05-09"],
            [],
        ),
    ],
)
def test_downstream_partition_has_valid_upstream_partitions(
    downstream_partitions_subset: BaseTimeWindowPartitionsSubset,
    upstream_partitions_def: TimeWindowPartitionsDefinition,
    allow_nonexistent_upstream_partitions: bool,
    current_time: datetime,
    valid_partitions_mapped_to: Sequence[str],
    required_but_nonexistent_partition_keys: Sequence[str],
):
    result = TimeWindowPartitionMapping(
        allow_nonexistent_upstream_partitions=allow_nonexistent_upstream_partitions
    ).get_upstream_mapped_partitions_result_for_partitions(
        downstream_partitions_subset=downstream_partitions_subset,
        upstream_partitions_def=upstream_partitions_def,
        current_time=current_time,
    )
    assert result.partitions_subset.get_partition_keys() == valid_partitions_mapped_to
    assert result.required_but_nonexistent_partition_keys == required_but_nonexistent_partition_keys


@pytest.mark.parametrize(
    "partition_key,expected_upstream_partition_key,expected_downstream_partition_key",
    [
        (
            "2023-11-04",
            "2023-11-03",
            "2023-11-05",
        ),
        (
            "2023-11-05",
            "2023-11-04",
            "2023-11-06",
        ),
        (
            "2023-11-06",
            "2023-11-05",
            "2023-11-07",
        ),
        (
            "2023-11-07",
            "2023-11-06",
            "2023-11-08",
        ),
        (
            "2024-03-09",
            "2024-03-08",
            "2024-03-10",
        ),
        (
            "2024-03-10",
            "2024-03-09",
            "2024-03-11",
        ),
        (
            "2024-03-11",
            "2024-03-10",
            "2024-03-12",
        ),
        (
            "2024-03-12",
            "2024-03-11",
            "2024-03-13",
        ),
    ],
)
def test_dst_transition_with_daily_partitions(
    partition_key: str, expected_upstream_partition_key: str, expected_downstream_partition_key: str
):
    partitions_def = DailyPartitionsDefinition("2023-11-01", timezone="America/Los_Angeles")
    time_partition_mapping = TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)
    current_time = datetime(2024, 3, 20, 0)

    subset = partitions_def.subset_with_partition_keys([partition_key])
    upstream = time_partition_mapping.get_upstream_mapped_partitions_result_for_partitions(
        subset, partitions_def, current_time=current_time
    )
    assert upstream.partitions_subset.get_partition_keys(current_time=current_time) == [
        expected_upstream_partition_key
    ]
    downstream = time_partition_mapping.get_downstream_partitions_for_partitions(
        subset, partitions_def, current_time=current_time
    )
    assert downstream.get_partition_keys(current_time=current_time) == [
        expected_downstream_partition_key
    ]


def test_mar_2024_dst_transition_with_hourly_partitions():
    partitions_def = HourlyPartitionsDefinition("2023-11-01-00:00", timezone="America/Los_Angeles")
    time_partition_mapping = TimeWindowPartitionMapping(start_offset=-1, end_offset=-1)
    current_time = datetime(2024, 3, 20, 0)

    assert "2023-03-10-02:00" not in partitions_def.get_partition_keys(current_time=current_time)

    subset = partitions_def.subset_with_partition_keys(["2024-03-10-03:00"])
    upstream = time_partition_mapping.get_upstream_mapped_partitions_result_for_partitions(
        subset, partitions_def, current_time=current_time
    )
    assert upstream.partitions_subset.get_partition_keys(current_time=current_time) == [
        "2024-03-10-01:00",
    ]
    downstream = time_partition_mapping.get_downstream_partitions_for_partitions(
        subset, partitions_def, current_time=current_time
    )
    assert downstream.get_partition_keys(current_time=current_time) == [
        "2024-03-10-04:00",
    ]
