from dagster import AssetKey, AutoMaterializePolicy, DailyPartitionsDefinition, SourceAsset, asset, repository
from dagster._core.definitions.freshness_policy import FreshnessPolicy
lazy_downstream_1_source = SourceAsset(AssetKey(['lazy_downstream_1']))

@asset(auto_materialize_policy=AutoMaterializePolicy.lazy())
def lazy_downstream_2(lazy_downstream_1):
    if False:
        for i in range(10):
            print('nop')
    return lazy_downstream_1 + 2

@asset(auto_materialize_policy=AutoMaterializePolicy.lazy())
def lazy_downstream_3(lazy_downstream_1):
    if False:
        for i in range(10):
            print('nop')
    return lazy_downstream_1 + 3

@asset(auto_materialize_policy=AutoMaterializePolicy.lazy(), freshness_policy=FreshnessPolicy(maximum_lag_minutes=1440))
def lazy_downstream_4(lazy_downstream_2, lazy_downstream_3):
    if False:
        print('Hello World!')
    return lazy_downstream_2 + lazy_downstream_3
daily_partitions_def = DailyPartitionsDefinition(start_date='2023-02-01')
lazy_downstream_1_source_partitioned = SourceAsset(AssetKey(['lazy_downstream_1_partitioned']))

@asset(partitions_def=daily_partitions_def, auto_materialize_policy=AutoMaterializePolicy.lazy())
def lazy_downstream_2_partitioned(lazy_downstream_1_partitioned):
    if False:
        print('Hello World!')
    return lazy_downstream_1_partitioned + 2

@asset(partitions_def=daily_partitions_def, auto_materialize_policy=AutoMaterializePolicy.lazy())
def lazy_downstream_3_partitioned(lazy_downstream_1_partitioned):
    if False:
        for i in range(10):
            print('nop')
    return lazy_downstream_1_partitioned + 3

@asset(partitions_def=daily_partitions_def, auto_materialize_policy=AutoMaterializePolicy.lazy(), freshness_policy=FreshnessPolicy(maximum_lag_minutes=1440))
def lazy_downstream_4_partitioned(lazy_downstream_2_partitioned, lazy_downstream_3_partitioned):
    if False:
        print('Hello World!')
    return lazy_downstream_2_partitioned + lazy_downstream_3_partitioned

@repository
def lazy_auto_materialize_repo_2():
    if False:
        for i in range(10):
            print('nop')
    return [lazy_downstream_2, lazy_downstream_3, lazy_downstream_1_source, lazy_downstream_4, lazy_downstream_2_partitioned, lazy_downstream_3_partitioned, lazy_downstream_1_source_partitioned, lazy_downstream_4_partitioned]