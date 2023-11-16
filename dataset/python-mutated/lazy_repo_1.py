from dagster import AutoMaterializePolicy, DailyPartitionsDefinition, asset, repository
daily_partitions_def = DailyPartitionsDefinition(start_date='2023-02-01')

@asset(auto_materialize_policy=AutoMaterializePolicy.lazy())
def lazy_upstream():
    if False:
        i = 10
        return i + 15
    return 1

@asset(auto_materialize_policy=AutoMaterializePolicy.lazy())
def lazy_downstream_1(lazy_upstream):
    if False:
        for i in range(10):
            print('nop')
    return lazy_upstream + 1

@asset(auto_materialize_policy=AutoMaterializePolicy.lazy(), partitions_def=daily_partitions_def)
def lazy_upstream_partitioned():
    if False:
        return 10
    return 1

@asset(auto_materialize_policy=AutoMaterializePolicy.lazy(), partitions_def=daily_partitions_def)
def lazy_downstream_1_partitioned(lazy_upstream_partitioned):
    if False:
        for i in range(10):
            print('nop')
    return lazy_upstream_partitioned + 1

@repository
def lazy_auto_materialize_repo_1():
    if False:
        print('Hello World!')
    return [lazy_upstream, lazy_downstream_1, lazy_upstream_partitioned, lazy_downstream_1_partitioned]