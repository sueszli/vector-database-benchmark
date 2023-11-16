from dagster import AutoMaterializePolicy, DailyPartitionsDefinition, asset, repository

@asset(auto_materialize_policy=AutoMaterializePolicy.eager())
def eager_upstream():
    if False:
        while True:
            i = 10
    return 3

@asset(auto_materialize_policy=AutoMaterializePolicy.eager())
def eager_downstream_0_point_5(eager_upstream):
    if False:
        for i in range(10):
            print('nop')
    return eager_upstream + 1

@asset(auto_materialize_policy=AutoMaterializePolicy.eager(), deps=[eager_upstream])
def eager_downstream_1(eager_downstream_0_point_5):
    if False:
        for i in range(10):
            print('nop')
    return eager_downstream_0_point_5 + 1
daily_partitions_def = DailyPartitionsDefinition(start_date='2023-02-01')

@asset(auto_materialize_policy=AutoMaterializePolicy.eager(), partitions_def=daily_partitions_def)
def eager_upstream_partitioned():
    if False:
        return 10
    return 3

@asset(auto_materialize_policy=AutoMaterializePolicy.eager(), partitions_def=daily_partitions_def)
def eager_downstream_0_point_5_partitioned(eager_upstream_partitioned):
    if False:
        for i in range(10):
            print('nop')
    return eager_upstream_partitioned + 1

@asset(auto_materialize_policy=AutoMaterializePolicy.eager(), partitions_def=daily_partitions_def, deps=[eager_downstream_0_point_5_partitioned])
def eager_downstream_1_partitioned(eager_upstream_partitioned):
    if False:
        print('Hello World!')
    return eager_upstream_partitioned + 1

@repository
def auto_materialize_repo_1():
    if False:
        i = 10
        return i + 15
    return [eager_upstream, eager_downstream_0_point_5, eager_downstream_1, eager_upstream_partitioned, eager_downstream_1_partitioned, eager_downstream_0_point_5_partitioned]