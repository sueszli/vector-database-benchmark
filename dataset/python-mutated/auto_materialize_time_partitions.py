from dagster import AutoMaterializePolicy, DailyPartitionsDefinition, asset

@asset(partitions_def=DailyPartitionsDefinition(start_date='2020-10-10'), auto_materialize_policy=AutoMaterializePolicy.eager())
def asset1():
    if False:
        for i in range(10):
            print('nop')
    ...

@asset(partitions_def=DailyPartitionsDefinition(start_date='2020-10-10'), auto_materialize_policy=AutoMaterializePolicy.eager(), deps=[asset1])
def asset2():
    if False:
        while True:
            i = 10
    ...