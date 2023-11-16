from dagster import AssetIn, StaticPartitionMapping, StaticPartitionsDefinition, asset

@asset(partitions_def=StaticPartitionsDefinition(['a', 'b', 'c']))
def static_partitioned_asset1(context):
    if False:
        while True:
            i = 10
    assert context.partition_key

@asset(partitions_def=StaticPartitionsDefinition(['1', '2', '3']), ins={'static_partitioned_asset1': AssetIn(partition_mapping=StaticPartitionMapping({'a': '1', 'b': '2', 'c': '3'}))})
def static_partitioned_asset2(static_partitioned_asset1):
    if False:
        for i in range(10):
            print('nop')
    return static_partitioned_asset1