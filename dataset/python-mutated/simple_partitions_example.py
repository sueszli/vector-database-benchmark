import random
from dagster import AssetIn, DailyPartitionsDefinition, TimeWindowPartitionMapping, asset
partitions_def = DailyPartitionsDefinition(start_date='2023-01-01', end_date='2023-02-01')

@asset(group_name='partitions', partitions_def=partitions_def, name='my_daily_partitioned_asset', compute_kind='wandb', metadata={'wandb_artifact_configuration': {'type': 'dataset'}})
def create_my_daily_partitioned_asset(context):
    if False:
        i = 10
        return i + 15
    'Example writing an Artifact with daily partitions and custom metadata.'
    if context.has_partition_key:
        partition_key = context.asset_partition_key_for_output()
        context.log.info(f'Creating partitioned asset for {partition_key}')
        return random.randint(0, 100)
    partition_key_range = context.asset_partition_key_range
    context.log.info(f'Creating partitioned assets for window {partition_key_range}')
    return random.randint(0, 100)

@asset(group_name='partitions', compute_kind='wandb', ins={'my_daily_partitioned_asset': AssetIn()}, output_required=False)
def read_all_partitions(context, my_daily_partitioned_asset):
    if False:
        print('Hello World!')
    'Example reading all Artifact partitions from the first asset.'
    for (partition, content) in my_daily_partitioned_asset.items():
        context.log.info(f'partition={partition}, content={content}')

@asset(group_name='partitions', partitions_def=partitions_def, compute_kind='wandb', ins={'my_daily_partitioned_asset': AssetIn(partition_mapping=TimeWindowPartitionMapping(start_offset=-1))}, output_required=False)
def read_specific_partitions(context, my_daily_partitioned_asset):
    if False:
        return 10
    'Example reading specific Artifact partitions from the first asset.'
    for (partition, content) in my_daily_partitioned_asset.items():
        context.log.info(f'partition={partition}, content={content}')