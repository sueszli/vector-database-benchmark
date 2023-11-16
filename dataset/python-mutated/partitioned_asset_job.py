from dagster import AssetSelection, Definitions, HourlyPartitionsDefinition, asset, define_asset_job
hourly_partitions_def = HourlyPartitionsDefinition(start_date='2022-05-31-00:00')

@asset(partitions_def=hourly_partitions_def)
def asset1():
    if False:
        while True:
            i = 10
    ...

@asset(partitions_def=hourly_partitions_def)
def asset2():
    if False:
        return 10
    ...
partitioned_asset_job = define_asset_job(name='asset_1_and_2_job', selection=AssetSelection.assets(asset1, asset2), partitions_def=hourly_partitions_def)
defs = Definitions(assets=[asset1, asset2], jobs=[partitioned_asset_job])