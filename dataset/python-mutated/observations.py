from dagster import job, op, OpExecutionContext

def read_df():
    if False:
        while True:
            i = 10
    return range(372)

def read_df_for_date(_):
    if False:
        for i in range(10):
            print('nop')
    return 1

def persist_to_storage(_):
    if False:
        return 10
    return 'tmp'

def calculate_bytes(_):
    if False:
        for i in range(10):
            print('nop')
    return 1.0
from dagster import AssetObservation, op

@op
def observation_op(context: OpExecutionContext):
    if False:
        print('Hello World!')
    df = read_df()
    context.log_event(AssetObservation(asset_key='observation_asset', metadata={'num_rows': len(df)}))
    return 5
from dagster import AssetMaterialization, Config, op, OpExecutionContext

class MyOpConfig(Config):
    date: str

@op
def partitioned_dataset_op(context: OpExecutionContext, config: MyOpConfig):
    if False:
        return 10
    partition_date = config.date
    df = read_df_for_date(partition_date)
    context.log_event(AssetObservation(asset_key='my_partitioned_dataset', partition=partition_date))
    return df
from dagster import AssetMaterialization, AssetObservation, MetadataValue, op

@op
def observes_dataset_op(context: OpExecutionContext):
    if False:
        i = 10
        return i + 15
    df = read_df()
    remote_storage_path = persist_to_storage(df)
    context.log_event(AssetObservation(asset_key='my_dataset', metadata={'text_metadata': 'Text-based metadata for this event', 'path': MetadataValue.path(remote_storage_path), 'dashboard_url': MetadataValue.url('http://mycoolsite.com/url_for_my_data'), 'size (bytes)': calculate_bytes(df)}))
    context.log_event(AssetMaterialization(asset_key='my_dataset'))
    return remote_storage_path

@job
def my_observation_job():
    if False:
        while True:
            i = 10
    observation_op()

@job
def my_dataset_job():
    if False:
        return 10
    observes_dataset_op()