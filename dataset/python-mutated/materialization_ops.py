def read_df():
    if False:
        while True:
            i = 10
    return 1

def read_df_for_date(_):
    if False:
        for i in range(10):
            print('nop')
    return 1

def persist_to_storage(df):
    if False:
        i = 10
        return i + 15
    return 'tmp'

def calculate_bytes(df):
    if False:
        while True:
            i = 10
    return 1.0
from dagster import op, OpExecutionContext

@op
def my_simple_op():
    if False:
        for i in range(10):
            print('nop')
    df = read_df()
    remote_storage_path = persist_to_storage(df)
    return remote_storage_path
from dagster import AssetMaterialization, op

@op
def my_materialization_op(context: OpExecutionContext):
    if False:
        return 10
    df = read_df()
    remote_storage_path = persist_to_storage(df)
    context.log_event(AssetMaterialization(asset_key='my_dataset', description='Persisted result to storage'))
    return remote_storage_path
from dagster import AssetMaterialization, Config, op, OpExecutionContext

class MyOpConfig(Config):
    date: str

@op
def my_partitioned_asset_op(context: OpExecutionContext, config: MyOpConfig):
    if False:
        while True:
            i = 10
    partition_date = config.date
    df = read_df_for_date(partition_date)
    remote_storage_path = persist_to_storage(df)
    context.log_event(AssetMaterialization(asset_key='my_dataset', partition=partition_date))
    return remote_storage_path
from dagster import AssetMaterialization, MetadataValue, op, OpExecutionContext

@op
def my_metadata_materialization_op(context: OpExecutionContext):
    if False:
        return 10
    df = read_df()
    remote_storage_path = persist_to_storage(df)
    context.log_event(AssetMaterialization(asset_key='my_dataset', description='Persisted result to storage', metadata={'text_metadata': 'Text-based metadata for this event', 'path': MetadataValue.path(remote_storage_path), 'dashboard_url': MetadataValue.url('http://mycoolsite.com/url_for_my_data'), 'size (bytes)': calculate_bytes(df)}))
    return remote_storage_path
from dagster import AssetKey, AssetMaterialization, Output, job, op, OpExecutionContext

@op
def my_asset_key_materialization_op(context: OpExecutionContext):
    if False:
        while True:
            i = 10
    df = read_df()
    remote_storage_path = persist_to_storage(df)
    yield AssetMaterialization(asset_key=AssetKey(['dashboard', 'my_cool_site']), description='Persisted result to storage', metadata={'dashboard_url': MetadataValue.url('http://mycoolsite.com/dashboard'), 'size (bytes)': calculate_bytes(df)})
    yield Output(remote_storage_path)

@job
def my_asset_job():
    if False:
        for i in range(10):
            print('nop')
    my_materialization_op()