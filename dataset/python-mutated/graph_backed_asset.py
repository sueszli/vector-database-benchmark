from dagster import AssetKey, load_assets_from_current_module, Out, Output, AssetSelection, define_asset_job, Definitions, OpExecutionContext
from mock import MagicMock

def create_db_connection():
    if False:
        print('Hello World!')
    return MagicMock()
import pandas as pd
from dagster import graph_asset, op
from dagster_slack import SlackResource

@op
def fetch_files_from_slack(slack: SlackResource) -> pd.DataFrame:
    if False:
        while True:
            i = 10
    files = slack.get_client().files_list(channel='#random')
    return pd.DataFrame([{'id': file.get('id'), 'created': file.get('created'), 'title': file.get('title'), 'permalink': file.get('permalink')} for file in files])

@op
def store_files(files):
    if False:
        for i in range(10):
            print('nop')
    return files.to_sql(name='slack_files', con=create_db_connection())

@graph_asset
def slack_files_table():
    if False:
        print('Hello World!')
    return store_files(fetch_files_from_slack())
slack_mock = MagicMock()
store_slack_files = define_asset_job('store_slack_files', selection=AssetSelection.assets(slack_files_table))
from dagster import asset, graph_asset, op

@asset
def upstream_asset():
    if False:
        while True:
            i = 10
    return 1

@op
def add_one(input_num):
    if False:
        for i in range(10):
            print('nop')
    return input_num + 1

@op
def multiply_by_two(input_num):
    if False:
        while True:
            i = 10
    return input_num * 2

@graph_asset
def middle_asset(upstream_asset):
    if False:
        print('Hello World!')
    return multiply_by_two(add_one(upstream_asset))

@asset
def downstream_asset(middle_asset):
    if False:
        for i in range(10):
            print('nop')
    return middle_asset + 7
basic_deps_job = define_asset_job('basic_deps_job', AssetSelection.assets(upstream_asset, middle_asset, downstream_asset))

@op(out={'one': Out(), 'two': Out()})
def two_outputs(upstream):
    if False:
        for i in range(10):
            print('nop')
    yield Output(output_name='one', value=upstream)
    yield Output(output_name='two', value=upstream)
from dagster import AssetOut, graph_multi_asset

@graph_multi_asset(outs={'first_asset': AssetOut(), 'second_asset': AssetOut()})
def two_assets(upstream_asset):
    if False:
        for i in range(10):
            print('nop')
    (one, two) = two_outputs(upstream_asset)
    return {'first_asset': one, 'second_asset': two}
second_basic_deps_job = define_asset_job('second_basic_deps_job', AssetSelection.assets(upstream_asset, two_assets))
from dagster import AssetOut, graph_multi_asset

@graph_multi_asset(outs={'asset_one': AssetOut(), 'asset_two': AssetOut()})
def one_and_two(upstream_asset):
    if False:
        return 10
    (one, two) = two_outputs(upstream_asset)
    return {'asset_one': one, 'asset_two': two}
explicit_deps_job = define_asset_job('explicit_deps_job', AssetSelection.assets(upstream_asset, one_and_two))
defs = Definitions(assets=load_assets_from_current_module(), jobs=[basic_deps_job, store_slack_files, second_basic_deps_job, explicit_deps_job], resources={'slack': slack_mock})