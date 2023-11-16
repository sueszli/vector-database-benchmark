"""This is used to generate the image on code snippet on the dbt front page.

We pull off some dark magic so that generating the screenshot doesn't involve a whole setup with
Fivetran and a database.
"""
from dagster import asset

class dagster_fivetran:

    @staticmethod
    def build_fivetran_assets(connector_id, table_names):
        if False:
            while True:
                i = 10

        @asset(compute_kind='fivetran')
        def users():
            if False:
                while True:
                    i = 10
            ...

        @asset(compute_kind='fivetran')
        def orders():
            if False:
                for i in range(10):
                    print('nop')
            ...
        return [users, orders]

class dagster_dbt:

    @staticmethod
    def load_assets_from_dbt_manifest(manifest):
        if False:
            for i in range(10):
                print('nop')

        @asset(non_argument_deps={'users'}, compute_kind='dbt')
        def stg_users():
            if False:
                while True:
                    i = 10
            'Users with test accounts removed.'
            ...

        @asset(non_argument_deps={'orders'}, compute_kind='dbt')
        def stg_orders():
            if False:
                return 10
            'Cleaned orders table.'
            ...

        @asset(non_argument_deps={'stg_users', 'stg_orders'}, compute_kind='dbt')
        def daily_order_summary():
            if False:
                return 10
            'Summary of daily orders, by user.'
            raise ValueError()
        return [stg_users, stg_orders, daily_order_summary]
from pathlib import Path
from dagster_dbt import DbtCliResource, dbt_assets, get_asset_key_for_model
from dagster_fivetran import build_fivetran_assets
from dagster import AssetExecutionContext, asset
fivetran_assets = build_fivetran_assets(connector_id='postgres', destination_tables=['users', 'orders'])

@dbt_assets(manifest=Path('manifest.json'))
def dbt_project_assets(context: AssetExecutionContext, dbt: DbtCliResource):
    if False:
        print('Hello World!')
    yield from dbt.cli(['build'], context=context).stream()

@asset(compute_kind='tensorflow', deps=[get_asset_key_for_model([dbt_project_assets], 'daily_order_summary')])
def predicted_orders():
    if False:
        return 10
    ...