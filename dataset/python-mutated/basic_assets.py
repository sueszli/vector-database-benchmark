from dagster import AssetSelection, MetadataValue, asset, define_asset_job

@asset(group_name='basic_assets')
def basic_asset_1():
    if False:
        i = 10
        return i + 15
    ...

@asset(group_name='basic_assets')
def basic_asset_2(basic_asset_1):
    if False:
        print('Hello World!')
    ...

@asset(group_name='basic_assets')
def basic_asset_3(basic_asset_1):
    if False:
        return 10
    ...

@asset(group_name='basic_assets')
def basic_asset_4(basic_asset_2, basic_asset_3):
    if False:
        return 10
    ...
basic_assets_job = define_asset_job('basic_assets_job', selection=AssetSelection.groups('basic_assets'), metadata={'owner': 'data team', 'link': MetadataValue.url(url='https://dagster.io')})