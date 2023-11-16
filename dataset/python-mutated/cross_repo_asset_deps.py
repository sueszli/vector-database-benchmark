from dagster import AssetKey, SourceAsset, asset, repository

@asset
def derived_asset():
    if False:
        print('Hello World!')
    return 5

@repository
def upstream_assets_repository():
    if False:
        return 10
    return [derived_asset]
source_assets = [SourceAsset(AssetKey('derived_asset')), SourceAsset('always_source_asset')]

@asset
def downstream_asset1(derived_asset, always_source_asset):
    if False:
        return 10
    assert derived_asset

@asset
def downstream_asset2(derived_asset, always_source_asset):
    if False:
        while True:
            i = 10
    assert derived_asset

@repository
def downstream_assets_repository1():
    if False:
        i = 10
        return i + 15
    return [downstream_asset1, *source_assets]

@repository
def downstream_assets_repository2():
    if False:
        return 10
    return [downstream_asset2, *source_assets]