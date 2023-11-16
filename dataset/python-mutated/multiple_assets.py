from dagster import AssetKey, SourceAsset, asset
source_asset = SourceAsset(AssetKey('source_asset'))

@asset
def asset1(source_asset):
    if False:
        print('Hello World!')
    assert source_asset

@asset
def asset2():
    if False:
        i = 10
        return i + 15
    pass