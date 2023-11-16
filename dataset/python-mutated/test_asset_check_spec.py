from dagster import AssetCheckSpec, AssetKey, SourceAsset, asset

def test_coerce_asset_key():
    if False:
        i = 10
        return i + 15
    assert AssetCheckSpec(asset='foo', name='check1').asset_key == AssetKey('foo')

def test_asset_def():
    if False:
        return 10

    @asset
    def foo():
        if False:
            for i in range(10):
                print('nop')
        ...
    assert AssetCheckSpec(asset=foo, name='check1').asset_key == AssetKey('foo')

def test_source_asset():
    if False:
        while True:
            i = 10
    foo = SourceAsset('foo')
    assert AssetCheckSpec(asset=foo, name='check1').asset_key == AssetKey('foo')