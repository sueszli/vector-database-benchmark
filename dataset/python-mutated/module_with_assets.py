from dagster import AssetKey, SourceAsset, asset
from .asset_subpackage.another_module_with_assets import miles_davis
assert miles_davis
elvis_presley = SourceAsset(key=AssetKey('elvis_presley'))

@asset
def chuck_berry(elvis_presley, miles_davis):
    if False:
        print('Hello World!')
    pass