from dagster import AssetKey, Definitions, SourceAsset, asset
code_location_1_source_asset = SourceAsset(key=AssetKey('code_location_1_asset'))

@asset
def code_location_2_asset(code_location_1_asset):
    if False:
        i = 10
        return i + 15
    return code_location_1_asset + 6
defs = Definitions(assets=[code_location_2_asset, code_location_1_source_asset])