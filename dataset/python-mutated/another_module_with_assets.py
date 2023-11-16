from dagster import AssetKey, SourceAsset, asset
patsy_cline = SourceAsset(key=AssetKey('patsy_cline'))

@asset
def miles_davis():
    if False:
        i = 10
        return i + 15
    pass