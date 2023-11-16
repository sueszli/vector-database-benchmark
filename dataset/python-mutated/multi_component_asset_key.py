from dagster import AssetIn, asset

@asset(key_prefix=['one', 'two', 'three'])
def upstream_asset():
    if False:
        i = 10
        return i + 15
    return [1, 2, 3]

@asset(ins={'upstream_asset': AssetIn(key_prefix='one/two/three')})
def downstream_asset(upstream_asset):
    if False:
        return 10
    return upstream_asset + [4]