from dagster import AssetIn, asset

@asset
def upstream_asset():
    if False:
        print('Hello World!')
    return [1, 2, 3]

@asset(ins={'upstream': AssetIn('upstream_asset')})
def downstream_asset(upstream):
    if False:
        while True:
            i = 10
    return upstream + [4]