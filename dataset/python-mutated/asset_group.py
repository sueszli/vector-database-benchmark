from dagster import asset

@asset
def upstream_asset():
    if False:
        return 10
    return [1, 2, 3]

@asset
def downstream_asset(upstream_asset):
    if False:
        while True:
            i = 10
    return upstream_asset + [4]