from dagster import asset

@asset
def upstream_asset():
    if False:
        print('Hello World!')
    return [1, 2, 3]

@asset
def downstream_asset(upstream_asset):
    if False:
        for i in range(10):
            print('nop')
    return upstream_asset + [4]