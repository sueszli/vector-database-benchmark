from dagster import AssetOut, Output, asset, multi_asset

@asset(code_version='1')
def asset_with_version():
    if False:
        for i in range(10):
            print('nop')
    return 100

@multi_asset(outs={'a': AssetOut(code_version='1'), 'b': AssetOut(code_version='2')})
def multi_asset_with_versions():
    if False:
        while True:
            i = 10
    yield Output(100, 'a')
    yield Output(200, 'b')