from dagster import AssetCheckResult, AssetCheckSeverity, Definitions, asset, asset_check

@asset
def my_asset():
    if False:
        print('Hello World!')
    ...

@asset_check(asset=my_asset)
def my_check():
    if False:
        for i in range(10):
            print('nop')
    is_serious = ...
    return AssetCheckResult(passed=False, severity=AssetCheckSeverity.ERROR if is_serious else AssetCheckSeverity.WARN)
defs = Definitions(assets=[my_asset], asset_checks=[my_check])