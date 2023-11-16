import datetime
from dagster import AssetOut, DataVersion, Output, SourceAsset, asset, multi_asset, observable_source_asset

@observable_source_asset
def observable_different_version():
    if False:
        print('Hello World!')
    return DataVersion(str(datetime.datetime.now()))

@observable_source_asset
def observable_same_version():
    if False:
        i = 10
        return i + 15
    return DataVersion('5')
non_observable_source = SourceAsset('non_observable_source')

@asset(code_version='1', deps=[observable_different_version])
def has_code_version1(context):
    if False:
        for i in range(10):
            print('nop')
    ...

@asset(code_version='1', deps=[observable_same_version])
def has_code_version2():
    if False:
        i = 10
        return i + 15
    ...

@asset(deps=[observable_different_version, observable_same_version, non_observable_source], code_version='1')
def has_code_version_multiple_deps():
    if False:
        i = 10
        return i + 15
    ...

@asset(code_version='1', deps=[has_code_version1])
def downstream_of_code_versioned():
    if False:
        for i in range(10):
            print('nop')
    ...

@asset
def root_asset_no_code_version(context):
    if False:
        return 10
    return 100

@asset(deps=[root_asset_no_code_version])
def downstream_of_no_code_version():
    if False:
        while True:
            i = 10
    ...

@multi_asset(outs={'code_versioned_multi_asset1': AssetOut(code_version='1'), 'code_versioned_multi_asset2': AssetOut(code_version='3')}, deps=[downstream_of_no_code_version])
def code_versioned_multi_asset():
    if False:
        print('Hello World!')
    yield Output(None, 'code_versioned_multi_asset1')
    yield Output(None, 'code_versioned_multi_asset2')

@asset(deps=['code_versioned_multi_asset2'])
def downstream_of_code_versioned_multi_asset():
    if False:
        print('Hello World!')
    ...