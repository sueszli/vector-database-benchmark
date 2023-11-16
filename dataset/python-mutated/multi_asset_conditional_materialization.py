import random
from dagster import AssetOut, Output, asset, multi_asset

@multi_asset(outs={'asset1': AssetOut(is_required=False), 'asset2': AssetOut(is_required=False)})
def assets_1_and_2():
    if False:
        i = 10
        return i + 15
    if random.randint(1, 10) < 5:
        yield Output([1, 2, 3, 4], output_name='asset1')
    if random.randint(1, 10) < 5:
        yield Output([6, 7, 8, 9], output_name='asset2')

@asset
def downstream1(asset1):
    if False:
        print('Hello World!')
    return asset1 + [5]

@asset
def downstream2(asset2):
    if False:
        print('Hello World!')
    return asset2 + [10]