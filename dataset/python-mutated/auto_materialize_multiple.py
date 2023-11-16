from dagster import AutoMaterializePolicy, Definitions, asset, load_assets_from_current_module

@asset
def asset1():
    if False:
        while True:
            i = 10
    ...

@asset(deps=[asset1])
def asset2():
    if False:
        while True:
            i = 10
    ...
defs = Definitions(assets=load_assets_from_current_module(auto_materialize_policy=AutoMaterializePolicy.eager()))