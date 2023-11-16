from dagster import AutoMaterializePolicy, asset

@asset
def asset1():
    if False:
        while True:
            i = 10
    ...

@asset(auto_materialize_policy=AutoMaterializePolicy.eager(), deps=[asset1])
def asset2():
    if False:
        return 10
    ...