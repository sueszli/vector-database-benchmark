from dagster import AutoMaterializePolicy, Definitions, asset, load_assets_from_current_module

@asset
def root1():
    if False:
        print('Hello World!')
    ...

@asset
def root2():
    if False:
        while True:
            i = 10
    ...

@asset
def diamond_left(root1):
    if False:
        i = 10
        return i + 15
    ...

@asset
def diamond_right(root1):
    if False:
        for i in range(10):
            print('nop')
    ...

@asset
def diamond_sink(diamond_left, diamond_right):
    if False:
        print('Hello World!')
    ...

@asset
def after_both_roots(root1, root2):
    if False:
        while True:
            i = 10
    ...
defs = Definitions(assets=load_assets_from_current_module(group_name='eager_reconciliation', key_prefix='eager_reconciliation', auto_materialize_policy=AutoMaterializePolicy.eager()))