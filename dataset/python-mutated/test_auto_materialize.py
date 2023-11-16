from dagster import Definitions, load_assets_from_modules
from docs_snippets.concepts.assets import auto_materialize_eager, auto_materialize_lazy, auto_materialize_lazy_transitive, auto_materialize_observable_source_asset, auto_materialize_time_partitions

def test_auto_materialize_eager_asset_defs():
    if False:
        print('Hello World!')
    Definitions(assets=load_assets_from_modules([auto_materialize_eager]))

def test_auto_materialize_lazy_asset_defs():
    if False:
        print('Hello World!')
    Definitions(assets=load_assets_from_modules([auto_materialize_lazy]))

def test_auto_materialize_lazy_transitive_asset_defs():
    if False:
        while True:
            i = 10
    Definitions(assets=load_assets_from_modules([auto_materialize_lazy_transitive]))

def test_auto_materialize_observable_source_asset():
    if False:
        print('Hello World!')
    Definitions(assets=load_assets_from_modules([auto_materialize_observable_source_asset]))

def test_auto_materialize_time_partitions():
    if False:
        i = 10
        return i + 15
    Definitions(assets=load_assets_from_modules([auto_materialize_time_partitions]))