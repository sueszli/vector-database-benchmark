from dagster import load_assets_from_modules
from dagster import materialize
from docs_snippets.guides.dagster.managing_ml import managing_ml_code

def assets_test():
    if False:
        print('Hello World!')
    assets = load_assets_from_modules([managing_ml_code])
    result = materialize(assets)
    assert result.success