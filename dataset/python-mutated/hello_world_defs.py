from dagster import Definitions, asset

@asset
def hello_asset():
    if False:
        return 10
    pass
defs = Definitions(assets=[hello_asset])