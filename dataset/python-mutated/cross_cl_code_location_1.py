from dagster import Definitions, asset

@asset
def code_location_1_asset():
    if False:
        return 10
    return 5
defs = Definitions(assets=[code_location_1_asset])