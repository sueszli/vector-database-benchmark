from dagster import asset

@asset
def iris_dataset():
    if False:
        for i in range(10):
            print('nop')
    return None
from dagster_deltalake import DeltaLakePyarrowIOManager, LocalConfig
from dagster import Definitions
defs = Definitions(assets=[iris_dataset], resources={'io_manager': DeltaLakePyarrowIOManager(root_uri='path/to/deltalake', storage_options=LocalConfig(), schema='iris')})