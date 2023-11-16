from dagster import Definitions, load_assets_from_modules
from dagster_test.toys.partitioned_assets import failing_partitions

def test_assets():
    if False:
        print('Hello World!')
    Definitions(assets=load_assets_from_modules([failing_partitions]))