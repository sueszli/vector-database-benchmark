from dagster import Definitions, load_assets_from_modules
from docs_snippets.concepts.partitions_schedules_sensors import partition_mapping

def test_definitions():
    if False:
        print('Hello World!')
    Definitions(assets=load_assets_from_modules([partition_mapping]))