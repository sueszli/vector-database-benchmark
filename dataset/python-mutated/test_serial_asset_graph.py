from dagster._core.definitions.load_assets_from_modules import assets_from_modules
from dagster._core.definitions.materialize import materialize
from docs_snippets.guides.dagster.asset_tutorial import serial_asset_graph
from docs_snippets.intro_tutorial.test_util import patch_cereal_requests

@patch_cereal_requests
def test_serial_asset_graph():
    if False:
        while True:
            i = 10
    (assets, source_assets, _) = assets_from_modules([serial_asset_graph])
    assert materialize([*assets, *source_assets])