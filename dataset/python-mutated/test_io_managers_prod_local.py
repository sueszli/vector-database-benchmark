import os
from unittest import mock
from docs_snippets.concepts.assets.asset_io_manager_prod_local import defs

@mock.patch.dict(os.environ, {'ENV': 'prod'})
def test_prod_assets():
    if False:
        print('Hello World!')
    assert len(defs.get_repository_def().assets_defs_by_key.keys()) == 2

@mock.patch.dict(os.environ, {'ENV': 'local'})
def test_local_assets():
    if False:
        i = 10
        return i + 15
    assert len(defs.get_repository_def().assets_defs_by_key.keys()) == 2