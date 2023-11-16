from dagster._core.definitions.events import AssetKey
from docs_snippets.concepts.assets.code_versions import asset_with_version, multi_asset_with_versions

def test_single_asset_with_version():
    if False:
        for i in range(10):
            print('nop')
    assert asset_with_version.code_versions_by_key == {AssetKey(['asset_with_version']): '1'}

def test_multi_asset_with_version():
    if False:
        print('Hello World!')
    assert multi_asset_with_versions.code_versions_by_key == {AssetKey(['a']): '1', AssetKey(['b']): '2'}