from dagster._core.definitions import ResourceDefinition, SourceAsset
from dagster._core.definitions.events import AssetKey
from dagster._core.definitions.metadata import MetadataValue

def test_source_asset_metadata():
    if False:
        print('Hello World!')
    sa = SourceAsset(key=AssetKey('foo'), metadata={'foo': 'bar', 'baz': object()})
    assert sa.metadata == {'foo': MetadataValue.text('bar'), 'baz': MetadataValue.text('[object] (unserializable)')}

def test_source_asset_key_args():
    if False:
        while True:
            i = 10
    assert SourceAsset(key='foo').key == AssetKey(['foo'])
    assert SourceAsset(key=['bar', 'foo']).key == AssetKey(['bar', 'foo'])

def test_source_asset_with_bare_resource():
    if False:
        i = 10
        return i + 15

    class BareResourceObject:
        pass
    source_asset = SourceAsset(key='foo', resource_defs={'bare_resource': BareResourceObject()})
    assert isinstance(source_asset.resource_defs['bare_resource'], ResourceDefinition)