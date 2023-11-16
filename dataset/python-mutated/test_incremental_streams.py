from airbyte_cdk.models import SyncMode
from pytest import fixture
from source_kyve.source import KYVEStream as IncrementalKyveStream
from . import config, pool_data

@fixture
def patch_incremental_base_class(mocker):
    if False:
        print('Hello World!')
    mocker.patch.object(IncrementalKyveStream, 'path', 'v0/example_endpoint')
    mocker.patch.object(IncrementalKyveStream, 'primary_key', 'test_primary_key')
    mocker.patch.object(IncrementalKyveStream, '__abstractmethods__', set())

def test_cursor_field(patch_incremental_base_class):
    if False:
        while True:
            i = 10
    stream = IncrementalKyveStream(config, pool_data)
    expected_cursor_field = 'offset'
    assert stream.cursor_field == expected_cursor_field

def test_get_updated_state(patch_incremental_base_class):
    if False:
        return 10
    stream = IncrementalKyveStream(config, pool_data)
    inputs = {'current_stream_state': None, 'latest_record': None}
    expected_state = {}
    assert stream.get_updated_state(**inputs) == expected_state

def test_stream_slices(patch_incremental_base_class):
    if False:
        while True:
            i = 10
    stream = IncrementalKyveStream(config, pool_data)
    inputs = {'sync_mode': SyncMode.incremental, 'cursor_field': [], 'stream_state': {}}
    expected_stream_slice = [None]
    assert stream.stream_slices(**inputs) == expected_stream_slice

def test_supports_incremental(patch_incremental_base_class, mocker):
    if False:
        for i in range(10):
            print('nop')
    mocker.patch.object(IncrementalKyveStream, 'cursor_field', 'dummy_field')
    stream = IncrementalKyveStream(config, pool_data)
    assert stream.supports_incremental

def test_source_defined_cursor(patch_incremental_base_class):
    if False:
        print('Hello World!')
    stream = IncrementalKyveStream(config, pool_data)
    assert stream.source_defined_cursor

def test_stream_checkpoint_interval(patch_incremental_base_class):
    if False:
        return 10
    stream = IncrementalKyveStream(config, pool_data)
    expected_checkpoint_interval = None
    assert stream.state_checkpoint_interval == expected_checkpoint_interval