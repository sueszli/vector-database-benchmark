from datetime import datetime, timezone
from typing import Dict, Generator
from unittest.mock import MagicMock, Mock
from airbyte_cdk.models import AirbyteMessage, AirbyteRecordMessage, AirbyteStateMessage, AirbyteStream, ConfiguredAirbyteCatalog, ConfiguredAirbyteStream, DestinationSyncMode, SyncMode, Type
from faunadb import _json
from faunadb import query as q
from source_fauna import SourceFauna
from test_util import CollectionConfig, config, expand_columns_query, mock_logger, ref
NOW = 1234512987

def results(modified, after):
    if False:
        for i in range(10):
            print('nop')
    modified_obj = {'data': modified}
    if after is not None:
        modified_obj['after'] = after
    return modified_obj

def record(stream: str, data: dict[str, any]) -> AirbyteMessage:
    if False:
        return 10
    return AirbyteMessage(type=Type.RECORD, record=AirbyteRecordMessage(data=data, stream=stream, emitted_at=NOW))

def state(data: dict[str, any]) -> AirbyteMessage:
    if False:
        while True:
            i = 10
    return AirbyteMessage(type=Type.STATE, state=AirbyteStateMessage(data=data, emitted_at=NOW))

def test_read_no_updates_or_creates_but_removes_present():
    if False:
        for i in range(10):
            print('nop')

    def find_index_for_stream(collection: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'ts'

    def read_updates_hardcoded(logger, stream: ConfiguredAirbyteStream, conf: CollectionConfig, state: Dict[str, any], index: str, page_size: int) -> Generator[any, None, None]:
        if False:
            print('Hello World!')
        return []

    def read_removes_hardcoded(logger, stream: ConfiguredAirbyteStream, conf, state, deletion_column: str) -> Generator[any, None, None]:
        if False:
            print('Hello World!')
        yield {'ref': '555', 'ts': 5, 'my_deleted_column': 5}
        yield {'ref': '123', 'ts': 3, 'my_deleted_column': 3}
    source = SourceFauna()
    source._setup_client = Mock()
    source.read_all = Mock()
    source.find_index_for_stream = find_index_for_stream
    source.read_updates = read_updates_hardcoded
    source.read_removes = read_removes_hardcoded
    source.client = MagicMock()
    source.find_emitted_at = Mock(return_value=NOW)
    logger = mock_logger()
    result = list(source.read(logger, config({'collection': {'name': 'my_stream_name', 'deletions': {'deletion_mode': 'deleted_field', 'column': 'my_deleted_column'}}}), ConfiguredAirbyteCatalog(streams=[ConfiguredAirbyteStream(sync_mode=SyncMode.incremental, destination_sync_mode=DestinationSyncMode.append_dedup, stream=AirbyteStream(name='my_stream_name', json_schema={}, supported_sync_modes=[SyncMode.incremental, SyncMode.full_refresh]))]), state={}))
    assert result == [record('my_stream_name', {'ref': '555', 'ts': 5, 'my_deleted_column': 5}), record('my_stream_name', {'ref': '123', 'ts': 3, 'my_deleted_column': 3}), state({'my_stream_name': {'remove_cursor': {}, 'updates_cursor': {}}})]
    assert source._setup_client.called
    assert not source.read_all.called
    assert not logger.error.called

def test_read_updates_ignore_deletes():
    if False:
        print('Hello World!')
    was_called = False

    def find_index_for_stream(collection: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'my_stream_name_ts'

    def read_updates_hardcoded(logger, stream: ConfiguredAirbyteStream, conf, state: dict[str, any], index: str, page_size: int) -> Generator[any, None, None]:
        if False:
            for i in range(10):
                print('nop')
        yield {'some_document': 'data_here', 'ts': 5}
        yield {'more_document': 'data_here', 'ts': 3}

    def read_removes_hardcoded(logger, stream: ConfiguredAirbyteStream, conf, state, deletion_column: str) -> Generator[any, None, None]:
        if False:
            for i in range(10):
                print('nop')
        nonlocal was_called
        was_called = True
        yield {'ref': '555', 'ts': 5, 'my_deleted_column': 5}
        yield {'ref': '123', 'ts': 3, 'my_deleted_column': 3}
    source = SourceFauna()
    source._setup_client = Mock()
    source.read_all = Mock()
    source.find_index_for_stream = find_index_for_stream
    source.read_updates = read_updates_hardcoded
    source.read_removes = read_removes_hardcoded
    source.client = MagicMock()
    source.find_emitted_at = Mock(return_value=NOW)
    logger = mock_logger()
    result = list(source.read(logger, config({'collection': {'name': 'my_stream_name', 'deletions': {'deletion_mode': 'ignore'}}}), ConfiguredAirbyteCatalog(streams=[ConfiguredAirbyteStream(sync_mode=SyncMode.incremental, destination_sync_mode=DestinationSyncMode.append_dedup, stream=AirbyteStream(name='my_stream_name', json_schema={}, supported_sync_modes=[SyncMode.incremental, SyncMode.full_refresh]))]), state={}))
    assert result == [record('my_stream_name', {'some_document': 'data_here', 'ts': 5}), record('my_stream_name', {'more_document': 'data_here', 'ts': 3}), state({'my_stream_name': {'updates_cursor': {}}})]
    assert source._setup_client.called
    assert not was_called
    assert not source.read_all.called
    assert not logger.error.called

def test_read_removes_resume_from_partial_failure():
    if False:
        for i in range(10):
            print('nop')
    PAGE_SIZE = 12344315
    FIRST_AFTER_TOKEN = ['some magical', 3, 'data']
    SECOND_AFTER_TOKEN = ['even more magical', 3, 'data']

    def make_query(after):
        if False:
            return 10
        return q.map_(q.lambda_('x', {'ref': q.select('document', q.var('x')), 'ts': q.select('ts', q.var('x'))}), q.filter_(q.lambda_('x', q.equals(q.select(['action'], q.var('x')), 'remove')), q.paginate(q.documents(q.collection('foo')), events=True, size=PAGE_SIZE, after=after)))
    current_query = 0
    QUERIES = [make_query(after={'ts': 0, 'action': 'remove'}), make_query(after=FIRST_AFTER_TOKEN), make_query(after=SECOND_AFTER_TOKEN), make_query(after={'ts': 12345, 'action': 'remove', 'resource': q.ref(q.collection('foo'), '3')})]
    QUERY_RESULTS = [results([{'ref': ref(100), 'ts': 99}], after=FIRST_AFTER_TOKEN), results([{'ref': ref(5), 'ts': 999}], after=SECOND_AFTER_TOKEN), results([{'ref': ref(3), 'ts': 12345}], after=None), results([{'ref': ref(3), 'ts': 12345}], after=None)]
    failed_yet = False

    def find_index_for_stream(collection: str) -> str:
        if False:
            while True:
                i = 10
        return 'foo_ts'

    def query_hardcoded(expr):
        if False:
            while True:
                i = 10
        nonlocal current_query
        nonlocal failed_yet
        assert expr == QUERIES[current_query]
        result = QUERY_RESULTS[current_query]
        if current_query == 2 and (not failed_yet):
            failed_yet = True
            raise ValueError('something has gone terribly wrong')
        current_query += 1
        return result
    source = SourceFauna()
    source._setup_client = Mock()
    source.client = MagicMock()
    source.find_index_for_stream = find_index_for_stream
    source.client.query = query_hardcoded
    logger = mock_logger()
    stream = Mock()
    stream.stream.name = 'foo'
    state = {}
    config = CollectionConfig(page_size=PAGE_SIZE)
    outputs = []
    try:
        for output in source.read_removes(logger, stream, config, state, deletion_column='deletes_here'):
            outputs.append(output)
    except ValueError:
        pass
    assert outputs == [{'ref': '100', 'ts': 99, 'deletes_here': datetime.utcfromtimestamp(99 / 1000000).isoformat()}, {'ref': '5', 'ts': 999, 'deletes_here': datetime.utcfromtimestamp(999 / 1000000).isoformat()}]
    assert state == {'after': _json.to_json(SECOND_AFTER_TOKEN)}
    result = list(source.read_removes(logger, stream, config, state, deletion_column='deletes_here'))
    assert result == [{'ref': '3', 'ts': 12345, 'deletes_here': datetime.utcfromtimestamp(12345 / 1000000).isoformat()}]
    assert state == {'ts': 12345, 'ref': '3'}
    result = list(source.read_removes(logger, stream, config, state, deletion_column='deletes_here'))
    assert result == []
    assert state == {'ts': 12345, 'ref': '3'}
    assert not source._setup_client.called
    assert current_query == 4
    assert failed_yet
    assert not logger.error.called

def test_read_remove_deletions():
    if False:
        while True:
            i = 10
    DATE = datetime(2022, 4, 3).replace(tzinfo=timezone.utc)
    TS = DATE.timestamp() * 1000000
    PAGE_SIZE = 12344315

    def make_query(after):
        if False:
            return 10
        return q.map_(q.lambda_('x', {'ref': q.select('document', q.var('x')), 'ts': q.select('ts', q.var('x'))}), q.filter_(q.lambda_('x', q.equals(q.select(['action'], q.var('x')), 'remove')), q.paginate(q.documents(q.collection('foo')), events=True, size=PAGE_SIZE, after=after)))
    current_query = 0
    QUERIES = [make_query(after={'ts': 0, 'action': 'remove'}), make_query(after={'ts': TS, 'action': 'remove', 'resource': q.ref(q.collection('foo'), '100')}), make_query(after={'ts': TS, 'action': 'remove', 'resource': q.ref(q.collection('foo'), '100')})]
    QUERY_RESULTS = [results([{'ref': ref(100), 'ts': TS}], after=None), results([{'ref': ref(100), 'ts': TS}], after=None), results([{'ref': ref(100), 'ts': TS}, {'ref': ref(300), 'ts': TS + 1000000}], after=None)]

    def find_index_for_stream(collection: str) -> str:
        if False:
            i = 10
            return i + 15
        return 'foo_ts'

    def query_hardcoded(expr):
        if False:
            return 10
        nonlocal current_query
        assert expr == QUERIES[current_query]
        result = QUERY_RESULTS[current_query]
        current_query += 1
        return result
    source = SourceFauna()
    source._setup_client = Mock()
    source.client = MagicMock()
    source.find_index_for_stream = find_index_for_stream
    source.client.query = query_hardcoded
    logger = mock_logger()
    stream = Mock()
    stream.stream.name = 'foo'
    state = {}
    config = CollectionConfig(page_size=PAGE_SIZE)
    outputs = list(source.read_removes(logger, stream, config, state, deletion_column='my_deleted_column'))
    assert outputs == [{'ref': '100', 'ts': TS, 'my_deleted_column': '2022-04-03T00:00:00'}]
    assert state == {'ts': TS, 'ref': '100'}
    outputs = list(source.read_removes(logger, stream, config, state, deletion_column='my_deleted_column'))
    assert outputs == []
    assert state == {'ts': TS, 'ref': '100'}
    outputs = list(source.read_removes(logger, stream, config, state, deletion_column='my_deleted_column'))
    assert outputs == [{'ref': '300', 'ts': TS + 1000000, 'my_deleted_column': '2022-04-03T00:00:01'}]
    assert state == {'ts': TS + 1000000, 'ref': '300'}
    assert not source._setup_client.called
    assert current_query == 3
    assert not logger.error.called

def test_read_updates_query():
    if False:
        print('Hello World!')
    '\n    Validates that read_updates() queries the database correctly.\n    '
    PAGE_SIZE = 12344315
    INDEX = 'my_index_name'
    FIRST_AFTER_TOKEN = ['some magical', 3, 'data']
    SECOND_AFTER_TOKEN = ['even more magical', 3, 'data']
    state = {}

    def make_query(after, start=[0]):
        if False:
            for i in range(10):
                print('nop')
        return q.map_(q.lambda_('x', expand_columns_query(q.select(1, q.var('x')))), q.paginate(q.range(q.match(q.index(INDEX)), start, []), after=after, size=PAGE_SIZE))
    current_query = 0
    QUERIES = [make_query(after=None), make_query(after=FIRST_AFTER_TOKEN), make_query(after=SECOND_AFTER_TOKEN), make_query(after=None, start=[999, q.ref(q.collection('my_stream_name'), '10')]), make_query(after=None, start=[999, q.ref(q.collection('my_stream_name'), '10')])]
    QUERY_RESULTS = [results([{'ref': '3', 'ts': 99}], after=FIRST_AFTER_TOKEN), results([{'ref': '5', 'ts': 123}], after=SECOND_AFTER_TOKEN), results([{'ref': '10', 'ts': 999}], after=None), results([{'ref': '10', 'ts': 999}], after=None), results([{'ref': '10', 'ts': 999}, {'ref': '11', 'ts': 1000}], after=None)]

    def query_hardcoded(expr):
        if False:
            for i in range(10):
                print('nop')
        nonlocal current_query
        assert expr == QUERIES[current_query]
        result = QUERY_RESULTS[current_query]
        current_query += 1
        return result
    source = SourceFauna()
    source._setup_client = Mock()
    source.client = MagicMock()
    source.find_index_for_stream = Mock()
    source.client.query = query_hardcoded
    source.find_emitted_at = Mock(return_value=NOW)
    logger = mock_logger()
    result = list(source.read_updates(logger, ConfiguredAirbyteStream(sync_mode=SyncMode.incremental, destination_sync_mode=DestinationSyncMode.append_dedup, stream=AirbyteStream(name='my_stream_name', json_schema={}, supported_sync_modes=[SyncMode.incremental, SyncMode.full_refresh])), CollectionConfig(page_size=PAGE_SIZE), state=state, index=INDEX, page_size=PAGE_SIZE))
    assert result == [{'ref': '3', 'ts': 99}, {'ref': '5', 'ts': 123}, {'ref': '10', 'ts': 999}]
    assert state == {'ref': '10', 'ts': 999}
    result = list(source.read_updates(logger, ConfiguredAirbyteStream(sync_mode=SyncMode.incremental, destination_sync_mode=DestinationSyncMode.append_dedup, stream=AirbyteStream(name='my_stream_name', json_schema={}, supported_sync_modes=[SyncMode.incremental, SyncMode.full_refresh])), CollectionConfig(page_size=PAGE_SIZE), state=state, index=INDEX, page_size=PAGE_SIZE))
    assert result == []
    assert state == {'ref': '10', 'ts': 999}
    result = list(source.read_updates(logger, ConfiguredAirbyteStream(sync_mode=SyncMode.incremental, destination_sync_mode=DestinationSyncMode.append_dedup, stream=AirbyteStream(name='my_stream_name', json_schema={}, supported_sync_modes=[SyncMode.incremental, SyncMode.full_refresh])), CollectionConfig(page_size=PAGE_SIZE), state=state, index=INDEX, page_size=PAGE_SIZE))
    assert result == [{'ref': '11', 'ts': 1000}]
    assert state == {'ref': '11', 'ts': 1000}
    assert not source._setup_client.called
    assert not source.find_index_for_stream.called
    assert not logger.error.called
    assert current_query == 5

def test_read_updates_resume():
    if False:
        i = 10
        return i + 15
    '\n    Validates that read_updates() queries the database correctly, and resumes\n    a failed query correctly.\n    '
    PAGE_SIZE = 12344315
    INDEX = 'my_index_name'
    FIRST_AFTER_TOKEN = ['some magical', 3, 'data']
    SECOND_AFTER_TOKEN = ['even more magical', 3, 'data']

    def make_query(after):
        if False:
            print('Hello World!')
        return q.map_(q.lambda_('x', expand_columns_query(q.select(1, q.var('x')))), q.paginate(q.range(q.match(q.index(INDEX)), [0], []), after=after, size=PAGE_SIZE))
    current_query = 0
    QUERIES = [make_query(after=None), make_query(after=FIRST_AFTER_TOKEN), make_query(after=SECOND_AFTER_TOKEN)]
    QUERY_RESULTS = [results([{'ref': '3', 'ts': 99}], after=FIRST_AFTER_TOKEN), results([{'ref': '5', 'ts': 123}], after=SECOND_AFTER_TOKEN), results([{'ref': '10', 'ts': 999}], after=None)]
    failed_yet = False

    def query_hardcoded(expr):
        if False:
            print('Hello World!')
        nonlocal current_query
        nonlocal failed_yet
        assert expr == QUERIES[current_query]
        result = QUERY_RESULTS[current_query]
        if current_query == 1 and (not failed_yet):
            failed_yet = True
            raise ValueError('oh no something went wrong')
        current_query += 1
        return result
    source = SourceFauna()
    source._setup_client = Mock()
    source.client = MagicMock()
    source.find_index_for_stream = Mock()
    source.client.query = query_hardcoded
    source.find_emitted_at = Mock(return_value=NOW)
    state = {}
    logger = mock_logger()
    result = []
    got_error = False
    try:
        for record in source.read_updates(logger, ConfiguredAirbyteStream(sync_mode=SyncMode.incremental, destination_sync_mode=DestinationSyncMode.append_dedup, stream=AirbyteStream(name='my_stream_name', json_schema={}, supported_sync_modes=[SyncMode.incremental, SyncMode.full_refresh])), CollectionConfig(page_size=PAGE_SIZE), state=state, index=INDEX, page_size=PAGE_SIZE):
            result.append(record)
    except ValueError:
        got_error = True
    assert 'ts' not in state
    assert 'ref' not in state
    assert 'after' in state
    assert got_error
    assert current_query == 1
    assert result == [{'ref': '3', 'ts': 99}]
    assert list(source.read_updates(logger, ConfiguredAirbyteStream(sync_mode=SyncMode.incremental, destination_sync_mode=DestinationSyncMode.append_dedup, stream=AirbyteStream(name='my_stream_name', json_schema={}, supported_sync_modes=[SyncMode.incremental, SyncMode.full_refresh])), CollectionConfig(page_size=PAGE_SIZE), state=state, index=INDEX, page_size=PAGE_SIZE)) == [{'ref': '5', 'ts': 123}, {'ref': '10', 'ts': 999}]
    assert state['ts'] == 999
    assert state['ref'] == '10'
    assert 'after' not in state
    assert not source._setup_client.called
    assert not source.find_index_for_stream.called
    assert not logger.error.called
    assert current_query == 3