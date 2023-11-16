from unittest.mock import MagicMock, Mock
from faunadb import _json
from faunadb import query as q
from source_fauna import SourceFauna
from test_util import CollectionConfig, expand_columns_query, mock_logger

def results(modified, after):
    if False:
        while True:
            i = 10
    modified_obj = {'data': modified}
    if after is not None:
        modified_obj['after'] = after
    return modified_obj

def test_read_all():
    if False:
        return 10
    TS = 12342134
    PAGE_SIZE = 12344315
    FIRST_AFTER_TOKEN = ['some magical', 3, 'data']
    current_query = 0
    QUERIES = [q.at(TS, q.map_(q.lambda_('x', expand_columns_query(q.var('x'))), q.paginate(q.documents(q.collection('my_stream_name')), size=PAGE_SIZE))), q.at(TS, q.map_(q.lambda_('x', expand_columns_query(q.var('x'))), q.paginate(q.documents(q.collection('my_stream_name')), size=PAGE_SIZE, after=FIRST_AFTER_TOKEN)))]
    QUERY_RESULTS = [results([{'ref': '3', 'ts': 12345, 'data': {'foo': 'bar'}}], after=FIRST_AFTER_TOKEN), results([{'ref': '5', 'ts': 9999999, 'data': {'more': 'data here'}}], after=None)]

    def query_hardcoded(expr):
        if False:
            i = 10
            return i + 15
        nonlocal current_query
        assert expr == QUERIES[current_query]
        result = QUERY_RESULTS[current_query]
        current_query += 1
        return result
    source = SourceFauna()
    source._setup_client = Mock()
    source.client = MagicMock()
    source.client.query = query_hardcoded
    logger = mock_logger()
    stream = Mock()
    stream.stream.name = 'my_stream_name'
    result = list(source.read_all(logger, stream, conf=CollectionConfig(page_size=PAGE_SIZE), state={'full_sync_cursor': {'ts': TS}}))
    assert result == [{'ref': '3', 'ts': 12345, 'data': {'foo': 'bar'}}, {'ref': '5', 'ts': 9999999, 'data': {'more': 'data here'}}]
    assert not source._setup_client.called
    assert current_query == 2
    assert not logger.error.called

def test_read_all_extra_columns():
    if False:
        print('Hello World!')

    def expand_columns_query_with_extra(ref):
        if False:
            for i in range(10):
                print('nop')
        doc = q.var('document')
        return q.let({'document': q.get(ref)}, {'ref': q.select(['ref', 'id'], doc), 'ts': q.select('ts', doc), 'data': q.select('data', doc, {}), 'ttl': q.select('ttl', doc, None)})
    TS = 12342134
    PAGE_SIZE = 12344315
    current_query = 0
    QUERIES = [q.at(TS, q.map_(q.lambda_('x', expand_columns_query_with_extra(q.var('x'))), q.paginate(q.documents(q.collection('my_stream_name')), size=PAGE_SIZE)))]
    QUERY_RESULTS = [results([{'ref': '3', 'ts': 12345, 'data': {'my_column': 'fancy string here', 'optional_data': 3}}, {'ref': '5', 'ts': 123459, 'data': {'my_column': 'another fancy string here', 'optional_data': 5}}, {'ref': '7', 'ts': 1234599, 'data': {'my_column': 'even more fancy string here', 'optional_data': None}}], after=None)]

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
    source.client.query = query_hardcoded
    logger = mock_logger()
    stream = Mock()
    stream.stream.name = 'my_stream_name'
    result = list(source.read_all(logger, stream, conf=CollectionConfig(page_size=PAGE_SIZE), state={'full_sync_cursor': {'ts': TS}}))
    assert result == [{'ref': '3', 'ts': 12345, 'data': {'my_column': 'fancy string here', 'optional_data': 3}}, {'ref': '5', 'ts': 123459, 'data': {'my_column': 'another fancy string here', 'optional_data': 5}}, {'ref': '7', 'ts': 1234599, 'data': {'my_column': 'even more fancy string here', 'optional_data': None}}]
    assert not source._setup_client.called
    assert current_query == 1
    assert not logger.error.called

def test_read_all_resume():
    if False:
        return 10
    TS = 12342134
    PAGE_SIZE = 12344315
    FIRST_AFTER_TOKEN = ['some magical', 3, 'data']
    SECOND_AFTER_TOKEN = ['even more magical', 3, 'data']

    def make_query(after):
        if False:
            print('Hello World!')
        return q.at(TS, q.map_(q.lambda_('x', expand_columns_query(q.var('x'))), q.paginate(q.documents(q.collection('foo')), size=PAGE_SIZE, after=after)))
    current_query = 0
    QUERIES = [make_query(after=None), make_query(after=FIRST_AFTER_TOKEN), make_query(after=SECOND_AFTER_TOKEN)]
    QUERY_RESULTS = [results([{'ref': '3', 'ts': 12345, 'data': {'foo': 'bar'}}], after=FIRST_AFTER_TOKEN), results([{'ref': '5', 'ts': 9999999, 'data': {'more': 'data here'}}], after=SECOND_AFTER_TOKEN), results([{'ref': '100', 'ts': 92321341234, 'data': {'last data': 'some data'}}], after=None)]
    failed_yet = False

    def query_hardcoded(expr):
        if False:
            print('Hello World!')
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
    source.client.query = query_hardcoded
    logger = mock_logger()
    stream = Mock()
    stream.stream.name = 'foo'
    state = {'full_sync_cursor': {'ts': TS}}
    config = CollectionConfig(page_size=PAGE_SIZE)
    outputs = []
    try:
        for output in source.read_all(logger, stream, config, state):
            outputs.append(output)
    except ValueError:
        pass
    assert outputs == [{'ref': '3', 'ts': 12345, 'data': {'foo': 'bar'}}, {'ref': '5', 'ts': 9999999, 'data': {'more': 'data here'}}]
    assert state == {'full_sync_cursor': {'ts': TS, 'after': _json.to_json(SECOND_AFTER_TOKEN)}}
    result = list(source.read_all(logger, stream, config, state))
    assert result == [{'ref': '100', 'ts': 92321341234, 'data': {'last data': 'some data'}}]
    assert not source._setup_client.called
    assert current_query == 3
    assert failed_yet
    assert not logger.error.called