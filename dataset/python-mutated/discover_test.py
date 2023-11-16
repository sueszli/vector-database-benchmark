from unittest.mock import MagicMock, Mock
from airbyte_cdk.models import AirbyteStream
from faunadb import query as q
from faunadb.objects import Ref
from source_fauna import SourceFauna
from test_util import config, mock_logger

def mock_source() -> SourceFauna:
    if False:
        while True:
            i = 10
    source = SourceFauna()
    source._setup_client = Mock()
    source.client = MagicMock()
    return source

def schema(properties) -> dict:
    if False:
        for i in range(10):
            print('nop')
    return {'$schema': 'http://json-schema.org/draft-07/schema#', 'type': 'object', 'properties': properties}

def query_hardcoded(expr):
    if False:
        print('Hello World!')
    print(expr)
    if expr == q.now():
        return 0
    elif expr == q.paginate(q.collections()):
        return {'data': [Ref('foo', Ref('collections')), Ref('bar', Ref('collections'))]}
    elif expr == q.paginate(q.indexes()):
        return {'data': [Ref('ts', Ref('indexes'))]}
    elif expr == q.get(Ref('ts', Ref('indexes'))):
        return {'source': Ref('foo', Ref('collections')), 'name': 'ts', 'values': [{'field': 'ts'}, {'field': 'ref'}], 'terms': []}
    else:
        raise ValueError(f'invalid query {expr}')

def test_simple_discover():
    if False:
        for i in range(10):
            print('nop')
    source = SourceFauna()
    source._setup_client = Mock()
    source.client = MagicMock()
    source.client.query = query_hardcoded
    logger = mock_logger()
    result = source.discover(logger, config=config({}))
    assert result.streams == [AirbyteStream(name='foo', json_schema={'$schema': 'http://json-schema.org/draft-07/schema#', 'type': 'object', 'properties': {'data': {'type': 'object'}, 'ref': {'type': 'string'}, 'ts': {'type': 'integer'}, 'ttl': {'type': ['null', 'integer']}}}, supported_sync_modes=['full_refresh', 'incremental'], source_defined_cursor=True, default_cursor_field=['ts'], source_defined_primary_key=None, namespace=None), AirbyteStream(name='bar', json_schema={'$schema': 'http://json-schema.org/draft-07/schema#', 'type': 'object', 'properties': {'data': {'type': 'object'}, 'ref': {'type': 'string'}, 'ts': {'type': 'integer'}, 'ttl': {'type': ['null', 'integer']}}}, supported_sync_modes=['full_refresh'], source_defined_cursor=True, default_cursor_field=['ts'], source_defined_primary_key=None, namespace=None)]
    assert not logger.info.called
    assert not logger.error.called
    assert source._setup_client.called