from unittest.mock import MagicMock, Mock
from airbyte_cdk.models import Status
from faunadb import query as q
from faunadb.errors import Unauthorized
from faunadb.objects import Ref
from source_fauna import SourceFauna
from test_util import config, mock_logger

def query_hardcoded(expr):
    if False:
        return 10
    print(expr)
    if expr == q.now():
        return 0
    elif expr == q.paginate(q.collections()):
        return [{'ref': Ref('foo', Ref('collections'))}]
    elif expr == q.paginate(q.indexes()):
        return [{'source': Ref('foo', Ref('collections')), 'values': [{'field': 'ts'}, {'field': 'ref'}], 'terms': []}]
    else:
        raise ValueError(f'invalid query {expr}')

def test_valid_query():
    if False:
        return 10
    source = SourceFauna()
    source._setup_client = Mock()
    source.client = MagicMock()
    source.client.query = query_hardcoded
    logger = mock_logger()
    result = source.check(logger, config=config({}))
    print(result)
    assert result.status == Status.SUCCEEDED
    assert source._setup_client.called
    assert not logger.error.called

def test_invalid_check():
    if False:
        for i in range(10):
            print('nop')
    source = SourceFauna()
    source._setup_client = Mock()
    source.client = MagicMock()
    source.client.query = query_hardcoded
    request_result = MagicMock()
    request_result.response_content = {'errors': [{'code': '403', 'description': 'Unauthorized'}]}
    source.client.query = Mock(side_effect=Unauthorized(request_result))
    print(source.client)
    logger = mock_logger()
    result = source.check(logger, config=config({'secret': 'some invalid secret'}))
    assert result.status == Status.FAILED
    assert result.message == 'Failed to connect to database: Unauthorized'
    assert source._setup_client.called
    assert not logger.error.called