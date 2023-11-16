from superset import app, db
from superset.common.db_query_status import QueryStatus
from superset.models.core import Database
from superset.models.sql_lab import Query
from superset.sql_lab import execute_sql_statements
from superset.utils.dates import now_as_float

def test_non_async_execute(non_async_example_db: Database, example_query: Query):
    if False:
        i = 10
        return i + 15
    'Test query.tracking_url is attached for Presto and Hive queries'
    result = execute_sql_statements(example_query.id, 'select 1 as foo;', store_results=False, return_results=True, session=db.session, start_time=now_as_float(), expand_data=True, log_params=dict())
    assert result
    assert result['query_id'] == example_query.id
    assert result['status'] == QueryStatus.SUCCESS
    assert result['data'] == [{'foo': 1}]
    if non_async_example_db.db_engine_spec.engine == 'presto':
        assert example_query.tracking_url
        assert '/ui/query.html?' in example_query.tracking_url
        app.config['TRACKING_URL_TRANSFORMER'] = lambda url, query: url.replace('/ui/query.html?', f'/{query.client_id}/')
        assert f'/{example_query.client_id}/' in example_query.tracking_url
        app.config['TRACKING_URL_TRANSFORMER'] = lambda url: url + '&foo=bar'
        assert example_query.tracking_url.endswith('&foo=bar')
    if non_async_example_db.db_engine_spec.engine_name == 'hive':
        assert example_query.tracking_url_raw