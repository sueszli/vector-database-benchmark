import json

def test_column_attributes_on_query():
    if False:
        while True:
            i = 10
    from superset.daos.query import QueryDAO
    from superset.models.core import Database
    from superset.models.sql_lab import Query
    db = Database(database_name='my_database', sqlalchemy_uri='sqlite://')
    query_obj = Query(client_id='foo', database=db, tab_name='test_tab', sql_editor_id='test_editor_id', sql='select * from bar', select_sql='select * from bar', executed_sql='select * from bar', limit=100, select_as_cta=False, rows=100, error_message='none', results_key='abc')
    columns = [{'name': 'test', 'is_dttm': False, 'type': 'INT'}]
    payload = {'columns': columns}
    QueryDAO.save_metadata(query_obj, payload)
    assert 'column_name' in json.loads(query_obj.extra_json).get('columns')[0]