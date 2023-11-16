"""Unit tests for Superset"""
import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest import mock
import prison
import pytest
from superset import app, db
from superset.common.utils.query_cache_manager import QueryCacheManager
from superset.connectors.sqla.models import SqlaTable, SqlMetric, TableColumn
from superset.constants import CacheRegion
from superset.daos.exceptions import DatasourceNotFound, DatasourceTypeNotSupportedError
from superset.datasets.commands.exceptions import DatasetNotFoundError
from superset.exceptions import SupersetGenericDBErrorException
from superset.models.core import Database
from superset.utils.core import backend, get_example_default_schema
from superset.utils.database import get_example_database, get_main_database
from tests.integration_tests.base_tests import db_insert_temp_object, SupersetTestCase
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from tests.integration_tests.fixtures.datasource import get_datasource_post

@contextmanager
def create_test_table_context(database: Database):
    if False:
        return 10
    schema = get_example_default_schema()
    full_table_name = f'{schema}.test_table' if schema else 'test_table'
    with database.get_sqla_engine_with_context() as engine:
        engine.execute(f'CREATE TABLE IF NOT EXISTS {full_table_name} AS SELECT 1 as first, 2 as second')
        engine.execute(f'INSERT INTO {full_table_name} (first, second) VALUES (1, 2)')
        engine.execute(f'INSERT INTO {full_table_name} (first, second) VALUES (3, 4)')
    yield db.session
    with database.get_sqla_engine_with_context() as engine:
        engine.execute(f'DROP TABLE {full_table_name}')

class TestDatasource(SupersetTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        db.session.begin(subtransactions=True)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        db.session.rollback()

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_external_metadata_for_physical_table(self):
        if False:
            return 10
        self.login(username='admin')
        tbl = self.get_table(name='birth_names')
        url = f'/datasource/external_metadata/table/{tbl.id}/'
        resp = self.get_json_resp(url)
        col_names = {o.get('column_name') for o in resp}
        self.assertEqual(col_names, {'num_boys', 'num', 'gender', 'name', 'ds', 'state', 'num_girls'})

    def test_always_filter_main_dttm(self):
        if False:
            for i in range(10):
                print('nop')
        self.login(username='admin')
        session = db.session
        database = get_example_database()
        sql = f'SELECT DATE() as default_dttm, DATE() as additional_dttm, 1 as metric;'
        if database.backend == 'sqlite':
            pass
        elif database.backend in ['postgresql', 'mysql']:
            sql = sql.replace('DATE()', 'NOW()')
        else:
            return
        query_obj = {'columns': ['metric'], 'filter': [], 'from_dttm': datetime.now() - timedelta(days=1), 'granularity': 'additional_dttm', 'orderby': [], 'to_dttm': datetime.now() + timedelta(days=1), 'series_columns': [], 'row_limit': 1000, 'row_offset': 0}
        table = SqlaTable(table_name='dummy_sql_table', database=database, schema=get_example_default_schema(), main_dttm_col='default_dttm', columns=[TableColumn(column_name='default_dttm', type='DATETIME', is_dttm=True), TableColumn(column_name='additional_dttm', type='DATETIME', is_dttm=True)], sql=sql)
        session.add(table)
        session.commit()
        table.always_filter_main_dttm = False
        result = str(table.get_sqla_query(**query_obj).sqla_query.whereclause)
        assert 'default_dttm' not in result and 'additional_dttm' in result
        table.always_filter_main_dttm = True
        result = str(table.get_sqla_query(**query_obj).sqla_query.whereclause)
        assert 'default_dttm' in result and 'additional_dttm' in result
        session.delete(table)
        session.commit()

    def test_external_metadata_for_virtual_table(self):
        if False:
            for i in range(10):
                print('nop')
        self.login(username='admin')
        session = db.session
        table = SqlaTable(table_name='dummy_sql_table', database=get_example_database(), schema=get_example_default_schema(), sql="select 123 as intcol, 'abc' as strcol")
        session.add(table)
        session.commit()
        table = self.get_table(name='dummy_sql_table')
        url = f'/datasource/external_metadata/table/{table.id}/'
        resp = self.get_json_resp(url)
        assert {o.get('column_name') for o in resp} == {'intcol', 'strcol'}
        session.delete(table)
        session.commit()

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_external_metadata_by_name_for_physical_table(self):
        if False:
            for i in range(10):
                print('nop')
        self.login(username='admin')
        tbl = self.get_table(name='birth_names')
        params = prison.dumps({'datasource_type': 'table', 'database_name': tbl.database.database_name, 'schema_name': tbl.schema, 'table_name': tbl.table_name, 'normalize_columns': tbl.normalize_columns, 'always_filter_main_dttm': tbl.always_filter_main_dttm})
        url = f'/datasource/external_metadata_by_name/?q={params}'
        resp = self.get_json_resp(url)
        col_names = {o.get('column_name') for o in resp}
        self.assertEqual(col_names, {'num_boys', 'num', 'gender', 'name', 'ds', 'state', 'num_girls'})

    def test_external_metadata_by_name_for_virtual_table(self):
        if False:
            return 10
        self.login(username='admin')
        session = db.session
        table = SqlaTable(table_name='dummy_sql_table', database=get_example_database(), schema=get_example_default_schema(), sql="select 123 as intcol, 'abc' as strcol")
        session.add(table)
        session.commit()
        tbl = self.get_table(name='dummy_sql_table')
        params = prison.dumps({'datasource_type': 'table', 'database_name': tbl.database.database_name, 'schema_name': tbl.schema, 'table_name': tbl.table_name, 'normalize_columns': tbl.normalize_columns, 'always_filter_main_dttm': tbl.always_filter_main_dttm})
        url = f'/datasource/external_metadata_by_name/?q={params}'
        resp = self.get_json_resp(url)
        assert {o.get('column_name') for o in resp} == {'intcol', 'strcol'}
        session.delete(tbl)
        session.commit()

    def test_external_metadata_by_name_from_sqla_inspector(self):
        if False:
            for i in range(10):
                print('nop')
        self.login(username='admin')
        example_database = get_example_database()
        with create_test_table_context(example_database):
            params = prison.dumps({'datasource_type': 'table', 'database_name': example_database.database_name, 'table_name': 'test_table', 'schema_name': get_example_default_schema(), 'normalize_columns': False, 'always_filter_main_dttm': False})
            url = f'/datasource/external_metadata_by_name/?q={params}'
            resp = self.get_json_resp(url)
            col_names = {o.get('column_name') for o in resp}
            self.assertEqual(col_names, {'first', 'second'})
        params = prison.dumps({'datasource_type': 'table', 'database_name': 'foo', 'table_name': 'bar', 'normalize_columns': False, 'always_filter_main_dttm': False})
        url = f'/datasource/external_metadata_by_name/?q={params}'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, DatasetNotFoundError.status)
        self.assertEqual(json.loads(resp.data.decode('utf-8')).get('error'), DatasetNotFoundError.message)
        params = prison.dumps({'datasource_type': 'table', 'database_name': example_database.database_name, 'table_name': 'fooooooooobarrrrrr', 'normalize_columns': False, 'always_filter_main_dttm': False})
        url = f'/datasource/external_metadata_by_name/?q={params}'
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, DatasetNotFoundError.status)
        self.assertEqual(json.loads(resp.data.decode('utf-8')).get('error'), DatasetNotFoundError.message)
        params = prison.dumps({'datasource_type': 'table'})
        url = f'/datasource/external_metadata_by_name/?q={params}'
        resp = self.get_json_resp(url)
        self.assertIn('error', resp)

    def test_external_metadata_for_virtual_table_template_params(self):
        if False:
            for i in range(10):
                print('nop')
        self.login(username='admin')
        session = db.session
        table = SqlaTable(table_name='dummy_sql_table_with_template_params', database=get_example_database(), schema=get_example_default_schema(), sql='select {{ foo }} as intcol', template_params=json.dumps({'foo': '123'}))
        session.add(table)
        session.commit()
        table = self.get_table(name='dummy_sql_table_with_template_params')
        url = f'/datasource/external_metadata/table/{table.id}/'
        resp = self.get_json_resp(url)
        assert {o.get('column_name') for o in resp} == {'intcol'}
        session.delete(table)
        session.commit()

    def test_external_metadata_for_malicious_virtual_table(self):
        if False:
            return 10
        self.login(username='admin')
        table = SqlaTable(table_name='malicious_sql_table', database=get_example_database(), schema=get_example_default_schema(), sql='delete table birth_names')
        with db_insert_temp_object(table):
            url = f'/datasource/external_metadata/table/{table.id}/'
            resp = self.get_json_resp(url)
            self.assertEqual(resp['error'], 'Only `SELECT` statements are allowed')

    def test_external_metadata_for_multistatement_virtual_table(self):
        if False:
            print('Hello World!')
        self.login(username='admin')
        table = SqlaTable(table_name='multistatement_sql_table', database=get_example_database(), schema=get_example_default_schema(), sql="select 123 as intcol, 'abc' as strcol;select 123 as intcol, 'abc' as strcol")
        with db_insert_temp_object(table):
            url = f'/datasource/external_metadata/table/{table.id}/'
            resp = self.get_json_resp(url)
            self.assertEqual(resp['error'], 'Only single queries supported')

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @mock.patch('superset.connectors.sqla.models.SqlaTable.external_metadata')
    def test_external_metadata_error_return_400(self, mock_get_datasource):
        if False:
            for i in range(10):
                print('nop')
        self.login(username='admin')
        tbl = self.get_table(name='birth_names')
        url = f'/datasource/external_metadata/table/{tbl.id}/'
        mock_get_datasource.side_effect = SupersetGenericDBErrorException('oops')
        pytest.raises(SupersetGenericDBErrorException, lambda : db.session.query(SqlaTable).filter_by(id=tbl.id).one_or_none().external_metadata())
        resp = self.client.get(url)
        assert resp.status_code == 400

    def compare_lists(self, l1, l2, key):
        if False:
            print('Hello World!')
        l2_lookup = {o.get(key): o for o in l2}
        for obj1 in l1:
            obj2 = l2_lookup.get(obj1.get(key))
            for k in obj1:
                if k not in 'id' and obj1.get(k):
                    self.assertEqual(obj1.get(k), obj2.get(k))

    def test_save(self):
        if False:
            return 10
        self.login(username='admin')
        tbl_id = self.get_table(name='birth_names').id
        datasource_post = get_datasource_post()
        datasource_post['id'] = tbl_id
        datasource_post['owners'] = [1]
        data = dict(data=json.dumps(datasource_post))
        resp = self.get_json_resp('/datasource/save/', data)
        for k in datasource_post:
            if k == 'columns':
                self.compare_lists(datasource_post[k], resp[k], 'column_name')
            elif k == 'metrics':
                self.compare_lists(datasource_post[k], resp[k], 'metric_name')
            elif k == 'database':
                self.assertEqual(resp[k]['id'], datasource_post[k]['id'])
            elif k == 'owners':
                self.assertEqual([o['id'] for o in resp[k]], datasource_post['owners'])
            else:
                print(k)
                self.assertEqual(resp[k], datasource_post[k])

    def test_save_default_endpoint_validation_success(self):
        if False:
            print('Hello World!')
        self.login(username='admin')
        tbl_id = self.get_table(name='birth_names').id
        datasource_post = get_datasource_post()
        datasource_post['id'] = tbl_id
        datasource_post['owners'] = [1]
        datasource_post['default_endpoint'] = 'http://localhost/superset/1'
        data = dict(data=json.dumps(datasource_post))
        resp = self.client.post('/datasource/save/', data=data)
        assert resp.status_code == 200

    def save_datasource_from_dict(self, datasource_post):
        if False:
            while True:
                i = 10
        data = dict(data=json.dumps(datasource_post))
        resp = self.get_json_resp('/datasource/save/', data)
        return resp

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_change_database(self):
        if False:
            i = 10
            return i + 15
        self.login(username='admin')
        admin_user = self.get_user('admin')
        tbl = self.get_table(name='birth_names')
        tbl_id = tbl.id
        db_id = tbl.database_id
        datasource_post = get_datasource_post()
        datasource_post['id'] = tbl_id
        datasource_post['owners'] = [admin_user.id]
        new_db = self.create_fake_db()
        datasource_post['database']['id'] = new_db.id
        resp = self.save_datasource_from_dict(datasource_post)
        self.assertEqual(resp['database']['id'], new_db.id)
        datasource_post['database']['id'] = db_id
        resp = self.save_datasource_from_dict(datasource_post)
        self.assertEqual(resp['database']['id'], db_id)
        self.delete_fake_db()

    def test_save_duplicate_key(self):
        if False:
            while True:
                i = 10
        self.login(username='admin')
        admin_user = self.get_user('admin')
        tbl_id = self.get_table(name='birth_names').id
        datasource_post = get_datasource_post()
        datasource_post['id'] = tbl_id
        datasource_post['owners'] = [admin_user.id]
        datasource_post['columns'].extend([{'column_name': '<new column>', 'filterable': True, 'groupby': True, 'expression': '<enter SQL expression here>', 'id': 'somerandomid'}, {'column_name': '<new column>', 'filterable': True, 'groupby': True, 'expression': '<enter SQL expression here>', 'id': 'somerandomid2'}])
        data = dict(data=json.dumps(datasource_post))
        resp = self.get_json_resp('/datasource/save/', data, raise_on_error=False)
        self.assertIn('Duplicate column name(s): <new column>', resp['error'])

    def test_get_datasource(self):
        if False:
            for i in range(10):
                print('nop')
        self.login(username='admin')
        admin_user = self.get_user('admin')
        tbl = self.get_table(name='birth_names')
        datasource_post = get_datasource_post()
        datasource_post['id'] = tbl.id
        datasource_post['owners'] = [admin_user.id]
        data = dict(data=json.dumps(datasource_post))
        self.get_json_resp('/datasource/save/', data)
        url = f'/datasource/get/{tbl.type}/{tbl.id}/'
        resp = self.get_json_resp(url)
        self.assertEqual(resp.get('type'), 'table')
        col_names = {o.get('column_name') for o in resp['columns']}
        self.assertEqual(col_names, {'num_boys', 'num', 'gender', 'name', 'ds', 'state', 'num_girls', 'num_california'})

    def test_get_datasource_with_health_check(self):
        if False:
            print('Hello World!')

        def my_check(datasource):
            if False:
                while True:
                    i = 10
            return 'Warning message!'
        app.config['DATASET_HEALTH_CHECK'] = my_check
        self.login(username='admin')
        tbl = self.get_table(name='birth_names')
        datasource = db.session.query(SqlaTable).filter_by(id=tbl.id).one_or_none()
        assert datasource.health_check_message == 'Warning message!'
        app.config['DATASET_HEALTH_CHECK'] = None

    def test_get_datasource_failed(self):
        if False:
            while True:
                i = 10
        from superset.daos.datasource import DatasourceDAO
        pytest.raises(DatasourceNotFound, lambda : DatasourceDAO.get_datasource(db.session, 'table', 9999999))
        self.login(username='admin')
        resp = self.get_json_resp('/datasource/get/table/500000/', raise_on_error=False)
        self.assertEqual(resp.get('error'), 'Datasource does not exist')

    def test_get_datasource_invalid_datasource_failed(self):
        if False:
            return 10
        from superset.daos.datasource import DatasourceDAO
        pytest.raises(DatasourceTypeNotSupportedError, lambda : DatasourceDAO.get_datasource(db.session, 'druid', 9999999))
        self.login(username='admin')
        resp = self.get_json_resp('/datasource/get/druid/500000/', raise_on_error=False)
        self.assertEqual(resp.get('error'), "'druid' is not a valid DatasourceType")

def test_get_samples(test_client, login_as_admin, virtual_dataset):
    if False:
        while True:
            i = 10
    '\n    Dataset API: Test get dataset samples\n    '
    uri = f'/datasource/samples?datasource_id={virtual_dataset.id}&datasource_type=table'
    test_client.post(uri, json={})
    rv = test_client.post(uri, json={})
    assert rv.status_code == 200
    assert len(rv.json['result']['data']) == 10
    assert QueryCacheManager.has(rv.json['result']['cache_key'], region=CacheRegion.DATA)
    assert rv.json['result']['is_cached']
    uri2 = f'/datasource/samples?datasource_id={virtual_dataset.id}&datasource_type=table&force=true'
    test_client.post(uri2, json={})
    rv2 = test_client.post(uri2, json={})
    assert rv2.status_code == 200
    assert len(rv2.json['result']['data']) == 10
    assert QueryCacheManager.has(rv2.json['result']['cache_key'], region=CacheRegion.DATA)
    assert not rv2.json['result']['is_cached']
    assert 'colnames' in rv2.json['result']
    assert 'coltypes' in rv2.json['result']
    assert 'data' in rv2.json['result']
    eager_samples = virtual_dataset.database.get_df(f"select * from ({virtual_dataset.sql}) as tbl limit {app.config['SAMPLES_ROW_LIMIT']}")
    eager_samples['col3'] = eager_samples['col3'].apply(float)
    eager_samples = eager_samples.to_dict(orient='records')
    assert eager_samples == rv2.json['result']['data']

def test_get_samples_with_incorrect_cc(test_client, login_as_admin, virtual_dataset):
    if False:
        for i in range(10):
            print('nop')
    TableColumn(column_name='DUMMY CC', type='VARCHAR(255)', table=virtual_dataset, expression='INCORRECT SQL')
    db.session.merge(virtual_dataset)
    uri = f'/datasource/samples?datasource_id={virtual_dataset.id}&datasource_type=table'
    rv = test_client.post(uri, json={})
    assert rv.status_code == 422
    assert 'error' in rv.json
    if virtual_dataset.database.db_engine_spec.engine_name == 'PostgreSQL':
        assert 'INCORRECT SQL' in rv.json.get('error')

def test_get_samples_on_physical_dataset(test_client, login_as_admin, physical_dataset):
    if False:
        while True:
            i = 10
    uri = f'/datasource/samples?datasource_id={physical_dataset.id}&datasource_type=table'
    rv = test_client.post(uri, json={})
    assert rv.status_code == 200
    assert QueryCacheManager.has(rv.json['result']['cache_key'], region=CacheRegion.DATA)
    assert len(rv.json['result']['data']) == 10

def test_get_samples_with_filters(test_client, login_as_admin, virtual_dataset):
    if False:
        for i in range(10):
            print('nop')
    uri = f'/datasource/samples?datasource_id={virtual_dataset.id}&datasource_type=table'
    rv = test_client.post(uri, json=None)
    assert rv.status_code == 415
    rv = test_client.post(uri, json={})
    assert rv.status_code == 200
    rv = test_client.post(uri, json={'foo': 'bar'})
    assert rv.status_code == 400
    rv = test_client.post(uri, json={'filters': [{'col': 'col1', 'op': 'INVALID', 'val': 0}]})
    assert rv.status_code == 400
    rv = test_client.post(uri, json={'filters': [{'col': 'col2', 'op': '==', 'val': 'a'}, {'col': 'col1', 'op': '==', 'val': 0}]})
    assert rv.status_code == 200
    assert rv.json['result']['colnames'] == ['col1', 'col2', 'col3', 'col4', 'col5']
    assert rv.json['result']['rowcount'] == 1
    rv = test_client.post(uri, json={'filters': [{'col': 'col2', 'op': '==', 'val': 'x'}]})
    assert rv.status_code == 200
    assert rv.json['result']['colnames'] == []
    assert rv.json['result']['rowcount'] == 0

def test_get_samples_with_time_filter(test_client, login_as_admin, physical_dataset):
    if False:
        i = 10
        return i + 15
    uri = f'/datasource/samples?datasource_id={physical_dataset.id}&datasource_type=table'
    payload = {'granularity': 'col5', 'time_range': '2000-01-02 : 2000-01-04'}
    rv = test_client.post(uri, json=payload)
    assert len(rv.json['result']['data']) == 2
    if physical_dataset.database.backend != 'sqlite':
        assert [row['col5'] for row in rv.json['result']['data']] == [946771200000.0, 946857600000.0]
    assert rv.json['result']['page'] == 1
    assert rv.json['result']['per_page'] == app.config['SAMPLES_ROW_LIMIT']
    assert rv.json['result']['total_count'] == 2

def test_get_samples_with_multiple_filters(test_client, login_as_admin, physical_dataset):
    if False:
        print('Hello World!')
    uri = f'/datasource/samples?datasource_id={physical_dataset.id}&datasource_type=table'
    payload = {'granularity': 'col5', 'time_range': '2000-01-02 : 2000-01-04', 'filters': [{'col': 'col4', 'op': 'IS NOT NULL'}]}
    rv = test_client.post(uri, json=payload)
    assert len(rv.json['result']['data']) == 0
    payload = {'granularity': 'col5', 'time_range': '2000-01-02 : 2000-01-04', 'filters': [{'col': 'col2', 'op': '==', 'val': 'c'}], 'extras': {'where': 'col3 = 1.2 and col4 is null'}}
    rv = test_client.post(uri, json=payload)
    assert len(rv.json['result']['data']) == 1
    assert rv.json['result']['total_count'] == 1
    assert '2000-01-02' in rv.json['result']['query']
    assert '2000-01-04' in rv.json['result']['query']
    assert 'col3 = 1.2' in rv.json['result']['query']
    assert 'col4 is null' in rv.json['result']['query']
    assert "col2 = 'c'" in rv.json['result']['query']

def test_get_samples_pagination(test_client, login_as_admin, virtual_dataset):
    if False:
        return 10
    uri = f'/datasource/samples?datasource_id={virtual_dataset.id}&datasource_type=table'
    rv = test_client.post(uri, json={})
    assert rv.json['result']['page'] == 1
    assert rv.json['result']['per_page'] == app.config['SAMPLES_ROW_LIMIT']
    assert rv.json['result']['total_count'] == 10
    per_pages = (app.config['SAMPLES_ROW_LIMIT'] + 1, 0, 'xx')
    for per_page in per_pages:
        uri = f'/datasource/samples?datasource_id={virtual_dataset.id}&datasource_type=table&per_page={per_page}'
        rv = test_client.post(uri, json={})
        assert rv.status_code == 400
    uri = f'/datasource/samples?datasource_id={virtual_dataset.id}&datasource_type=table&page=xx'
    rv = test_client.post(uri, json={})
    assert rv.status_code == 400
    uri = f'/datasource/samples?datasource_id={virtual_dataset.id}&datasource_type=xx'
    rv = test_client.post(uri, json={})
    assert rv.status_code == 400
    uri = f'/datasource/samples?datasource_id={virtual_dataset.id}&datasource_type=table&per_page=2&page=1'
    rv = test_client.post(uri, json={})
    assert rv.json['result']['page'] == 1
    assert rv.json['result']['per_page'] == 2
    assert rv.json['result']['total_count'] == 10
    assert [row['col1'] for row in rv.json['result']['data']] == [0, 1]
    uri = f'/datasource/samples?datasource_id={virtual_dataset.id}&datasource_type=table&per_page=2&page=2'
    rv = test_client.post(uri, json={})
    assert rv.json['result']['page'] == 2
    assert rv.json['result']['per_page'] == 2
    assert rv.json['result']['total_count'] == 10
    assert [row['col1'] for row in rv.json['result']['data']] == [2, 3]
    uri = f'/datasource/samples?datasource_id={virtual_dataset.id}&datasource_type=table&per_page=2&page=6'
    rv = test_client.post(uri, json={})
    assert rv.json['result']['page'] == 6
    assert rv.json['result']['per_page'] == 2
    assert rv.json['result']['total_count'] == 10
    assert [row['col1'] for row in rv.json['result']['data']] == []