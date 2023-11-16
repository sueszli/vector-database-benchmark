"""Unit tests for Superset CSV upload"""
import json
import logging
import os
import shutil
from typing import Optional, Union
from unittest import mock
import pandas as pd
import pytest
import superset.utils.database
from superset.sql_parse import Table
from tests.integration_tests.conftest import ADMIN_SCHEMA_NAME
from superset import db
from superset import security_manager
from superset.models.core import Database
from superset.utils import core as utils
from tests.integration_tests.test_app import app, login
from tests.integration_tests.base_tests import get_resp, SupersetTestCase
logger = logging.getLogger(__name__)
test_client = app.test_client()
CSV_UPLOAD_DATABASE = 'csv_explore_db'
CSV_FILENAME1 = 'testCSV1.csv'
CSV_FILENAME2 = 'testCSV2.csv'
EXCEL_FILENAME = 'testExcel.xlsx'
PARQUET_FILENAME1 = 'testZip/testParquet1.parquet'
PARQUET_FILENAME2 = 'testZip/testParquet2.parquet'
ZIP_DIRNAME = 'testZip'
ZIP_FILENAME = 'testZip.zip'
EXCEL_UPLOAD_TABLE = 'excel_upload'
CSV_UPLOAD_TABLE = 'csv_upload'
PARQUET_UPLOAD_TABLE = 'parquet_upload'
CSV_UPLOAD_TABLE_W_SCHEMA = 'csv_upload_w_schema'
CSV_UPLOAD_TABLE_W_EXPLORE = 'csv_upload_w_explore'

def _setup_csv_upload():
    if False:
        while True:
            i = 10
    upload_db = superset.utils.database.get_or_create_db(CSV_UPLOAD_DATABASE, app.config['SQLALCHEMY_EXAMPLES_URI'])
    extra = upload_db.get_extra()
    extra['explore_database_id'] = superset.utils.database.get_example_database().id
    upload_db.extra = json.dumps(extra)
    upload_db.allow_file_upload = True
    db.session.commit()
    yield
    upload_db = get_upload_db()
    with upload_db.get_sqla_engine_with_context() as engine:
        engine.execute(f'DROP TABLE IF EXISTS {EXCEL_UPLOAD_TABLE}')
        engine.execute(f'DROP TABLE IF EXISTS {CSV_UPLOAD_TABLE}')
        engine.execute(f'DROP TABLE IF EXISTS {PARQUET_UPLOAD_TABLE}')
        engine.execute(f'DROP TABLE IF EXISTS {CSV_UPLOAD_TABLE_W_SCHEMA}')
        engine.execute(f'DROP TABLE IF EXISTS {CSV_UPLOAD_TABLE_W_EXPLORE}')
    db.session.delete(upload_db)
    db.session.commit()

@pytest.fixture(scope='module')
def setup_csv_upload(login_as_admin):
    if False:
        print('Hello World!')
    yield from _setup_csv_upload()

@pytest.fixture(scope='module')
def setup_csv_upload_with_context():
    if False:
        return 10
    with app.app_context():
        login(test_client, username='admin')
        yield from _setup_csv_upload()

@pytest.fixture(scope='module')
def create_csv_files():
    if False:
        while True:
            i = 10
    with open(CSV_FILENAME1, 'w+') as test_file:
        for line in ['a,b', 'john,1', 'paul,2']:
            test_file.write(f'{line}\n')
    with open(CSV_FILENAME2, 'w+') as test_file:
        for line in ['b,c,d', 'john,1,x', 'paul,2,']:
            test_file.write(f'{line}\n')
    yield
    os.remove(CSV_FILENAME1)
    os.remove(CSV_FILENAME2)

@pytest.fixture()
def create_excel_files():
    if False:
        print('Hello World!')
    pd.DataFrame({'a': ['john', 'paul'], 'b': [1, 2]}).to_excel(EXCEL_FILENAME)
    yield
    os.remove(EXCEL_FILENAME)

@pytest.fixture()
def create_columnar_files():
    if False:
        for i in range(10):
            print('nop')
    os.mkdir(ZIP_DIRNAME)
    pd.DataFrame({'a': ['john', 'paul'], 'b': [1, 2]}).to_parquet(PARQUET_FILENAME1)
    pd.DataFrame({'a': ['max', 'bob'], 'b': [3, 4]}).to_parquet(PARQUET_FILENAME2)
    shutil.make_archive(ZIP_DIRNAME, 'zip', ZIP_DIRNAME)
    yield
    os.remove(ZIP_FILENAME)
    shutil.rmtree(ZIP_DIRNAME)

def get_upload_db():
    if False:
        while True:
            i = 10
    return db.session.query(Database).filter_by(database_name=CSV_UPLOAD_DATABASE).one()

def upload_csv(filename: str, table_name: str, extra: Optional[dict[str, str]]=None, dtype: Union[str, None]=None):
    if False:
        i = 10
        return i + 15
    csv_upload_db_id = get_upload_db().id
    form_data = {'csv_file': open(filename, 'rb'), 'delimiter': ',', 'table_name': table_name, 'database': csv_upload_db_id, 'if_exists': 'fail', 'index_label': 'test_label', 'overwrite_duplicate': False}
    if (schema := utils.get_example_default_schema()):
        form_data['schema'] = schema
    if extra:
        form_data.update(extra)
    if dtype:
        form_data['dtype'] = dtype
    return get_resp(test_client, '/csvtodatabaseview/form', data=form_data)

def upload_excel(filename: str, table_name: str, extra: Optional[dict[str, str]]=None):
    if False:
        i = 10
        return i + 15
    excel_upload_db_id = get_upload_db().id
    form_data = {'excel_file': open(filename, 'rb'), 'name': table_name, 'database': excel_upload_db_id, 'sheet_name': 'Sheet1', 'if_exists': 'fail', 'index_label': 'test_label', 'mangle_dupe_cols': False}
    if (schema := utils.get_example_default_schema()):
        form_data['schema'] = schema
    if extra:
        form_data.update(extra)
    return get_resp(test_client, '/exceltodatabaseview/form', data=form_data)

def upload_columnar(filename: str, table_name: str, extra: Optional[dict[str, str]]=None):
    if False:
        return 10
    columnar_upload_db_id = get_upload_db().id
    form_data = {'columnar_file': open(filename, 'rb'), 'name': table_name, 'database': columnar_upload_db_id, 'if_exists': 'fail', 'index_label': 'test_label'}
    if (schema := utils.get_example_default_schema()):
        form_data['schema'] = schema
    if extra:
        form_data.update(extra)
    return get_resp(test_client, '/columnartodatabaseview/form', data=form_data)

def mock_upload_to_s3(filename: str, upload_prefix: str, table: Table) -> str:
    if False:
        print('Hello World!')
    '\n    HDFS is used instead of S3 for the unit tests.integration_tests.\n\n    :param filename: The file to upload\n    :param upload_prefix: The S3 prefix\n    :param table: The table that will be created\n    :returns: The HDFS path to the directory with external table files\n    '
    import docker
    client = docker.from_env()
    container = client.containers.get('namenode')
    src = os.path.join('/tmp/superset_uploads', os.path.basename(filename))
    dest_dir = os.path.join('/tmp/external/superset_uploads/', str(table))
    container.exec_run(f'hdfs dfs -mkdir -p {dest_dir}')
    dest = os.path.join(dest_dir, os.path.basename(filename))
    container.exec_run(f'hdfs dfs -put {src} {dest}')
    return dest_dir

def escaped_double_quotes(text):
    if False:
        i = 10
        return i + 15
    return f'\\&#34;{text}\\&#34;'

def escaped_parquet(text):
    if False:
        return 10
    return escaped_double_quotes(f'[&#39;{text}&#39;]')

@pytest.mark.usefixtures('setup_csv_upload_with_context')
@pytest.mark.usefixtures('create_csv_files')
@mock.patch('superset.models.core.config', {**app.config, 'ALLOWED_USER_CSV_SCHEMA_FUNC': lambda d, u: ['admin_database']})
@mock.patch('superset.db_engine_specs.hive.upload_to_s3', mock_upload_to_s3)
@mock.patch('superset.views.database.views.event_logger.log_with_context')
def test_import_csv_enforced_schema(mock_event_logger):
    if False:
        print('Hello World!')
    if utils.backend() == 'sqlite':
        pytest.skip("Sqlite doesn't support schema / database creation")
    if utils.backend() == 'mysql':
        pytest.skip('This test is flaky on MySQL')
    full_table_name = f'admin_database.{CSV_UPLOAD_TABLE_W_SCHEMA}'
    resp = upload_csv(CSV_FILENAME1, full_table_name)
    assert 'Table name cannot contain a schema' in resp
    resp = upload_csv(CSV_FILENAME1, CSV_UPLOAD_TABLE_W_SCHEMA, extra={'schema': None})
    assert f"Database {escaped_double_quotes(CSV_UPLOAD_DATABASE)} schema {escaped_double_quotes('None')} is not allowed for csv uploads" in resp
    success_msg = f'CSV file {escaped_double_quotes(CSV_FILENAME1)} uploaded to table {escaped_double_quotes(full_table_name)}'
    resp = upload_csv(CSV_FILENAME1, CSV_UPLOAD_TABLE_W_SCHEMA, extra={'schema': 'admin_database', 'if_exists': 'replace'})
    assert success_msg in resp
    mock_event_logger.assert_called_with(action='successful_csv_upload', database=get_upload_db().name, schema='admin_database', table=CSV_UPLOAD_TABLE_W_SCHEMA)
    with get_upload_db().get_sqla_engine_with_context() as engine:
        data = engine.execute(f'SELECT * from {ADMIN_SCHEMA_NAME}.{CSV_UPLOAD_TABLE_W_SCHEMA} ORDER BY b').fetchall()
        assert data == [('john', 1), ('paul', 2)]
    resp = upload_csv(CSV_FILENAME1, CSV_UPLOAD_TABLE_W_SCHEMA, extra={'schema': 'gold'})
    assert f"Database {escaped_double_quotes(CSV_UPLOAD_DATABASE)} schema {escaped_double_quotes('gold')} is not allowed for csv uploads" in resp
    if utils.backend() == 'hive':
        pytest.skip("Hive database doesn't support append csv uploads.")
    resp = upload_csv(CSV_FILENAME1, CSV_UPLOAD_TABLE_W_SCHEMA, extra={'schema': 'admin_database', 'if_exists': 'append'})
    assert success_msg in resp
    with get_upload_db().get_sqla_engine_with_context() as engine:
        engine.execute(f'DROP TABLE {full_table_name}')

@mock.patch('superset.db_engine_specs.hive.upload_to_s3', mock_upload_to_s3)
def test_import_csv_explore_database(setup_csv_upload_with_context, create_csv_files):
    if False:
        while True:
            i = 10
    schema = utils.get_example_default_schema()
    full_table_name = f'{schema}.{CSV_UPLOAD_TABLE_W_EXPLORE}' if schema else CSV_UPLOAD_TABLE_W_EXPLORE
    if utils.backend() == 'sqlite':
        pytest.skip("Sqlite doesn't support schema / database creation")
    resp = upload_csv(CSV_FILENAME1, CSV_UPLOAD_TABLE_W_EXPLORE)
    assert f'CSV file {escaped_double_quotes(CSV_FILENAME1)} uploaded to table {escaped_double_quotes(full_table_name)}' in resp
    table = SupersetTestCase.get_table(name=CSV_UPLOAD_TABLE_W_EXPLORE)
    assert table.database_id == superset.utils.database.get_example_database().id

@pytest.mark.usefixtures('setup_csv_upload_with_context')
@pytest.mark.usefixtures('create_csv_files')
@mock.patch('superset.db_engine_specs.hive.upload_to_s3', mock_upload_to_s3)
@mock.patch('superset.views.database.views.event_logger.log_with_context')
def test_import_csv(mock_event_logger):
    if False:
        i = 10
        return i + 15
    schema = utils.get_example_default_schema()
    full_table_name = f'{schema}.{CSV_UPLOAD_TABLE}' if schema else CSV_UPLOAD_TABLE
    success_msg_f1 = f'CSV file {escaped_double_quotes(CSV_FILENAME1)} uploaded to table {escaped_double_quotes(full_table_name)}'
    test_db = get_upload_db()
    resp = upload_csv(CSV_FILENAME1, CSV_UPLOAD_TABLE)
    assert success_msg_f1 in resp
    fail_msg = f'Unable to upload CSV file {escaped_double_quotes(CSV_FILENAME1)} to table {escaped_double_quotes(CSV_UPLOAD_TABLE)}'
    resp = upload_csv(CSV_FILENAME1, CSV_UPLOAD_TABLE)
    assert fail_msg in resp
    if utils.backend() != 'hive':
        resp = upload_csv(CSV_FILENAME1, CSV_UPLOAD_TABLE, extra={'if_exists': 'append'})
        assert success_msg_f1 in resp
        mock_event_logger.assert_called_with(action='successful_csv_upload', database=test_db.name, schema=schema, table=CSV_UPLOAD_TABLE)
    resp = upload_csv(CSV_FILENAME1, CSV_UPLOAD_TABLE, extra={'if_exists': 'replace'})
    assert success_msg_f1 in resp
    resp = upload_csv(CSV_FILENAME2, CSV_UPLOAD_TABLE, extra={'if_exists': 'append'})
    fail_msg_f2 = f'Unable to upload CSV file {escaped_double_quotes(CSV_FILENAME2)} to table {escaped_double_quotes(CSV_UPLOAD_TABLE)}'
    assert fail_msg_f2 in resp
    resp = upload_csv(CSV_FILENAME2, CSV_UPLOAD_TABLE, extra={'if_exists': 'replace'})
    success_msg_f2 = f'CSV file {escaped_double_quotes(CSV_FILENAME2)} uploaded to table {escaped_double_quotes(full_table_name)}'
    assert success_msg_f2 in resp
    table = SupersetTestCase.get_table(name=CSV_UPLOAD_TABLE)
    assert 'd' in table.column_names
    assert security_manager.find_user('admin') in table.owners
    upload_csv(CSV_FILENAME2, CSV_UPLOAD_TABLE, extra={'null_values': '["", "john"]', 'if_exists': 'replace'})
    with test_db.get_sqla_engine_with_context() as engine:
        data = engine.execute(f'SELECT * from {CSV_UPLOAD_TABLE} ORDER BY c').fetchall()
        assert data == [(None, 1, 'x'), ('paul', 2, None)]
        upload_csv(CSV_FILENAME2, CSV_UPLOAD_TABLE, extra={'if_exists': 'replace'})
        data = engine.execute(f'SELECT * from {CSV_UPLOAD_TABLE} ORDER BY c').fetchall()
        assert data == [('john', 1, 'x'), ('paul', 2, None)]
    with get_upload_db().get_sqla_engine_with_context() as engine:
        engine.execute(f'DROP TABLE {full_table_name}')
    upload_csv(CSV_FILENAME1, CSV_UPLOAD_TABLE, dtype='{"a": "string", "b": "float64"}')
    with test_db.get_sqla_engine_with_context() as engine:
        data = engine.execute(f'SELECT * from {CSV_UPLOAD_TABLE} ORDER BY b').fetchall()
        assert data == [('john', 1), ('paul', 2)]
    with get_upload_db().get_sqla_engine_with_context() as engine:
        engine.execute(f'DROP TABLE {full_table_name}')
    resp = upload_csv(CSV_FILENAME1, CSV_UPLOAD_TABLE, dtype='{"a": "int"}')
    fail_msg = f'Unable to upload CSV file {escaped_double_quotes(CSV_FILENAME1)} to table {escaped_double_quotes(CSV_UPLOAD_TABLE)}'
    assert fail_msg in resp

@pytest.mark.usefixtures('setup_csv_upload_with_context')
@pytest.mark.usefixtures('create_excel_files')
@mock.patch('superset.db_engine_specs.hive.upload_to_s3', mock_upload_to_s3)
@mock.patch('superset.views.database.views.event_logger.log_with_context')
def test_import_excel(mock_event_logger):
    if False:
        i = 10
        return i + 15
    if utils.backend() == 'hive':
        pytest.skip("Hive doesn't excel upload.")
    schema = utils.get_example_default_schema()
    full_table_name = f'{schema}.{EXCEL_UPLOAD_TABLE}' if schema else EXCEL_UPLOAD_TABLE
    test_db = get_upload_db()
    success_msg = f'Excel file {escaped_double_quotes(EXCEL_FILENAME)} uploaded to table {escaped_double_quotes(full_table_name)}'
    resp = upload_excel(EXCEL_FILENAME, EXCEL_UPLOAD_TABLE)
    assert success_msg in resp
    mock_event_logger.assert_called_with(action='successful_excel_upload', database=test_db.name, schema=schema, table=EXCEL_UPLOAD_TABLE)
    table = SupersetTestCase.get_table(name=EXCEL_UPLOAD_TABLE)
    assert security_manager.find_user('admin') in table.owners
    fail_msg = f'Unable to upload Excel file {escaped_double_quotes(EXCEL_FILENAME)} to table {escaped_double_quotes(EXCEL_UPLOAD_TABLE)}'
    resp = upload_excel(EXCEL_FILENAME, EXCEL_UPLOAD_TABLE)
    assert fail_msg in resp
    if utils.backend() != 'hive':
        resp = upload_excel(EXCEL_FILENAME, EXCEL_UPLOAD_TABLE, extra={'if_exists': 'append'})
        assert success_msg in resp
    resp = upload_excel(EXCEL_FILENAME, EXCEL_UPLOAD_TABLE, extra={'if_exists': 'replace'})
    assert success_msg in resp
    mock_event_logger.assert_called_with(action='successful_excel_upload', database=test_db.name, schema=schema, table=EXCEL_UPLOAD_TABLE)
    with test_db.get_sqla_engine_with_context() as engine:
        data = engine.execute(f'SELECT * from {EXCEL_UPLOAD_TABLE} ORDER BY b').fetchall()
        assert data == [(0, 'john', 1), (1, 'paul', 2)]

@pytest.mark.usefixtures('setup_csv_upload_with_context')
@pytest.mark.usefixtures('create_columnar_files')
@mock.patch('superset.db_engine_specs.hive.upload_to_s3', mock_upload_to_s3)
@mock.patch('superset.views.database.views.event_logger.log_with_context')
def test_import_parquet(mock_event_logger):
    if False:
        i = 10
        return i + 15
    if utils.backend() == 'hive':
        pytest.skip("Hive doesn't allow parquet upload.")
    schema = utils.get_example_default_schema()
    full_table_name = f'{schema}.{PARQUET_UPLOAD_TABLE}' if schema else PARQUET_UPLOAD_TABLE
    test_db = get_upload_db()
    success_msg_f1 = f'Columnar file {escaped_parquet(PARQUET_FILENAME1)} uploaded to table {escaped_double_quotes(full_table_name)}'
    resp = upload_columnar(PARQUET_FILENAME1, PARQUET_UPLOAD_TABLE)
    assert success_msg_f1 in resp
    fail_msg = f'Unable to upload Columnar file {escaped_parquet(PARQUET_FILENAME1)} to table {escaped_double_quotes(PARQUET_UPLOAD_TABLE)}'
    resp = upload_columnar(PARQUET_FILENAME1, PARQUET_UPLOAD_TABLE)
    assert fail_msg in resp
    if utils.backend() != 'hive':
        resp = upload_columnar(PARQUET_FILENAME1, PARQUET_UPLOAD_TABLE, extra={'if_exists': 'append'})
        assert success_msg_f1 in resp
        mock_event_logger.assert_called_with(action='successful_columnar_upload', database=test_db.name, schema=schema, table=PARQUET_UPLOAD_TABLE)
    resp = upload_columnar(PARQUET_FILENAME1, PARQUET_UPLOAD_TABLE, extra={'if_exists': 'replace', 'usecols': '["a"]'})
    assert success_msg_f1 in resp
    table = SupersetTestCase.get_table(name=PARQUET_UPLOAD_TABLE, schema=None)
    assert 'b' not in table.column_names
    assert security_manager.find_user('admin') in table.owners
    resp = upload_columnar(PARQUET_FILENAME1, PARQUET_UPLOAD_TABLE, extra={'if_exists': 'replace'})
    assert success_msg_f1 in resp
    with test_db.get_sqla_engine_with_context() as engine:
        data = engine.execute(f'SELECT * from {PARQUET_UPLOAD_TABLE} ORDER BY b').fetchall()
        assert data == [('john', 1), ('paul', 2)]
    resp = upload_columnar(ZIP_FILENAME, PARQUET_UPLOAD_TABLE, extra={'if_exists': 'replace'})
    success_msg_f2 = f'Columnar file {escaped_parquet(ZIP_FILENAME)} uploaded to table {escaped_double_quotes(full_table_name)}'
    assert success_msg_f2 in resp
    with test_db.get_sqla_engine_with_context() as engine:
        data = engine.execute(f'SELECT * from {PARQUET_UPLOAD_TABLE} ORDER BY b').fetchall()
        assert data == [('john', 1), ('paul', 2), ('max', 3), ('bob', 4)]