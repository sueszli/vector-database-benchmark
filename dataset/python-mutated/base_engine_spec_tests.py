import datetime
from unittest import mock
import pytest
from superset.connectors.sqla.models import TableColumn
from superset.db_engine_specs import load_engine_specs
from superset.db_engine_specs.base import BaseEngineSpec, BasicParametersMixin, builtin_time_grains, LimitMethod
from superset.db_engine_specs.mysql import MySQLEngineSpec
from superset.db_engine_specs.sqlite import SqliteEngineSpec
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.sql_parse import ParsedQuery
from superset.utils.database import get_example_database
from tests.integration_tests.db_engine_specs.base_tests import TestDbEngineSpec
from tests.integration_tests.test_app import app
from ..fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from ..fixtures.energy_dashboard import load_energy_table_data, load_energy_table_with_slice
from ..fixtures.pyodbcRow import Row

class TestDbEngineSpecs(TestDbEngineSpec):

    def test_extract_limit_from_query(self, engine_spec_class=BaseEngineSpec):
        if False:
            print('Hello World!')
        q0 = 'select * from table'
        q1 = 'select * from mytable limit 10'
        q2 = 'select * from (select * from my_subquery limit 10) where col=1 limit 20'
        q3 = 'select * from (select * from my_subquery limit 10);'
        q4 = 'select * from (select * from my_subquery limit 10) where col=1 limit 20;'
        q5 = 'select * from mytable limit 20, 10'
        q6 = 'select * from mytable limit 10 offset 20'
        q7 = 'select * from mytable limit'
        q8 = 'select * from mytable limit 10.0'
        q9 = 'select * from mytable limit x'
        q10 = 'select * from mytable limit 20, x'
        q11 = 'select * from mytable limit x offset 20'
        self.assertEqual(engine_spec_class.get_limit_from_sql(q0), None)
        self.assertEqual(engine_spec_class.get_limit_from_sql(q1), 10)
        self.assertEqual(engine_spec_class.get_limit_from_sql(q2), 20)
        self.assertEqual(engine_spec_class.get_limit_from_sql(q3), None)
        self.assertEqual(engine_spec_class.get_limit_from_sql(q4), 20)
        self.assertEqual(engine_spec_class.get_limit_from_sql(q5), 10)
        self.assertEqual(engine_spec_class.get_limit_from_sql(q6), 10)
        self.assertEqual(engine_spec_class.get_limit_from_sql(q7), None)
        self.assertEqual(engine_spec_class.get_limit_from_sql(q8), None)
        self.assertEqual(engine_spec_class.get_limit_from_sql(q9), None)
        self.assertEqual(engine_spec_class.get_limit_from_sql(q10), None)
        self.assertEqual(engine_spec_class.get_limit_from_sql(q11), None)

    def test_wrapped_semi_tabs(self):
        if False:
            return 10
        self.sql_limit_regex('SELECT * FROM a  \t \n   ; \t  \n  ', 'SELECT * FROM a\nLIMIT 1000')

    def test_simple_limit_query(self):
        if False:
            i = 10
            return i + 15
        self.sql_limit_regex('SELECT * FROM a', 'SELECT * FROM a\nLIMIT 1000')

    def test_modify_limit_query(self):
        if False:
            print('Hello World!')
        self.sql_limit_regex('SELECT * FROM a LIMIT 9999', 'SELECT * FROM a LIMIT 1000')

    def test_limit_query_with_limit_subquery(self):
        if False:
            for i in range(10):
                print('nop')
        self.sql_limit_regex('SELECT * FROM (SELECT * FROM a LIMIT 10) LIMIT 9999', 'SELECT * FROM (SELECT * FROM a LIMIT 10) LIMIT 1000')

    def test_limit_query_without_force(self):
        if False:
            i = 10
            return i + 15
        self.sql_limit_regex('SELECT * FROM a LIMIT 10', 'SELECT * FROM a LIMIT 10', limit=11)

    def test_limit_query_with_force(self):
        if False:
            return 10
        self.sql_limit_regex('SELECT * FROM a LIMIT 10', 'SELECT * FROM a LIMIT 11', limit=11, force=True)

    def test_limit_with_expr(self):
        if False:
            i = 10
            return i + 15
        self.sql_limit_regex("\n            SELECT\n                'LIMIT 777' AS a\n                , b\n            FROM\n            table\n            LIMIT 99990", "SELECT\n                'LIMIT 777' AS a\n                , b\n            FROM\n            table\n            LIMIT 1000")

    def test_limit_expr_and_semicolon(self):
        if False:
            for i in range(10):
                print('nop')
        self.sql_limit_regex("\n                SELECT\n                    'LIMIT 777' AS a\n                    , b\n                FROM\n                table\n                LIMIT         99990            ;", "SELECT\n                    'LIMIT 777' AS a\n                    , b\n                FROM\n                table\n                LIMIT         1000")

    def test_get_datatype(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('VARCHAR', BaseEngineSpec.get_datatype('VARCHAR'))

    def test_limit_with_implicit_offset(self):
        if False:
            print('Hello World!')
        self.sql_limit_regex("\n                SELECT\n                    'LIMIT 777' AS a\n                    , b\n                FROM\n                table\n                LIMIT 99990, 999999", "SELECT\n                    'LIMIT 777' AS a\n                    , b\n                FROM\n                table\n                LIMIT 99990, 1000")

    def test_limit_with_explicit_offset(self):
        if False:
            return 10
        self.sql_limit_regex("\n                SELECT\n                    'LIMIT 777' AS a\n                    , b\n                FROM\n                table\n                LIMIT 99990\n                OFFSET 999999", "SELECT\n                    'LIMIT 777' AS a\n                    , b\n                FROM\n                table\n                LIMIT 1000\n                OFFSET 999999")

    def test_limit_with_non_token_limit(self):
        if False:
            for i in range(10):
                print('nop')
        self.sql_limit_regex("SELECT 'LIMIT 777'", "SELECT 'LIMIT 777'\nLIMIT 1000")

    def test_limit_with_fetch_many(self):
        if False:
            i = 10
            return i + 15

        class DummyEngineSpec(BaseEngineSpec):
            limit_method = LimitMethod.FETCH_MANY
        self.sql_limit_regex('SELECT * FROM table', 'SELECT * FROM table', DummyEngineSpec)

    def test_engine_time_grain_validity(self):
        if False:
            return 10
        time_grains = set(builtin_time_grains.keys())
        for engine in load_engine_specs():
            if engine is not BaseEngineSpec:
                self.assertGreater(len(engine.get_time_grain_expressions()), 0)
                defined_grains = {grain.duration for grain in engine.get_time_grains()}
                intersection = time_grains.intersection(defined_grains)
                self.assertSetEqual(defined_grains, intersection, engine)

    def test_get_time_grain_expressions(self):
        if False:
            return 10
        time_grains = MySQLEngineSpec.get_time_grain_expressions()
        self.assertEqual(list(time_grains.keys()), [None, 'PT1S', 'PT1M', 'PT1H', 'P1D', 'P1W', 'P1M', 'P3M', 'P1Y', '1969-12-29T00:00:00Z/P1W'])

    def test_get_table_names(self):
        if False:
            return 10
        inspector = mock.Mock()
        inspector.get_table_names = mock.Mock(return_value=['schema.table', 'table_2'])
        inspector.get_foreign_table_names = mock.Mock(return_value=['table_3'])
        ' Make sure base engine spec removes schema name from table name\n        ie. when try_remove_schema_from_table_name == True. '
        base_result_expected = {'table', 'table_2'}
        base_result = BaseEngineSpec.get_table_names(database=mock.ANY, schema='schema', inspector=inspector)
        assert base_result_expected == base_result

    @pytest.mark.usefixtures('load_energy_table_with_slice')
    def test_column_datatype_to_string(self):
        if False:
            while True:
                i = 10
        example_db = get_example_database()
        sqla_table = example_db.get_table('energy_usage')
        dialect = example_db.get_dialect()
        if example_db.backend == 'presto':
            return
        col_names = [example_db.db_engine_spec.column_datatype_to_string(c.type, dialect) for c in sqla_table.columns]
        if example_db.backend == 'postgresql':
            expected = ['VARCHAR(255)', 'VARCHAR(255)', 'DOUBLE PRECISION']
        elif example_db.backend == 'hive':
            expected = ['STRING', 'STRING', 'FLOAT']
        else:
            expected = ['VARCHAR(255)', 'VARCHAR(255)', 'FLOAT']
        self.assertEqual(col_names, expected)

    def test_convert_dttm(self):
        if False:
            print('Hello World!')
        dttm = self.get_dttm()
        self.assertIsNone(BaseEngineSpec.convert_dttm('', dttm, db_extra=None))

    def test_pyodbc_rows_to_tuples(self):
        if False:
            while True:
                i = 10
        data = [Row((1, 1, datetime.datetime(2017, 10, 19, 23, 39, 16, 660000))), Row((2, 2, datetime.datetime(2018, 10, 19, 23, 39, 16, 660000)))]
        expected = [(1, 1, datetime.datetime(2017, 10, 19, 23, 39, 16, 660000)), (2, 2, datetime.datetime(2018, 10, 19, 23, 39, 16, 660000))]
        result = BaseEngineSpec.pyodbc_rows_to_tuples(data)
        self.assertListEqual(result, expected)

    def test_pyodbc_rows_to_tuples_passthrough(self):
        if False:
            while True:
                i = 10
        data = [(1, 1, datetime.datetime(2017, 10, 19, 23, 39, 16, 660000)), (2, 2, datetime.datetime(2018, 10, 19, 23, 39, 16, 660000))]
        result = BaseEngineSpec.pyodbc_rows_to_tuples(data)
        self.assertListEqual(result, data)

    @mock.patch('superset.models.core.Database.db_engine_spec', BaseEngineSpec)
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_calculated_column_in_order_by_base_engine_spec(self):
        if False:
            i = 10
            return i + 15
        table = self.get_table(name='birth_names')
        TableColumn(column_name='gender_cc', type='VARCHAR(255)', table=table, expression="\n            case\n              when gender='boy' then 'male'\n              else 'female'\n            end\n            ")
        table.database.sqlalchemy_uri = 'sqlite://'
        query_obj = {'groupby': ['gender_cc'], 'is_timeseries': False, 'filter': [], 'orderby': [['gender_cc', True]]}
        sql = table.get_query_str(query_obj)
        assert "ORDER BY case\n             when gender='boy' then 'male'\n             else 'female'\n         end ASC;" in sql

def test_is_readonly():
    if False:
        while True:
            i = 10

    def is_readonly(sql: str) -> bool:
        if False:
            print('Hello World!')
        return BaseEngineSpec.is_readonly_query(ParsedQuery(sql))
    assert is_readonly('SHOW LOCKS test EXTENDED')
    assert not is_readonly("SET hivevar:desc='Legislators'")
    assert not is_readonly('UPDATE t1 SET col1 = NULL')
    assert is_readonly('EXPLAIN SELECT 1')
    assert is_readonly('SELECT 1')
    assert is_readonly('WITH (SELECT 1) bla SELECT * from bla')
    assert is_readonly('SHOW CATALOGS')
    assert is_readonly('SHOW TABLES')

def test_time_grain_denylist():
    if False:
        return 10
    config = app.config.copy()
    app.config['TIME_GRAIN_DENYLIST'] = ['PT1M', 'SQLITE_NONEXISTENT_GRAIN']
    with app.app_context():
        time_grain_functions = SqliteEngineSpec.get_time_grain_expressions()
        assert not 'PT1M' in time_grain_functions
        assert not 'SQLITE_NONEXISTENT_GRAIN' in time_grain_functions
    app.config = config

def test_time_grain_addons():
    if False:
        return 10
    config = app.config.copy()
    app.config['TIME_GRAIN_ADDONS'] = {'PTXM': 'x seconds'}
    app.config['TIME_GRAIN_ADDON_EXPRESSIONS'] = {'sqlite': {'PTXM': 'ABC({col})'}}
    with app.app_context():
        time_grains = SqliteEngineSpec.get_time_grains()
        time_grain_addon = time_grains[-1]
        assert 'PTXM' == time_grain_addon.duration
        assert 'x seconds' == time_grain_addon.label
    app.config = config

def test_get_time_grain_with_config():
    if False:
        i = 10
        return i + 15
    'Should concatenate from configs and then sort in the proper order'
    config = app.config.copy()
    app.config['TIME_GRAIN_ADDON_EXPRESSIONS'] = {'mysql': {'PT2H': 'foo', 'PT4H': 'foo', 'PT6H': 'foo', 'PT8H': 'foo', 'PT10H': 'foo', 'PT12H': 'foo', 'PT1S': 'foo'}}
    with app.app_context():
        time_grains = MySQLEngineSpec.get_time_grain_expressions()
        assert set(time_grains.keys()) == {None, 'PT1S', 'PT1M', 'PT1H', 'PT2H', 'PT4H', 'PT6H', 'PT8H', 'PT10H', 'PT12H', 'P1D', 'P1W', 'P1M', 'P3M', 'P1Y', '1969-12-29T00:00:00Z/P1W'}
    app.config = config

def test_get_time_grain_with_unknown_values():
    if False:
        print('Hello World!')
    'Should concatenate from configs and then sort in the proper order\n    putting unknown patterns at the end'
    config = app.config.copy()
    app.config['TIME_GRAIN_ADDON_EXPRESSIONS'] = {'mysql': {'PT2H': 'foo', 'weird': 'foo', 'PT12H': 'foo'}}
    with app.app_context():
        time_grains = MySQLEngineSpec.get_time_grain_expressions()
        assert list(time_grains)[-1] == 'weird'
    app.config = config

@mock.patch('superset.db_engine_specs.base.is_hostname_valid')
@mock.patch('superset.db_engine_specs.base.is_port_open')
def test_validate(is_port_open, is_hostname_valid):
    if False:
        return 10
    is_hostname_valid.return_value = True
    is_port_open.return_value = True
    properties = {'parameters': {'host': 'localhost', 'port': 5432, 'username': 'username', 'password': 'password', 'database': 'dbname', 'query': {'sslmode': 'verify-full'}}}
    errors = BasicParametersMixin.validate_parameters(properties)
    assert errors == []

def test_validate_parameters_missing():
    if False:
        return 10
    properties = {'parameters': {'host': '', 'port': None, 'username': '', 'password': '', 'database': '', 'query': {}}}
    with app.app_context():
        errors = BasicParametersMixin.validate_parameters(properties)
        assert errors == [SupersetError(message='One or more parameters are missing: database, host, port, username', error_type=SupersetErrorType.CONNECTION_MISSING_PARAMETERS_ERROR, level=ErrorLevel.WARNING, extra={'missing': ['database', 'host', 'port', 'username']})]

@mock.patch('superset.db_engine_specs.base.is_hostname_valid')
def test_validate_parameters_invalid_host(is_hostname_valid):
    if False:
        for i in range(10):
            print('nop')
    is_hostname_valid.return_value = False
    properties = {'parameters': {'host': 'localhost', 'port': None, 'username': 'username', 'password': 'password', 'database': 'dbname', 'query': {'sslmode': 'verify-full'}}}
    with app.app_context():
        errors = BasicParametersMixin.validate_parameters(properties)
        assert errors == [SupersetError(message='One or more parameters are missing: port', error_type=SupersetErrorType.CONNECTION_MISSING_PARAMETERS_ERROR, level=ErrorLevel.WARNING, extra={'missing': ['port']}), SupersetError(message="The hostname provided can't be resolved.", error_type=SupersetErrorType.CONNECTION_INVALID_HOSTNAME_ERROR, level=ErrorLevel.ERROR, extra={'invalid': ['host']})]

@mock.patch('superset.db_engine_specs.base.is_hostname_valid')
@mock.patch('superset.db_engine_specs.base.is_port_open')
def test_validate_parameters_port_closed(is_port_open, is_hostname_valid):
    if False:
        i = 10
        return i + 15
    is_hostname_valid.return_value = True
    is_port_open.return_value = False
    properties = {'parameters': {'host': 'localhost', 'port': 5432, 'username': 'username', 'password': 'password', 'database': 'dbname', 'query': {'sslmode': 'verify-full'}}}
    with app.app_context():
        errors = BasicParametersMixin.validate_parameters(properties)
        assert errors == [SupersetError(message='The port is closed.', error_type=SupersetErrorType.CONNECTION_PORT_CLOSED_ERROR, level=ErrorLevel.ERROR, extra={'invalid': ['port'], 'issue_codes': [{'code': 1008, 'message': 'Issue 1008 - The port is closed.'}]})]

def test_get_indexes():
    if False:
        print('Hello World!')
    indexes = [{'name': 'partition', 'column_names': ['a', 'b'], 'unique': False}]
    inspector = mock.Mock()
    inspector.get_indexes = mock.Mock(return_value=indexes)
    assert BaseEngineSpec.get_indexes(database=mock.Mock(), inspector=inspector, table_name='bar', schema='foo') == indexes