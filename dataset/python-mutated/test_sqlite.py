from __future__ import annotations
import pytest
from airflow.models.dag import DAG
from airflow.providers.sqlite.operators.sqlite import SqliteOperator
from airflow.utils import timezone
DEFAULT_DATE = timezone.datetime(2015, 1, 1)
DEFAULT_DATE_ISO = DEFAULT_DATE.isoformat()
DEFAULT_DATE_DS = DEFAULT_DATE_ISO[:10]
TEST_DAG_ID = 'unit_test_dag'

@pytest.mark.backend('sqlite')
class TestSqliteOperator:

    def setup_method(self):
        if False:
            while True:
                i = 10
        args = {'owner': 'airflow', 'start_date': DEFAULT_DATE}
        dag = DAG(TEST_DAG_ID, default_args=args)
        self.dag = dag

    def teardown_method(self):
        if False:
            print('Hello World!')
        tables_to_drop = ['test_airflow', 'test_airflow2']
        from airflow.providers.sqlite.hooks.sqlite import SqliteHook
        with SqliteHook().get_conn() as conn:
            cur = conn.cursor()
            for table in tables_to_drop:
                cur.execute(f'DROP TABLE IF EXISTS {table}')

    def test_sqlite_operator_with_one_statement(self):
        if False:
            print('Hello World!')
        sql = '\n        CREATE TABLE IF NOT EXISTS test_airflow (\n            dummy VARCHAR(50)\n        );\n        '
        op = SqliteOperator(task_id='basic_sqlite', sql=sql, dag=self.dag)
        op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)

    def test_sqlite_operator_with_multiple_statements(self):
        if False:
            return 10
        sql = ['CREATE TABLE IF NOT EXISTS test_airflow (dummy VARCHAR(50))', "INSERT INTO test_airflow VALUES ('X')"]
        op = SqliteOperator(task_id='sqlite_operator_with_multiple_statements', sql=sql, dag=self.dag)
        op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)

    def test_sqlite_operator_with_invalid_sql(self):
        if False:
            while True:
                i = 10
        sql = ['CREATE TABLE IF NOT EXISTS test_airflow (dummy VARCHAR(50))', "INSERT INTO test_airflow2 VALUES ('X')"]
        from sqlite3 import OperationalError
        try:
            op = SqliteOperator(task_id='sqlite_operator_with_multiple_statements', sql=sql, dag=self.dag)
            op.run(start_date=DEFAULT_DATE, end_date=DEFAULT_DATE, ignore_ti_state=True)
            pytest.fail('An exception should have been thrown')
        except OperationalError as e:
            assert 'no such table: test_airflow2' in str(e)