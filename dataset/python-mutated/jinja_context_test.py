import json
import pytest
from pytest_mock import MockFixture
from sqlalchemy.dialects import mysql
from superset.datasets.commands.exceptions import DatasetNotFoundError
from superset.jinja_context import dataset_macro, WhereInMacro

def test_where_in() -> None:
    if False:
        while True:
            i = 10
    '\n    Test the ``where_in`` Jinja2 filter.\n    '
    where_in = WhereInMacro(mysql.dialect())
    assert where_in([1, 'b', 3]) == "(1, 'b', 3)"
    assert where_in([1, 'b', 3], '"') == "(1, 'b', 3)\n-- WARNING: the `mark` parameter was removed from the `where_in` macro for security reasons\n"
    assert where_in(["O'Malley's"]) == "('O''Malley''s')"

def test_dataset_macro(mocker: MockFixture) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test the ``dataset_macro`` macro.\n    '
    from superset.connectors.sqla.models import SqlaTable, SqlMetric, TableColumn
    from superset.models.core import Database
    columns = [TableColumn(column_name='ds', is_dttm=1, type='TIMESTAMP'), TableColumn(column_name='num_boys', type='INTEGER'), TableColumn(column_name='revenue', type='INTEGER'), TableColumn(column_name='expenses', type='INTEGER'), TableColumn(column_name='profit', type='INTEGER', expression='revenue-expenses')]
    metrics = [SqlMetric(metric_name='cnt', expression='COUNT(*)')]
    dataset = SqlaTable(table_name='old_dataset', columns=columns, metrics=metrics, main_dttm_col='ds', default_endpoint='https://www.youtube.com/watch?v=dQw4w9WgXcQ', database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), offset=-8, description='This is the description', is_featured=1, cache_timeout=3600, schema='my_schema', sql=None, params=json.dumps({'remote_id': 64, 'database_name': 'examples', 'import_time': 1606677834}), perm=None, filter_select_enabled=1, fetch_values_predicate='foo IN (1, 2)', is_sqllab_view=0, template_params=json.dumps({'answer': '42'}), schema_perm=None, extra=json.dumps({'warning_markdown': '*WARNING*'}))
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = dataset
    assert dataset_macro(1) == '(\nSELECT ds AS ds,\n       num_boys AS num_boys,\n       revenue AS revenue,\n       expenses AS expenses,\n       revenue-expenses AS profit\nFROM my_schema.old_dataset\n) AS dataset_1'
    assert dataset_macro(1, include_metrics=True) == '(\nSELECT ds AS ds,\n       num_boys AS num_boys,\n       revenue AS revenue,\n       expenses AS expenses,\n       revenue-expenses AS profit,\n       COUNT(*) AS cnt\nFROM my_schema.old_dataset\nGROUP BY ds,\n         num_boys,\n         revenue,\n         expenses,\n         revenue-expenses\n) AS dataset_1'
    assert dataset_macro(1, include_metrics=True, columns=['ds']) == '(\nSELECT ds AS ds,\n       COUNT(*) AS cnt\nFROM my_schema.old_dataset\nGROUP BY ds\n) AS dataset_1'
    DatasetDAO.find_by_id.return_value = None
    with pytest.raises(DatasetNotFoundError) as excinfo:
        dataset_macro(1)
    assert str(excinfo.value) == 'Dataset 1 not found!'

def test_dataset_macro_mutator_with_comments(mocker: MockFixture) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Test ``dataset_macro`` when the mutator adds comment.\n    '

    def mutator(sql: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        A simple mutator that wraps the query in comments.\n        '
        return f'-- begin\n{sql}\n-- end'
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id().get_query_str_extended().sql = mutator('SELECT 1')
    assert dataset_macro(1) == '(\n-- begin\nSELECT 1\n-- end\n) AS dataset_1'