import pytest

@pytest.mark.parametrize('limit,original,expected', [(100, 'SEL TOP 1000 * FROM My_table', 'SEL TOP 100 * FROM My_table'), (100, 'SEL TOP 1000 * FROM My_table;', 'SEL TOP 100 * FROM My_table'), (10000, 'SEL TOP 1000 * FROM My_table;', 'SEL TOP 1000 * FROM My_table'), (1000, 'SEL TOP 1000 * FROM My_table;', 'SEL TOP 1000 * FROM My_table'), (100, 'SELECT TOP 1000 * FROM My_table', 'SELECT TOP 100 * FROM My_table'), (100, 'SEL SAMPLE 1000 * FROM My_table', 'SEL SAMPLE 100 * FROM My_table'), (10000, 'SEL SAMPLE 1000 * FROM My_table', 'SEL SAMPLE 1000 * FROM My_table')])
def test_apply_top_to_sql_limit(limit: int, original: str, expected: str) -> None:
    if False:
        return 10
    '\n    Ensure limits are applied to the query correctly\n    '
    from superset.db_engine_specs.teradata import TeradataEngineSpec
    assert TeradataEngineSpec.apply_top_to_sql(original, limit) == expected