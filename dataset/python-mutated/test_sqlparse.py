from __future__ import annotations
import pytest
from airflow.providers.common.sql.hooks.sql import DbApiHook

@pytest.mark.parametrize('line,parsed_statements', [('SELECT * FROM table', ['SELECT * FROM table']), ('SELECT * FROM table;', ['SELECT * FROM table;']), ('SELECT * FROM table; # comment', ['SELECT * FROM table;']), ('SELECT * FROM table; # comment;', ['SELECT * FROM table;']), (' SELECT * FROM table ; # comment;', ['SELECT * FROM table ;']), ('SELECT * FROM table; SELECT * FROM table2 # comment', ['SELECT * FROM table;', 'SELECT * FROM table2'])])
def test_sqlparse(line, parsed_statements):
    if False:
        while True:
            i = 10
    assert DbApiHook.split_sql_string(line) == parsed_statements