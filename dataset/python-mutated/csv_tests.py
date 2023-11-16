import io
import pandas as pd
import pyarrow as pa
import pytest
from superset.utils import csv

def test_escape_value():
    if False:
        print('Hello World!')
    result = csv.escape_value('value')
    assert result == 'value'
    result = csv.escape_value('-10')
    assert result == '-10'
    result = csv.escape_value('@value')
    assert result == "'@value"
    result = csv.escape_value('+value')
    assert result == "'+value"
    result = csv.escape_value('-value')
    assert result == "'-value"
    result = csv.escape_value('=value')
    assert result == "'=value"
    result = csv.escape_value('|value')
    assert result == "'\\|value"
    result = csv.escape_value('%value')
    assert result == "'%value"
    result = csv.escape_value("=cmd|' /C calc'!A0")
    assert result == "'=cmd\\|' /C calc'!A0"
    result = csv.escape_value('""=10+2')
    assert result == '\'""=10+2'
    result = csv.escape_value(' =10+2')
    assert result == "' =10+2"

def test_df_to_escaped_csv():
    if False:
        return 10
    csv_rows = [['col_a', '=func()'], ['-10', "=cmd|' /C calc'!A0"], ['a', '""=b'], [' =a', 'b']]
    csv_str = '\n'.join([','.join(row) for row in csv_rows])
    df = pd.read_csv(io.StringIO(csv_str))
    escaped_csv_str = csv.df_to_escaped_csv(df, encoding='utf8', index=False)
    escaped_csv_rows = [row.split(',') for row in escaped_csv_str.strip().split('\n')]
    assert escaped_csv_rows == [['col_a', "'=func()"], ['-10', "'=cmd\\|' /C calc'!A0"], ['a', "'=b"], ["' =a", 'b']]
    df = pa.array([1, None]).to_pandas(integer_object_nulls=True).to_frame()
    assert csv.df_to_escaped_csv(df, encoding='utf8', index=False) == '0\n1\n""\n'