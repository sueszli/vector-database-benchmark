"""Test the sql output adapter."""
from textwrap import dedent
from mycli.packages.tabular_output import sql_format
from cli_helpers.tabular_output import TabularOutputFormatter
from .utils import USER, PASSWORD, HOST, PORT, dbtest
import pytest
from mycli.main import MyCli
from pymysql.constants import FIELD_TYPE

@pytest.fixture
def mycli():
    if False:
        return 10
    cli = MyCli()
    cli.connect(None, USER, PASSWORD, HOST, PORT, None, init_command=None)
    return cli

@dbtest
def test_sql_output(mycli):
    if False:
        print('Hello World!')
    'Test the sql output adapter.'
    headers = ['letters', 'number', 'optional', 'float', 'binary']

    class FakeCursor(object):

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.data = [('abc', 1, None, 10.0, b'\xaa'), ('d', 456, '1', 0.5, b'\xaa\xbb')]
            self.description = [(None, FIELD_TYPE.VARCHAR), (None, FIELD_TYPE.LONG), (None, FIELD_TYPE.LONG), (None, FIELD_TYPE.FLOAT), (None, FIELD_TYPE.BLOB)]

        def __iter__(self):
            if False:
                print('Hello World!')
            return self

        def __next__(self):
            if False:
                print('Hello World!')
            if self.data:
                return self.data.pop(0)
            else:
                raise StopIteration()

        def description(self):
            if False:
                while True:
                    i = 10
            return self.description
    assert list(mycli.change_table_format('sql-update')) == [(None, None, None, 'Changed table format to sql-update')]
    mycli.formatter.query = ''
    output = mycli.format_output(None, FakeCursor(), headers)
    actual = '\n'.join(output)
    assert actual == dedent("            UPDATE `DUAL` SET\n              `number` = 1\n            , `optional` = NULL\n            , `float` = 10.0e0\n            , `binary` = X'aa'\n            WHERE `letters` = 'abc';\n            UPDATE `DUAL` SET\n              `number` = 456\n            , `optional` = '1'\n            , `float` = 0.5e0\n            , `binary` = X'aabb'\n            WHERE `letters` = 'd';")
    assert list(mycli.change_table_format('sql-update-2')) == [(None, None, None, 'Changed table format to sql-update-2')]
    mycli.formatter.query = ''
    output = mycli.format_output(None, FakeCursor(), headers)
    assert '\n'.join(output) == dedent("            UPDATE `DUAL` SET\n              `optional` = NULL\n            , `float` = 10.0e0\n            , `binary` = X'aa'\n            WHERE `letters` = 'abc' AND `number` = 1;\n            UPDATE `DUAL` SET\n              `optional` = '1'\n            , `float` = 0.5e0\n            , `binary` = X'aabb'\n            WHERE `letters` = 'd' AND `number` = 456;")
    assert list(mycli.change_table_format('sql-insert')) == [(None, None, None, 'Changed table format to sql-insert')]
    mycli.formatter.query = ''
    output = mycli.format_output(None, FakeCursor(), headers)
    assert '\n'.join(output) == dedent("            INSERT INTO `DUAL` (`letters`, `number`, `optional`, `float`, `binary`) VALUES\n              ('abc', 1, NULL, 10.0e0, X'aa')\n            , ('d', 456, '1', 0.5e0, X'aabb')\n            ;")
    assert list(mycli.change_table_format('sql-insert')) == [(None, None, None, 'Changed table format to sql-insert')]
    mycli.formatter.query = 'SELECT * FROM `table`'
    output = mycli.format_output(None, FakeCursor(), headers)
    assert '\n'.join(output) == dedent("            INSERT INTO table (`letters`, `number`, `optional`, `float`, `binary`) VALUES\n              ('abc', 1, NULL, 10.0e0, X'aa')\n            , ('d', 456, '1', 0.5e0, X'aabb')\n            ;")
    assert list(mycli.change_table_format('sql-insert')) == [(None, None, None, 'Changed table format to sql-insert')]
    mycli.formatter.query = 'SELECT * FROM `database`.`table`'
    output = mycli.format_output(None, FakeCursor(), headers)
    assert '\n'.join(output) == dedent("            INSERT INTO database.table (`letters`, `number`, `optional`, `float`, `binary`) VALUES\n              ('abc', 1, NULL, 10.0e0, X'aa')\n            , ('d', 456, '1', 0.5e0, X'aabb')\n            ;")