from textwrap import dedent
import psycopg
import pytest
from unittest.mock import patch, MagicMock
from pgspecial.main import PGSpecial, NO_QUERY
from utils import run, dbtest, requires_json, requires_jsonb
from pgcli.main import PGCli
from pgcli.packages.parseutils.meta import FunctionMetadata

def function_meta_data(func_name, schema_name='public', arg_names=None, arg_types=None, arg_modes=None, return_type=None, is_aggregate=False, is_window=False, is_set_returning=False, is_extension=False, arg_defaults=None):
    if False:
        return 10
    return FunctionMetadata(schema_name, func_name, arg_names, arg_types, arg_modes, return_type, is_aggregate, is_window, is_set_returning, is_extension, arg_defaults)

@dbtest
def test_conn(executor):
    if False:
        print('Hello World!')
    run(executor, 'create table test(a text)')
    run(executor, "insert into test values('abc')")
    assert run(executor, 'select * from test', join=True) == dedent('        +-----+\n        | a   |\n        |-----|\n        | abc |\n        +-----+\n        SELECT 1')

@dbtest
def test_copy(executor):
    if False:
        i = 10
        return i + 15
    executor_copy = executor.copy()
    run(executor_copy, 'create table test(a text)')
    run(executor_copy, "insert into test values('abc')")
    assert run(executor_copy, 'select * from test', join=True) == dedent('        +-----+\n        | a   |\n        |-----|\n        | abc |\n        +-----+\n        SELECT 1')

@dbtest
def test_bools_are_treated_as_strings(executor):
    if False:
        i = 10
        return i + 15
    run(executor, 'create table test(a boolean)')
    run(executor, 'insert into test values(True)')
    assert run(executor, 'select * from test', join=True) == dedent('        +------+\n        | a    |\n        |------|\n        | True |\n        +------+\n        SELECT 1')

@dbtest
def test_expanded_slash_G(executor, pgspecial):
    if False:
        while True:
            i = 10
    run(executor, 'create table test(a boolean)')
    run(executor, 'insert into test values(True)')
    results = run(executor, 'select * from test \\G', pgspecial=pgspecial)
    assert pgspecial.expanded_output == False

@dbtest
def test_schemata_table_views_and_columns_query(executor):
    if False:
        print('Hello World!')
    run(executor, 'create table a(x text, y text)')
    run(executor, 'create table b(z text)')
    run(executor, 'create view d as select 1 as e')
    run(executor, 'create schema schema1')
    run(executor, "create table schema1.c (w text DEFAULT 'meow')")
    run(executor, 'create schema schema2')
    assert set(executor.schemata()) >= {'public', 'pg_catalog', 'information_schema', 'schema1', 'schema2'}
    assert executor.search_path() == ['pg_catalog', 'public']
    assert set(executor.tables()) >= {('public', 'a'), ('public', 'b'), ('schema1', 'c')}
    assert set(executor.table_columns()) >= {('public', 'a', 'x', 'text', False, None), ('public', 'a', 'y', 'text', False, None), ('public', 'b', 'z', 'text', False, None), ('schema1', 'c', 'w', 'text', True, "'meow'::text")}
    assert set(executor.views()) >= {('public', 'd')}
    assert set(executor.view_columns()) >= {('public', 'd', 'e', 'integer', False, None)}

@dbtest
def test_foreign_key_query(executor):
    if False:
        for i in range(10):
            print('nop')
    run(executor, 'create schema schema1')
    run(executor, 'create schema schema2')
    run(executor, 'create table schema1.parent(parentid int PRIMARY KEY)')
    run(executor, 'create table schema2.child(childid int PRIMARY KEY, motherid int REFERENCES schema1.parent)')
    assert set(executor.foreignkeys()) >= {('schema1', 'parent', 'parentid', 'schema2', 'child', 'motherid')}

@dbtest
def test_functions_query(executor):
    if False:
        return 10
    run(executor, 'create function func1() returns int\n                     language sql as $$select 1$$')
    run(executor, 'create schema schema1')
    run(executor, 'create function schema1.func2() returns int\n                     language sql as $$select 2$$')
    run(executor, 'create function func3()\n                     returns table(x int, y int) language sql\n                     as $$select 1, 2 from generate_series(1,5)$$;')
    run(executor, 'create function func4(x int) returns setof int language sql\n                     as $$select generate_series(1,5)$$;')
    funcs = set(executor.functions())
    assert funcs >= {function_meta_data(func_name='func1', return_type='integer'), function_meta_data(func_name='func3', arg_names=['x', 'y'], arg_types=['integer', 'integer'], arg_modes=['t', 't'], return_type='record', is_set_returning=True), function_meta_data(schema_name='public', func_name='func4', arg_names=('x',), arg_types=('integer',), return_type='integer', is_set_returning=True), function_meta_data(schema_name='schema1', func_name='func2', return_type='integer')}

@dbtest
def test_datatypes_query(executor):
    if False:
        return 10
    run(executor, 'create type foo AS (a int, b text)')
    types = list(executor.datatypes())
    assert types == [('public', 'foo')]

@dbtest
def test_database_list(executor):
    if False:
        while True:
            i = 10
    databases = executor.databases()
    assert '_test_db' in databases

@dbtest
def test_invalid_syntax(executor, exception_formatter):
    if False:
        print('Hello World!')
    result = run(executor, 'invalid syntax!', exception_formatter=exception_formatter)
    assert 'syntax error at or near "invalid"' in result[0]

@dbtest
def test_invalid_column_name(executor, exception_formatter):
    if False:
        return 10
    result = run(executor, 'select invalid command', exception_formatter=exception_formatter)
    assert 'column "invalid" does not exist' in result[0]

@pytest.fixture(params=[True, False])
def expanded(request):
    if False:
        i = 10
        return i + 15
    return request.param

@dbtest
def test_unicode_support_in_output(executor, expanded):
    if False:
        while True:
            i = 10
    run(executor, 'create table unicodechars(t text)')
    run(executor, "insert into unicodechars (t) values ('é')")
    assert 'é' in run(executor, 'select * from unicodechars', join=True, expanded=expanded)

@dbtest
def test_not_is_special(executor, pgspecial):
    if False:
        return 10
    'is_special is set to false for database queries.'
    query = 'select 1'
    result = list(executor.run(query, pgspecial=pgspecial))
    (success, is_special) = result[0][5:]
    assert success == True
    assert is_special == False

@dbtest
def test_execute_from_file_no_arg(executor, pgspecial):
    if False:
        for i in range(10):
            print('nop')
    '\\i without a filename returns an error.'
    result = list(executor.run('\\i', pgspecial=pgspecial))
    (status, sql, success, is_special) = result[0][3:]
    assert 'missing required argument' in status
    assert success == False
    assert is_special == True

@dbtest
@patch('pgcli.main.os')
def test_execute_from_file_io_error(os, executor, pgspecial):
    if False:
        for i in range(10):
            print('nop')
    '\\i with an os_error returns an error.'
    os.path.expanduser.side_effect = OSError('test')
    result = list(executor.run('\\i test', pgspecial=pgspecial))
    (status, sql, success, is_special) = result[0][3:]
    assert status == 'test'
    assert success == False
    assert is_special == True

@dbtest
def test_execute_from_commented_file_that_executes_another_file(executor, pgspecial, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    sqlfile1 = tmpdir.join('test01.sql')
    sqlfile1.write('-- asdf \n\\h')
    sqlfile2 = tmpdir.join('test00.sql')
    sqlfile2.write('--An useless comment;\nselect now();\n-- another useless comment')
    rcfile = str(tmpdir.join('rcfile'))
    print(rcfile)
    cli = PGCli(pgexecute=executor, pgclirc_file=rcfile)
    assert cli != None
    statement = '--comment\n\\h'
    result = run(executor, statement, pgspecial=cli.pgspecial)
    assert result != None
    assert result[0].find('ALTER TABLE')

@dbtest
def test_execute_commented_first_line_and_special(executor, pgspecial, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    statement = '--comment\nselect now();'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[1].find('now') >= 0
    statement = '/*comment*/\nselect now();'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[1].find('now') >= 0
    statement = '--comment\n\\h'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[1].find('ALTER') >= 0
    assert result[1].find('ABORT') >= 0
    statement = '--comment1\n--comment2\n\\h'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[1].find('ALTER') >= 0
    assert result[1].find('ABORT') >= 0
    statement = '/*comment*/\n\\h;'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[1].find('ALTER') >= 0
    assert result[1].find('ABORT') >= 0
    statement = '/*comment1\n    comment2*/\n    \\h'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[1].find('ALTER') >= 0
    assert result[1].find('ABORT') >= 0
    statement = '/*comment1\n    comment2*/\n    /*comment 3\n    comment4*/\n    \\h'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[1].find('ALTER') >= 0
    assert result[1].find('ABORT') >= 0
    statement = '    /*comment*/\n\\h;'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[1].find('ALTER') >= 0
    assert result[1].find('ABORT') >= 0
    statement = '/*comment\ncomment line2*/\n\\h;'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[1].find('ALTER') >= 0
    assert result[1].find('ABORT') >= 0
    statement = '          /*comment\ncomment line2*/\n\\h;'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[1].find('ALTER') >= 0
    assert result[1].find('ABORT') >= 0
    statement = '\\h /*comment4 */'
    result = run(executor, statement, pgspecial=pgspecial)
    print(result)
    assert result != None
    assert result[0].find('No help') >= 0
    statement = '/*comment1*/\n    \\h\n    /*comment4 */'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[0].find('No help') >= 0
    statement = '/*comment1\n    comment3\n    comment2*/\n    \\h\n    /*comment4\n    comment5\n    comment6*/'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[0].find('No help') >= 0

@dbtest
def test_execute_commented_first_line_and_normal(executor, pgspecial, tmpdir):
    if False:
        print('Hello World!')
    statement = '--comment\nselect now();'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[1].find('now') >= 0
    statement = '/*comment*/\nselect now();'
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[1].find('now') >= 0
    statement = "VALUES (1, 'one'), (2, 'two'), (3, 'three');"
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[5].find('three') >= 0
    statement = "VALUES (1, 'one'),\n (2, 'two'), (3, 'three');"
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[5].find('three') >= 0
    statement = "VALUES (1, 'one'),\n     (2, 'two'), (3, 'three');"
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[5].find('three') >= 0
    statement = "--comment\nVALUES (1, 'one'), (2, 'two'), (3, 'three');"
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[5].find('three') >= 0
    statement = "--comment\n    VALUES (1,'one'),\n    (2, 'two'), (3, 'three');"
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[5].find('three') >= 0
    statement = "--comment\n--comment2\nVALUES (1,'one'), (2, 'two'), (3, 'three');"
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[5].find('three') >= 0
    statement = "--comment\n    --comment2\n    VALUES (1,'one'), (2, 'two'), (3, 'three');\n    "
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[5].find('three') >= 0
    statement = "/*comment\ncomment2\ncomment3*/\nVALUES (1,'one'),\n(2, 'two'), (3, 'three');"
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[5].find('three') >= 0
    statement = "/*comment\ncomment2\ncomment3*/\nVALUES (1,'one'),\n(2, 'two'), (3, 'three');\n--comment4\n--comment5"
    result = run(executor, statement, pgspecial=pgspecial)
    assert result != None
    assert result[5].find('three') >= 0

@dbtest
def test_multiple_queries_same_line(executor):
    if False:
        while True:
            i = 10
    result = run(executor, "select 'foo'; select 'bar'")
    assert len(result) == 12
    assert 'foo' in result[3]
    assert 'bar' in result[9]

@dbtest
def test_multiple_queries_with_special_command_same_line(executor, pgspecial):
    if False:
        i = 10
        return i + 15
    result = run(executor, "select 'foo'; \\d", pgspecial=pgspecial)
    assert len(result) == 11
    assert 'foo' in result[3]
    assert 'Schema' in result[7]

@dbtest
def test_multiple_queries_same_line_syntaxerror(executor, exception_formatter):
    if False:
        print('Hello World!')
    result = run(executor, "select 'fooé'; invalid syntax é", exception_formatter=exception_formatter)
    assert 'fooé' in result[3]
    assert 'syntax error at or near "invalid"' in result[-1]

@pytest.fixture
def pgspecial():
    if False:
        while True:
            i = 10
    return PGCli().pgspecial

@dbtest
def test_special_command_help(executor, pgspecial):
    if False:
        print('Hello World!')
    result = run(executor, '\\?', pgspecial=pgspecial)[1].split('|')
    assert 'Command' in result[1]
    assert 'Description' in result[2]

@dbtest
def test_bytea_field_support_in_output(executor):
    if False:
        print('Hello World!')
    run(executor, 'create table binarydata(c bytea)')
    run(executor, "insert into binarydata (c) values (decode('DEADBEEF', 'hex'))")
    assert '\\xdeadbeef' in run(executor, 'select * from binarydata', join=True)

@dbtest
def test_unicode_support_in_unknown_type(executor):
    if False:
        print('Hello World!')
    assert '日本語' in run(executor, "SELECT '日本語' AS japanese;", join=True)

@dbtest
def test_unicode_support_in_enum_type(executor):
    if False:
        i = 10
        return i + 15
    run(executor, "CREATE TYPE mood AS ENUM ('sad', 'ok', 'happy', '日本語')")
    run(executor, 'CREATE TABLE person (name TEXT, current_mood mood)')
    run(executor, "INSERT INTO person VALUES ('Moe', '日本語')")
    assert '日本語' in run(executor, 'SELECT * FROM person', join=True)

@requires_json
def test_json_renders_without_u_prefix(executor, expanded):
    if False:
        for i in range(10):
            print('nop')
    run(executor, 'create table jsontest(d json)')
    run(executor, 'insert into jsontest (d) values (\'{"name": "Éowyn"}\')')
    result = run(executor, 'SELECT d FROM jsontest LIMIT 1', join=True, expanded=expanded)
    assert '{"name": "Éowyn"}' in result

@requires_jsonb
def test_jsonb_renders_without_u_prefix(executor, expanded):
    if False:
        print('Hello World!')
    run(executor, 'create table jsonbtest(d jsonb)')
    run(executor, 'insert into jsonbtest (d) values (\'{"name": "Éowyn"}\')')
    result = run(executor, 'SELECT d FROM jsonbtest LIMIT 1', join=True, expanded=expanded)
    assert '{"name": "Éowyn"}' in result

@dbtest
def test_date_time_types(executor):
    if False:
        print('Hello World!')
    run(executor, 'SET TIME ZONE UTC')
    assert run(executor, "SELECT (CAST('00:00:00' AS time))", join=True).split('\n')[3] == '| 00:00:00 |'
    assert run(executor, "SELECT (CAST('00:00:00+14:59' AS timetz))", join=True).split('\n')[3] == '| 00:00:00+14:59 |'
    assert run(executor, "SELECT (CAST('4713-01-01 BC' AS date))", join=True).split('\n')[3] == '| 4713-01-01 BC |'
    assert run(executor, "SELECT (CAST('4713-01-01 00:00:00 BC' AS timestamp))", join=True).split('\n')[3] == '| 4713-01-01 00:00:00 BC |'
    assert run(executor, "SELECT (CAST('4713-01-01 00:00:00+00 BC' AS timestamptz))", join=True).split('\n')[3] == '| 4713-01-01 00:00:00+00 BC |'
    assert run(executor, "SELECT (CAST('-123456789 days 12:23:56' AS interval))", join=True).split('\n')[3] == '| -123456789 days, 12:23:56 |'

@dbtest
@pytest.mark.parametrize('value', ['10000000', '10000000.0', '10000000000000'])
def test_large_numbers_render_directly(executor, value):
    if False:
        i = 10
        return i + 15
    run(executor, 'create table numbertest(a numeric)')
    run(executor, f'insert into numbertest (a) values ({value})')
    assert value in run(executor, 'select * from numbertest', join=True)

@dbtest
@pytest.mark.parametrize('command', ['di', 'dv', 'ds', 'df', 'dT'])
@pytest.mark.parametrize('verbose', ['', '+'])
@pytest.mark.parametrize('pattern', ['', 'x', '*.*', 'x.y', 'x.*', '*.y'])
def test_describe_special(executor, command, verbose, pattern, pgspecial):
    if False:
        while True:
            i = 10
    sql = '\\{command}{verbose} {pattern}'.format(**locals())
    list(executor.run(sql, pgspecial=pgspecial))

@dbtest
@pytest.mark.parametrize('sql', ['invalid sql', 'SELECT 1; select error;'])
def test_raises_with_no_formatter(executor, sql):
    if False:
        i = 10
        return i + 15
    with pytest.raises(psycopg.ProgrammingError):
        list(executor.run(sql))

@dbtest
def test_on_error_resume(executor, exception_formatter):
    if False:
        i = 10
        return i + 15
    sql = 'select 1; error; select 1;'
    result = list(executor.run(sql, on_error_resume=True, exception_formatter=exception_formatter))
    assert len(result) == 3

@dbtest
def test_on_error_stop(executor, exception_formatter):
    if False:
        return 10
    sql = 'select 1; error; select 1;'
    result = list(executor.run(sql, on_error_resume=False, exception_formatter=exception_formatter))
    assert len(result) == 2

@dbtest
def test_nonexistent_function_definition(executor):
    if False:
        while True:
            i = 10
    with pytest.raises(RuntimeError):
        result = executor.view_definition('there_is_no_such_function')

@dbtest
def test_function_definition(executor):
    if False:
        return 10
    run(executor, '\n            CREATE OR REPLACE FUNCTION public.the_number_three()\n            RETURNS int\n            LANGUAGE sql\n            AS $function$\n              select 3;\n            $function$\n    ')
    result = executor.function_definition('the_number_three')

@dbtest
def test_view_definition(executor):
    if False:
        print('Hello World!')
    run(executor, 'create table tbl1 (a text, b numeric)')
    run(executor, 'create view vw1 AS SELECT * FROM tbl1')
    run(executor, 'create materialized view mvw1 AS SELECT * FROM tbl1')
    result = executor.view_definition('vw1')
    assert 'VIEW "public"."vw1" AS' in result
    assert 'FROM tbl1' in result
    result = executor.view_definition('mvw1')
    assert 'MATERIALIZED VIEW' in result

@dbtest
def test_nonexistent_view_definition(executor):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(RuntimeError):
        result = executor.view_definition('there_is_no_such_view')
    with pytest.raises(RuntimeError):
        result = executor.view_definition('mvw1')

@dbtest
def test_short_host(executor):
    if False:
        print('Hello World!')
    with patch.object(executor, 'host', 'localhost'):
        assert executor.short_host == 'localhost'
    with patch.object(executor, 'host', 'localhost.example.org'):
        assert executor.short_host == 'localhost'
    with patch.object(executor, 'host', 'localhost1.example.org,localhost2.example.org'):
        assert executor.short_host == 'localhost1'

class VirtualCursor:
    """Mock a cursor to virtual database like pgbouncer."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self.protocol_error = False
        self.protocol_message = ''
        self.description = None
        self.status = None
        self.statusmessage = 'Error'

    def execute(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.protocol_error = True
        self.protocol_message = 'Command not supported'

@dbtest
def test_exit_without_active_connection(executor):
    if False:
        print('Hello World!')
    quit_handler = MagicMock()
    pgspecial = PGSpecial()
    pgspecial.register(quit_handler, '\\q', '\\q', 'Quit pgcli.', arg_type=NO_QUERY, case_sensitive=True, aliases=(':q',))
    with patch.object(executor.conn, 'cursor', side_effect=psycopg.InterfaceError("I'm broken!")):
        run(executor, '\\q', pgspecial=pgspecial)
        quit_handler.assert_called_once()
        with pytest.raises(psycopg.InterfaceError):
            run(executor, 'select 1', pgspecial=pgspecial)

@dbtest
def test_virtual_database(executor):
    if False:
        print('Hello World!')
    virtual_connection = MagicMock()
    virtual_connection.cursor.return_value = VirtualCursor()
    with patch.object(executor, 'conn', virtual_connection):
        result = run(executor, 'select 1')
        assert 'Command not supported' in result