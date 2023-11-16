from pgcli.packages.formatter.sqlformatter import escape_for_sql_statement
from cli_helpers.tabular_output import TabularOutputFormatter
from pgcli.packages.formatter.sqlformatter import adapter, register_new_formatter

def test_escape_for_sql_statement_bytes():
    if False:
        while True:
            i = 10
    bts = b'837124ab3e8dc0f'
    escaped_bytes = escape_for_sql_statement(bts)
    assert escaped_bytes == "X'383337313234616233653864633066'"

def test_escape_for_sql_statement_number():
    if False:
        for i in range(10):
            print('nop')
    num = 2981
    escaped_bytes = escape_for_sql_statement(num)
    assert escaped_bytes == "'2981'"

def test_escape_for_sql_statement_str():
    if False:
        for i in range(10):
            print('nop')
    example_str = 'example str'
    escaped_bytes = escape_for_sql_statement(example_str)
    assert escaped_bytes == "'example str'"

def test_output_sql_insert():
    if False:
        while True:
            i = 10
    global formatter
    formatter = TabularOutputFormatter
    register_new_formatter(formatter)
    data = [[1, 'Jackson', 'jackson_test@gmail.com', '132454789', None, '2022-09-09 19:44:32.712343+08', '2022-09-09 19:44:32.712343+08']]
    header = ['id', 'name', 'email', 'phone', 'description', 'created_at', 'updated_at']
    table_format = 'sql-insert'
    kwargs = {'column_types': [int, str, str, str, str, str, str], 'sep_title': 'RECORD {n}', 'sep_character': '-', 'sep_length': (1, 25), 'missing_value': '<null>', 'integer_format': '', 'float_format': '', 'disable_numparse': True, 'preserve_whitespace': True, 'max_field_width': 500}
    formatter.query = 'SELECT * FROM "user";'
    output = adapter(data, header, table_format=table_format, **kwargs)
    output_list = [l for l in output]
    expected = ['INSERT INTO "user" ("id", "name", "email", "phone", "description", "created_at", "updated_at") VALUES', "  ('1', 'Jackson', 'jackson_test@gmail.com', '132454789', NULL, " + "'2022-09-09 19:44:32.712343+08', '2022-09-09 19:44:32.712343+08')", ';']
    assert expected == output_list

def test_output_sql_update():
    if False:
        print('Hello World!')
    global formatter
    formatter = TabularOutputFormatter
    register_new_formatter(formatter)
    data = [[1, 'Jackson', 'jackson_test@gmail.com', '132454789', '', '2022-09-09 19:44:32.712343+08', '2022-09-09 19:44:32.712343+08']]
    header = ['id', 'name', 'email', 'phone', 'description', 'created_at', 'updated_at']
    table_format = 'sql-update'
    kwargs = {'column_types': [int, str, str, str, str, str, str], 'sep_title': 'RECORD {n}', 'sep_character': '-', 'sep_length': (1, 25), 'missing_value': '<null>', 'integer_format': '', 'float_format': '', 'disable_numparse': True, 'preserve_whitespace': True, 'max_field_width': 500}
    formatter.query = 'SELECT * FROM "user";'
    output = adapter(data, header, table_format=table_format, **kwargs)
    output_list = [l for l in output]
    print(output_list)
    expected = ['UPDATE "user" SET', '  "name" = \'Jackson\'', ', "email" = \'jackson_test@gmail.com\'', ', "phone" = \'132454789\'', ', "description" = \'\'', ', "created_at" = \'2022-09-09 19:44:32.712343+08\'', ', "updated_at" = \'2022-09-09 19:44:32.712343+08\'', 'WHERE "id" = \'1\';']
    assert expected == output_list