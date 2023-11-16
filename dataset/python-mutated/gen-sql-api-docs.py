import os
from collections import namedtuple
from pyspark.java_gateway import launch_gateway
ExpressionInfo = namedtuple('ExpressionInfo', 'className name usage arguments examples note since deprecated')
_virtual_operator_infos = [ExpressionInfo(className='', name='!=', usage='expr1 != expr2 - Returns true if `expr1` is not equal to `expr2`, ' + 'or false otherwise.', arguments='\n    Arguments:\n      ' + '* expr1, expr2 - the two expressions must be same type or can be casted to\n                       a common type, and must be a type that can be used in equality comparison.\n                       Map type is not supported. For complex types such array/struct,\n                       the data types of fields must be orderable.', examples='\n    Examples:\n      ' + '> SELECT 1 != 2;\n      ' + ' true\n      ' + "> SELECT 1 != '2';\n      " + ' true\n      ' + '> SELECT true != NULL;\n      ' + ' NULL\n      ' + '> SELECT NULL != NULL;\n      ' + ' NULL', note='', since='1.0.0', deprecated=''), ExpressionInfo(className='', name='<>', usage='expr1 != expr2 - Returns true if `expr1` is not equal to `expr2`, ' + 'or false otherwise.', arguments='\n    Arguments:\n      ' + '* expr1, expr2 - the two expressions must be same type or can be casted to\n                       a common type, and must be a type that can be used in equality comparison.\n                       Map type is not supported. For complex types such array/struct,\n                       the data types of fields must be orderable.', examples='\n    Examples:\n      ' + '> SELECT 1 != 2;\n      ' + ' true\n      ' + "> SELECT 1 != '2';\n      " + ' true\n      ' + '> SELECT true != NULL;\n      ' + ' NULL\n      ' + '> SELECT NULL != NULL;\n      ' + ' NULL', note='', since='1.0.0', deprecated=''), ExpressionInfo(className='', name='between', usage='expr1 [NOT] BETWEEN expr2 AND expr3 - ' + 'evaluate if `expr1` is [not] in between `expr2` and `expr3`.', arguments='', examples='\n    Examples:\n      ' + '> SELECT col1 FROM VALUES 1, 3, 5, 7 WHERE col1 BETWEEN 2 AND 5;\n      ' + ' 3\n      ' + ' 5', note='', since='1.0.0', deprecated=''), ExpressionInfo(className='', name='case', usage='CASE expr1 WHEN expr2 THEN expr3 ' + '[WHEN expr4 THEN expr5]* [ELSE expr6] END - ' + 'When `expr1` = `expr2`, returns `expr3`; ' + 'when `expr1` = `expr4`, return `expr5`; else return `expr6`.', arguments='\n    Arguments:\n      ' + '* expr1 - the expression which is one operand of comparison.\n      ' + '* expr2, expr4 - the expressions each of which is the other ' + '  operand of comparison.\n      ' + '* expr3, expr5, expr6 - the branch value expressions and else value expression' + '  should all be same type or coercible to a common type.', examples='\n    Examples:\n      ' + "> SELECT CASE col1 WHEN 1 THEN 'one' " + "WHEN 2 THEN 'two' ELSE '?' END FROM VALUES 1, 2, 3;\n      " + ' one\n      ' + ' two\n      ' + ' ?\n      ' + "> SELECT CASE col1 WHEN 1 THEN 'one' " + "WHEN 2 THEN 'two' END FROM VALUES 1, 2, 3;\n      " + ' one\n      ' + ' two\n      ' + ' NULL', note='', since='1.0.1', deprecated=''), ExpressionInfo(className='', name='||', usage='expr1 || expr2 - Returns the concatenation of `expr1` and `expr2`.', arguments='', examples='\n    Examples:\n      ' + "> SELECT 'Spark' || 'SQL';\n      " + ' SparkSQL\n      ' + '> SELECT array(1, 2, 3) || array(4, 5) || array(6);\n      ' + ' [1,2,3,4,5,6]', note='\n    || for arrays is available since 2.4.0.\n', since='2.3.0', deprecated='')]

def _list_function_infos(jvm):
    if False:
        while True:
            i = 10
    '\n    Returns a list of function information via JVM. Sorts wrapped expression infos by name\n    and returns them.\n    '
    jinfos = jvm.org.apache.spark.sql.api.python.PythonSQLUtils.listBuiltinFunctionInfos()
    infos = _virtual_operator_infos
    for jinfo in jinfos:
        name = jinfo.getName()
        usage = jinfo.getUsage()
        usage = usage.replace('_FUNC_', name) if usage is not None else usage
        infos.append(ExpressionInfo(className=jinfo.getClassName(), name=name, usage=usage, arguments=jinfo.getArguments().replace('_FUNC_', name), examples=jinfo.getExamples().replace('_FUNC_', name), note=jinfo.getNote().replace('_FUNC_', name), since=jinfo.getSince(), deprecated=jinfo.getDeprecated()))
    return sorted(infos, key=lambda i: i.name)

def _make_pretty_usage(usage):
    if False:
        while True:
            i = 10
    '\n    Makes the usage description pretty and returns a formatted string if `usage`\n    is not an empty string. Otherwise, returns None.\n    '
    if usage is not None and usage.strip() != '':
        usage = '\n'.join(map(lambda u: u.strip(), usage.split('\n')))
        return '%s\n\n' % usage

def _make_pretty_arguments(arguments):
    if False:
        print('Hello World!')
    '\n    Makes the arguments description pretty and returns a formatted string if `arguments`\n    starts with the argument prefix. Otherwise, returns None.\n\n    Expected input:\n\n        Arguments:\n          * arg0 - ...\n              ...\n          * arg0 - ...\n              ...\n\n    Expected output:\n    **Arguments:**\n\n    * arg0 - ...\n        ...\n    * arg0 - ...\n        ...\n\n    '
    if arguments.startswith('\n    Arguments:'):
        arguments = '\n'.join(map(lambda u: u[6:], arguments.strip().split('\n')[1:]))
        return '**Arguments:**\n\n%s\n\n' % arguments

def _make_pretty_examples(examples):
    if False:
        return 10
    '\n    Makes the examples description pretty and returns a formatted string if `examples`\n    starts with the example prefix. Otherwise, returns None.\n\n    Expected input:\n\n        Examples:\n          > SELECT ...;\n           ...\n          > SELECT ...;\n           ...\n\n    Expected output:\n    **Examples:**\n\n    ```\n    > SELECT ...;\n     ...\n    > SELECT ...;\n     ...\n    ```\n\n    '
    if examples.startswith('\n    Examples:'):
        examples = '\n'.join(map(lambda u: u[6:], examples.strip().split('\n')[1:]))
        return '**Examples:**\n\n```\n%s\n```\n\n' % examples

def _make_pretty_note(note):
    if False:
        print('Hello World!')
    '\n    Makes the note description pretty and returns a formatted string if `note` is not\n    an empty string. Otherwise, returns None.\n\n    Expected input:\n\n        ...\n\n    Expected output:\n    **Note:**\n\n    ...\n\n    '
    if note != '':
        note = '\n'.join(map(lambda n: n[4:], note.split('\n')))
        return '**Note:**\n%s\n' % note

def _make_pretty_deprecated(deprecated):
    if False:
        print('Hello World!')
    '\n    Makes the deprecated description pretty and returns a formatted string if `deprecated`\n    is not an empty string. Otherwise, returns None.\n\n    Expected input:\n\n        ...\n\n    Expected output:\n    **Deprecated:**\n\n    ...\n\n    '
    if deprecated != '':
        deprecated = '\n'.join(map(lambda n: n[4:], deprecated.split('\n')))
        return '**Deprecated:**\n%s\n' % deprecated

def generate_sql_api_markdown(jvm, path):
    if False:
        return 10
    '\n    Generates a markdown file after listing the function information. The output file\n    is created in `path`.\n\n    Expected output:\n    ### NAME\n\n    USAGE\n\n    **Arguments:**\n\n    ARGUMENTS\n\n    **Examples:**\n\n    ```\n    EXAMPLES\n    ```\n\n    **Note:**\n\n    NOTE\n\n    **Since:** SINCE\n\n    **Deprecated:**\n\n    DEPRECATED\n\n    <br/>\n\n    '
    with open(path, 'w') as mdfile:
        mdfile.write('# Built-in Functions\n\n')
        for info in _list_function_infos(jvm):
            name = info.name
            usage = _make_pretty_usage(info.usage)
            arguments = _make_pretty_arguments(info.arguments)
            examples = _make_pretty_examples(info.examples)
            note = _make_pretty_note(info.note)
            since = info.since
            deprecated = _make_pretty_deprecated(info.deprecated)
            mdfile.write('### %s\n\n' % name)
            if usage is not None:
                mdfile.write('%s\n\n' % usage.strip())
            if arguments is not None:
                mdfile.write(arguments)
            if examples is not None:
                mdfile.write(examples)
            if note is not None:
                mdfile.write(note)
            if since is not None and since != '':
                mdfile.write('**Since:** %s\n\n' % since.strip())
            if deprecated is not None:
                mdfile.write(deprecated)
            mdfile.write('<br/>\n\n')
if __name__ == '__main__':
    jvm = launch_gateway().jvm
    spark_root_dir = os.path.dirname(os.path.dirname(__file__))
    markdown_file_path = os.path.join(spark_root_dir, 'sql/docs/index.md')
    generate_sql_api_markdown(jvm, markdown_file_path)