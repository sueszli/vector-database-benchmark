import itertools
import os
import re
from collections import namedtuple
from mkdocs.structure.pages import markdown
from pyspark.java_gateway import launch_gateway
ExpressionInfo = namedtuple('ExpressionInfo', 'name usage examples group')
groups = {'agg_funcs', 'array_funcs', 'datetime_funcs', 'json_funcs', 'map_funcs', 'window_funcs', 'math_funcs', 'conditional_funcs', 'generator_funcs', 'predicate_funcs', 'string_funcs', 'misc_funcs', 'bitwise_funcs', 'conversion_funcs', 'csv_funcs', 'xml_funcs', 'lambda_funcs', 'collection_funcs', 'url_funcs', 'hash_funcs', 'struct_funcs'}

def _list_grouped_function_infos(jvm):
    if False:
        print('Hello World!')
    '\n    Returns a list of function information grouped by each group value via JVM.\n    Sorts wrapped expression infos in each group by name and returns them.\n    '
    jinfos = jvm.org.apache.spark.sql.api.python.PythonSQLUtils.listBuiltinFunctionInfos()
    infos = []
    for jinfo in filter(lambda x: x.getGroup() in groups, jinfos):
        name = jinfo.getName()
        if name == 'raise_error':
            continue
        group = jinfo.getGroup()
        if group == 'lambda_funcs':
            group = 'collection_funcs'
        usage = jinfo.getUsage()
        usage = usage.replace('_FUNC_', name) if usage is not None else usage
        infos.append(ExpressionInfo(name=name, usage=usage, examples=jinfo.getExamples().replace('_FUNC_', name), group=group))
    grouped_infos = itertools.groupby(sorted(infos, key=lambda x: x.group), key=lambda x: x.group)
    return [(k, sorted(g, key=lambda x: x.name)) for (k, g) in grouped_infos]

def _make_pretty_usage(infos):
    if False:
        for i in range(10):
            print('nop')
    '\n    Makes the usage description pretty and returns a formatted string.\n\n    Expected input:\n\n        func(*) - ...\n\n        func(expr[, expr...]) - ...\n\n    Expected output:\n    <table class="table">\n      <thead>\n        <tr>\n          <th style="width:25%">Function</th>\n          <th>Description</th>\n        </tr>\n      </thead>\n      <tbody>\n        <tr>\n          <td>func(*)</td>\n          <td>...</td>\n        </tr>\n        <tr>\n          <td>func(expr[, expr...])</td>\n          <td>...</td>\n        </tr>\n      </tbody>\n      ...\n    </table>\n\n    '
    result = []
    result.append('<table class="table">')
    result.append('  <thead>')
    result.append('    <tr>')
    result.append('      <th style="width:25%">Function</th>')
    result.append('      <th>Description</th>')
    result.append('    </tr>')
    result.append('  </thead>')
    result.append('  <tbody>')
    for info in infos:
        func_name = info.name
        if info.name == '*' or info.name == '+':
            func_name = '\\' + func_name
        elif info.name == 'when':
            func_name = 'CASE WHEN'
        usages = iter(re.split('(.*%s.*) - ' % func_name, info.usage.strip())[1:])
        for (sig, description) in zip(usages, usages):
            result.append('    <tr>')
            result.append('      <td>%s</td>' % sig)
            result.append('      <td>%s</td>' % description.strip())
            result.append('    </tr>')
    result.append('  </tbody>')
    result.append('</table>\n')
    return '\n'.join(result)

def _make_pretty_examples(jspark, infos):
    if False:
        print('Hello World!')
    '\n    Makes the examples description pretty and returns a formatted string if `infos`\n    has any `examples` starting with the example prefix. Otherwise, returns None.\n\n    Expected input:\n\n        Examples:\n          > SELECT func(col)...;\n           ...\n          > SELECT func(col)...;\n           ...\n\n    Expected output:\n    <div class="codehilite"><pre><span></span>\n      <span class="c1">-- func</span>\n      <span class="k">SELECT</span>\n      ...\n    </pre></div>\n    ```\n\n    '
    pretty_output = ''
    for info in infos:
        if info.examples.startswith('\n    Examples:'):
            output = []
            output.append('-- %s' % info.name)
            query_examples = filter(lambda x: x.startswith('      > '), info.examples.split('\n'))
            for query_example in query_examples:
                query = query_example.lstrip('      > ')
                print('    %s' % query)
                query_output = jspark.sql(query).showString(20, 20, False)
                output.append(query)
                output.append(query_output)
            pretty_output += '\n' + '\n'.join(output)
    if pretty_output != '':
        return markdown.markdown('```sql%s```' % pretty_output, extensions=['codehilite', 'fenced_code'])

def generate_functions_table_html(jvm, html_output_dir):
    if False:
        i = 10
        return i + 15
    '\n    Generates a HTML file after listing the function information. The output file\n    is created under `html_output_dir`.\n\n    Expected output:\n\n    <table class="table">\n      <thead>\n        <tr>\n          <th style="width:25%">Function</th>\n          <th>Description</th>\n        </tr>\n      </thead>\n      <tbody>\n        <tr>\n          <td>func(*)</td>\n          <td>...</td>\n        </tr>\n        <tr>\n          <td>func(expr[, expr...])</td>\n          <td>...</td>\n        </tr>\n      </tbody>\n      ...\n    </table>\n\n    '
    for (key, infos) in _list_grouped_function_infos(jvm):
        function_table = _make_pretty_usage(infos)
        key = key.replace('_', '-')
        with open('%s/generated-%s-table.html' % (html_output_dir, key), 'w') as table_html:
            table_html.write(function_table)

def generate_functions_examples_html(jvm, jspark, html_output_dir):
    if False:
        i = 10
        return i + 15
    '\n    Generates a HTML file after listing and executing the function information.\n    The output file is created under `html_output_dir`.\n\n    Expected output:\n\n    <div class="codehilite"><pre><span></span>\n      <span class="c1">-- func</span>\n      <span class="k">SELECT</span>\n      ...\n    </pre></div>\n\n    '
    print('Running SQL examples to generate formatted output.')
    for (key, infos) in _list_grouped_function_infos(jvm):
        examples = _make_pretty_examples(jspark, infos)
        key = key.replace('_', '-')
        if examples is not None:
            with open('%s/generated-%s-examples.html' % (html_output_dir, key), 'w') as examples_html:
                examples_html.write(examples)
if __name__ == '__main__':
    jvm = launch_gateway().jvm
    jspark = jvm.org.apache.spark.sql.SparkSession.builder().getOrCreate()
    jspark.sparkContext().setLogLevel('ERROR')
    spark_root_dir = os.path.dirname(os.path.dirname(__file__))
    html_output_dir = os.path.join(spark_root_dir, 'docs')
    generate_functions_table_html(jvm, html_output_dir)
    generate_functions_examples_html(jvm, jspark, html_output_dir)