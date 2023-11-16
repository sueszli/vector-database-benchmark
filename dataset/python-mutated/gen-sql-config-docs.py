import os
import re
from collections import namedtuple
from textwrap import dedent
from mkdocs.structure.pages import markdown
from pyspark.java_gateway import launch_gateway
SQLConfEntry = namedtuple('SQLConfEntry', ['name', 'default', 'description', 'version'])

def get_sql_configs(jvm, group):
    if False:
        while True:
            i = 10
    if group == 'static':
        config_set = jvm.org.apache.spark.sql.api.python.PythonSQLUtils.listStaticSQLConfigs()
    else:
        config_set = jvm.org.apache.spark.sql.api.python.PythonSQLUtils.listRuntimeSQLConfigs()
    sql_configs = [SQLConfEntry(name=_sql_config._1(), default=_sql_config._2(), description=_sql_config._3(), version=_sql_config._4()) for _sql_config in config_set]
    return sql_configs

def generate_sql_configs_table_html(sql_configs, path):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates an HTML table at `path` that lists all public SQL\n    configuration options.\n\n    The table will look something like this:\n\n    ```html\n    <table class="table">\n    <tr><th>Property Name</th><th>Default</th><th>Meaning</th><th>Since Version</th></tr>\n\n    <tr>\n        <td><code>spark.sql.adaptive.enabled</code></td>\n        <td>true</td>\n        <td><p>When true, enable adaptive query execution.</p></td>\n        <td>1.6.0</td>\n    </tr>\n\n    ...\n\n    </table>\n    ```\n    '
    value_reference_pattern = re.compile('^<value of (\\S*)>$')
    with open(path, 'w') as f:
        f.write(dedent('\n            <table class="table">\n            <tr><th>Property Name</th><th>Default</th><th>Meaning</th><th>Since Version</th></tr>\n            '))
        for config in sorted(sql_configs, key=lambda x: x.name):
            if config.name == 'spark.sql.session.timeZone':
                default = '(value of local timezone)'
            elif config.name == 'spark.sql.warehouse.dir':
                default = '(value of <code>$PWD/spark-warehouse</code>)'
            elif config.default == '<undefined>':
                default = '(none)'
            elif config.default.startswith('<value of '):
                referenced_config_name = value_reference_pattern.match(config.default).group(1)
                default = '(value of <code>{}</code>)'.format(referenced_config_name)
            else:
                default = config.default
            if default.startswith('<'):
                raise RuntimeError("Unhandled reference in SQL config docs. Config '{name}' has default '{default}' that looks like an HTML tag.".format(name=config.name, default=config.default))
            f.write(dedent('\n                <tr>\n                    <td><code>{name}</code></td>\n                    <td>{default}</td>\n                    <td>{description}</td>\n                    <td>{version}</td>\n                </tr>\n                '.format(name=config.name, default=default, description=markdown.markdown(config.description), version=config.version)))
        f.write('</table>\n')
if __name__ == '__main__':
    jvm = launch_gateway().jvm
    docs_root_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    sql_configs = get_sql_configs(jvm, 'runtime')
    sql_configs_table_path = os.path.join(docs_root_dir, 'generated-runtime-sql-config-table.html')
    generate_sql_configs_table_html(sql_configs, path=sql_configs_table_path)
    sql_configs = get_sql_configs(jvm, 'static')
    sql_configs_table_path = os.path.join(docs_root_dir, 'generated-static-sql-config-table.html')
    generate_sql_configs_table_html(sql_configs, path=sql_configs_table_path)