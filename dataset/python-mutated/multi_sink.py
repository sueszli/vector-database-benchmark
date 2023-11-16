import logging
import sys
from pyflink.table import EnvironmentSettings, TableEnvironment, DataTypes
from pyflink.table.udf import udf

def multi_sink():
    if False:
        for i in range(10):
            print('nop')
    t_env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())
    table = t_env.from_elements(elements=[(1, 'Hello'), (2, 'World'), (3, 'Flink'), (4, 'PyFlink')], schema=['id', 'data'])
    t_env.execute_sql("\n        CREATE TABLE first_sink (\n            id BIGINT,\n            data VARCHAR\n        ) WITH (\n            'connector' = 'print'\n        )\n    ")
    t_env.execute_sql("\n        CREATE TABLE second_sink (\n            id BIGINT,\n            data VARCHAR\n        ) WITH (\n            'connector' = 'print'\n        )\n    ")
    statement_set = t_env.create_statement_set()
    statement_set.add_insert_sql('INSERT INTO first_sink SELECT * FROM %s WHERE id <= 3' % table)

    @udf(result_type=DataTypes.BOOLEAN())
    def contains_flink(data):
        if False:
            print('Hello World!')
        return 'Flink' in data
    second_table = table.where(contains_flink(table.data))
    statement_set.add_insert('second_sink', second_table)
    statement_set.execute().wait()
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    multi_sink()