import logging
import sys
from pyflink.table import EnvironmentSettings, TableEnvironment, DataTypes, TableDescriptor, Schema
from pyflink.table.expressions import col

def process_json_data():
    if False:
        print('Hello World!')
    t_env = TableEnvironment.create(EnvironmentSettings.in_streaming_mode())
    table = t_env.from_elements(elements=[(1, '{"name": "Flink", "tel": 123, "addr": {"country": "Germany", "city": "Berlin"}}'), (2, '{"name": "hello", "tel": 135, "addr": {"country": "China", "city": "Shanghai"}}'), (3, '{"name": "world", "tel": 124, "addr": {"country": "USA", "city": "NewYork"}}'), (4, '{"name": "PyFlink", "tel": 32, "addr": {"country": "China", "city": "Hangzhou"}}')], schema=['id', 'data'])
    t_env.create_temporary_table('sink', TableDescriptor.for_connector('print').schema(Schema.new_builder().column('id', DataTypes.BIGINT()).column('data', DataTypes.STRING()).build()).build())
    table = table.select(col('id'), col('data').json_value('$.addr.country', DataTypes.STRING()))
    table.execute_insert('sink').wait()
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    process_json_data()