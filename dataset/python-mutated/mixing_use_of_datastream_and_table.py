import logging
import sys
from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import DataTypes, TableDescriptor, Schema, StreamTableEnvironment
from pyflink.table.expressions import col
from pyflink.table.udf import udf

def mixing_use_of_datastream_and_table():
    if False:
        i = 10
        return i + 15
    env = StreamExecutionEnvironment.get_execution_environment()
    t_env = StreamTableEnvironment.create(stream_execution_environment=env)
    t_env.create_temporary_table('source', TableDescriptor.for_connector('datagen').schema(Schema.new_builder().column('id', DataTypes.BIGINT()).column('data', DataTypes.STRING()).build()).option('number-of-rows', '10').build())
    t_env.create_temporary_table('sink', TableDescriptor.for_connector('print').schema(Schema.new_builder().column('a', DataTypes.BIGINT()).build()).build())

    @udf(result_type=DataTypes.BIGINT())
    def length(data):
        if False:
            for i in range(10):
                print('nop')
        return len(data)
    table = t_env.from_path('source')
    table = table.select(col('id'), length(col('data')))
    ds = t_env.to_data_stream(table)
    ds = ds.map(lambda i: i[0] + i[1], output_type=Types.LONG())
    table = t_env.from_data_stream(ds, Schema.new_builder().column('f0', DataTypes.BIGINT()).build())
    table.execute_insert('sink').wait()
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    mixing_use_of_datastream_and_table()