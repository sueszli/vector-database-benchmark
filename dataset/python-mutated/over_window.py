import logging
import sys
from pyflink.common.time import Instant
from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import DataTypes, TableDescriptor, Schema, StreamTableEnvironment
from pyflink.table.expressions import col, row_interval, CURRENT_ROW
from pyflink.table.window import Over

def tumble_window_demo():
    if False:
        return 10
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)
    t_env = StreamTableEnvironment.create(stream_execution_environment=env)
    ds = env.from_collection(collection=[(Instant.of_epoch_milli(1000), 'Alice', 110.1), (Instant.of_epoch_milli(4000), 'Bob', 30.2), (Instant.of_epoch_milli(3000), 'Alice', 20.0), (Instant.of_epoch_milli(2000), 'Bob', 53.1), (Instant.of_epoch_milli(5000), 'Alice', 13.1), (Instant.of_epoch_milli(3000), 'Bob', 3.1), (Instant.of_epoch_milli(7000), 'Bob', 16.1), (Instant.of_epoch_milli(10000), 'Alice', 20.1)], type_info=Types.ROW([Types.INSTANT(), Types.STRING(), Types.FLOAT()]))
    table = t_env.from_data_stream(ds, Schema.new_builder().column_by_expression('ts', 'CAST(f0 AS TIMESTAMP(3))').column('f1', DataTypes.STRING()).column('f2', DataTypes.FLOAT()).watermark('ts', "ts - INTERVAL '3' SECOND").build()).alias('ts', 'name', 'price')
    t_env.create_temporary_table('sink', TableDescriptor.for_connector('print').schema(Schema.new_builder().column('name', DataTypes.STRING()).column('total_price', DataTypes.FLOAT()).build()).build())
    table = table.over_window(Over.partition_by(col('name')).order_by(col('ts')).preceding(row_interval(2)).following(CURRENT_ROW).alias('w')).select(col('name'), col('price').max.over(col('w')))
    table.execute_insert('sink').wait()
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    tumble_window_demo()