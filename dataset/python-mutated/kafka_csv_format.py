import logging
import sys
from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaProducer, FlinkKafkaConsumer
from pyflink.datastream.formats.csv import CsvRowSerializationSchema, CsvRowDeserializationSchema

def write_to_kafka(env):
    if False:
        for i in range(10):
            print('nop')
    type_info = Types.ROW([Types.INT(), Types.STRING()])
    ds = env.from_collection([(1, 'hi'), (2, 'hello'), (3, 'hi'), (4, 'hello'), (5, 'hi'), (6, 'hello'), (6, 'hello')], type_info=type_info)
    serialization_schema = CsvRowSerializationSchema.Builder(type_info).build()
    kafka_producer = FlinkKafkaProducer(topic='test_csv_topic', serialization_schema=serialization_schema, producer_config={'bootstrap.servers': 'localhost:9092', 'group.id': 'test_group'})
    ds.add_sink(kafka_producer)
    env.execute()

def read_from_kafka(env):
    if False:
        while True:
            i = 10
    type_info = Types.ROW([Types.INT(), Types.STRING()])
    deserialization_schema = CsvRowDeserializationSchema.Builder(type_info).build()
    kafka_consumer = FlinkKafkaConsumer(topics='test_csv_topic', deserialization_schema=deserialization_schema, properties={'bootstrap.servers': 'localhost:9092', 'group.id': 'test_group_1'})
    kafka_consumer.set_start_from_earliest()
    env.add_source(kafka_consumer).print()
    env.execute()
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    env = StreamExecutionEnvironment.get_execution_environment()
    env.add_jars('file:///path/to/flink-sql-connector-kafka-1.15.0.jar')
    print('start writing data to kafka')
    write_to_kafka(env)
    print('start reading data from kafka')
    read_from_kafka(env)