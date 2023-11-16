import logging
import sys
from pyflink.common import Types
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import FlinkKafkaProducer, FlinkKafkaConsumer
from pyflink.datastream.formats.avro import AvroRowSerializationSchema, AvroRowDeserializationSchema

def write_to_kafka(env):
    if False:
        while True:
            i = 10
    ds = env.from_collection([(1, 'hi'), (2, 'hello'), (3, 'hi'), (4, 'hello'), (5, 'hi'), (6, 'hello'), (6, 'hello')], type_info=Types.ROW([Types.INT(), Types.STRING()]))
    serialization_schema = AvroRowSerializationSchema(avro_schema_string='\n            {\n                "type": "record",\n                "name": "TestRecord",\n                "fields": [\n                    {"name": "id", "type": "int"},\n                    {"name": "name", "type": "string"}\n                ]\n            }')
    kafka_producer = FlinkKafkaProducer(topic='test_avro_topic', serialization_schema=serialization_schema, producer_config={'bootstrap.servers': 'localhost:9092', 'group.id': 'test_group'})
    ds.add_sink(kafka_producer)
    env.execute()

def read_from_kafka(env):
    if False:
        i = 10
        return i + 15
    deserialization_schema = AvroRowDeserializationSchema(avro_schema_string='\n            {\n                "type": "record",\n                "name": "TestRecord",\n                "fields": [\n                    {"name": "id", "type": "int"},\n                    {"name": "name", "type": "string"}\n                ]\n            }')
    kafka_consumer = FlinkKafkaConsumer(topics='test_avro_topic', deserialization_schema=deserialization_schema, properties={'bootstrap.servers': 'localhost:9092', 'group.id': 'test_group_1'})
    kafka_consumer.set_start_from_earliest()
    env.add_source(kafka_consumer).print()
    env.execute()
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')
    env = StreamExecutionEnvironment.get_execution_environment()
    env.add_jars('file:///path/to/flink-sql-avro-1.15.0.jar', 'file:///path/to/flink-sql-connector-kafka-1.15.0.jar')
    print('start writing data to kafka')
    write_to_kafka(env)
    print('start reading data from kafka')
    read_from_kafka(env)