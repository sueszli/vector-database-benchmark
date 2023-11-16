from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import kafka_errors
import traceback
import json
import sys

def producer_demo():
    if False:
        return 10
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'], key_serializer=lambda k: json.dumps(k).encode(), value_serializer=lambda v: json.dumps(v).encode())
    for i in range(0, 3):
        future = producer.send('serving_stream', key='test', value=str(i), partition=0)
        print('send {}'.format(str(i)))
        try:
            future.get(timeout=10)
        except kafka_errors:
            traceback.format_exc()
    producer.close()

def consumer_demo():
    if False:
        for i in range(10):
            print('nop')
    consumer = KafkaConsumer('cluster-serving_serving_stream', bootstrap_servers=['localhost:9092'])
    for message in consumer:
        print('receive, key: {}, value: {}'.format(json.loads(message.key.decode()), json.loads(message.value.decode())))
if __name__ == '__main__':
    globals()[sys.argv[1]]()